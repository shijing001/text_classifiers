# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from typing import Dict, List, Tuple

import torch.nn.functional as F

from lamb import Lamb

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForDiscriminator,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)#args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # # 10% of the time, we replace masked input tokens with random word
    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    # inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model_d, model_g, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_d_grouped_parameters = [
        {
            "params": [p for n, p in model_d.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_d.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    # optimizer_d = AdamW(optimizer_d_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer_d = Lamb(optimizer_d_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-6)
    scheduler_d = get_linear_schedule_with_warmup(
        optimizer_d, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    optimizer_g_grouped_parameters = [
        {
            "params": [p for n, p in model_g.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_g.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    # optimizer_g = AdamW(optimizer_g_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer_g = Lamb(optimizer_g_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-6)
    scheduler_g = get_linear_schedule_with_warmup(
        optimizer_g, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer_d.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler_d.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer_d.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer_d.pt")))
        scheduler_d.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler_d.pt")))
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer_g.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler_g.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer_g.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer_g.pt")))
        scheduler_g.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler_g.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_d, optimizer_d = amp.initialize(model_d, optimizer_d, opt_level=args.fp16_opt_level)
        model_g, optimizer_g = amp.initialize(model_g, optimizer_g, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_d = torch.nn.DataParallel(model_d)
        model_g = torch.nn.DataParallel(model_g)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_d = torch.nn.parallel.DistributedDataParallel(
            model_d, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
        model_g = torch.nn.parallel.DistributedDataParallel(
            model_g, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    model_to_resize_d = model_d.module if hasattr(model_d, "module") else model_d  # Take care of distributed/parallel training
    # model_to_resize_d.resize_token_embeddings(len(tokenizer))
    model_to_resize_g = model_g.module if hasattr(model_g, "module") else model_g  # Take care of distributed/parallel training
    # model_to_resize_g.resize_token_embeddings(len(tokenizer))

    # model_to_resize_d.bert.embeddings = model_to_resize_g.bert.embeddings

    tr_loss, logging_loss = 0.0, 0.0
    tr_loss_d, logging_loss_d = 0.0, 0.0
    tr_loss_g, logging_loss_g = 0.0, 0.0
    model_d.zero_grad()
    model_g.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model_d.train()
            model_g.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            # outputs = model(**inputs)
            # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            masked_input_ids, mask_labels = mask_tokens(inputs['input_ids'], tokenizer, args)
            outputs_g = model_g(input_ids=masked_input_ids.to(args.device),
                                masked_lm_labels=mask_labels.to(args.device),
                                attention_mask=inputs['attention_mask'].to(args.device),
                                token_type_ids=inputs['token_type_ids'].to(args.device))
            masked_lm_loss, prediction_scores_g = outputs_g[0], outputs_g[1]

            prediction_g = prediction_scores_g.max(dim=-1)[1].cpu()
            acc_g = (prediction_g[mask_labels >= 0] == mask_labels[mask_labels >= 0]).float().mean().item()

            prediction_probs_g = F.softmax(prediction_scores_g, dim=-1).cpu()
            bsz, seq_len, vocab_size = prediction_probs_g.size()
            prediction_samples_g = torch.multinomial(prediction_probs_g.view(-1,vocab_size),
                                                     num_samples=1)
            prediction_samples_g = prediction_samples_g.view(bsz, seq_len)
            input_ids_replace = inputs['input_ids'].clone()
            input_ids_replace[mask_labels>=0] = prediction_samples_g[mask_labels>=0]
            labels_d = input_ids_replace.eq(inputs['input_ids']).long()

            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs['input_ids'].tolist()
            ]
            labels_d.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=-100)
            padding_mask = inputs['input_ids'].eq(tokenizer.pad_token_id)
            labels_d.masked_fill_(padding_mask, value=-100)

            labels_d_ones = labels_d[labels_d>=0].float().mean().item()
            acc_replace = 1- ( (labels_d==0).sum().float() / (mask_labels>=0).sum().float() ).item()

            outputs_d = model_d(input_ids=input_ids_replace.to(args.device),
                                attention_mask=inputs['attention_mask'].to(args.device),
                                token_type_ids=inputs['token_type_ids'].to(args.device),
                                labels=labels_d.to(args.device))
            loss_d, prediction_scores_d = outputs_d[0], outputs_d[1]
            prediction_d = prediction_scores_d.max(dim=-1)[1].cpu()
            acc_d = (prediction_d[labels_d>=0] == labels_d[labels_d>=0]).float().mean().item()
            acc_d_0 = (prediction_d[labels_d==0] == labels_d[labels_d==0]).float().mean().item()
            acc_d_1 = (prediction_d[labels_d==1] == labels_d[labels_d==1]).float().mean().item()



            if args.n_gpu > 1:
                loss_d = loss_d.mean()  # mean() to average on multi-gpu parallel training
                masked_lm_loss = masked_lm_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss_d = loss_d / args.gradient_accumulation_steps
                masked_lm_loss = masked_lm_loss / args.gradient_accumulation_steps

            lambd = 50
            loss = loss_d * lambd + masked_lm_loss
            if args.fp16:
                loss_d = loss_d * lambd
                with amp.scale_loss(loss_d, optimizer_d) as scaled_loss_d:
                    scaled_loss_d.backward()
                with amp.scale_loss(masked_lm_loss, optimizer_g) as scaled_loss_g:
                    scaled_loss_g.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_loss_d += loss_d.item()
            tr_loss_g += masked_lm_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_d), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_g), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_d.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model_g.parameters(), args.max_grad_norm)

                optimizer_d.step()
                scheduler_d.step()  # Update learning rate schedule
                model_d.zero_grad()
                optimizer_g.step()
                scheduler_g.step()  # Update learning rate schedule
                model_g.zero_grad()

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    # if (
                    #     args.local_rank == -1 and args.evaluate_during_training
                    # ):  # Only evaluate when single GPU otherwise metrics may not average well
                    #     results = evaluate(args, model, tokenizer)
                    #     for key, value in results.items():
                    #         eval_key = "eval_{}".format(key)
                    #         logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    loss_scalar_d = (tr_loss_d - logging_loss_d) / args.logging_steps
                    loss_scalar_g = (tr_loss_g - logging_loss_g) / args.logging_steps
                    learning_rate_scalar_d = scheduler_d.get_lr()[0]
                    learning_rate_scalar_g = scheduler_g.get_lr()[0]
                    logs["learning_rate_d"] = learning_rate_scalar_d
                    logs["learning_rate_g"] = learning_rate_scalar_g
                    logs["loss"] = loss_scalar
                    logs["loss_d"] = loss_scalar_d
                    logs["loss_g"] = loss_scalar_g
                    logs["acc_repalce"] = acc_replace
                    logs["acc_d"] = acc_d
                    logs["acc_d_0"] = acc_d_0
                    logs["acc_d_1"] = acc_d_1
                    logs["acc_g"] = acc_g
                    logs["labels_d_ones"] = labels_d_ones
                    logs["masked_ratio"] = (mask_labels>=0).float().sum().item() / (labels_d>=0).sum().float().item()
                    logging_loss = tr_loss
                    logging_loss_d = tr_loss_d
                    logging_loss_g = tr_loss_g

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                # print(args.save_steps)
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_dir_d = os.path.join(output_dir, "checkpoint-d-{}".format(global_step))
                    output_dir_g = os.path.join(output_dir, "checkpoint-g-{}".format(global_step))
                    if not os.path.exists(output_dir_d):
                        os.makedirs(output_dir_d)
                    if not os.path.exists(output_dir_g):
                        os.makedirs(output_dir_g)
                    model_to_save_d = (
                        model_d.module if hasattr(model_d, "module") else model_d
                    )  # Take care of distributed/parallel training
                    model_to_save_g = (
                        model_g.module if hasattr(model_g, "module") else model_g
                    )  # Take care of distributed/parallel training
                    model_to_save_d.save_pretrained(output_dir_d)
                    model_to_save_g.save_pretrained(output_dir_g)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer_d.state_dict(), os.path.join(output_dir_d, "optimizer_d.pt"))
                    torch.save(scheduler_d.state_dict(), os.path.join(output_dir_d, "scheduler_d.pt"))
                    torch.save(optimizer_g.state_dict(), os.path.join(output_dir_d, "optimizer_g.pt"))
                    torch.save(scheduler_g.state_dict(), os.path.join(output_dir_d, "scheduler_g.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--cache_dir",
    #     default="",
    #     type=str,
    #     help="Where do you want to store the pre-trained models downloaded from s3",
    # )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    # args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    # num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config_class_d, model_class_d, model_class_g, tokenizer_class = BertConfig, BertForDiscriminator, BertForMaskedLM, BertTokenizer
    if args.config_name:
        config_d = config_class_d.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config_d = config_class_d.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config_d = config_class_d()
    config_g = copy.deepcopy(config_d)
    config_g.hidden_size = config_d.hidden_size // 3
    config_g.intermediate_size = config_d.intermediate_size // 3
    config_g.num_attention_heads = config_d.num_attention_heads // 3
    print('Hidden_size: {}, {}'.format(config_g.hidden_size, config_d.hidden_size))
    print('Intermediate_size: {}, {}'.format(config_g.intermediate_size, config_d.intermediate_size))
    print('Num_attention_head: {},{}'.format(config_g.num_attention_heads, config_d.num_attention_heads))
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    weight = torch.FloatTensor([1 / 0.15, 1.0])
    weight.requires_grad = False
    model_d = model_class_d(config=config_d, weight=weight)
    model_g = model_class_g(config=config_g)

    # model_d = model_class_d.from_pretrained(
    #     'tmp/ELECTRA_lamb/checkpoint-1500/checkpoint-d-1500',
    #     from_tf=False,
    #     config=config_d,
    #     cache_dir=args.cache_dir,
    # )
    # model_d.set_weight(weight=weight)
    # model_g = model_class_g.from_pretrained(
    #     'tmp/ELECTRA_lamb/checkpoint-1500/checkpoint-g-1500',
    #     from_tf=False,
    #     config=config_g,
    #     cache_dir=args.cache_dir,
    # )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_d.to(args.device)
    model_g.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model_d, model_g, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = (
    #         model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
    #
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    #
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    #     model.to(args.device)

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #         )
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
    #
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=prefix)
    #         result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    #         results.update(result)

    # return results


if __name__ == "__main__":
    main()
