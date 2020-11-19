import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torchnlp.datasets.imdb import imdb_dataset

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForSequenceClassification_label,
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
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers.data.processors.utils import InputExample

from sklearn.datasets import fetch_20newsgroups


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
    "bert": (BertConfig, BertForSequenceClassification_label, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

model_name_or_path = 'bert-base-uncased'
config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path,
    do_lower_case=True,
    cache_dir=None,
)

train_dataset = imdb_dataset('/home/ray/transformers/imdb', train=True)
examples = []
lengths = []
for i, inst in enumerate(train_dataset):
    examples.append(InputExample(guid='', text_a=inst['text'], label=inst['sentiment']))
    lengths.append(len(tokenizer.tokenize(inst['text'])))
    if i%1000==0:
        print('{}/{}: {}'.format(i,len(train_dataset), np.sum(np.array(lengths)>=512)))
pass

# train_dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# data, target = train_dataset.data, train_dataset.target
# assert len(data) == len(target)
# examples = []
# lengths = []
# for i, inst in enumerate(data):
#     examples.append(InputExample(guid='', text_a=data[i], label=target[i]))
#     lengths.append(len(tokenizer.tokenize(data[i])))
#     if i%1000==0:
#         print('{}/{}: {}'.format(i,len(train_dataset), np.sum(np.array(lengths)>=256)))

features = convert_examples_to_features(
    examples,
    tokenizer,
    label_list=['pos', 'neg'],#list(range(20)),
    max_length=512,
    output_mode="classification",
    pad_on_left=False,  # pad on the left for xlnet
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
    pad_token_segment_id=0,
)
pass

