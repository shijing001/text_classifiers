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
from sklearn.datasets import fetch_20newsgroups
from transformers.data.processors.utils import InputExample

from transformers import (
    AutoTokenizer,
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
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
from torchnlp.datasets.imdb import imdb_dataset


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

def get_init_label_embedding(immediate_tokens, soon_tokens, tokenizer, embedding, device):
    def get_embeds(tokens):
        embeds = []
        for t in tokens:
            ids = torch.tensor(tokenizer.convert_tokens_to_ids(t), dtype=torch.long, device=device)
            embeds.append(embedding(ids).mean(0))
        return torch.stack(embeds, dim=0).mean(0).detach().cpu().numpy()

    immediate_embeds = get_embeds(immediate_tokens)
    soon_embeds = get_embeds(soon_tokens)
    plain_embeds = np.random.normal(size=embedding.embedding_dim)
    return np.stack([plain_embeds, soon_embeds, immediate_embeds], axis=0)



immediate_tokens = [['all', '##ergic'], ['it', '##chy', 'bleeding'], ['difficulty', 'breathing'], ['short', '##ness', 'of', 'breathing'], ['chest', 'pain'], ['confusion'],
                    ['di', '##sor', '##ient', '##ation'],['loss', 'of', 'con', '##cious', '##ness'], ['occurred', 'today'], ['ing', '##est', '##ion', 'of', 'poison'],['numb', '##ness'],
                    ['severe'], ['para', '##lysis'], ['pain'], ['s', '##lu', '##rred'], ['g', '##ar', '##bled'], ['new', 'onset'], ['suicide'] ]
soon_tokens = [['loss', 'of', 'balance'], ['loss', 'of', 'coordination'], ['di', '##zzi', '##ness'], ['swelling'], ['headache'], ['nausea'], ['vomit'], ['p', '##al', '##pit', '##ation'],
               ['pain'], ['blurred', 'vision'], ['double', 'vision'], ['w', '##hee', '##zing']]

model_path = '/home/ray/transformers/biobert_v1.1_pubmed'
tokenizer = AutoTokenizer.from_pretrained(model_path)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
config = BertConfig.from_pretrained(model_path)
model = model_class.from_pretrained(model_path, from_tf=True, config=config)
device='cpu'
init_label_embedding = get_init_label_embedding(immediate_tokens, soon_tokens, tokenizer, model.bert.embeddings.word_embeddings, device=device)
w = torch.tensor(init_label_embedding, device=device)
model.bert.embeddings.label_embeddings = torch.nn.Embedding.from_pretrained(w, freeze=False).to(device)