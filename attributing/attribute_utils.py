import torch
import numpy as np
from typing import List
import os
import logging
import json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BartTokenizer
from utils import read_record

logger = logging.getLogger(__name__)

def PPL_on_batch(batch, logits, tokenizer: PreTrainedTokenizer, return_ppl=True): 
    log_likelihood = torch.log_softmax(logits.cpu().detach(), dim=2)
    # eval_dataset_length * max_length * vocab_size
    labels = batch['labels'].cpu().detach()
    lls = []
    sum_words = 0
    for i in range(len(labels)):
        tmp = 0
        for j in range(len(labels[i])):
            tmp += log_likelihood[i, j, labels[i, j]]
            sum_words += 1
            if labels[i][j] == tokenizer.eos_token_id:
                break
        lls.append(tmp)
    ppl = -np.sum(lls) / sum_words
    if return_ppl:
        return np.exp(ppl)
    else:
        return lls, sum_words

def model_forward(model, inputs: dict):
    with torch.enable_grad():
        outputs = model(**inputs, output_hidden_states=True)
        assert outputs.loss is not None
        loss = outputs.loss
        loss.backward(retain_graph=True)
        res1, res2 = outputs.encoder_hidden_states[0].cpu(), model.model.encoder.input_embeds_grads.cpu()
        model.model.encoder.clear_input_embeds_grad()
        return res1, res2


MAX_IDX = 100
def load_attrs(load_path: os.path, split_attrs: bool = False, DEBUG=False):
    res = []
    if os.path.exists(load_path):
        f = read_record(load_path)
        for i, line in enumerate(tqdm(f)):
            if DEBUG and i == MAX_IDX:
                break
            res.append(line)
    
    if split_attrs:
        attrs, splits = [], []
        for x in res:
            attrs.append(x[0])
            splits.append(x[1])
        logger.info("loading attrs {} done splitted from {}".format(len(res), load_path))
        return attrs, splits
    else:
        logger.info("loading attrs {} done from {}".format(len(res), load_path))
        return res


def save_attrs(attrs_tosave, save_path: os.path):
    with open(save_path, 'w') as f:
        for attr_on_single in tqdm(attrs_tosave):
            f.write(json.dumps(attr_on_single) + '\n')
    logger.info("save attrs done to {}".format(save_path))