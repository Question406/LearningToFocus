import os
import nltk
import json
import torch
import logging
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import numpy as np
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, BlenderbotTokenizer, BartTokenizer
from dataset.dataset import PersonaChatDataset, CNN_DAILYMAILDataset, AttrsDataset
from dataset.dataset_utils import DataColltorForFocusTrain, split2sentence
import modeling.copy_bart as copy_bart
import modeling.copy_blenderbot as copy_blenderbot
from modeling.copy_bart import copyBARTForConditionalGeneration
from modeling.copy_blenderbot import copyBlenderbotForConditionalGeneration
from attributing.attribute_utils import load_attrs
from utils import setSeed
from typing import List
from datasets import load_metric
from tqdm import tqdm
from bert_score import score as bertscore

transformers.logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_rouge(labels : List, preds : List):
    rouge = load_metric('rouge')
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) if not isinstance(label, List) else "\n".join(
        nltk.sent_tokenize(" ".join(label))) for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    result = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
    return result

def compute_bertscore(labels : List, preds : List):
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) if not isinstance(label, List) else [
        "\n".join(nltk.sent_tokenize(x.strip())) for x in label] for label in labels]
    bertscore_p, bertscore_r, bertscore_f1 = bertscore(preds, labels, lang='en')
    return {
        'bertscore_p': np.mean(bertscore_p.tolist()),
        'bertscore_r': np.mean(bertscore_r.tolist()),
        'bertscore_f1': np.mean(bertscore_f1.tolist())
    }

def compute_PPL(dataset : torch.utils.data.dataset, model : PreTrainedModel, tokenizer : PreTrainedTokenizer) : 
    def PPL_on_batch(batch, model):
        loss = model(**batch).loss
        labels = batch['labels'].cpu().detach().numpy()
        word_cnt = np.sum(labels != -100)
        return loss.item(), word_cnt 

    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                            collate_fn=DataColltorForFocusTrain(model=model, tokenizer=tokenizer))
    loss_all = 0
    word_all = 0
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k : v.to(model.device) for k, v in batch.items()}
            loss, word_cnt = PPL_on_batch(batch, model)
            loss_all += loss
            word_all += word_cnt
            cnt += 1
            del batch
    torch.cuda.empty_cache()
    model.train()
    return np.exp(loss_all / cnt)

def collect_predicts(model, tokenizer, dataset, gen_method, no_attn=False):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                            collate_fn=DataColltorForFocusTrain(model=model, tokenizer=tokenizer, padding='longest'))
    res_labels = []
    res_preds = []
    for idx, batch in tqdm(enumerate(dataloader), desc='collecting labels&preds'): 
        batch_labels = batch.pop('labels').numpy()
        batch_labels = np.where(batch_labels != -100, batch_labels, tokenizer.pad_token_id)
        if no_attn and 'attn_input_ids' in batch:
            batch.pop('attn_input_ids')
        batch = {k : v.to(model.device) for k, v in batch.items()}
        batch.update(gen_method)
        with torch.no_grad():
            batch_preds = model.generate(**batch)
        label_strs = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
        pred_strs = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)

        res_labels.extend(label_strs)
        res_preds.extend(pred_strs)

    return res_labels, res_preds


def evaluate(arg):
    # preapre tokenizer
    if arg.task == 'personachat':
        model_name_or_path = 'facebook/blenderbot-400M-distill'
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name_or_path)
        tokenizer = PersonaChatDataset.prepare_tokenizer(tokenizer)
        tokenizer.model_max_length = 256
    elif arg.task == 'dailymail':
        model_name_or_path = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    
    # prepare model
    if arg.task == 'personachat':
        model = copyBlenderbotForConditionalGeneration.from_pretrained(arg.pretrained_model_path)
        if arg.attn_onfly:
            # model should be only finetuned 
            assert model.get_attn_embed() is None
            copy_blenderbot.setATTN_CONST(3.0326394770521578)
        gen_method = {
            'num_beams': 10,
            'max_length': 128,
            'min_length': 20,
            'early_stopping': True,
            'no_repeat_ngram_size': 5
        }
    else:
        model = copyBARTForConditionalGeneration.from_pretrained(arg.pretrained_model_path)
        if arg.attn_onfly:
            assert model.get_attn_embed() is None
            copy_bart.setATTN_CONST(0.169120863099577)
        gen_method = {
            "length_penalty": 2.0,
            "max_length": 142,
            "min_length": 56,
            "num_beams": 4,
            'early_stopping': True
        }

    # prepare dataset 
    evaldataset = None
    if arg.do_amt_eval:
        # preapre AMT annotated dataset
        if arg.task == 'personachat':
            evaldataset = PersonaChatDataset.getAMTEvalDataset(tokenizer, './resource/AMT_dataset/persona_highlight.json', split='test', DEBUG=arg.DEBUG)
        elif arg.task == 'dailymail':
            evaldataset = CNN_DAILYMAILDataset.getAMTEvalDataset(tokenizer, './resource/AMT_dataset/dailymail_highlight.json', split='test', DEBUG=arg.DEBUG)

    else:
        if arg.do_focus_eval:
            if arg.task == 'personachat':
                evaldataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='test', MAX_LENGTH=256, DEBUG=arg.DEBUG)
                test_attrs, test_id_splits = load_attrs(os.path.join(arg.attr_dir, 'validation-attrs.json'), split_attrs=True)
                drop_method = {'random_drop' : False, 'random_k' : True, 'top_k' : 1}
                evaldataset = AttrsDataset.from_dataset(evaldataset, test_attrs, test_id_splits, drop_method=drop_method)
            elif arg.task == 'dailymail':
                evaldataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='split', DEBUG=arg.DEBUG)
                test_attrs, test_id_splits = load_attrs(os.path.join(arg.attr_dir, 'validation-attrs.json'), split_attrs=True)
                drop_method = {'random_drop' : False, tokenizer : tokenizer, 'ref_label' : True, 'get_sentence_fun' : split2sentence}
                evaldataset = AttrsDataset.from_dataset(evaldataset, test_attrs, test_id_splits, drop_method=drop_method)
        elif arg.do_finetune_eval:
            if arg.task == 'personachat':
                evaldataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='test', MAX_LENGTH=256, DEBUG=arg.DEBUG)
            elif arg.task == 'dailymail':
                evaldataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='split', DEBUG=arg.DEBUG)
        else:
            assert False, "Should not be here"

    # collect predicts and calculate metrics
    labels, predicts = collect_predicts(model, tokenizer, evaldataset, gen_method, no_attn=True if arg.do_finetune_eval else False)
    res_metrics = {}
    res_metrics['PPL'] = compute_PPL(evaldataset, model, tokenizer)
    res_metrics['bertscore'] = compute_bertscore(labels, predicts)
    res_metrics['rouge'] = compute_rouge(labels, predicts)
    logger.info("Evaluate Done! >>> \n" + json.dumps(res_metrics))
    

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--task', choices=['personachat', 'dailymail'], help="Which datasets is working on")
    argparser.add_argument('--attr_dir', help="The directory of attribution scores")
    argparser.add_argument('--do_finetune_eval', action='store_true', help="Evaluate finetune model")
    argparser.add_argument('--do_focus_eval', action='store_true', help="Evaluate focus vector")
    argparser.add_argument('--do_keyword_eval', action='store_true', help="Evaluate prompt model")
    argparser.add_argument('--do_amt_eval', action='store_true', help="Evaluate on highlighg dataset")
    argparser.add_argument('--attn_onfly', action='store_true', help="Evaluate onfly attention model")
    argparser.add_argument('--pretrained_model_path', required=True, help="The pretrained model to be evaluated")
    argparser.add_argument('--DEBUG', action='store_true')

    setSeed()
    args = argparser.parse_args()
    evaluate(args)
