import os
import gc
import json
import torch
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from modeling.copy_bart import copyBARTForConditionalGeneration
from modeling.copy_blenderbot import copyBlenderbotAttention, copyBlenderbotForConditionalGeneration
from utils import createIfNotExist, read_record
from transformers import BartTokenizer, BlenderbotTokenizer, BartConfig, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset.dataset import CNN_DAILYMAILDataset, PersonaChatDataset
from attributing.attribute_methods import sentence_attr_methods
from attributing.attribute_utils import load_attrs, save_attrs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_attrs(dataset, model, tokenizer, attr_fn, save_dir, save_name):
    save_path = os.path.join(save_dir, save_name + "-attrs.json")
    createIfNotExist(save_dir)
    tosave_attrs = load_attrs(save_path, split_attrs=False)
    for idx, feature in enumerate(tqdm(dataset.features, desc=dataset.name)):
        if idx < len(tosave_attrs):
            continue
        res = attr_fn(model, tokenizer, feature, model.device) 
        tosave_attrs.append((res['attributions'], res['input_id_splits']))
        if (idx + 1) % 500 == 0:
            save_attrs(tosave_attrs, save_path)
            torch.cuda.empty_cache()
    save_attrs(tosave_attrs, save_path)
def collect_attrs(dataset, model, tokenizer, save_dir):
    for attr_method_name, attr_fn in sentence_attr_methods.items():
        model.eval()
        run_attrs(dataset, model, tokenizer, attr_fn, os.path.join(save_dir, attr_method_name), dataset.split)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--task', choices=['personachat', 'dailymail'], help="Which dataset is working on")
    argparser.add_argument('--save_dir', default='./resource/attrs', help="Which directory to save result attributions")
    argparser.add_argument('--finetune_model_path', required=True, help="Which model do we use to attribute")
    argparser.add_argument('--DEBUG', action='store_true')
    args = argparser.parse_args()

    createIfNotExist(args.save_dir)    
    if args.task == 'personachat':
        model_name_or_path = 'facebook/blenderbot-400M-distill'
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name_or_path)
        tokenizer = PersonaChatDataset.prepare_tokenizer(tokenizer) 
        model = copyBlenderbotForConditionalGeneration.from_pretrained(args.finetune_model_path)
            
        for split in ['train', 'validation', 'test']:
            dataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split=split, MAX_LENGTH=256, DEBUG=args.DEBUG)
            collect_attrs(dataset, model, tokenizer, args.save_dir)
    elif args.task == 'dailymail':
        model_name_or_path = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        tokenizer = PersonaChatDataset.prepare_tokenizer(tokenizer)
        config = BartConfig.from_pretrained(model_name_or_path)
        model = copyBARTForConditionalGeneration.from_pretrained(args.finetune_model_path)
        for split in ['train', 'validation', 'test']:
            dataset = CNN_DAILYMAILDataset.preprocess(config, tokenizer, split=split, DEBUG=args.DEBUG)
            collect_attrs(dataset, model, tokenizer, args.save_dir)
    
