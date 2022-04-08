import json

import argparse
import logging
import math
import os

from transformers import BartForConditionalGeneration, BartTokenizer, \
                         BlenderbotTokenizer, DataCollatorForSeq2Seq, \
                         Seq2SeqTrainer, Seq2SeqTrainingArguments, optimization
import transformers
from dataset.dataset import AttrsDataset, CNN_DAILYMAILDataset, PersonaChatDataset, toKeywordDataset
from dataset.dataset_utils import DataColltorForFocusTrain
from torch.optim import AdamW
from dataset.dataset_utils import split2sentence
from utils import setSeed

from modeling.copy_blenderbot import copyBlenderbotConfig, copyBlenderbotForConditionalGeneration
from modeling.copy_bart import copyBARTConfig, copyBARTForConditionalGeneration 
from attributing.attribute_utils import load_attrs

transformers.logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def finetune(arg):
    if arg.task == 'personachat':
        model_name_or_path = 'facebook/blenderbot-400M-distill'
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name_or_path)
        tokenizer = PersonaChatDataset.prepare_tokenizer(tokenizer)
        tokenizer.model_max_length = 256
        MAX_LENGTH = 256
        model = copyBlenderbotForConditionalGeneration.from_pretrained(model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
        model.resize_position_embedding(MAX_LENGTH)
        traindataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='train', MAX_LENGTH=MAX_LENGTH, DEBUG=arg.DEBUG)
        validataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='validation', MAX_LENGTH=MAX_LENGTH, DEBUG=arg.DEBUG)
        output_dir = "personachat"

    elif arg.task == 'dailymail':
        model_name_or_path = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        traindataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='train', DEBUG=arg.DEBUG)
        validataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='validation', DEBUG=arg.DEBUG)
        output_dir = "dailymail"

    train_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join('./trained_models', output_dir, f"lr-{arg.lr}"),
        logging_dir=os.path.join('./trained_models', output_dir, f"lr-{arg.lr}"),
        learning_rate=arg.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        weight_decay=0.01,
        save_total_limit=4,
        do_train=True,
        do_eval=True,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        warmup_ratio=0.1,
        num_train_epochs=10,
        logging_first_step=True,
        logging_steps=40,
        fp16=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model, train_args,
        train_dataset=traindataset,
        eval_dataset=validataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    res = trainer.train()
    logger.info("Training Done: \n " + json.dumps(res))


def focus_train(arg):
    if arg.attr_dir.split('/')[-1] == '':
        attr_name = arg.attr_dir.split('/')[-2]
    else:
        attr_name = arg.attr_dir.split('/')[-1]

    if arg.task == 'personachat':
        base_model_name = 'facebook/blenderbot-400M-distill'
        tokenizer = BlenderbotTokenizer.from_pretrained(base_model_name)
        tokenizer = PersonaChatDataset.prepare_tokenizer(tokenizer)
        config = copyBlenderbotConfig.from_pretrained(arg.finetune_model_path, 
                                                      attn_type=arg.attn_type, 
                                                      attn_init_type='xavier')
        model = copyBlenderbotForConditionalGeneration.from_pretrained(
                            arg.finetune_model_path, config=config) # we use config to set

        # raw dataset
        traindataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='train', MAX_LENGTH=256, DEBUG=arg.DEBUG)
        validdataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='validation', MAX_LENGTH=256, DEBUG=arg.DEBUG)
        # load attribution list 
        train_attrs, train_id_splits = load_attrs(os.path.join(arg.attr_dir, 'train-attrs.json'), split_attrs=True)
        valid_attrs, valid_id_splits = load_attrs(os.path.join(arg.attr_dir, 'validation-attrs.json'), split_attrs=True)
        # merge raw-dataset with attribution
        drop_method = {'random_drop' : False, 'random_k' : True, 'top_k' : 1}
        trainattrsdataset = AttrsDataset.from_dataset(traindataset, train_attrs, train_id_splits, drop_method=drop_method)
        validattrsdataset = AttrsDataset.from_dataset(validdataset, valid_attrs, valid_id_splits, drop_method=drop_method)
        print(trainattrsdataset[2])
        output_dir = f'personachat_focus'

    elif arg.task == 'dailymail':
        base_model_name = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(base_model_name)
        config = copyBARTConfig.from_pretrained(arg.finetune_model_path, 
                                                attn_type=arg.attn_type, 
                                                attn_init_type='xavier')
        model = copyBARTForConditionalGeneration.from_pretrained(
                                arg.finetune_model_path, config=config) # we use config to set

        # raw dataset
        traindataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='train', DEBUG=arg.DEBUG)
        validdataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='validation', DEBUG=arg.DEBUG)
        
        train_attrs, train_id_splits = load_attrs(os.path.join(arg.attr_dir, 'train-attrs.json'), split_attrs=True)
        valid_attrs, valid_id_splits = load_attrs(os.path.join(arg.attr_dir, 'validation-attrs.json'), split_attrs=True)
        # merge raw-dataset with attribution
        drop_method = {'random_drop' : False, 'tokenizer' : tokenizer, 'ref_label' : True, 'get_sentence_fun' : split2sentence}
        trainattrsdataset = AttrsDataset.from_dataset(traindataset, train_attrs, train_id_splits, drop_method=drop_method)
        validattrsdataset = AttrsDataset.from_dataset(validdataset, valid_attrs, valid_id_splits, drop_method=drop_method)
        print(trainattrsdataset[2])
        output_dir = f'dailymail_focus'
    
    def custom_train(traindataset, evaldataset, model, tokenizer, train_config):
        
        def get_optimizer(model):
            if train_config['model_lr']:
                optimizer = AdamW([{'params': model.get_attn_embed().parameters(), 'lr': train_config['learning_rate']},
                                   {'params': [para for name, para in model.named_parameters() if
                                               not ('attn_embed' in name or 'layer_attn_embeds' in name)],
                                    'lr': train_config['model_lr']}])
                # 'lr': 0.0}])
            else:
                optimizer = AdamW(model.get_attn_embed().parameters(), lr=train_config['learning_rate'])
            return optimizer
    
        def get_lr_scheduler(traindataset, training_args):
            steps = math.ceil(len(traindataset) / training_args.train_batch_size)
            num_update_steps_per_epoch = steps // training_args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
            logger.info("all steps: {}".format(max_steps))
            lr_scheduler = optimization.get_linear_schedule_with_warmup(optimizer, 0.1 * max_steps, max_steps)
            return lr_scheduler
        
        setSeed(42)
        train_arguments = train_config['train_config']
        optimizer = get_optimizer(model)
        lr_scheduler = get_lr_scheduler(traindataset, train_arguments)
        trainer = Seq2SeqTrainer(
            model=model,
            args=train_arguments,
            train_dataset=traindataset,
            eval_dataset=evaldataset,
            data_collator=DataColltorForFocusTrain(model=model, tokenizer=tokenizer, padding='longest'),
            tokenizer=tokenizer,
            optimizers=(optimizer, lr_scheduler),
        )
        res = trainer.train()
        logger.info("Focus training done >>> \n" + json.dumps(res))

    run_name = args.attn_type + "_" + attr_name + f'_lr-{arg.lr}'
    if arg.modellr:
        run_name += f'_modellr-{arg.modellr}'

    attn_trainconfig = {
        'attn_type': arg.attn_type,
        'model_lr': arg.modellr if arg.modellr else None,
        'learning_rate': arg.lr,
        'train_config': Seq2SeqTrainingArguments(
            output_dir=os.path.join('./trained_models', output_dir, run_name),
            logging_dir=os.path.join('./trained_models', output_dir, run_name),
            num_train_epochs=4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            save_strategy='epoch',
            save_total_limit=4,
            do_train=True,
            do_eval=True,
            evaluation_strategy='epoch',
            learning_rate=arg.lr,
            weight_decay=0.01,
            gradient_accumulation_steps=8,
            eval_accumulation_steps=8,
            logging_first_step=True,
            logging_steps=20,
            fp16=False,
            seed=42,
        )
    }
    custom_train(trainattrsdataset, validattrsdataset, model, tokenizer, attn_trainconfig)

def keyword_train(arg):
    drop_method = None # we don't pass attn_input_ids as we're not training focus vector
    if arg.attr_dir.split('/')[-1] == '':
        attr_name = arg.attr_dir.split('/')[-2]
    else:
        attr_name = arg.attr_dir.split('/')[-1]

    if arg.task == 'personachat':
        model_name_or_path = 'facebook/blenderbot-400M-distill'
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name_or_path)
        tokenizer = PersonaChatDataset.prepare_tokenizer(tokenizer)
        MAX_LENGTH = 256
        model = copyBlenderbotForConditionalGeneration.from_pretrained(model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
        model.resize_position_embedding(MAX_LENGTH)
        traindataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='train', MAX_LENGTH=MAX_LENGTH, DEBUG=arg.DEBUG)
        validdataset = PersonaChatDataset.preprocess(tokenizer, './resource/personachat/', split='validation', MAX_LENGTH=MAX_LENGTH, DEBUG=arg.DEBUG)
        train_attrs, train_id_splits = load_attrs(os.path.join(arg.attr_dir, 'train-attrs.json'), split_attrs=True)
        valid_attrs, valid_id_splits = load_attrs(os.path.join(arg.attr_dir, 'validation-attrs.json'), split_attrs=True)
        traindataset = AttrsDataset.from_dataset(traindataset, train_attrs, train_id_splits, drop_method=drop_method)
        validdataset = AttrsDataset.from_dataset(validdataset, valid_attrs, valid_id_splits, drop_method=drop_method)
        # change to keyword dataset
        train_keyword_dataset = toKeywordDataset(traindataset, tokenizer, training=True)
        valid_keyword_dataset = toKeywordDataset(validdataset, tokenizer, training=False)

        output_dir = "personachat_prompt"

    elif arg.task == 'dailymail':
        model_name_or_path = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        traindataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='train', DEBUG=arg.DEBUG)
        validdataset = CNN_DAILYMAILDataset.preprocess(tokenizer, split='validation', DEBUG=arg.DEBUG)
        train_attrs, train_id_splits = load_attrs(os.path.join(arg.attr_dir, 'train-attrs.json'), split_attrs=True)
        valid_attrs, valid_id_splits = load_attrs(os.path.join(arg.attr_dir, 'validation-attrs.json'), split_attrs=True)
        traindataset = AttrsDataset.from_dataset(traindataset, train_attrs, train_id_splits, drop_method=drop_method)
        validdataset = AttrsDataset.from_dataset(validdataset, valid_attrs, valid_id_splits, drop_method=drop_method)
        # change to keyword dataset
        train_keyword_dataset = toKeywordDataset(traindataset, tokenizer, training=True)
        valid_keyword_dataset = toKeywordDataset(validdataset, tokenizer, training=False)

        output_dir = "dailymail_prompt"

    run_name = f'{attr_name}_lr-{args.lr}'

    train_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join('./trained_models', output_dir, run_name),
        logging_dir=os.path.join('./trained_models', output_dir, run_name),
        learning_rate=arg.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        weight_decay=0.01,
        save_total_limit=4,
        do_train=True,
        do_eval=True,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        warmup_ratio=0.1,
        num_train_epochs=10,
        logging_first_step=True,
        logging_steps=40,
        fp16=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model, train_args,
        train_dataset=train_keyword_dataset,
        eval_dataset=valid_keyword_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    res = trainer.train()
    logger.info("Training Done: \n " + json.dumps(res))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--task', choices=['personachat', 'dailymail'])
    argparser.add_argument('--do_finetune', action='store_true', help="Finetune base model on dataset")
    argparser.add_argument('--do_focus_train', action='store_true', help="Train focus-embedding with finetuned model")
    argparser.add_argument('--do_keyword_train', action='store_true', help="Train prompt with finetuned model")
    argparser.add_argument('--finetune_model_path')
    argparser.add_argument('--attr_dir', help="The directory of attribution scores")
    argparser.add_argument('--attn_type', choices=['none', 'input', 'layer', 'attention', 'mul_input', 'mul_layer', 'lin_input', 'lin_layer', 'mul_flayer'])
    argparser.add_argument('--lr', type=float, help='learning rate of our main goal, namely model parameter when finetune; focus vector when focus-train')
    argparser.add_argument('--modellr', type=float, help="learning rate for model parameter if we don't fix them")
    argparser.add_argument('--DEBUG', action='store_true')
    args = argparser.parse_args()

    setSeed(42)
    if args.do_finetune:
        finetune(args)
    elif args.do_focus_train:
        focus_train(args)
    elif args.do_keyword_train:
        keyword_train(args) 