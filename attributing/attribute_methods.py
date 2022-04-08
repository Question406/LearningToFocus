import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer, BartTokenizer, BlenderbotTokenizer
from dataset.dataset import Feature
from attributing.attribute_utils import PPL_on_batch, model_forward
from dataset.dataset_utils import split2sentence
from typing import List


def prepare_inputs(feature, tokenizer, device, **kwargs):
    DEBUG = kwargs.get('debug', False)
    
    input_ids = torch.LongTensor(feature.input_ids).to(device).unsqueeze(0)
    batch_size, seq_len = input_ids.size()
    assert batch_size == 1
    prefix_input_ids = torch.LongTensor(feature.labels).to(device).unsqueeze(0)
    
    res = {'input_ids': input_ids,
           'labels': prefix_input_ids,
           }
    input_id_splits = split2sentence(feature.input_ids, tokenizer, debug=DEBUG)
    res['input_id_splits'] = input_id_splits
    
    return res



def att_score_attribute(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, feature: Feature,
                        device: torch.device, **kwargs):
    res = prepare_inputs(feature, tokenizer, device, **kwargs)
    input_ids = res['input_ids']
    labels = res['labels']
    
    assert input_ids.shape[0] == 1 and labels.shape[0] == 1
    raw_inputs = {'input_ids': input_ids, 'labels': labels}
    # 'attention_mask': torch.LongTensor([[1] * input_ids.shape[-1]]).to(device)}
    model.eval()
    res_attributions = []

    input_id_splits = res['input_id_splits']
    with torch.no_grad():
        outputs = model(**raw_inputs, output_attentions=True)
        cross_attentions = torch.stack(outputs.cross_attentions).squeeze().cpu()
        cross_attentions = cross_attentions.permute(2, 3, 0, 1)
        input_id_scores = torch.zeros(input_ids[0].shape, dtype=torch.float)
        for i in range(len(input_ids[0])):
            for j in range(len(labels[0])):
                input_id_scores[i] += torch.sum(cross_attentions[j][i])
        input_id_scores /= cross_attentions.shape[2] * cross_attentions.shape[3]
    res_attributions = []
    for split in input_id_splits:
        l, r = split
        res_attributions.append(torch.sum(input_id_scores[l: r + 1]).item())
    del raw_inputs
    torch.cuda.empty_cache()
    res = {
        'attributions': res_attributions,
        'input_id_splits': input_id_splits
    }
    
    model.train()
    model.zero_grad()
    
    return res


def l2_attribute(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, feature: Feature,
                 device: torch.device, **kwargs):
    res = prepare_inputs(feature, tokenizer, device, **kwargs)
    input_ids = res['input_ids']
    labels = res['labels']
    on_sentence = kwargs.pop('on_sentence', False)
    
    assert input_ids.shape[0] == 1 and labels.shape[0] == 1
    model.eval()
    # run_model(model, {'input_ids': input_ids, 'labels': labels})
    _, input_embeds_grad = model_forward(model, {'input_ids': input_ids, 'labels': labels})
    # embed_layer = model.model.shared
    res_attributions = []
    for i in range(len(input_ids[0])):
        res_attributions.append(torch.norm(input_embeds_grad[0][i], p=2).item())
    model.train()
    model.zero_grad()
    
    input_id_splits = res['input_id_splits']
    sentence_attributions = []
    for split in input_id_splits:
        start, end = split
        split_attribution = np.sqrt(np.sum(np.square(res_attributions[start: end + 1]))) / (end - start + 1)
        sentence_attributions.append(split_attribution)
        
    res = {
        'attributions': sentence_attributions,
        'input_id_splits': input_id_splits
    }
    
    return res



def dot_attribute(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                  feature: Feature,
                  device: torch.device, **kwargs):
    res = prepare_inputs(feature, tokenizer, device, **kwargs)
    input_ids = res['input_ids']
    labels = res['labels']
    on_sentence = kwargs.pop('on_sentence', False)
    
    assert input_ids.shape[0] == 1 and labels.shape[0] == 1
    model.eval()
    input_embeds, input_embeds_grad = model_forward(model, {'input_ids': input_ids, 'labels': labels})
    res_attributions = []
    for i in range(len(input_ids[0])):
        res_attributions.append((input_embeds[0][i] @ -input_embeds_grad[0][i]).item())
    model.train()
    model.zero_grad()

    input_id_splits = res['input_id_splits']
    sentence_attributions = []
    for split in input_id_splits:
        start, end = split
        split_attribution = np.mean(res_attributions[start: end + 1])
        sentence_attributions.append(split_attribution)
        
    res = {
        'attributions': sentence_attributions,
        'input_id_splits': input_id_splits
    }
    
    return res


def loo_attribute(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, feature: Feature,
                  device: torch.device, **kwargs):
    res = prepare_inputs(feature, tokenizer, device, **kwargs)
    input_ids = res['input_ids']
    labels = res['labels']
    on_sentence = kwargs.pop('on_sentence', False)
    
    assert input_ids.shape[0] == 1 and labels.shape[0] == 1
    raw_inputs = {'input_ids': input_ids, 'labels': labels}
    # 'attention_mask': torch.LongTensor([[1] * input_ids.shape[-1]]).to(device)}
    model.eval()
    res_attributions = []

    with torch.no_grad():
        base_outputs = model(**raw_inputs)
        base_ppl = PPL_on_batch(raw_inputs, base_outputs.logits, tokenizer, return_ppl=True)
        input_id_splits = res['input_id_splits']
        for i, pos in enumerate(input_id_splits):
            start, end = pos
            old_ids = raw_inputs['input_ids'][0][start: end + 1].clone()
            raw_inputs['input_ids'][0][start: end + 1] = tokenizer.pad_token_id
            outputs = model(**raw_inputs)
            ppl = PPL_on_batch(raw_inputs, outputs.logits, tokenizer, return_ppl=True)
            res_attributions.append(ppl - base_ppl)
            raw_inputs['input_ids'][0][start: end + 1] = old_ids
            del old_ids
    del raw_inputs
    torch.cuda.empty_cache()
    res = {
        'attributions': res_attributions,
        'input_id_splits': input_id_splits
    }
    model.zero_grad()
    model.train()
    return res


sentence_attr_methods = {
    'att_score': att_score_attribute,
    'l2': l2_attribute,
    'dot': dot_attribute,
    'loo': loo_attribute,
}