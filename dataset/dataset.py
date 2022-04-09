import os
import json
import copy
import yake
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizer, PretrainedConfig
from datasets import load_dataset
from typing import List, Optional
from functools import partial

from dataset.dataset_utils import split2sentence 

class Feature:
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
    
    def __dict__(self):
        return {
            'input_ids' : self.input_ids,
            'labels' : self.labels
        }
    
    def __repr__(self) -> str:
        return str(self.__dict__())

class AttrsFeature(Feature):
    def __init__(self, input_ids: list, labels: list, attrs: list,
                 id_splits: list,
                 highlights: Optional[list] = None):
        assert len(attrs) == len(id_splits)
        super(AttrsFeature, self).__init__(input_ids, labels)
        self.attrs = attrs
        self.id_splits = id_splits
        self.highlights = highlights
    
    def __copy__(self):
        res = AttrsFeature(
            copy.deepcopy(self.input_ids),
            copy.deepcopy(self.labels),
            copy.deepcopy(self.attrs),
            copy.deepcopy(self.id_splits),
            copy.deepcopy(self.highlights)
        )
        return res
    
    def __dict__(self, drop_method: dict = None):
        res = super().__dict__()
        if drop_method is not None:  # attribtion feature
            if 'running_eval' in drop_method:
                todrops = self.highlights
                tmp = np.zeros_like(self.input_ids)
                for todrop in todrops:
                    pos = self.id_splits[todrop]
                    start, end = pos
                    tmp[start: end + 1] = 1
                res['attn_input_ids'] = tmp.tolist()
            elif 'trivial_eval' in drop_method:
                todrops = self.highlights
                tmp = np.zeros_like(self.input_ids)
                for todrop in todrops:
                    pos = self.id_splits[todrop]
                    start, end = pos
                    tmp[start: end + 1] = 1
                res['input_ids'] = np.where(tmp == 1, res['input_ids'], 0)
                res['attention_mask'] = tmp
                res['attn_input_ids'] = tmp.tolist()
            else:
                if drop_method.get('trivial_train', False):
                    if len(self.id_splits) >= 3:
                        left = int(len(self.id_splits) * 0.3)
                        right = max(int(len(self.id_splits) * 0.5), left + 1)
                        num_of_mask = np.random.randint(left, right)
                        num_of_mask = max(2, num_of_mask)
                        indexes = np.random.choice(range(len(self.id_splits)), num_of_mask, replace=False)
                        for todo in indexes:
                            l, r = self.id_splits[todo]
                            l = max(0, l)
                            r = min(r, len(res['input_ids']) - 1)
                            for pos in range(l, r + 1):
                                res['input_ids'][pos] = 0
                else:
                    random_drop = drop_method['random_drop']
                    if 'random_k' in drop_method and drop_method['random_k']:
                        # top_k = random.randint(1, 4)
                        if len(self.attrs) // 4 == 0:
                            top_k = 1
                        elif len(self.attrs) // 4 >= 4:
                            top_k = 4
                        else:
                            top_k = len(self.attrs) // 4
                        # top_k = min(len(self.attrs) / 4, )
                    else:
                        if drop_method.get('ref_label', False):
                            tokenizer = drop_method.get('tokenizer', None)
                            get_sentence_fun = drop_method.get('get_sentence_fun', None)
                            assert tokenizer is not None, "You need tokenizer to get top_k referring labels"
                            assert get_sentence_fun is not None, "You need get_sentence function to get top_k referring labels"
                            splits = get_sentence_fun(self.labels, tokenizer)
                            top_k = len(splits)
                        else:
                            top_k = drop_method.get('top_k', None)
                    
                    top_p = drop_method.get('top_p', None)
                    
                    tmp_attrs = sorted([(x, i) for i, x in enumerate(self.attrs)], reverse=True)
                    if top_p is not None:
                        num_of_mask = max(int(len(tmp_attrs) * top_p), 1)
                    elif top_k is not None:
                        num_of_mask = top_k
                    else:
                        raise ValueError(f"Unkown strategy {drop_method}")
                    
                    if random_drop:
                        indexes = np.random.choice(range(len(tmp_attrs)), num_of_mask, replace=False)
                        todrops = [tmp_attrs[x] for x in indexes]
                    else:
                        todrops = tmp_attrs[:num_of_mask]
                    
                    tmp = np.zeros_like(self.input_ids)
                    for todrop in todrops:
                        pos = self.id_splits[todrop[1]]
                        start, end = pos
                        tmp[start: end + 1] = 1
                    res['attn_input_ids'] = tmp.tolist()
        
        return res
    

class AttrsDataset:
    
    def __init__(self, datasetname, features: List[AttrsFeature] = None,
                 drop_method: dict = None, **kwargs):
        self.name = datasetname
        self.features = features
        self.drop_method = drop_method
        self.split = kwargs.get('split', 'unkown')
        if drop_method is not None and 'running_eval' in drop_method:
            for x in features:
                assert x.highlights is not None, 'All features must have highlights when running evaluation'
    
    def set_drop_method(self, new_drop_method: dict):
        self.drop_method = new_drop_method
    
    @classmethod
    def from_dataset(cls, olddataset,
                     attrs: List[List],
                     input_splits: List[List],
                     drop_method: dict = None):
        assert len(input_splits) == len(attrs) <= len(
            olddataset), f"splits: {len(input_splits)}, attrs:{len(attrs)}, dataset:{len(olddataset)}"
        
        attrs_features = []
        for i in range(len(input_splits)):
            newfeature = AttrsFeature(
                olddataset.features[i].input_ids,
                olddataset.features[i].labels,
                attrs[i],
                input_splits[i])
            attrs_features.append(newfeature)
        newdataset = cls(datasetname=f"{olddataset.get_name()}_attrs",
                         sample_cls=AttrsFeature,
                         features=attrs_features,
                         drop_method=drop_method,
                         split=olddataset.split)
        return newdataset
    
    def get_subdataset(self, startidx: int, len: int):
        return AttrsDataset(self.name + f"-sub[{startidx}:{startidx + len}]",
                            self.sample_cls,
                            self.features[startidx: startidx + len],
                            self.drop_method)
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        res = self.features[idx].__dict__(self.drop_method)
        return res
    
    def showSample(self, idx, tokenizer, sortAttn: False):
        print(f"{self.name} , Idx: {idx}")
        self.features[idx].showSample(tokenizer, sortAttn)


class PersonaChatDataset:
    
    def __init__(self, features : List[Feature], split):
        self.name = "PersonaChat"
        self.features = features
        self.split = split
    
    def get_name(self):
        return self.name

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx].__dict__()
    
    @staticmethod
    def prepare_tokenizer(tokenizer: PreTrainedTokenizer):
        if not "[SOP]" in tokenizer.all_special_tokens:
            special_tokens_dict = {'additional_special_tokens': ['[SOP]', '[EOP]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            added = tokenizer.get_added_vocab()
            print("added tokens: ")
            print(added)
            return tokenizer
    
    @staticmethod
    def preprocess(tokenizer: PreTrainedTokenizer, rawpath: os.path, split,
                   **kwargs):
        
        def preprocess_persona(personas, tokenizer: PreTrainedTokenizer, **kwargs):
            res = {'input_ids': [], 'labels': []}
            sop_id = tokenizer.convert_tokens_to_ids('[SOP]')
            eop_id = tokenizer.convert_tokens_to_ids('[EOP]')
            sep_id = tokenizer.sep_token_id
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            # construct persona_prefix
            persona_prefix = []
            for i, x in enumerate(personas['personas']):
                ids = tokenizer(x, add_special_tokens=False).input_ids
                if i == 0:
                    persona_prefix.extend([sop_id] + ids + [sep_id])
                elif i == len(personas['personas']) - 1:
                    persona_prefix.extend(ids + [eop_id])
                else:
                    persona_prefix.extend(ids + [sep_id])
            # construct samples
            utt_ids = [tokenizer(x, add_special_tokens=False).input_ids for x in personas['dialogues']]
            utt_lens = [len(x) for x in utt_ids]
            for i in range(0, len(utt_ids) - 1, 2):
                input_ids = []
                k = 0
                sum_len = len(persona_prefix) + np.sum(utt_lens[0] + 2)
                if sum_len > max_length:
                    continue
                for k in range(0, i, 2):
                    sum_len = len(persona_prefix) + np.sum(utt_lens[k:i + 1]) + 2 * (i + 1 - k)
                    if sum_len <= max_length:
                        break
                if sum_len > max_length:
                    continue
                for j in range(k, i + 1):
                    input_ids.extend([bos_id] + utt_ids[j] + [eos_id])
                res['input_ids'].append(persona_prefix + input_ids)
                res['labels'].append([bos_id] + utt_ids[i + 1] + [eos_id])
                assert sum_len == len(res['input_ids'][-1])
            return res
        
        def get_single_personas(path):
            def get_dialogue(line: str):
                splits = line.split('\t')
                res = []
                for x in splits:
                    if x.strip() == '':
                        continue
                    if '|' in x:
                        break
                    else:
                        if x.strip() != '':
                            res.append(x.strip())
                return res
            
            with open(path, 'r') as f:
                last = None
                res = None
                for line in f:
                    # print(line)
                    line = line.strip()
                    line = line[line.find(' ') + 1:]
                    if res is None:
                        res = {'personas': [], 'dialogues': []}
                    
                    if line.startswith("your persona:") and last == 'dialogue':
                        yield res
                        res = {'personas': [], 'dialogues': []}
                    
                    if 'your persona:' in line:
                        res['personas'].append(line[line.find(':') + 1:].strip())
                        last = 'persona'
                    else:
                        dialogue = get_dialogue(line)
                        res['dialogues'].extend(dialogue)
                        last = 'dialogue'
                
                yield res
        
        # PersonaChatDataset.prepare_tokenizer(tokenizer)
        max_length = kwargs.pop('max_length', 256)
        max_response_length = 128
        
        mode = 'self'
        original_revised = 'original'
        
        allfeatures = []
        splitname = 'valid' if split == 'validation' else split
        idx = 0
        for personas in tqdm(get_single_personas(os.path.join(rawpath,
                                                              f'{splitname}_{mode}_{original_revised}.txt'))):
            idx += 1
            # continue
            # print(personas)
            samples = preprocess_persona(personas, tokenizer)
            allfeatures.extend([Feature(input_ids, labels) for input_ids, labels in
                                zip(samples['input_ids'], samples['labels'])])
            if kwargs.get("DEBUG", False) and len(allfeatures) >= 50:
                break
        print(split, " : ", idx)
        dataset = PersonaChatDataset(features=allfeatures, split=split)

        return dataset
    
    @staticmethod
    def getAMTEvalDataset(tokenizer, rawpath, split, **kwargs):
        raw_data = json.load(open(rawpath, 'r'))

        allfeatures = []
        sop_id = tokenizer.convert_tokens_to_ids('[SOP]')
        eop_id = tokenizer.convert_tokens_to_ids('[EOP]')
        sep_id = tokenizer.sep_token_id
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        for item in raw_data:

            if item['split'] != split:
                continue

            input_ids = []
            id_splits = []
            persona_prefix = []
            for i, x in enumerate(item['persona_splits']):
                split_ids = tokenizer(x, add_special_tokens=False).input_ids
                id_splits.append((len(input_ids), len(input_ids) + len(split_ids)))
                input_ids.extend(split_ids)
                if i != len(item['persona_splits']) - 1:
                    persona_prefix.extend([eop_id])
            for i, x in enumerate(item['utterance_history']):
                if i == 0:
                    input_ids.append(bos_id)
                split_ids = tokenizer(x, add_special_tokens=False).input_ids
                id_splits.append((len(input_ids), len(input_ids) + len(split_ids)))
                input_ids.extend(split_ids)
                if i != len(item['utterance_history']) - 1:
                    input_ids.append(eos_id)
            
            highlights = []
            highlights.extend(item['persona_highlight_index'])
            for x in item['utterance_highlight_index']:
                highlights.append(x + len(item['persona_splits']))
                
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(item['response']).input_ids

            allfeatures.append(AttrsFeature(
                input_ids=input_ids,
                labels=labels,            
                attrs=[0] * len(id_splits),
                id_splits=id_splits,
                highlights=highlights,
            ))

            if kwargs.get("DEBUG", False) and len(allfeatures) == 20:
                break
            
        return AttrsDataset(
            datasetname=f"PersonaChat_AMTeval",
            sample_cls=AttrsFeature,
            features=allfeatures,
            drop_method={'running_eval' : True},
            split='test'
        )


class CNN_DAILYMAILDataset:
    
    def __init__(self, features : List[Feature], split):
        self.name = 'dailymail'
        self.features = features
        self.split = split
    
    def __getitem__(self, idx):
        return self.features[idx].__dict__()

    def __len__(self):
        return len(self.features)
    
    def get_name(self):
        return self.name
    
    @staticmethod
    def getAMTEvalDataset(tokenizer, rawpath, split, **kwargs):
        raw_data = json.load(open(rawpath, 'r'))
        allfeatures = []
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        for item in raw_data:
            if item['split'] != split:
                continue

            input_ids = [bos_token_id]
            id_splits = []
            highlights = []
            for article_split in item['article_splits']:
                split_ids = tokenizer(article_split, add_special_tokens=False).input_ids
                id_splits.append((len(input_ids), len(input_ids) + len(split_ids)))
                id_splits.extend(split_ids)
            input_ids.append(eos_token_id)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(item['summary'], max_length=128, truncation=True)
            allfeatures.append(AttrsFeature(
                input_ids=input_ids,
                labels=labels,            
                attrs=[0] * len(id_splits),
                id_splits=id_splits,
                highlights=item['highlight_split_index']
            ))

            if kwargs.get("DEBUG", False) and len(allfeatures) == 20:
                break

        return AttrsDataset(
            datasetname=f"CNN_Dailymail_AMT_eval",
            sample_cls=AttrsFeature,
            features=allfeatures,
            drop_method={'running_eval' : True},
            split='test'
        )
    
    @staticmethod
    def preprocess(tokenizer: PreTrainedTokenizer, 
                   split = 'train', add_Keyword=False, **kwargs):
        max_length = tokenizer.model_max_length
        max_response_length = 128
        
        def process_examples(examples, add_Keyword=False):
            model_inputs = tokenizer(examples['article'], max_length=max_length, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples['highlights'], max_length=max_response_length, truncation=True)
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
            
        if kwargs.get("DEBUG", False):
            dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
        else:
            dataset = load_dataset('cnn_dailymail', '3.0.0', split=f"{split}[:50]")
            
        dataset = dataset.map(partial(process_examples, add_Keyword=add_Keyword), batched=True, num_proc=4)
        featuresAll = [Feature(input_ids=x['input_ids'], labels=x['labels']) for x in tqdm(dataset)]
        dataset = CNN_DAILYMAILDataset(features=featuresAll, split=split)

        return dataset



def extract_keyword(text, dataset_name):
    language = "en"
    max_ngram_size = 3 if 'personachat' in dataset_name else 5
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 2 if 'personachat' in dataset_name else 3
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                         dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                         features=None)
    return [x[0] for x in kw_extractor.extract_keywords(text)]
 
def keywordOfFeature(feature, tokenizer, dataset_name, training):
    # extract keyword from raw reference in train mode
    # extract keyword from highlight in test mode
    source_text = ""
    if training:
        source_text = tokenizer.decode(feature.labels, skip_special_tokens=True)
    else:
        if feature.highlights:
            for highlight in feature.highlights:
                start, end = feature.id_splits[highlight]
                source_text += tokenizer.decode(feature.input_ids[start: end + 1], skip_special_tokens=True)
        else:
            for highlight in [0]:
                start, end = feature.id_splits[highlight]
                source_text += tokenizer.decode(feature.input_ids[start: end + 1], skip_special_tokens=True)
    return extract_keyword(source_text, dataset_name)

def toKeywordPromptFeature(feature, tokenizer, dataset_name, training):
    MAX_LENGTH = tokenizer.model_max_length
    keywords = keywordOfFeature(feature, tokenizer, dataset_name, training)
    if 'personachat' in dataset_name:
        raw_input = tokenizer.decode(feature.input_ids)
        # KEYWORD: k1, k2, k3. CONTEXT: raw_input
        sop_id = tokenizer.convert_tokens_to_ids('[SOP]')
        eos_id = tokenizer.eos_token_id
        newinput = "KEYWORD: " + ", ".join(keywords) + "." + raw_input
        newinput_ids = tokenizer(newinput, add_special_tokens=False, max_length=MAX_LENGTH).input_ids
        if (len(newinput_ids) >= MAX_LENGTH):
            newinput_ids = newinput_ids[:MAX_LENGTH - 1] + [eos_id]
        feature.input_ids = newinput_ids
        assert len(feature.input_ids) <= MAX_LENGTH, "Exceed length limit"
        return
    elif 'dailymail' in dataset_name:
        raw_input = tokenizer.decode(feature.input_ids[1:])
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        newinput = "KEYWORD: " + ", ".join(keywords) + "." + "CONTEXT:" + raw_input
        newinput_ids = [bos_id] + tokenizer(newinput, add_special_tokens=False, max_length=MAX_LENGTH).input_ids
        if (len(newinput_ids) >= MAX_LENGTH):
            newinput_ids = newinput_ids[:MAX_LENGTH - 1] + [eos_id]
        feature.input_ids = newinput_ids
        assert len(feature.input_ids) <= MAX_LENGTH, "Exceed length limit"
        return
    assert False, "Should never be here"

def toKeywordDataset(olddataset, tokenizer, training):
    for feature in olddataset.features:
        toKeywordPromptFeature(feature, tokenizer, olddataset.name.lower(), training)        
    return olddataset
