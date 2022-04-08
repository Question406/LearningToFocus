import numpy as np
from typing import List
from transformers import BartTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class DataColltorForFocusTrain:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    
    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        
        input_ids = [feature["attn_input_ids"] for feature in features] if "attn_input_ids" in features[
            0].keys() else None
        if input_ids is not None:
            max_length = max(len(l) for l in input_ids)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_length - len(feature["attn_input_ids"]))
                feature["attn_input_ids"] = (
                    feature["attn_input_ids"] + remainder if padding_side == "right" else remainder + feature[
                        "attn_input_ids"]
                )
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        
        return features



def split2sentence(input_ids: List[int], tokenizer: PreTrainedTokenizer, **kwargs):
    if isinstance(tokenizer, BartTokenizer):
        DEBUG = kwargs.pop('debug', False)
        
        split_marks = ['.', '?', '!', '."', '?"']
        split_ids = [tokenizer.convert_tokens_to_ids(x) for x in split_marks]
        split_ids.extend(tokenizer(' '.join(split_marks), add_special_tokens=False).input_ids)
        split_ids.extend([x for x in tokenizer.all_special_ids if x != tokenizer.pad_token_id])
        # split_ids.extend([x for x in tokenizer.all_special_ids]) split_ids.append(50118)  # / '\n', used in cnn_dailymail labels
        split_ids.append(479)  # another r'.'
        
        split_pos = []  # position of all split tokens
        for i, id in enumerate(input_ids):
            if id in split_ids:
                if len(split_pos) != 0:
                    if i - split_pos[-1] <= 3:
                        continue  # omit those sentences that're too short
                
                split_pos.append(i)
        
        res_splits = []
        
        # split tokens
        def not_working(start_pos, end_pos):
            if DEBUG:
                print('try: ', tokenizer.decode(input_ids[start_pos + 1: end_pos]))
            if end_pos - start_pos <= 4:
                return True
            if tokenizer.decode(input_ids[end_pos - 1:end_pos + 1]).strip().lower() in ['mr.', 'mrs.', 'jr.', ] or \
                    tokenizer.decode(input_ids[end_pos - 2:end_pos + 1]).strip().lower() in ['mr.', 'mrs.', 'jr.'] or \
                    tokenizer.decode(input_ids[end_pos - 3:end_pos + 1]).strip().lower() in ['u.s', 'u. s.']:
                return True
            return False
        
        start_pos = split_pos[0]
        end_pos = split_pos[1]
        nowidx = 1
        while True:
            if not_working(start_pos, end_pos):
                nowidx += 1
                if nowidx == len(split_pos):
                    break
                end_pos = split_pos[nowidx]
            else:
                res_splits.append((start_pos + 1, end_pos - 1))
                start_pos = end_pos
                nowidx += 1
                if nowidx == len(split_pos):
                    break
                end_pos = split_pos[nowidx]
        
        if DEBUG:
            print("RAW: ", tokenizer.decode(input_ids))
            print("Splited: ")
            for pos in res_splits:
                print(tokenizer.decode(input_ids[pos[0]: pos[1] + 1]))
        
        return res_splits
    else:
        # return persona, utterances
        return_split = False
        sop_id = tokenizer.convert_tokens_to_ids('[SOP]')
        eop_id = tokenizer.convert_tokens_to_ids('[EOP]')
        sep_id = tokenizer.sep_token_id
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        
        sop_pos = input_ids.index(sop_id)
        eop_pos = input_ids.index(eop_id)
        persona_split_pos = [sop_pos]
        persona_split_pos.extend(np.where(np.array(input_ids[sop_pos:eop_pos]) == sep_id)[0].tolist())
        persona_splits = [(persona_split_pos[i] + 1, persona_split_pos[i + 1] - 1) for i in
                          range(len(persona_split_pos) - 1)]
        utt_splits = []
        offset = eop_pos + 1
        last_bos = None
        for i, id in enumerate(input_ids[eop_pos + 1:]):
            if last_bos is None and id == bos_id:
                last_bos = i
                continue
            if last_bos is not None and id == eos_id:
                utt_splits.append((last_bos + 1 + offset, i + offset - 1))
                last_bos = None
        if return_split:
            return persona_splits, utt_splits
        else:
            persona_splits.extend(utt_splits)
            return persona_splits