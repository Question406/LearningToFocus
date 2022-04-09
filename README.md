# Learning To Focus

This repo contains the code and our collected dataset with highlight information, details in [Controlling the Focus of Pretrained Language Generation Models](https://arxiv.org/abs/2203.01146).

## Download dataset
CNN_Dailymail dataset is loaded by huggingface datasets and you don't need to provide a specific path to the dataset files.

Personachat dataset is downloaded within ParlAI and you should provide a specific path to stored directory when loading it, see `train.py` for loading detail.

## Some scripts 
You can change `task` argument to train on different datasets.

finetune base model on two datasets
```bash 
python train.py --task dailymail --do_finetune --lr 1e-5
python train.py --task personachat --do_finetune --lr 1e-5
```
attribute on dataset cnn_dailymail with finetuned model
```bash
python attribute.py --task dailymail --finetune_model_path {trained_model_dir} --save_dir {save_attrs_dir}
```
train focus vector on finetuned model
```bash
python train.py --task dailymail --do_focus_train --attn_type layer --finetune_model_path {trained_model_dir} --attr_dir {save_attrs_dir} --lr 1e-5 
```
evaluate trained model on AMT dataset
```bash 
python evaluate.py --task dailymail --do_amt_eval --pretrained_model_path {pretrained_model_dir}
```

## Highlight dataset sample
We put the AMT collected highlight dataset at `resource/AMT_dataset/`, here is a sample
```json
{
    "article_splits": ["Nairobi, Kenya (CNN)University of Nairobi students were terrified Sunday morning when they heard explosions -- caused by a faulty electrical cable -- and believed it was a terror attack, the school said", "..........", " said Vice Chancellor Peter M", "F. Mbithi in a statement", " He called on the students, staff and public to remain calm", " CNN's Lillian Leposo reported from Nairobi and Ashley Fantz wrote this story in Atlanta"],
    "summary": "Students stampeded; some jumped from a fifth story at a dorm; one student died, school officials say.\nThe blasts were caused by faulty electrical cable, and Kenya Power is at the school.\nThe panic came less than two weeks after terrorists attacked Kenya's Garissa University.",
    "highlight_split_index": [0, 1, 3, 4, 9], // which of the split is highlight to the summary
    "split" : "test"
}
```

## Cite
```bibtex
@article{ji2022controlling,
  title={Controlling the Focus of Pretrained Language Generation Models},
  author={Ji, Jiabao and Kim, Yoon and Glass, James and He, Tianxing},
  journal={arXiv preprint arXiv:2203.01146},
  year={2022}
}
```
