import os
import json
import chardet
import torch
import random
import numpy as np

def setSeed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        data = open(path, "rb").read()
        charset = chardet.detect(data)["encoding"]
        with open(path, "r", encoding=charset) as f:
            return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_record(path):
    with open(path, "r") as f:
        for line in f:
            line = json.loads(line.strip())
            yield line


def write_json_record(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_json(path):
    return json.loads(read_file(path))


def write_json(path, obj):
    write_file(path, json.dumps(obj))


def createIfNotExist(path):
    path = os.path.expandvars(path)
    if not os.path.exists(path):
        os.mkdir(path)


