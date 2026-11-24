import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import json
import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import urllib.parse
import socket
import urllib

def get_empatheticdialogues(
    name: str,
    split: str,
    silent: bool = False,
    cache_dir: str = None,
    num_turns: int = 1,
    data_fraction: float = 1.0
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    '''
        {
        'conv_id': 'hit:0_conv:1',
        'utterance_idx': 1,
        'context': 'sentimental',
        'prompt': 'I remember going to the fireworks ...',
        'speaker_idx': 1,
        'utterance': 'I remember going to see the fireworks ...',
        'selfeval': '5|5|5_2|2|5',
        'tags': ''
        }
    '''
    debug = False
    debug = True
    

    # 从cache 加载：无法联网从本地下载之后上传
    # dataset_path = "/slurm/home/yrd/cailab/yinmin/.cache/huggingface/datasets/facebook___empathetic_dialogues"
    # print(f"Loading dataset from local path: {dataset_path}")
# 
    # dataset = load_dataset(dataset_path, split=split)

    # 本地 csv 文件路径
    data_file = f"C:/Users/Vsky/Desktop/ccl/personality/code/LoRI/dataset/empatheticdialogues/{split}.csv"

    print(f"Loading dataset from: {data_file}")

    # 直接从本地 CSV 加载
    dataset = load_dataset("csv", data_files={split: data_file})[split]

    print(">>> dataset type:", type(dataset))   # 打印 dataset 类型
    print(">>> dataset len:", len(dataset))    # 打印数据量

    # dataset = load_dataset("facebook/empathetic_dialogues", split=split, cache_dir=cache_dir)

    if debug:
        print(f"Dataset columns: {dataset.column_names}")
        print("First 3 samples:")
        for i in range(min(3, len(dataset))):
            print(dataset[i])

    num_conversations = len(dataset)
    dataset = dataset.select(range(int(num_conversations * data_fraction)))

    data = defaultdict(lambda: defaultdict(list))

    def generate_prompt(context, prompt_text):
        return f"""Below is a conversation context. Write the next response.

### Context:
{context}

### Prompt:
{prompt_text}

### Response:
"""

    for i in tqdm.tqdm(range(len(dataset)), desc=f'Processing {name}', disable=silent):
        
        row = dataset[i]   # row is dict

        if i == 0:
            print(">>> first row type:", type(row))   # 打印 row 的类型
            print(">>> first row content:", row)      # 打印 row 的内容

        context = row.get("context", "")
        prompt_text = row.get("prompt", "")
        response = row.get("utterance", "")

        prompt = generate_prompt(context, prompt_text)

        data[prompt]["sft_target"] = response
        data[prompt]["pairs"] = []
        data[prompt]["responses"] = []

    return data

data = get_empatheticdialogues(
    name="empathetic_dialogues",
    split="train",   # 可以是 "train" "test" "valid"
    cache_dir=None
)

print("样本数量:", len(data))
for k in list(data.keys())[:1]:
    print("Prompt 示例:\n", k)
    print("Target 示例:\n", data[k]["sft_target"])
