import re
import torch
import os
from get_datasets import get_batch_iterator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import pad_to_length, all_gather_if_needed
from rouge_score import rouge_scorer
from peft import PeftModel
import numpy as np
import argparse
import yaml
from eval_model import evaluate

cache_dir = os.getenv("PROJECT_CACHE", "~/.cache")

commonsense_tasks = [
    'boolq',
    'piqa',
    'social_i_qa',
    'arc-challenge',
    'arc-easy',
    'openbookqa',
    'hellaswag',
    'winogrande',
]

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')
parser.add_argument('--adapter_path', type=str, default=['mistralai/Mistral-7B-v0.1'], nargs='+')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--sample', action='store_true')
parser.add_argument('--datasets', type=str, default='commonsense,gsm8k')
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('--combination_type', type=str, default='cat')
parser.add_argument('--density', type=float, default=0.7)
parser.add_argument('--weights', type=float, default=[1.0, 1.0], nargs='+')  # 改为2个权重
parser.add_argument('--adapter_names', type=str, default=['adapter1', 'adapter2'], nargs='+')  # 新增：自定义适配器名称

args = parser.parse_args()

# 处理数据集
args.datasets = args.datasets.replace("commonsense", ','.join(commonsense_tasks))
args.datasets = args.datasets.split(',')

# 验证适配器路径和权重数量
if len(args.adapter_path) != 2:
    raise ValueError("merge_2_loras requires exactly 2 adapter paths")
if len(args.weights) != 2:
    raise ValueError("merge_2_loras requires exactly 2 weights")

# 读取模型配置
model_yaml_path = os.path.join('config', 'model', f"{args.model_name}.yaml")
with open(model_yaml_path, 'r') as file:
    model_config = yaml.safe_load(file)
    load_path = model_config.get('name_or_path')

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    load_path,
    torch_dtype=torch.float32,
    device_map='auto',
)

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(load_path)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.resize_token_embeddings(len(tokenizer))

# 加载第一个适配器作为基础
model = PeftModel.from_pretrained(base_model, args.adapter_path[0], adapter_name=args.adapter_names[0])

# 加载第二个适配器
model.load_adapter(args.adapter_path[1], adapter_name=args.adapter_names[1])

# 定义适配器组合（只有2个）
adapters = [args.adapter_names[0], args.adapter_names[1]]
adapter_name = "merge"

# 按权重合并两个适配器
model.add_weighted_adapter(
    adapters, 
    args.weights, 
    adapter_name, 
    combination_type=args.combination_type, 
    density=args.density
)
model.set_adapter("merge")
model.merge_and_unload()

# 评估
for run_idx in range(args.num_runs):
    print(f"Run {run_idx + 1}/{args.num_runs}")
    for dataset in args.datasets:
        print(f"Evaluating on dataset: {dataset}")
        acc = evaluate(dataset, model, tokenizer, args)
        
        # 保存结果
        os.makedirs(os.path.join(args.results_path, "results"), exist_ok=True)
        with open(f"{args.results_path}/results/merge_2_loras.txt", "a") as f:
            f.write(f"Run: {run_idx + 1}\n")
            f.write(f"Model: {args.adapter_path}\n")
            f.write(f"Adapter names: {args.adapter_names}\n")
            f.write(f"Combination_type: {args.combination_type}, density: {args.density}, weights: {args.weights}\n")
            f.write(f"Dataset: {dataset}, Accuracy: {acc * 100:.2f}%\n")
            f.write("-" * 50 + "\n")
        
        print(f"Dataset: {dataset}, Accuracy: {acc * 100:.2f}%")

print("Evaluation completed!")