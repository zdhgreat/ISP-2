import os
import json
import random
import json
import os
import re
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any




def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()



def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print("Saved to", save_path)


def load_prompt(data_name, prompt_type):
    if data_name in ['gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps', 'gsm8k']:
        data_name = "gsm8k"
    if data_name in ['math', 'math-minival']:
        data_name = "math"
    if prompt_type in ['platypus_fs', 'wizard_zs']:
        prompt_type = "cot"
    prompt_path = "./prompts/{}/{}.md".format(prompt_type, data_name)
    if not os.path.exists(prompt_path):
        prompt_path = "./prompts/{}.md".format(prompt_type)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            prompt = fp.read().strip() + "\n\n"
    else:
        print(f"Error: prompt file {prompt_path} not found")
        prompt = ""
    return prompt



def construct_prompt(args, example):
    demo_prompt = load_prompt(args.data_name, args.prompt_type)
    if args.use_train_prompt_format:
        full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
    elif "tora" in args.prompt_type or "pot" in args.prompt_type:
        context = f"Question: {example['question']}\n\nSolution:"
        full_prompt = demo_prompt + context
    elif "cr" in args.prompt_type:
        context = f"## Problem: {example['question']}\n\nSolution: Let's think step by step."
        full_prompt = demo_prompt + context
    elif args.prompt_type in ["direct", "cot"]:
        context = f"Question: {example['question']}\nAnswer:"
        full_prompt = demo_prompt + context
    elif args.prompt_type == "pal":
        context = f"Question: {example['question']}\n\nSolution:\n"
        full_prompt = demo_prompt + context
    elif args.prompt_type == "wizard_zs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "platypus_fs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt.format(instruction=demo_prompt + f"Question: {example['question']}\nAnswer:")
    elif args.prompt_type == "dd":
        context = f"## Problem: {example['question']}\n\nSolution: Let's think step by step."
        full_prompt = context
    else:
        raise NotImplementedError(args.prompt_type)
    return full_prompt

def show_sample(sample):
    print("=="*20)
    print("idx:", sample['idx'])
    for key in ["type", "level"]:
        if key in sample:
            print("{}: {}".format(key, sample[key]))
    print("question:", sample['question'])
    if 'code' in sample:
        for code in sample['code']:
            print('-'*20)
            print("code:", code)
        print("execution", sample['report'])
    for key in ["pred", "gt", "score", "unit", "gt_cot"]:
        if key in sample:
            print("{}: {}".format(key, sample[key]))
    print()




def construct_entity_prompt(args, example):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'entity')
    if args.data_name == "StrategyQA":
        context = f"## Term: {example['term']}\n## Description: {example['description']}\n## Problem: {example['question']}\n\nLet's think step by step.:"
    else:
        context = f"## Problem: {example['question']}\n\nLet's think step by step.:"
    full_prompt = demo_prompt + context
    return full_prompt






def load_all_prompt(data_name, prompt_type, stage):
    if data_name in ['gsm-hard', 'tabmwp', 'asdiv', 'mawps', 'gsm8k']:
        data_name = "gsm8k"
    if data_name in ['math', 'math-minival']:
        data_name = "math"
    if prompt_type in ['platypus_fs', 'wizard_zs']:
        prompt_type = "cot"
    prompt_path = "./prompts/{}/{}/{}.md".format(data_name, prompt_type, stage)
    if not os.path.exists(prompt_path):
        prompt_path = "./prompts/{}.md".format(prompt_type)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            prompt = fp.read().strip() + "\n\n"
    else:
        print(f"Error: prompt file {prompt_path} not found")
        prompt = ""
    return prompt




def construct_scores_prompt(args, example, entity_hint):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'score')
    if args.data_name == "StrategyQA":
        problem_context = f"## Term: {example['term']}\n## Description: {example['description']}\n## Problem: {example['question']}\n\n"
    else:
        problem_context = f"## Problem: {example['question']}\n\n"
    entity_hint_prompt = f"### Entity Event Hints:\n"
    entity_hint_context = str(entity_hint)
    full_prompt = demo_prompt + problem_context + entity_hint_prompt + entity_hint_context + "\n\nJust score the event hints."
    #print(full_prompt)
    return full_prompt



def construct_summary_prompt(args, sample, first_entity_hint, second_entity_hint, entities_hints):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'summary')
    if args.data_name == "StrategyQA":
        problem_context = f"## Term: {sample['term']}\n## Description: {sample['description']}\n## Problem: {sample['question']}\n\n"
    else:
        problem_context = f"## Problem: {sample['question']}\n\n"
    entity_hint_prompt = f"### Entity Event Hints:\n"
    first_entity_hint_prompt = str(entities_hints.find_entity(first_entity_hint))
    second_entity_hint_prompt = str(entities_hints.find_entity(second_entity_hint))
    full_prompt = demo_prompt + problem_context + entity_hint_prompt + first_entity_hint_prompt +"\n" + second_entity_hint_prompt +"\n\nLet's think step by step."
    #print(full_prompt)
    return full_prompt


def construct_score_prompt(args, example, entity_hint, entity_name):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'score')
    if args.data_name == "StrategyQA":
        problem_context = f"## Term: {example['term']}\n## Description: {example['description']}\n## Problem: {example['question']}\n\n"
    else:
        problem_context = f"## Problem: {example['question']}\n\n"
    entity_hint_prompt = f"### Entity Event Hints:\n"
    entity_hint_context = entity_hint.find_entity(entity_name)
    full_prompt = demo_prompt + problem_context + entity_hint_prompt + entity_hint_context + "\n\nJust score the event hints."
    #print(full_prompt)
    return full_prompt



def construct_final_prompt(args, example, entities, hint):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'final_complex')
    if args.data_name == "StrategyQA":
        problem_context = f"## Term: {example['term']}\n## Description: {example['description']}\n## Problem: {example['question']}\n\n"
    else:
        problem_context = f"## Problem: {example['question']}\n\n"
    entities_prompt = f"### Entities:\n"
    entity_context = ', '.join(entities)
    entity_hint_prompt = f"\n\n### Event Hints:\n"
    entity_hint_context = hint.only_hint()
    full_prompt = demo_prompt + problem_context + entities_prompt + entity_context + entity_hint_prompt + entity_hint_context + "\n\nLet's think this through step by step with the help of hints."
    #print(full_prompt)
    return full_prompt



def construct_finalcode_prompt(args, example, entities, hint):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'final_code')
    problem_context = f"## Problem: {example['question']}\n\n"
    entities_prompt = f"### Entities:\n"
    entity_context = ', '.join(entities)
    entity_hint_prompt = f"\n\n### Event Hints:\n"
    entity_hint_context = hint.only_hint()
    full_prompt = demo_prompt + problem_context + entities_prompt + entity_context + entity_hint_prompt + entity_hint_context + "\n\nLet's think this through step by step with the help of hints."
    #print(full_prompt)
    return full_prompt


def construct_finalcode_prompt_byread(args, example, entities, hints):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'final_code')
    problem_context = f"## Problem: {example['question']}\n\n"
    entities_prompt = f"### Entities:\n"
    entity_context = ', '.join(entities)
    entity_hint_prompt = f"\n\n### Event Hints:\n"
    entity_hint_context = '\n'.join(hints)
    full_prompt = demo_prompt + problem_context + entities_prompt + entity_context + entity_hint_prompt + entity_hint_context + "\n\nLet's think this through step by step with the help of hints."
    #print(full_prompt)
    return full_prompt

def construct_final_prompt_byread(args, example, entities, hints):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'final_complex')
    if args.data_name == "StrategyQA":
        problem_context = f"## Term: {example['term']}\n## Description: {example['description']}\n## Problem: {example['question']}\n\n"
    else:
        problem_context = f"## Problem: {example['question']}\n\n"
    entities_prompt = f"### Entities:\n"
    entity_context = ', '.join(entities)
    entity_hint_prompt = f"\n\n### Event Hints:\n"
    entity_hint_context = '\n'.join(hints)
    full_prompt = demo_prompt + problem_context + entities_prompt + entity_context + entity_hint_prompt + entity_hint_context + "\n\nLet's think this through step by step with the help of hints."
    #print(full_prompt)
    return full_prompt

def construct_final_prompt_combyread(args, example):
    demo_prompt = load_all_prompt(args.data_name, args.prompt_type, 'final')
    if args.data_name == "svamp":
        problem_context = f"## Question: {example['question']}"
    full_prompt = demo_prompt + problem_context
    return full_prompt

def extract_str_entity(text):
    """
    从提供的文本中提取所有的实体名称。
    
    参数:
    - text (str): 包含实体定义的字符串。
    
    返回:
    - List[str]: 包含所有实体名称的列表。
    """
    # 使用正则表达式查找所有匹配的实体名称
    entities = re.findall(r'Entity: (.+)', text)
    return entities

def extract_str_hints(text):
    """
    从文本中提取所有带编号的行，并去除行首的编号。
    
    参数:
    - text (str): 包含带编号行的多行字符串。
    
    返回:
    - List[str]: 包含去除编号后的行内容的列表。
    """
    # 使用正则表达式找到所有带编号的行
    numbered_lines = re.findall(r'^\d+\.\s*(.*)', text, re.MULTILINE)
    return numbered_lines


if __name__ == '__main__':
    # 示例文本
    text = """
    idx:0

    Question:Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

    Entity: Breakfast
    Scores:
    0.6
    0.4
    Entity: Ducks
    Scores:
    0.8
    0.7
    0.7
    Entity: Eggs
    Scores:
    0.8
    0.6
    0.7
    0.8
    0.6
    Entity: Farmers' market
    Scores:
    0.8
    0.7
    0.9
    Entity: Friends
    Scores:
    0.7
    0.6
    Entity: Janet
    Scores:
    0.7
    0.6
    0.8
    0.9
    Entity: Muffins
    Scores:
    0.7
    0.7
    0.6
    """

    text2 = "1. John drives for 3 hours at 60 mph before realizing he forgot something at home.\n2. He attempts to return home within 4 hours, encountering 2 hours of standstill traffic.\n3. After the traffic delay, he drives at varying speeds to reach home.\n4. John drives at 30 mph for half an hour and at 80 mph for the remaining time.\n5. The total duration of John's journey is 4 hours.\n6. Calculate the total distance covered by John during the journey.\n7. Determine the distance from home at the end of 4 hours."
    # 调用函数
    entities = extract_str_entity(text)
    number_line = extract_str_hints(text2)
    print("Extracted Entities:", entities)
    for line in number_line:
        print(line)

