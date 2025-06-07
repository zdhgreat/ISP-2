import os
import argparse
import re 
import torch
import guidance
import random
import shutil
import json

from guidance import models
from tqdm import tqdm
from utils.parser import *
from guidance import gen, select
from utils.python_executor import PythonExecutor
from eval.evaluate import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.entity_summary import llm_find_optimal, llm_alter_find_optimal
from utils.dataload import load_data
from utils.self_consistency import aggregate_final_answer

# 禁用 tokenizers 的并行化以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_global_data_cache = {}

def load_example(file_path):
    """
    加载 JSON 文件并返回其内容。
    如果文件已经加载过，则直接从缓存中返回。
    """
    # 检查是否已经加载过
    if file_path in _global_data_cache:
        return _global_data_cache[file_path]
    
    # 确保文件存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 加载 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 缓存数据
    _global_data_cache[file_path] = data

    return data


# 动态选择数据列表
def get_examples_section(dataset_name, format_type, datasets):
    return datasets.get(dataset_name, {}).get(format_type, [])


def split_options_with_letters(options_str):
    # 清理多余的空格和换行符
    options_str = options_str.strip()
    
    # 使用正则表达式匹配选项
    # 匹配模式：以字母加右括号开头（如 A)），后跟选项内容，直到下一个选项或字符串结束
    pattern = r'([A-Z]\))(.*?)(?=[A-Z]\)|$)'
    
    # 使用 re.findall 提取所有匹配的选项
    matches = re.findall(pattern, options_str, re.DOTALL)
    
    # 组织结果：去掉每个选项内容中的多余空格
    result = [f"{match[0]}{match[1].strip()}" for match in matches]
    
    return result



def is_string_not_convertible_to_float(var):
    # 检查变量是否是字符串类型
    if isinstance(var, str):
        try:
            # 尝试将字符串转换为浮点数
            float(var)
            # 如果转换成功，则返回False
            return False
        except ValueError:
            # 如果转换失败（抛出ValueError），则返回True
            return True
    else:
        # 如果变量不是字符串，直接返回False
        return False


def get_parser():
    parser = argparse.ArgumentParser(description="dd")
    parser.add_argument('--prompt_type', type=str, default="pot", help='prompt method')
    parser.add_argument('--format', type=str, default="pot", choices=["cot","comcot","pot"], help="how to format prompt")
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
    parser.add_argument('--max_tokens', type=int, default=256, help='max tokens')
    parser.add_argument('--sc_num', type=int, choices=range(1, 30), default=5, help='number of self consistency')
    parser.add_argument('--model', type=str, default='./Llama-3.1-8B', help='model to use')
    parser.add_argument('--dataset', type=str, default='gsm8k', help='dataset to use')
    parser.add_argument("--split", default="only_debug", type=str)
    parser.add_argument("--max_func_call", default=5, type=int)
    parser.add_argument("--max_code_fix_retries", default=4, type=int)
    # 是否开启详细日志输出，默认关闭
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--scnum", default=5, type=int)   
    # 是否将代码段进行拼接，默认不拼接
    parser.add_argument("--code_concat", action="store_true")     
    return parser





def cal_acc(args, final_outputs, samples, out_file):
    correct = 0
    final_output_dir = out_file + '/final'
    final_true_output_dir = final_output_dir + '/true'
    final_false_output_dir = final_output_dir + '/false'
    os.makedirs(final_true_output_dir, exist_ok=True)
    os.makedirs(final_false_output_dir, exist_ok=True)

    for sample, answer in zip(samples, final_outputs):
        if args.dataset == "gsm8k" or args.dataset == "svamp" or args.dataset == "AddSub":
            number_string = sample['gt'].replace(',', '')  # 删除逗号
            source_path = final_output_dir + f"/{sample['idx']}.txt"
            answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(number_string)
            print(answer_match)
            integer_part = answer.split('.')[0]
            flag = is_string_not_convertible_to_float(answer)
            if answer is None or flag == True:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
            elif float(integer_part) == float(number_string):
                correct += 1
                destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
            else:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
        
        elif args.dataset == "SQA" or args.dataset == "AQUA" or args.dataset == "CSQA":
            ground_answer = str(sample['gt'])
            source_path = final_output_dir + f"/{sample['idx']}.txt"
            answer_match = str(sample['idx']) + " answer:" + str(answer) +" ground truth:" + str(ground_answer)
            print(answer_match)
            if answer == None:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"
            elif answer.lower() == ground_answer.lower():
                correct += 1
                destination_path = final_true_output_dir + f"/{sample['idx']}.txt"
            else:
                destination_path = final_false_output_dir + f"/{sample['idx']}.txt"

        shutil.copy(source_path, destination_path)
        print(f"File copy from {source_path} to {destination_path}")

    acc = correct/len(samples)
    str_acc = "accuracy:" + str(acc)
    print(str_acc)


@guidance
def gen_cot(lm, args, examples_section, problem, options=None):
    if options:
        option_list = split_options_with_letters(options)
    
    example_output = f'''\
### Instruction:
Suppose you are a seasoned scholar tasked with analyzing  an issue. Your main objective is to solve the problem.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
'''
    
    if options and args.dataset == "CSQA":
        example_output += f'\nOptions: {options}'
    example_output += f'''
### CoT Response:
Let's think step by step.
**start**
{gen(name="thought", stop='ending', temperature=args.temperature, max_tokens=args.max_tokens)}
**ending**
'''
    if options and (args.dataset == "CSQA" or args.dataset == "AQUA" or args.dataset == "SQA"):
        example_output += f'''
### Answer Response(option choice):
**start**
{select(option_list, name="answer")}
**ending**
'''
    else:
        example_output += f'''
### Answer Response(arabic numerals):
**start**
{gen(name="answer", stop='ending', temperature=args.temperature, max_tokens=8)}
**ending**
'''
    return lm + example_output



@guidance
def gen_pot(lm, args, examples_section, problem, options=None):
    if options:
        option_list = split_options_with_letters(options)
    
    example_output = f'''\
### Instruction:
Suppose you are a seasoned scholar tasked with analyzing  an issue. Your main objective is to solve the problem.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
'''
    
    if options and args.dataset == "CSQA":
        example_output += f'\nOptions: {options}'
    example_output += f'''
### POT Code Response:
**start**
{gen(name="code", stop='ending', temperature=args.temperature, max_tokens=args.max_tokens)}
**ending**
'''
    
    return lm + example_output




@guidance
def gen_entities(lm, args, examples_section, problem, options=None):
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar with analyzing an issue. Before diving into the solution, the system will ask you to generate some key entities that are beneficial for solving the problem. 
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
'''
    if options and args.dataset == "CSQA":
        prompt += f'\n"Options": "{options}"'
    prompt += f'''\
Your main objective is to generate key entities in order to fully reveal both the explicit and implicit knowledge associated with this problem. Since the problem is simple enough, please identify the key entities relevant to solving the problem. Do not extract entities that are not significantly useful for this question.
### Response:
"Key Entities":
**start**
{gen(name="entities", stop="**ending**", temperature=args.temperature, max_tokens=64)}
**ending**
'''
    return lm + prompt



@guidance
def gen_hints_num(lm, args, examples_section, problem, entity_list, options=None):
    prompt = f"""\
### Instruction:
Suppose you are a seasoned scholar with analyzing an issue. Before diving into the solution, the system will ask you to generate some key hints that are beneficial for solving the problem. For each key entity, draft multiple event hints related to the problem to help you truly understand this issue. 
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
""" 
    if options and args.dataset == "CSQA":
        prompt += f'\n"Options": "{options}"'
    prompt += f'''\
Your main objective is to generate hints in order to fully reveal both the explicit and implicit knowledge associated with this problem. Since the problem is simple enough, please identify the key hints relevant to solving the problem.
### Response:
"Hints":
**start**
'''
    lm += prompt
    for entity in entity_list:
        lm += f'''{entity}:\nHere are "{gen(name="num", stop=['"', 'hints', ':'], list_append=True, temperature=args.temperature, max_tokens=8)}" hints:\n'''
    return lm 


@guidance
def gen_hints(lm, args, examples_section, problem, entity_list, num_list, options=None):
    prompt = f"""\
### Instruction:
Suppose you are a seasoned scholar with analyzing an issue. Before diving into the solution, the system will ask you to generate some key hints that are beneficial for solving the problem. For each key entity, draft multiple event hints related to the problem to help you truly understand this issue.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
""" 
    if options and args.dataset == "CSQA":
        prompt += f'\n"Options": "{options}"'
    prompt += '''\
Your main objective is to generate hints in order to fully reveal both the explicit and implicit knowledge associated with this problem. Since the problem is simple enough, please identify the key hints relevant to solving the problem.
### Response:
"Hints":
**start**
'''   
    lm += prompt
    for entity, num in zip(entity_list, num_list):
        lm += f'''"{entity}":\nHere are {num} hints.\n'''
        for i in range(num):
            lm += f'''"{i+1}.{gen(name="hints", list_append=True, stop='"', temperature=args.temperature, max_tokens=128)}"\n'''
    return lm 


@guidance
def gen_score(lm, args, examples_section, problem, hint_list, options=None):
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar with analyzing an issue. In understanding this problem, key entities within the question and corresponding event hints for these entities are provided to help you truly understand the problem. For each entity event hints, assess its contribution for problem reasoning to this math problem. 
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
'''
    if options and args.dataset == "CSQA":
        prompt += f'\n"Options": "{options}"'
    prompt += '''\
Your primary goal is to score the event hints of key entities to fully reveal their priority relevance to the problem. Additionally, your scoring should be differentiated from other scores to reflect the reliability rating of each hint.
### Response:
"Hints Score":
**start**
'''    
    lm += prompt
    for entity, hints in hint_list:
        lm += f'''{entity}\n'''
        for i, hint in enumerate(hints, start=1):
            lm += f'''{i}.{hint}. --Score:"{gen(name="score", list_append=True, stop=['"','--'], max_tokens=8, temperature=args.temperature)}"---\n'''
    return lm 



@guidance
def gen_summary(lm, examples_section, problem, hint_list):
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar with analyzing an issue. When understanding this problem, we will provide the two key entities with the highest priority of reasoning in the problem, as well as the corresponding event hints of the entities to help you truly understand the problem. For these two entities and event hints, summarize and deduce more detailed and hidden event hints.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
"Problem": {problem}
Your primary goal is to summarize and explore the given two key entities and event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the problem.
"Hints":
'''
    lm += prompt
    hint_instruction = ""
    for entity, hints in hint_list:
        hint_instruction += f'''{entity}:\n'''
        for i, hint in enumerate(hints, start=1):
            hint_instruction += f'''{i}."{hint}"\n'''
    lm += hint_instruction
    lm += f'''### Respone\n"Summary Hints":\n**start**{{\n{gen(name="all", stop=["}", "ending"], max_tokens=256, temperature=0.7)}\n}}**ending**'''
    return lm



@guidance
def gen_summary_hints(lm, examples_section, problem, hint_list, key_entity, num):
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar with analyzing an issue. When understanding this problem, we will provide the two key entities with the highest priority of reasoning in the problem, as well as the corresponding event hints of the entities to help you truly understand the problem. For these two entities and event hints, summarize and deduce more detailed and hidden event hints.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
Your primary goal is to summarize and explore the given two key entities and event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the problem.
"Hints":
'''
    lm += prompt
    hint_instruction = ""
    for entity, hints in hint_list:
        hint_instruction += f'''{entity}\n'''
        for i, hint in enumerate(hints, start=1):
            hint_instruction += f'''{i}."{hint}"\n'''
    lm += hint_instruction
    lm += f'''### Respone\n"Summary Hints":\n**start**\n{key_entity}:\nHere are {num} hints:\n'''
    for i in range(int(num)):
        lm += f'''{i+1}."{gen(name="hints", list_append=True, stop='"', temperature=args.temperature, max_tokens=args.max_tokens)}"\n'''
    return lm


@guidance
def gen_res(lm, args, problem, entity_list=None, hints_list=None, final_hints=None, options=None):
    # 加载 JSON 数据
    if options:
        option_list = split_options_with_letters(options)
    elif args.dataset == "SQA":
        option_list = ['Yes', 'No']

    if args.prompt_type == "OE" and hints_list is None and final_hints is None:
        example_data = load_example("prompts/guidance_llm/oe_llama_prompt.json")
        # 动态选择 COT 数据
        examples_section = oe_prompt(get_examples_section(args.dataset, args.format, example_data))
        return lm + _gen_oe(examples_section, problem, entity_list, option_list, options)
    
    elif args.prompt_type == "OIP" and final_hints is None and hints_list is not None:
        example_data = load_example("prompts/guidance_llm/oip_llama_prompt.json")
        # 动态选择 OIP_COT 数据
        examples_section = oip_prompt(get_examples_section(args.dataset, args.format, example_data))
        return lm + _gen_oip(examples_section, problem, hints_list, option_list, options)
    
    else:
        example_data = load_example("prompts/guidance_llm/isp_llama_prompt.json")
        if args.dataset in ["gsm8k", "svamp", "AddSub"]:
            # 动态选择 CLOZE_ISP_COT 数据
            examples_section = isp_prompt_cloze(get_examples_section(args.dataset, args.format, example_data))
            return lm + _gen_isp_cloze(examples_section, problem, entity_list, final_hints)
        
        elif args.dataset == "CSQA":
            format_prompt = "cot" if args.format == "comcot" else args.format
            # 动态选择 CSQA 或者 SQA数据
            examples_section = isp_prompt_MC(args, get_examples_section(args.dataset, format_prompt, example_data))
            return lm + _gen_isp_MC(examples_section, problem, entity_list, final_hints, option_list=option_list, options=options)
        
        elif args.dataset == "SQA":
            format_prompt = "cot" if args.format == "comcot" else args.format
            examples_section = isp_prompt_MC(args, get_examples_section(args.dataset, args.format, example_data))
            return lm + _gen_isp_MC(examples_section, problem, entity_list, final_hints, option_list=option_list)
        elif args.dataset == "AQUA":
            format_prompt = "cot" if args.format == "comcot" else args.format
            # 动态选择 AQUA 数据
            examples_section = isp_prompt_MC(args, get_examples_section(args.dataset, args.format, example_data))
            return lm + _gen_isp_MC(examples_section, problem, entity_list, final_hints, option_list=option_list)



@guidance
def _gen_oe(lm, args, examples_section, problem, entity_list, option_list=None, options=None):
    entity_instruction = ', '.join(entity_list)
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar tasked with analyzing an issue. Your main objective is to solve the problem.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
'''
    if args.dataset in ["gsm8k", "AddSub", "svamp"]:
        prompt = f'''\
### Input:
Problem: {problem}
Entities: {entity_instruction}
### CoT Response:
Let's think step by step.
**start**
{gen(name="thought", stop='"', temperature=args.temperature)}
**ending**
### Answer Response(arabic numerals):
**start**
{gen(name="answer", stop='"', temperature=args.temperature)}
**ending**
'''
    elif args.dataset in ["AQUA", "CSQA", "SQA"] and option_list is not None:
        prompt += f'''\
### Input:
Problem: {problem}
'''
        if args.dataset == 'CSQA':
            prompt += f'''\nOptions:{options}'''
        prompt += f'''
Entities: {entity_instruction}
### CoT Response:
Let's think step by step.
**start**
{gen(name="thought", stop='"', temperature=args.temperature)}
**ending**
### Answer Response(option choice):
**start**
{select(option_list, name="answer")}
**ending**
'''
    lm += prompt
    return lm





@guidance
def _gen_oip(lm, args, examples_section, problem, hints_list, option_list=None, options=None):
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar tasked with analyzing an issue. Your main objective is to solve the problem.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
'''
    hint_instruction = ""
    for entity, hints in hints_list.items():
        hint_instruction += f'''{entity}\n'''
        for i, hint in enumerate(hints, start=1):
            hint_instruction += f'''{i}.{hint}."\n'''
    
    if args.dataset in ["gsm8k", "AddSub", "svamp"]:
        prompt += f'''\
Hints: 
{hint_instruction}
### CoT Response:
Let's think step by step.
**start**
{gen(name="thought", stop='ending', temperature=args.temperature, max_tokens=256)}
**ending**
### Answer Response(arabic numerals):
**start**
{gen(name="answer", stop='ending', temperature=args.temperature)}
**ending**
'''
    elif args.dataset in ["AQUA", "CSQA", "SQA"] and option_list is not None:
        if args.dataset == "CSQA":
            prompt += f'''\nOptions:{options}'''
        prompt += f'''
Hints: 
{hint_instruction}
### CoT Response:
Let's think step by step.
**start**
{gen(name="thought", stop='ending', temperature=args.temperature, max_tokens=256)}
**ending**
### Answer Response(option choice):
**start**
{select(option_list, name="answer")}
**ending**
'''
    lm += prompt
    return lm



@guidance
def _gen_isp_cloze(lm, examples_section, problem, entity_list, final_hints):
    entity_instruction = ', '.join(entity_list)
    numbered_hints = [f"  {i + 1}. {hint}" for i, hint in enumerate(final_hints)]
    hint_instruction = "\n".join(numbered_hints)
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar tasked with analyzing an issue. Your main objective is to solve the problem.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
Entities: {entity_instruction}
Hints: 
{hint_instruction}
### CoT Response:
Let's think step by step.
**start**
{gen(name="thought", stop='ending', temperature=args.temperature, max_tokens=256)}
**ending**
### Answer Response(arabic numerals):
**start**
{gen(name="answer", stop='ending', temperature=args.temperature, max_tokens=16)}
**ending**
'''
    lm += prompt
    return lm




@guidance
def _gen_isp_MC(lm, examples_section, problem, entity_list, final_hints, option_list=None, options=None):
    entity_instruction = ', '.join(entity_list)
    numbered_hints = [f"  {i + 1}. {hint}" for i, hint in enumerate(final_hints)]
    hint_instruction = "\n".join(numbered_hints)
    prompt = f'''\
### Instruction:
Suppose you are a seasoned scholar tasked with analyzing an issue. Your main objective is to solve the problem.
----
{{~! display the example format for the problem ~}}
{examples_section}
{{~! place the real question at the end ~}}
### Input:
Problem: {problem}
'''
    if options:
        prompt += f'''\nOptions: "{options}"'''
    prompt += f'''
Entities: {entity_instruction}
Hints: 
{hint_instruction}
### CoT Response:
Let's think step by step.
**start**
{gen(name="thought", stop='ending', temperature=args.temperature, max_tokens=256)}
**ending**
### Answer Response(option choice):
**start**
{select(option_list, name="answer")}
**ending**
'''
    lm += prompt
    return lm





def cot_prompt(examples):
    examples_section = ""
    for example in examples:
        problem = example.get("problem", "")  # 获取问题
        options = example.get("options", None)  # 获取选项（如果存在）
        cot = example.get("COT", "")  # 获取推理过程
        answer = example.get("answer", "")  # 获取答案
        # 构建每个示例的输出部分
        example_output = f'''\
### Input:
"Problem": "{problem}"'''
        
        # 如果存在选项，则添加到输出中
        if options:
            example_output += f'\n"Options": "{options}"'
        
        # 添加推理和答案部分
        example_output += f'''\
### CoT Response:
Let's think step by step.
**start**
{cot}
**ending**
'''
        if options and (args.dataset == "CSQA" or args.dataset == "AQUA" or args.dataset == "SQA"):
            example_output += f'\n### Answer Response(option choice):'
        else:
            example_output += f'\n### Answer Response(arabic numerals):'

        example_output += f'''
**start**
{answer}
**ending**
'''
        # 将当前示例追加到总输出中
        examples_section += example_output
    
    return examples_section



def pot_prompt(examples):
    examples_section = ""
    for example in examples:
        problem = example.get("problem", "")  # 获取问题
        pot = example.get("code", "")  # 获取推理过程
        # 构建每个示例的输出部分
        example_output = f'''\
### Input:
"Problem": "{problem}"'''
        
        
        # 添加推理和答案部分
        example_output += f'''\
### POT Code Response:
**start**
```python
{pot}
```
**ending**
'''
        # 将当前示例追加到总输出中
        examples_section += example_output
    
    return examples_section





def entity_prompt(args, examples):
    examples_section = ""
    for example in examples:
        problem = example["problem"]
        options = example.get("options", None)  # 获取选项（如果存在）
        key_entities = "\n".join([f"- {entity}" for entity in example["key_entities"]])
        example_output = f'''\
### Input:
"Problem": "{problem}"'''
        
        if options and args.dataset == "CSQA":
            example_output += f'\n"Options": "{options}"'

        example_output += f"""\
Your main objective is to generate key entities in order to fully reveal both the explicit and implicit knowledge associated with this problem. Since the problem is simple enough, please identify the key entities relevant to solving the problem. Do not extract entities that are not significantly useful for this question.
### Response:
"Key Entities":
**start**
{key_entities}
**ending**
"""
            
        examples_section += example_output
        
    return examples_section



def hint_prompt(args, examples):
    examples_section = ""
    for example in examples:
        problem = example["problem"]
        options = example.get("options", None)  # 获取选项（如果存在）
        hint_entities = ""
        example_output = f'''\
### Input:
"Problem": "{problem}"'''
        
        if options and args.dataset == "CSQA":
            example_output += f'\n"Options": "{options}"'

        for entity in example['entity_event_hints']:
            hint_entities += f'''"{entity['key']}":\n'''
            hint_entities += f"Here are {len(entity['hints'])} hints.\n"
            hint_entities += '\n'.join([f'"{hint}"' for hint in entity['hints']]) + '\n'
        example_output += f"""\
Your main objective is to generate hints in order to fully reveal both the explicit and implicit knowledge associated with this problem. Since the problem is simple enough, please identify the key hints relevant to solving the problem.
### Response:
"Hints":
**start**
{hint_entities}
**ending**
"""
        examples_section += example_output
        
    return examples_section



def score_prompt(args, examples):
    examples_section = ""
    for example in examples:
        problem = example["problem"]
        options = example.get("options", None) 
        hint_score = ""
        example_output = f'''\
### Input:
Problem: {problem}'''
        
        if options and args.dataset == "CSQA":
            example_output += f'\n"Options": "{options}"'

        for entity in example['entity_event_hints']:
            hint_score += f"{entity['key']}\n"
            hint_score += '\n'.join([f'{hint}' for hint in entity['hints']]) + '\n'
        example_output += f"""\
Your primary goal is to score the event hints of key entities to fully reveal their priority relevance to the problem. Additionally, your scoring should be differentiated from other scores to reflect the reliability rating of each hint.
### Response:
"Hints Score":
**start**
{hint_score}
**ending**
"""
        examples_section += example_output

    return examples_section



def summary_prompt(examples):
    examples_section = ""
    for example in examples:
        problem = example["problem"]
        summary_hint = ""
        entity_hint = ""
        for entity in example['entity_event_hints']:
            entity_hint += f"{entity['key']}:\n"
            entity_hint += '\n'.join([f'"{hint}"' for hint in entity['hints']]) + '\n'
        for entity in example['summary_hints']:
            summary_hint += f'''"{entity['key']}":\n'''
            summary_hint += '\n'.join([f'"{hint}"' for hint in entity['hints']]) + '\n'
        example_output = f'''\
### Input:
"Problem": {problem}\n'''

        example_output += f"""\
Your primary goal is to summarize and explore the given two key entities and event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the problem.
"Hints":
{entity_hint}
### Response:
"Summary Hints":
**start**{{
{summary_hint}
}}**ending**
"""
        examples_section += example_output

    return examples_section


def oip_prompt(args, examples):
    examples_section = ""
    for example in examples:
        problem = example["problem"]
        options = example.get("options", None)  # 获取选项（如果存在）
        entity_hint = ""
        for entity in example['entity_event_hints']:
            entity_hint += f"{entity['key']}\n"
            entity_hint += '\n'.join([f'"{hint}"' for hint in entity['hints']]) + '\n'
        entities = ', '.join(example['key_entities'])
        Thought = example["COT"]
        Answer = example["answer"]
        example_output = f'''\
### Input:
"Problem": "{problem}"'''

        if options and (args.dataset == "CSQA" or args.dataset == "AQUA" or args.dataset == "SQA"):
            example_output += f'\n"Options": "{options}"'

        example_output += f"""\
Your primary goal is to summarize and explore the given event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the problem.
Entities: {entities}
Hints:
{entity_hint}
### Response:
**start**
Let's think step by step.
{Thought}
**ending**
"""       
        if options and (args.dataset == "CSQA" or args.dataset == "AQUA" or args.dataset == "SQA"):
            example_output += f'\n### Answer Response(option choice):'
        else:
            example_output += f'\n### Answer Response(arabic numerals):'
        
        example_output += f'''\
**start**
{Answer}
**ending**
'''

        examples_section += example_output     

    return examples_section



def oe_prompt(args, examples):
    examples_section = ""
    for example in examples:
        options = example.get("options", None)  # 获取选项（如果存在）
        problem = example["problem"]
        entities = ', '.join(example['key_entities'])
        Thought = example["COT"]
        Answer = example["answer"]
        example_output = f'''\
### Input:
"Problem": "{problem}"
'''
        

        if options and args.dataset == "CSQA":
            example_output += f'\nOptions: {options}'

        example_output += f"""
Your primary goal is to summarize and explore the given key entities, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the problem.
Entities: {entities}
### CoT Response:
Let's think step by step.
**start**
{Thought}
**ending**
"""      
        if options and (args.dataset == "CSQA" or args.dataset == "AQUA" or args.dataset == "SQA"):
            example_output += f'\n### Answer Response(option choice):'
        else:
            example_output += f'\n### Answer Response(arabic numerals):'
        
        example_output += f'''\
**start**
{Answer}
**ending**
'''
        
        examples_section += example_output
    return examples_section       



def isp_prompt_cloze(examples):
    examples_section = ""
    for example in examples:
        problem = example["problem"]
        Thought = example['COT']
        Answer = example['answer']
        entities = ', '.join(example['key_entities'])
        Hints = '\n '.join(example['hints'])
        examples_section += f'''\
### Input:
Problem: {problem}
Your primary goal is to summarize and explore the given two key entities and event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the problem.
Entities: {entities}
Hints:
{Hints}
### CoT Response:
Let's think step by step.
**start**
{Thought}
**ending**
### Answer Response(arabic numerals):
**start**
{Answer}
**ending**
'''
    return examples_section


def isp_prompt_MC(args, examples):
    examples_section = ""
    for example in examples:
        problem = example["problem"]
        options = example.get("options", None)
        Thought = example['COT']
        Choice = example['choice'] 
        entities = ', '.join(example['key_entities'])
        Hints = '\n '.join(example['hints'])
        example_output = f'''\
### Input:
Problem: {problem}'''
        
        if options and args.dataset == "CSQA":
            example_output += f'\nOptions: "{options}"'

        examples_section += f'''\
Your primary goal is to summarize and explore the given two key entities and event hints, in order to fully deduce summary event hints and hidden reasoning knowledge that can solve the problem.
Entities: {entities}
Hints:
{Hints}
### CoT Response:
Let's think step by step.
**start**
{Thought}
**ending**
### Answer Response(option choice):
**start**
{Choice}
**ending**
'''
    return examples_section



def entity_convert_output(str_entity):
    entity_list = [line.strip().lstrip('- ').strip() for line in str_entity.split('\n') if line.strip()]
    return entity_list


def hints_convert_num(str_num):
    num_list = []
    for string in str_num:
        match = re.search(r'\d+', string)  # 查找第一个连续数字
        if match:
            num = int(match.group())    # 转换为整数
        else:
            # 处理没有数字的情况（根据需求选择
            num = 3  # 默认值方案
            # 或者抛出异常：raise ValueError(f"No number found in: {hint_str}")
        num_list.append(num)
    return num_list



def hint_convert_output(hints, count_list, entity_list):
    """
    根据数量列表切割hint列表，并关联对应实体

    Args:
    hints (list): 原始hint列表
    count_list (list): 每个子列表应包含的元素数量列表
    entity_list (list): 与count_list对应的实体标识列表

    Returns:
    list: 包含(实体, 子列表)元组的嵌套列表结构

    Raises:
    ValueError: 当count_list和entity_list长度不一致时

    输出：
    [
    ('FirstGroup', [
        'There are 4 friends who came to visit Jerome.',
        'The first friend pressed on the doorbell 20 times before Jerome opened.'
    ]),
    ('SecondGroup', [
        'The fourth friend pressed on the doorbell 60 times.',
        'The first friend pressed on the doorbell 60 times.'
    ])
    ]
    """

    if len(count_list) != len(entity_list):
        raise ValueError("count_list和entity_list的长度必须一致")
        
    result = []
    current_index = 0
    for count, entity in zip(count_list, entity_list):
        sub_list = hints[current_index:current_index + count]
        if not sub_list:  # 当剩余元素不足时终止循环
            break
        result.append((entity, sub_list))
        current_index += count
    return result




def score_convert_output(score_list, num_list, entity_list):
    """
    根据 score_list、num_list 和 entity_list 生成每个实体对应的分数。
    
    参数:
        score_list (list): 分数列表。
        num_list (list): 每段的长度列表。
        entity_list (list): 实体列表，长度应与 num_list 相同。
    
    返回:
        list: 每个实体对应的分数列表。
    """
    # 检查输入的有效性
    if len(num_list) != len(entity_list):
        raise ValueError("num_list 和 entity_list 的长度必须相等")
    
    # 将 score_list 中的字符串转换为浮点数
    try:
        score_list = [float(score) for score in score_list]
    except ValueError:
        raise ValueError("score_list 中的元素必须是可以转换为浮点数的字符串或数字")
    
    resc_list = []  # 存储每段分数的平均值
    start_index = 0  # 切割的起始索引

    for i, count in enumerate(num_list):
        # 根据当前的 count 值切割 score_list
        segment = score_list[start_index:start_index + count]
        
        # 计算当前段的平均分
        if len(segment) > 0:  # 防止空列表导致除以零的错误
            average_score = sum(segment) / len(segment)
        else:
            average_score = 0  # 如果段为空，默认平均分为 0
        
        # 将当前实体和对应的分数添加到结果列表中
        resc_list.append((entity_list[i], average_score))
        
        # 更新起始索引
        start_index += count

    return resc_list



def extract_entities_hints(hint_list, target_entities):
    """
    从hint_list中提取多个指定实体的hints，并保持原有格式
    
    Args:
        hint_list (list): 包含(实体, 子列表)元组的嵌套列表结构
        target_entities (list): 要提取的目标实体名称列表
    
    Returns:
        list: 包含目标实体及其对应hints的嵌套列表结构（与hint_list格式一致）
    """
    result = []
    for entity, hints in hint_list:
        if entity in target_entities:
            result.append([entity, hints])
    return result



def entity_to_text(entity, out_file):
    with open(out_file, 'w') as file:
        if len(entity) == 0:
            file.write("No entity")
        for item in entity:
            file.write(item + '\n')    


def hints_to_txt(hint_list, out_file):
    with open(out_file, 'w') as file:
        if len(hint_list) == 0:
            file.write('No entity and hint')
        else:
            for title, hints in hint_list:  # 直接遍历列表中的元组
                file.write(title + ':\n')  # 写入标题
                for index, hint in enumerate(hints, start=1):
                    file.write(f"{index}. {hint}\n")  # 写入带编号的提示


def last_hints_to_txt(final_list, out_file):
    with open(out_file, 'w') as file:
        if len(final_list) == 0:   
            file.write("No final hints")
        else:
            for index, hint in enumerate(final_list, start=1):
                file.write(f"{index}. {hint}\n")  

def final_answer_to_txt(answer, thought, out_file, final_output):
    with open(out_file, 'w') as file:
        final_output += f"Thought:\n{thought}\nAnswer:\n{answer}"
        file.write(final_output)
             


def extract_summary(input_str):
    # 分割行并过滤空行
    lines = [line.strip() for line in input_str.split('\n') if line.strip()]
    
    result = []
    current_title = None
    current_hints = []
    
    for line in lines:
        # 去除首尾的双引号
        stripped_line = line.strip('"')
        
        # 检测标题行（以冒号结尾）
        if re.search(r":$", stripped_line):
            # 遇到新标题时保存前一个标题的内容
            if current_title is not None:
                result.append([current_title, current_hints])
            current_title = stripped_line[:-1].strip()  # 去掉末尾冒号
            current_hints = []
        else:
            # 提取hint内容（去掉数字前缀）
            hint = re.sub(r"^\d+\.\s*", "", stripped_line).strip()
            current_hints.append(hint)
    
    # 添加最后一个标题的内容
    if current_title:
        result.append([current_title, current_hints])
    
    return result




def extract_summary_newentitiy(input_string):
    """
    提取字符串中第一个换行符（\n）之前的部分。

    参数:
        input_string (str): 输入的字符串。

    返回:
        str: 第一个换行符之前的部分。如果字符串中没有换行符，则返回整个字符串。
    """
    # 使用字符串的 split 方法按 '\n' 分割，最多分割一次
    parts = input_string.split('\n', 1)
    # 返回分割后的第一部分
    return parts[0]


def extract_summary_number(input_string):
    """
    提取字符串中第一个换行符前的数字。
    如果没有换行符，则提取整个字符串中的数字。
    
    参数:
        s (str): 输入的字符串。
        
    返回:
        int 或 None: 提取到的数字。如果未找到数字，则返回 None。
    """
    # 检查是否有换行符
    if '\n' in input_string:
        # 分割字符串，取第一个换行符前的部分
        before_newline = input_string.split('\n')[0]
    else:
        # 如果没有换行符，使用整个字符串
        before_newline = input_string
    
    # 使用正则表达式提取数字
    match = re.search(r'\d+', before_newline)
    if match:
        number_str = match.group()  # 提取匹配到的第一个数字（字符串形式）
        # 将数字字符串拆分为单个数字字符并重新组合
        single_digits_str = ''.join([char for char in number_str])
        return single_digits_str
    else:
        return None  # 如果没有找到数字，返回 None



def validate_hints_strict(hints):
    """
    严格验证 hints 列表结构：
    1. 必须是列表且长度为1（只允许一个章节）
    2. 唯一的元素必须是长度为2的列表
    3. 第一个元素是字符串（标题）且不能是纯空白
    4. 第二个元素是字符串列表（至少1个元素），每个字符串不能是纯空白
    """
    # 直接检查根列表长度是否为1
    if not (isinstance(hints, list) and len(hints) == 1):
        return False
    
    item = hints[0]
    
    # 检查章节结构是否为长度为2的列表
    if not (isinstance(item, list) and len(item) == 2):
        return False
    
    title_part, content_part = item
    
    # 检查标题是否符合要求
    if not (isinstance(title_part, str) and title_part.strip()):
        return False
    
    # 检查内容部分是否是列表且至少包含一个非空字符串
    if not (isinstance(content_part, list) and len(content_part) >= 1):
        return False
    
    # 检查内容中的每个元素是否为非空字符串
    for line in content_part:
        if not (isinstance(line, str) and line.strip()):
            return False
    
    return True


def merge_titles(data, scores):
    # 创建统一的标题清理函数
    def clean_title(title):
        return title.strip('"')
    
    # 处理内容合并
    content_dict = {}
    for item in data:
        title = clean_title(item[0])
        if title in content_dict:
            content_dict[title].extend(item[1])
        else:
            content_dict[title] = item[1].copy()
    
    # 处理分数合并（保留最高分）
    score_dict = {}
    for title, score in scores:
        cleaned_title = clean_title(title)
        # 保留最高分
        if cleaned_title in score_dict:
            if score > score_dict[cleaned_title]:
                score_dict[cleaned_title] = score
        else:
            score_dict[cleaned_title] = score
    
    # 建立最终分数字典（以data为准）
    final_scores = {}
    # 1. 处理data中存在的标题
    for title in content_dict:
        if title in score_dict:
            final_scores[title] = score_dict[title]
        else:
            # 生成0-1之间的随机数补充
            final_scores[title] = round(random.random(), 2)
    
    # 2. 删除score中多余的标题（不需要处理，因为直接以data为准）
    
    # 生成最终结果（保持顺序一致）
    merged_data = []
    merged_scores = []
    for title in content_dict:  # 利用字典的有序性
        merged_data.append([title, content_dict[title]])
        merged_scores.append((title, final_scores[title]))
    
    return merged_data, merged_scores


def convert_semicolon_to_newline(code_str):
    """
    将分号分隔的Python代码转换为换行符格式
    示例输入: 'a=1; b=2; print(a+b)\n'
    示例输出: 'a=1\nb=2\nprint(a+b)'
    """
    # 按分号分割语句
    statements = code_str.split(';')
    
    # 清理每个语句并过滤空行
    cleaned = [
        stmt.strip()  # 去除前后空格
        for stmt in statements
        if stmt.strip() != ''  # 跳过空语句
    ]
    
    # 用换行符连接，并移除原字符串末尾的换行符
    return '\n'.join(cleaned).rstrip('\n')


def gen_pot_answer(llama, args, executor, code_output_file, problem, options=None):
    max_func_call = 1 if args.prompt_type in ['cot', 'pot'] else args.max_func_call
    MAX_CODE_FIX_RETRIES = args.max_code_fix_retries
    pot_example_data = load_example("prompts/guidance_llm/pot_llama_prompt.json")
    for epoch in range(max_func_call):
        pass_key = False
        print("=" * 50, "Epoch program", epoch)  
        examples_pot_section = pot_prompt(get_examples_section(args.dataset, args.format, pot_example_data))
        lm = llama + gen_pot(args, examples_pot_section, problem, options)   
        code = lm['code'] 
        code_output = extract_program(code)
        run_code = convert_semicolon_to_newline(code_output)
        code_result = executor.batch_apply(run_code)
        pred, report = code_result[0]
        pred, report = str(pred).strip(), str(report).strip()
        exec_result = pred if pred else report

        if exec_result == "":
            exec_result += "<warning>\nDid you forget to use print()?\n</warning>\n"
        elif "Error" in exec_result:
            # Split the query string
            split_query = examples_pot_section.split("Tried Times: 0")

            # Check if the split result has at least one element and if the last element is not empty
            if split_query and split_query[-1]:
                # Count the occurrences of the warning message in the last part of the split query
                tried_times = split_query[-1].count("<warning>\nThe previous code block") + 1
            else:
                # If the split result is empty or the last element is empty, set tried_times to 0
                tried_times = 0 
            # Convert the integer tried_times to a string and append the warning message to exec_result
            if tried_times <= (MAX_CODE_FIX_RETRIES - 1):
                if args.verbose: print("Errors haven been occured.\n<extracted program>\n", code_output, "\n</extracted program>\n")
                exec_result += "<warning>\nThe previous code block is not executable, will be removed from the code execution context. Please rewrite and fix this code block. (Tried Times: " + str(tried_times) + ")\n</warning>\n"
                if args.code_concat and tried_times >= 1:
                    exec_result += "<current_full_code_context>\n" + code_output + "\n</current_full_code_context>\n"
            else:
                exec_result += "<warning>\nYou have tried to execute the code block " + str(tried_times) + " times, but it is still not executable. Please stop writing code for this question and try to solve this question manually.\n</warning>\nLet's think step by step, without using code. "
        else:
            pass_key = True        

        if pass_key == True:
            with open(code_output_file, 'w') as f:
                f.write(code_output)
            break
        else:
            if epoch == max_func_call - 2:
                examples_pot_section += "\n<system>\nReach the max reponse limit, you must finish your reasoning and give your final solution in next reponse without resorting python code.\n</system>\n"
            elif epoch == max_func_call - 1:
                examples_pot_section += "\nReach max function call limit."
            else:
                examples_pot_section += exec_result

    match = re.search(r'\d+\.?\d*', exec_result)
    first_number = 0
    if match:
        first_number = int(float(match.group()))
        print("The first number found is:", first_number)
    else:
        print("No number found.")  
    return float(first_number), True  





def iter_summary(llama, args, problem, score_list, or_hints_list, num_list, output_for_file, for_output):
    hints_list = [list(item) for item in or_hints_list]
    score_example_date = load_example("prompts/guidance_llm/score_prompt.json")
    summary_example_data = load_example("prompts/guidance_llm/summary_prompt.json")
    max_func_call = args.max_func_call
    final_hints = None
    if len(score_list) > 0 and len(hints_list) > 0 and len(score_list) == len(hints_list):
        for epoch in range(max_func_call):
            hints_list, score_list = merge_titles(hints_list, score_list)
            if len(hints_list) == 0 and len(score_list) == 0:
                first_hint = []
                final_hints = first_hint
                break
            print("=" * 50, "Epoch", epoch)
            print(hints_list)
            print(score_list)      
            if len(hints_list) == 1:
                for_output += str(hints_list)
                # 提取第一个提示的 hints 部分
                first_hint = hints_list[0]  # 获取子列表中的第二个元素（即 hints）
                final_hints = first_hint
                break      
            if args.prompt_type == "SAISP":
                first_entity, second_entity, match = llm_alter_find_optimal(hints_list, score_list)
            else:
                first_entity, second_entity, match = llm_find_optimal(hints_list, score_list)#这里要重写一个函数
            review_turn = 0
            
            examples_score_section = score_prompt(args, get_examples_section(args.dataset, args.format, score_example_date))
            examples_hint_section = summary_prompt(get_examples_section(args.dataset, args.format, summary_example_data))
            while not match:
                if review_turn >= 5:
                    break
                lm = llama + gen_score(examples_score_section, problem, hints_list)
                score_list = score_convert_output(lm['score'], num_list)
                if args.prompt_type == "SAISP":
                    first_entity, second_entity, match = llm_alter_find_optimal(hints_list, score_list)
                else:
                    first_entity, second_entity, match = llm_find_optimal(hints_list, score_list)#这里要重写一个函数
                review_turn += 1
            
            if review_turn >= 5:
                for_output += str(hints_list)
                final_hints = next(iter(hints_list.values()))
                break

            select_hints = extract_entities_hints(hints_list, [first_entity, second_entity])
            summary_flag = False
            review_turn = 0 
            while not summary_flag:
                if review_turn > 6:
                    # 添加空列表检查
                    selected_entity = [item for item in hints_list if item[0] == second_entity]
                    new_list = selected_entity
                    break
                
                lm = llama + gen_summary(examples_hint_section, problem, select_hints)
                new_list = extract_summary(lm['all'])
                
                # 添加对空列表的防御
                if not new_list:
                    new_list = []  # 或者设置默认值
                
                if validate_hints_strict(new_list):
                    summary_flag = True
                review_turn += 1        


            print("=" * 50, 'new list\n', new_list)
            for_output = for_output + str(hints_list) + "\n\n"

            remove_entity_hints_inplace(hints_list, first_entity)
            score_list = remove_score(score_list, first_entity)
            print("After removing first_entity hints:", hints_list)
            print("Length:", len(hints_list))  # 应为 1（假设只剩 second_entity）

            if len(new_list) == 1 and len(new_list[0]) == 2 and len(new_list[0][1]) >= 1:
                remove_entity_hints_inplace(hints_list, second_entity)
                score_list = remove_score(score_list, second_entity)
                print("After removing second_entity hints:", hints_list)
                print("Final length:", len(hints_list)) 

                hints_list.extend(new_list)
                print("After extending with new_list:", hints_list)
                print("Length:", len(hints_list)) 

            if len(hints_list) == 1:
                for_output += str(hints_list)
                # 提取第一个提示的 hints 部分
                first_hint = hints_list[0]  # 获取子列表中的第二个元素（即 hints）
                final_hints = first_hint
                break

            score_flag = False   
            loop_count = 0 

            new_num_list = []
            new_entity_list = []
            while not score_flag:
                loop_count += 1

                if len(new_list) != 1 or len(new_list[0]) == 0:
                    break

                lm = llama + gen_score(args, examples_score_section, problem, new_list)
                for title, hints in new_list:
                    count = len(hints)
                    new_num_list.append(count)
                    new_entity_list.append(title)
                new_score_list = score_convert_output(lm['score'], new_num_list, new_entity_list)
                if any(new_entity_list[0] == entity for entity, score in new_score_list):
                    score_flag = True
                    score_list.extend(new_score_list)
                if loop_count > 5:
                    score_list[new_entity_list[0]] = random.random()
                    break
            

    with open(output_for_file, 'w') as f:
        if final_hints == None or len(final_hints) == 0:
            f.write("No iterative hints")
        else:
            f.write(for_output)

    return final_hints







def remove_entity_hints_inplace(hint_list, target_entities):
    """
    原地从hint_list中删除指定实体及其对应的hints
    
    Args:
        hint_list (list): 包含(实体, 子列表)元组的嵌套列表结构
        target_entities (str or list): 要删除的目标实体名称（单个或多个）
    """
    # 如果target_entities是单个字符串，则转换为列表以便统一处理
    if isinstance(target_entities, str):
        target_entities = [target_entities]
    
    # 使用列表推导式过滤，并重新赋值给原列表
    hint_list[:] = [[entity, hints] for entity, hints in hint_list if entity not in target_entities]



def remove_score(score_list, entities_to_remove):
    """
    从 score_list 中删除指定的实体及其对应的分数。
    
    参数:
        score_list (list): 每个元素是一个元组 (entity, score)。
        entities_to_remove (list or str): 需要删除的实体名称（可以是单个字符串或列表）。
    
    返回:
        list: 删除指定实体后的 score_list。
    """
    # 如果 entities_to_remove 是单个字符串，则将其转换为列表
    if isinstance(entities_to_remove, str):
        entities_to_remove = [entities_to_remove]
    
    # 使用列表推导式过滤掉需要删除的实体
    filtered_score_list = [(entity, score) for entity, score in score_list if entity not in entities_to_remove]
    
    return filtered_score_list




def parse_options(option_str):
    """将选项字符串转换为标准化的选项列表"""
    # 正则匹配所有 "字母)数值+单位" 模式的选项
    options = re.findall(r'([A-Za-z]\)\s*[\d.%+-]+[\s\S]*?(?=\s+[A-Za-z]\)|$))', option_str)
    # 清理多余空格并标准化格式（例如去掉末尾空格）
    return [opt.strip() for opt in options]

def find_closest_option(option_str, answer):
    """主函数：输入选项字符串和答案，返回最接近的选项标签"""
    answer = float(answer)
    options = parse_options(option_str)
    
    closest_label = None
    min_diff = float('inf')
    
    for option in options:
        # 匹配标签和数值部分（允许数值后存在单位符号）
        match = re.match(r'^([A-Za-z])\)\s*([-+]?\d+\.?\d*)\D*', option)
        if not match:
            raise ValueError(f"Invalid option format: {option}")
        label = match.group(1)
        value = float(match.group(2))
        
        diff = abs(value - answer)
        if diff < min_diff or (diff == min_diff and closest_label is None):
            min_diff = diff
            closest_label = label
    
    return closest_label




def main(args):
    if "Llama" in args.model:
        llama3 = models.Transformers(
            args.model, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
    else:
        #deepseek旗下模型，guidance版本问题以及需要原模型的参数
        llama3 = models.Transformers(
            args.model, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            tokenizer=AutoTokenizer.from_pretrained("Qwen2.5-7B-Instruct-Int8-W8A16")
        )

    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # llama3 = models.Transformers(
    #     "DeepSeek-R1-Distill-Llama-8B", 
    #     device_map="auto", 
    #     torch_dtype=torch.bfloat16,
    #     tokenizer=AutoTokenizer.from_pretrained("llama-3.1-8B")
    # )

    questions = load_data(args.dataset, args.split)
    model_name = args.model.split("/")[1]
    out_file = f'outputs/{model_name}/{args.prompt_type}/{args.format}/{args.dataset}'

    samples = []


    for question in tqdm(questions, total=len(questions)):

        if args.dataset in ["gsm8k", "svamp", "AddSub", "AQUA"]:
            if args.dataset in ["gsm8k", "svamp", "AQUA"]:
                idx = question.get('idx', -1)  # 如果没有 idx，就设为 -1 或者其他默认值
            else:
                idx = question['qid']

            # parse question and answer
            question['question'] = parse_question(question, args.dataset)
            gt_cot, gt_ans = parse_math_ground_truth(question, args.dataset)
            if args.dataset in ["gsm8k", "svamp", "AddSub"]:
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'gt': gt_ans}
            else:
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'options': question['options'], 'gt': gt_ans}

            # add remain fields
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in question:
                    sample[key] = question[key]
            samples.append(sample)  

        elif args.dataset in ["SQA", "CSQA"]:
            idx = question['idx']
            question['question'] = parse_question(question, args.dataset)
            if args.dataset == "SQA":
                gt_ans, gt_cot = parse_sqa_ground_truth(question)
            else:
                gt_cot, gt_ans = parse_math_ground_truth(question, args.dataset)
            #gt_ans = question['answer']
            if args.dataset == "CSQA":
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'options': question['options'], 'gt': gt_ans}
            else:
                sample = {'idx': idx, 'question': question['question'], 'gt_cot': gt_cot, 'gt': gt_ans}
            for key in ['level', 'type', 'subject', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type']:
                if key in question:
                    sample[key] = question[key]
            samples.append(sample)    

    print("dataset:", args.dataset, "samples:", len(samples))


    sc_turn = args.sc_num
    answer_lists = []



    cot_example_data = load_example("prompts/guidance_llm/cot_llama_prompt.json")
    entity_example_data = load_example("prompts/guidance_llm/entity_prompt.json")
    hint_example_data = load_example("prompts/guidance_llm/hint_prompt.json")
    score_example_date = load_example("prompts/guidance_llm/score_prompt.json")


    for sample in tqdm(samples, desc=f'{args.prompt_type}'):
        problem = sample['question']
        idx = sample['idx']
        options = None
        if args.dataset in ["AQUA", "CSQA"]:
            options = sample['options']
        en_dir = out_file + '/entity'
        for_dir = out_file + '/for'
        final_dir = out_file + '/final'
        hint_dir = out_file + '/hint'
        last_dir = out_file + '/last'
        os.makedirs(for_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        self_consistency_answer = []
        thought_lists = []
        final_file = final_dir + f'/{idx}.txt'
        final_output = f"idx:{idx}\n\nSelf-Consistency:{sc_turn}\n\nQuestion:{sample['question']}\n\nGround:{sample['gt']}\n\n"
        for step in range(sc_turn):
            flag = False
            print("=" * 50, f"idx:{idx}\tself-consistency epoch 1-5:", step)
            if args.prompt_type == "cot" or args.prompt_type == "comcot" or args.prompt_type == "pot":
                gen_res_turn = 0
                while not flag:
                    gen_res_turn += 1
                    if args.prompt_type == "pot":
                        code_out_dir = out_file + '/code' + f'/{idx}'
                        os.makedirs(code_out_dir, exist_ok=True)
                        code_out_file = out_file + '/code' + f'/{idx}' + f'/{step}.txt'
                        answer, flag = gen_pot_answer(llama3, args, executor, code_out_file, problem, options)
                        if args.dataset == "AQUA":
                            print(options)
                        print(answer)
                        if args.dataset == "AQUA":
                            self_consistency_answer.append(find_closest_option(options, int(answer)))
                            print(find_closest_option(options, int(answer)))
                            final_output = final_output + "\n\nself-consistency epoch:" + str(step)  + "\n\nanswer:" + str(find_closest_option(options, int(answer)))
                        else:
                            self_consistency_answer.append(answer)
                            final_output = final_output + "\n\nself-consistency epoch:" + str(step)  + "\n\nanswer:" + str(answer)

                    else:   
                        examples_section = cot_prompt(get_examples_section(args.dataset, args.format, cot_example_data))
                        lm = llama3 + gen_cot(args, examples_section, problem, options)
                        final_thought = lm['thought']
                        print(final_thought)
                        answer = lm['answer']
                        if args.dataset == "SQA":
                            match = re.search(r'\b(Yes|No)\b', answer, re.IGNORECASE)
                            final_answer = "None"
                        elif args.dataset == "AQUA" or args.dataset == "CSQA":
                            match = re.search(r'(?<![A-Za-z0-9])[A-E](?![A-Za-z0-9])', answer)
                            final_answer = "None"
                        else:
                            match = re.search(r'[\d\.]+', answer)
                            final_answer = "None"
                        if match:
                            final_answer = match.group(0)
                            print("Extracted answer:", final_answer)
                            flag = True
                            thought_lists.append(final_thought)
                            self_consistency_answer.append(final_answer)
                            final_output = final_output + "\n\nself-consistency epoch:" + str(step) + final_thought + "\n\nanswer:" + str(final_answer)
                        else:
                            print("No number found.")
                            if gen_res_turn > 5:
                                flag = True
                                final_answer = "None"
                                self_consistency_answer.append(final_answer)
                                final_output = final_output + "\n\nself-consistency epoch:" + str(step) + final_thought + "\n\nanswer:" + str("None")

            else:

                if args.prompt_type in ["OE", "OIP", "SAISP", "ISP"]:
                    #生成实体文件路径
                    en_sc_dir = en_dir + '/sc' + str(step)
                    os.makedirs(en_sc_dir, exist_ok=True)
                    entity_sc_file = en_sc_dir + f'/{idx}.txt'

                    #生成实体
                    examples_section = entity_prompt(args, get_examples_section(args.dataset, args.format, entity_example_data))
                    lm = llama3 + gen_entities(args, examples_section, problem, options)
                    entity_list = entity_convert_output(lm["entities"])
                    entity_to_text(entity_list, entity_sc_file)
                    final_output += f"Entities:\n{entity_list}"
                
                if args.prompt_type in ["OIP", "SAISP", "ISP"]:
                    #生成提示文件路径
                    hint_sc_dir = hint_dir + '/sc' + str(step)
                    os.makedirs(hint_sc_dir, exist_ok=True)
                    hint_sc_file = hint_sc_dir + f'/{idx}.txt'   


                    #生成Hint
                    examples_section = hint_prompt(args, get_examples_section(args.dataset, args.format, hint_example_data))
                    lm = llama3 + gen_hints_num(args, examples_section, problem, entity_list, options)
                    num_list = hints_convert_num(lm['num'])
                    lm = llama3 + gen_hints(args, examples_section, problem, entity_list, num_list, options)
                    hints_list = hint_convert_output(lm['hints'], num_list, entity_list)
                    hints_to_txt(hints_list, hint_sc_file)

                if args.prompt_type in ["SAISP", "ISP"]:
                    #生成迭代文件路径
                    for_sc_dir = for_dir + '/sc' + str(step)
                    os.makedirs(for_sc_dir, exist_ok=True)
                    output_for_file = for_sc_dir + f'/{idx}.txt'
                    for_output = f"idx:{idx}\n\nQuestion:{sample['question']}\n\n"
                    #最终生成提示文件路径
                    last_sc_dir = last_dir + '/sc' + str(step)
                    os.makedirs(last_sc_dir, exist_ok=True)
                    last_sc_file = last_sc_dir + f'/{idx}.txt'
                    #生成分数
                    examples_section = score_prompt(args, get_examples_section(args.dataset, args.format, score_example_date))
                    lm = llama3 + gen_score(args, examples_section, problem, hints_list, options)
                    score_list = score_convert_output(lm['score'], num_list, entity_list)
                    #进行迭代操作
                    final_hints = iter_summary(llama3, args, problem, score_list, hints_list, num_list, output_for_file, for_output)
                    if final_hints == None:
                        final_hints = []
                    last_hints_to_txt(final_hints, last_sc_file)
                    final_output += f"Final Hints:\n{final_hints}"


                #输出答案文件路径
                final_sc_dir = final_dir + '/sc' + str(step)
                os.makedirs(final_sc_dir, exist_ok=True)
                final_sc_file = final_sc_dir + f'/{idx}.txt'


                #答案生成
                flag = False
                gen_res_turn = 0
                while not flag:
                    gen_res_turn += 1
                    final_lm = llama3 + gen_res(args, problem, entity_list=entity_list, hints_list=hints_list, final_hints=final_hints, options=options)
                    answer = final_lm['answer']
                    print(answer)
                    final_thought = final_lm['thought']
                    print(final_thought)
                    if args.dataset == "SQA":
                        match = re.search(r'\b(Yes|No)\b', answer, re.IGNORECASE)
                        final_answer = "None"
                    elif args.dataset == "AQUA" or args.dataset == "CSQA":
                        match = re.search(r'(?<![A-Za-z0-9])[A-E](?![A-Za-z0-9])', answer)
                        final_answer = "None"
                    else:
                        match = re.search(r'[\d\.]+', answer)
                        final_answer = "None"
                    if match:
                        final_answer = match.group(0)
                        print("Extracted answer:", final_answer)
                        final_answer_to_txt(final_answer, final_thought, final_sc_file, final_output)
                        self_consistency_answer.append(final_answer)
                        flag = True
                    else:
                        print("No number found.")
                        if gen_res_turn > 5:
                            flag = True
                            final_answer = None
                            final_answer_to_txt(final_answer, final_thought, final_sc_file, final_output)



        common_answer = aggregate_final_answer(self_consistency_answer)
        answer_lists.append(common_answer) 
        final_output = final_output + "\n\nfinal common answer:" + str(common_answer)

        with open(final_file, 'w') as f:
            f.write(final_output)
    
    cal_acc(args, answer_lists, samples, out_file)




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)