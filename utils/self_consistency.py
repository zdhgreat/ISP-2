import random
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any


# 根据给定的prompt和question，通过语言模型采样生成多个输出
def sample_outputs(prompt: str, question: str, language_model, num_samples: int) -> List[Tuple[str, str]]:
    sampled_outputs = []
    for _ in range(num_samples):
        output = language_model.generate(prompt, question)  # 调用模型生成答案
        reasoning_path, answer = parse_output(output)  # 解析输出为推理路径和答案
        sampled_outputs.append((reasoning_path, answer))  # 收集所有输出
    return sampled_outputs

# 解析生成的输出为推理路径和答案
def parse_output(output: str) -> Tuple[str, str]:
    reasoning_path = output.split('Answer:')[0].strip()  # 分离推理路径
    answer = output.split('Answer:')[1].strip()  # 分离答案
    return reasoning_path, answer

# 获取所有输出中应该被拒绝的原因
def get_rejection_reasons(sampled_outputs: List[Tuple[str, str]]) -> Dict[str, Any]:
    rejection_reasons = {}
    for reasoning_path, answer in sampled_outputs:
        reason = check_rejection_reason(reasoning_path, answer)  # 检查每个输出是否有拒绝的理由
        if reason:
            rejection_reasons[reasoning_path] = reason  # 如果有拒绝理由，则记录
    return rejection_reasons

# 检查推理路径和答案是否有被拒绝的理由
def check_rejection_reason(reasoning_path: str, answer: str) -> Optional[Any]:
    if "incorrect" in reasoning_path.lower():  # 示例：如果推理路径中包含"incorrect"，则认为是错误的
        return "Incorrect reasoning"
    return None

# 根据拒绝的理由调整输出
def adjust_outputs(sampled_outputs: List[Tuple[str, str]], rejection_reasons: Dict[str, Any]) -> List[Tuple[str, str]]:
    adjusted_outputs = []
    for reasoning_path, answer in sampled_outputs:
        if reasoning_path not in rejection_reasons:
            adjusted_outputs.append((reasoning_path, answer))  # 如果没有被拒绝，则保留原样
        else:
            adjusted_reasoning_path = adjust_reasoning_path(reasoning_path, rejection_reasons[reasoning_path])  # 如果有拒绝理由，则进行调整
            adjusted_outputs.append((adjusted_reasoning_path, answer))
    return adjusted_outputs

# 调整推理路径
def adjust_reasoning_path(reasoning_path: str, rejection_reason: Any) -> str:
    return reasoning_path + " [adjusted based on feedback: {}]".format(rejection_reason)  # 返回调整后的推理路径

# 聚合所有的答案
def aggregate_answers(sampled_outputs: List[Tuple[str, str]]) -> List[str]:
    answers = [answer for _, answer in sampled_outputs]  # 提取所有答案
    return answers

# 找出最一致的答案
def find_most_consistent_answer(aggregated_answers: List[str]) -> str:
    counter = Counter(aggregated_answers)  # 统计每个答案出现的次数
    most_consistent_answer, _ = counter.most_common(1)[0]  # 获取出现次数最多的答案
    return most_consistent_answer

# 执行自洽检查
def self_consistency(prompt: str, question: str, language_model, num_samples: int) -> str:
    sampled_outputs = sample_outputs(prompt, question, language_model, num_samples)  # 生成输出
    rejection_reasons = get_rejection_reasons(sampled_outputs)  # 获取拒绝理由
    adjusted_outputs = adjust_outputs(sampled_outputs, rejection_reasons)  # 调整输出
    aggregated_answers = aggregate_answers(adjusted_outputs)  # 聚合答案
    most_consistent_answer = find_most_consistent_answer(aggregated_answers)  # 找到最一致的答案
    return most_consistent_answer



def aggregate_final_answer(self_consistency_answers: List[Any]) -> Any:
    """
    聚合一批自洽性检查得到的答案，选择最一致的答案作为最终答案。

    参数:
    - self_consistency_answers: 包含多个答案的列表。

    返回:
    - 最一致的答案。
    """
    # 使用Counter来统计每个答案出现的频次
    answer_counter = Counter(self_consistency_answers)
    
    common_answers = answer_counter.most_common()
    
    # 检查最常见的答案是否为"None"，如果是则选择第二多的答案
    if common_answers[0][0] == "None":
        if len(common_answers) > 1:
            return common_answers[1][0]
        else:
            return None  # 如果列表中只有"None"，返回None或其他默认值
    else:
        return common_answers[0][0] 

# 示例使用
self_consistency_answers = ["Answer A", "Answer B", "Answer A", "Answer C", "Answer A", "Answer B"]
final_answer = aggregate_final_answer(self_consistency_answers)
print("The most consistent answer is:", final_answer)
