from collections import defaultdict
import warnings
import json
import re

FULL_RELATION_TYPES = [
    "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF",
    "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"
]


def get_confusion_counts(y_true, y_pred, labels):
    """
    计算每个类别的 TP, FP, FN, support
    返回格式：
    {
        "LABEL_1": {"tp": tp, "fp": fp, "fn": fn, "support": support},
        "LABEL_2": {"tp": tp, "fp": fp, "fn": fn, "support": support},
        ...
    }
    """
    confusion = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    
    for true, pred in zip(y_true, y_pred):
        for label in labels:
            if true == label:
                confusion[label]["support"] += 1  # 真实标签是该类别的样本数
                if pred == label:
                    confusion[label]["tp"] += 1   # True Positive
                else:
                    confusion[label]["fn"] += 1   # False Negative
            elif pred == label:
                confusion[label]["fp"] += 1       # False Positive
    
    return dict(confusion)


def compute_micro_metrics(confusion_counts, target_labels=None):
    """
    计算 Micro Precision, Recall, F1
    - 如果 target_labels=None，计算所有类别
    - 如果 target_labels=[...]，只计算这些类别的 Micro P/R/F1
    - 对于不存在的标签，会跳过并给出 warning
    """
    if target_labels is None:
        target_labels = confusion_counts.keys()
    else:
        # 检查 target_labels 是否存在
        valid_labels = []
        for label in target_labels:
            if label not in confusion_counts:
                warnings.warn(f"Label '{label}' not found in confusion_counts! Skipping...")
            else:
                valid_labels.append(label)
        target_labels = valid_labels  # 只保留有效标签
    
    # 如果所有 target_labels 都不存在，返回全零
    if not target_labels:
        warnings.warn("No valid labels found in confusion_counts! Returning zeros.")
        return {
            "Micro-P": 0.0,
            "Micro-R": 0.0,
            "Micro-F1": 0.0,
            "support": 0,
        }
    
    # 计算 TP/FP/FN 总和
    total_tp = sum(confusion_counts[label]["tp"] for label in target_labels)
    total_fp = sum(confusion_counts[label]["fp"] for label in target_labels)
    total_fn = sum(confusion_counts[label]["fn"] for label in target_labels)
    
    # 计算 Micro P/R/F1
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * (micro_p * micro_r) / (micro_p + micro_r) 
        if (micro_p + micro_r) > 0 else 0.0
    )
    
    return {
        "Micro-P": micro_p,
        "Micro-R": micro_r,
        "Micro-F1": micro_f1,
        "support": sum(confusion_counts[label]["support"] for label in target_labels),
    }


def evaluate_predictions(decoded_seq, correct_rels):
    """
    输入：
        - decoded_seq: List[str]，大模型生成的回答
        - correct_rels: List[str]，真实关系标签（字符串，如"USED-FOR"）
    输出：
        - batch_p, batch_r, batch_f1: float
    """

    y_true = []
    y_pred = []

    for pred_text, true_rel in zip(decoded_seq, correct_rels):
        _, pred_rel = get_pred_rel(pred_text)
        if pred_rel is None or pred_rel == "Fail":
            continue
        pred_rel = pred_rel.strip()
        true_rel = true_rel.strip()

        y_pred.append(pred_rel)
        y_true.append(true_rel)

    if not y_true:
        return 0.0, 0.0, 0.0

    present_labels = list(set(y_true + y_pred) & set(FULL_RELATION_TYPES))
    confusion = get_confusion_counts(y_true, y_pred, present_labels)
    metrics = compute_micro_metrics(confusion)
    return metrics["Micro-P"], metrics["Micro-R"], metrics["Micro-F1"]


def extract_reward_content(full_response: str) -> str:
    """按照用户四步思路实现的强化版结构化提取函数"""
    # Phase 1: 删除Relation Definitions到Real Data之间的中间部分
    # 分割为任务定义头和剩余部分
    header_split = re.split(
        r'^#+\s*-Relation\s*Definitions-\s*#+',  # 匹配Relation Definitions标题
        full_response, 
        flags=re.MULTILINE | re.IGNORECASE,
        maxsplit=1
    )
    processed = header_split[0].strip()  # 保留任务定义头
    
    # Phase 2: 提取Real Data之后的内容
    real_data_split = re.split(
        r'^#+\s*-Real\s*Data-\s*#+',  # 匹配Real Data标题
        header_split[-1], 
        flags=re.MULTILINE | re.IGNORECASE,
        maxsplit=1
    )
    if len(real_data_split) > 1:
        # 合并任务定义头和Real Data部分
        processed += '\n\n' + real_data_split[-1].strip()

    # Phase 3: 分割Answer前后内容
    answer_split = re.split(
        r'\nAnswer:\s*\n',  # 匹配Answer声明
        processed, 
        flags=re.IGNORECASE,
        maxsplit=1
    )
    front_part = answer_split[0].strip()
    answer_content = answer_split[1].strip() if len(answer_split) > 1 else ""

    # Phase 4: 解析并重构Answer
    parsed_relation = parse_relationship(answer_content)
    structured_answer = f'{{\"relationship\": \"{parsed_relation}\"}}'

    # 最终拼接
    return f"\n{front_part}\n\nAnswer:\n{structured_answer}"


def get_pred_rel(full_response: str):
    header_split = re.split(
        r'^#+\s*-Relation\s*Definitions-\s*#+',
        full_response, 
        flags=re.MULTILINE | re.IGNORECASE,
        maxsplit=1
    )
    processed = header_split[0].strip()
    
    real_data_split = re.split(
        r'^#+\s*-Real\s*Data-\s*#+',
        header_split[-1], 
        flags=re.MULTILINE | re.IGNORECASE,
        maxsplit=1
    )
    if len(real_data_split) > 1:
        processed += '\n\n' + real_data_split[-1].strip()

    answer_split = re.split(
        r'\nAnswer:\s*\n',
        processed, 
        flags=re.IGNORECASE,
        maxsplit=1
    )
    front_part = answer_split[0].strip()
    answer_content = answer_split[1].strip() if len(answer_split) > 1 else ""

    parsed_relation = parse_relationship(answer_content)
    structured_answer = f'{{\"relationship\": \"{parsed_relation}\"}}'

    return f"\n{front_part}", parsed_relation
    

def parse_relationship(answer_content: str) -> str:
    """
    解析 answer_content 并提取 relationship 字段。
    如果解析失败，返回 None 或者一个默认值。
    """
    # 尝试直接解析为 JSON
    try:
        # 如果 answer_content 是一个字符串，确保它是有效的 JSON
        json_str = answer_content.strip()
        # 有时候模型可能会多输出一些内容，尝试找到第一个 JSON 对象
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        data = json.loads(json_str)
        relationship = data.get("relationship", None)
        if relationship and isinstance(relationship, str):
            return relationship
        # else:
        #     print("JSON解析成功，但缺少 'relationship' 字段或字段类型不正确。")
    except json.JSONDecodeError as e:
        pass
        # print(f"JSON解析失败: {e}")
  
    # 如果直接解析失败，尝试使用正则表达式提取 relationship
    try:
        # 假设 relationship 是一个双引号包裹的字符串
        relationship_match = re.search(r'"relationship"\s*:\s*"([^"]+)"', answer_content)
        if relationship_match:
            relationship = relationship_match.group(1)
            return relationship
    except Exception as e:
        pass
        # print(f"使用正则表达式提取 'relationship' 失败: {e}")
  
    # 如果仍然失败，尝试提取第一行包含关系的词
    try:
        # 假设关系是引号中的第一个词
        relationship_match = re.search(r'\"relationship\":\s*\"(\w+)\"', answer_content)
        if relationship_match:
            relationship = relationship_match.group(1)
            return relationship
    except Exception as e:
        pass
        # print(f"尝试提取 'relationship' 时出错: {e}")
  
    # 最后，返回 None 或者一个默认关系
    # print("无法从 answer_content 中提取 'relationship'。")
    return "Fail"
