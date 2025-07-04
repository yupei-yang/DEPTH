import logging
import json

from openrlhf.models import Actor
from utils import extract_corrected_relation
from prompt import MY_Prompt_4_Refinement

from openrlhf.utils import get_strategy, get_tokenizer
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from datetime import timedelta
import argparse
import csv
import warnings
import os

POS_RELATION_TYPES = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION"]
FULL_RELATION_TYPES = POS_RELATION_TYPES + ["NO-RELATION"]


def save_confusion_matrix_csv(confusion_data, filename):
    """将混淆矩阵保存为 CSV 文件"""
    # 提取所有类别
    labels = list(confusion_data.keys())
    metrics = ["tp", "fp", "fn", "support"]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["Label"] + metrics)
        # 写入每行数据
        for label in labels:
            row = [label] + [confusion_data[label][m] for m in metrics]
            writer.writerow(row)
    log_info(f"混淆矩阵已保存到: {filename}")


def is_rank_0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_logger(logger_name):
    logger = logging.getLogger("DatasetLogger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    if is_rank_0():
        file_handler = logging.FileHandler(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_info(message):
    if is_rank_0():
        logger.info(message)


def get_data(data_path):    
    with open(data_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results


def prepare_model(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))
    
    # configure model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    actor = strategy.prepare(actor)
    actor.eval()

    return actor, tokenizer


def perform_refinement(merged_results, actor, tokenizer, args):
    results = []
    all_pairs = []  # tuple: (sample_data, (e1, e2), extract_prompt, source_entity, target_entity, true_label, cand_rel)

    # 准备所有待推理的实体对和prompt
    for sample_idx, item in enumerate(merged_results):
        predicted_rels = item["predicted_rels"]
        entities = set()
        for e1, e2, _ in predicted_rels:
            entities.update([e1, e2])
        entity_list = list(entities)

        seen = set()
        for i1 in range(len(entity_list)):
            for i2 in range(i1 + 1, len(entity_list)):
                pair = tuple(sorted([entity_list[i1], entity_list[i2]]))
                if pair not in seen:
                    # ------- 以下生成prompt及相关变量 -------
                    e1, e2 = pair
                    relation_types = FULL_RELATION_TYPES

                    # 匹配预测中的实体对
                    matched = [
                        (pred_e1, pred_e2, rel_type)
                        for pred_e1, pred_e2, rel_type in predicted_rels
                        if set([pred_e1, pred_e2]) == set([e1, e2])
                    ]
                    if matched:
                        source_entity, target_entity, cand_rel = matched[0]
                    else:
                        source_entity, target_entity = e1, e2
                        cand_rel = "None"

                    true_rels = item.get("true_rels", [])
                    true_label = next(
                        (rel_type for t1, t2, rel_type in true_rels if set([t1, t2]) == set([source_entity, target_entity])),
                        "NO-RELATION"
                    )

                    context_base = {
                        "raw_sentence": item["raw_sentence"],
                        "source_entity": source_entity,
                        "target_entity": target_entity,
                        "dp_description": item.get("sentence_llm_dp_info_4_refinement", ""),
                        "relation_types": relation_types,
                        "predicted_rels": predicted_rels,
                        "cand_rel": cand_rel,
                    }

                    try:
                        extract_prompt = MY_Prompt_4_Refinement.format(**context_base)
                    except Exception as e:
                        log_info(f"[ERROR: Prompt format failed] Pair ({source_entity}, {target_entity}): {e}")
                        continue

                    all_pairs.append((
                        item,
                        (e1, e2),
                        extract_prompt,
                        source_entity,
                        target_entity,
                        true_label,
                        cand_rel
                    ))
                    seen.add(pair)

    # ========== batch 推理 ==========
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(all_pairs), batch_size)):
        batch_end = min(batch_start + batch_size, len(all_pairs))
        batch_tuples = all_pairs[batch_start:batch_end]
        prompts = [x[2] for x in batch_tuples]

        # batch tokenization      
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}        

        with torch.no_grad():
            output = actor.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=False,
                num_beams=1,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded_answers = tokenizer.batch_decode(output, skip_special_tokens=True)

        # extract & 记录
        for bi, decoded_answer in enumerate(decoded_answers):
            item, (e1, e2), _, source_entity, target_entity, true_label, cand_rel = batch_tuples[bi]
            corrected_pairs = extract_corrected_relation(
                model_output=decoded_answer,
                true_rel=(source_entity, target_entity, true_label),
                cand_rel=(source_entity, target_entity, cand_rel)
            )
            results.extend(corrected_pairs)

            # 可选日志输出
            log_info("=" * 60)
            log_info(f"[Pair {batch_start + bi}] Sentence: {item['raw_sentence']}")
            log_info(f"[Pair {batch_start + bi}] Entity Pair: ({source_entity}, {target_entity})")
            log_info(f"[Pair {batch_start + bi}] True Relation: {true_label}")
            log_info(f"[Pair {batch_start + bi}] Candidate Relation: {cand_rel}")
            log_info(f"[Pair {batch_start + bi}] Corrected Pair: {corrected_pairs}")

    return results


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


def run_relation_correct(merged_results, actor, tokenizer, args):   
    log_info("\n===== Experiment: Refinement Module =====")
    full_results = perform_refinement(merged_results, actor, tokenizer, args)

    # 获取正确的与预测的关系
    full_true = [item[0] for item in full_results]  # true_rel
    full_pred = [item[1] for item in full_results]  # predicted_rel
    
    # 计算混淆矩阵
    d2_confusion = get_confusion_counts(full_true, full_pred, FULL_RELATION_TYPES)
    
    # 计算全局 Micro 指标
    d2_global_metrics = compute_micro_metrics(d2_confusion)
    log_info(
        f"[D2 Global Metrics] Micro-P: {d2_global_metrics['Micro-P']:.4f}, "
        f"Micro-R: {d2_global_metrics['Micro-R']:.4f}, "
        f"Micro-F1: {d2_global_metrics['Micro-F1']:.4f}"
    )

    # 计算正样本（POS_RELATION_TYPES）的 Micro 指标
    d2_pos_metrics = compute_micro_metrics(d2_confusion, target_labels=POS_RELATION_TYPES)
    log_info(
        f"[D2 POS-only Metrics] Micro-P: {d2_pos_metrics['Micro-P']:.4f}, "
        f"Micro-R: {d2_pos_metrics['Micro-R']:.4f}, "
        f"Micro-F1: {d2_pos_metrics['Micro-F1']:.4f}"
    )

    # 提取 NO-RELATION 的指标
    no_rel_metrics = d2_confusion.get("NO-RELATION", {})
    if no_rel_metrics:
        no_rel_p = (
            no_rel_metrics["tp"] / (no_rel_metrics["tp"] + no_rel_metrics["fp"])
            if (no_rel_metrics["tp"] + no_rel_metrics["fp"]) > 0 else 0.0
        )
        no_rel_r = (
            no_rel_metrics["tp"] / (no_rel_metrics["tp"] + no_rel_metrics["fn"])
            if (no_rel_metrics["tp"] + no_rel_metrics["fn"]) > 0 else 0.0
        )
        no_rel_f1 = (
            2 * (no_rel_p * no_rel_r) / (no_rel_p + no_rel_r)
            if (no_rel_p + no_rel_r) > 0 else 0.0
        )
        
        log_info(
            f"[D2 NO-RELATION Metrics] Precision: {no_rel_p:.4f}, "
            f"Recall: {no_rel_r:.4f}, F1: {no_rel_f1:.4f}"
        )
    else:
        log_info("No NO-RELATION samples in D2")
    
    # 输出详细统计量
    log_info("\n===== Detailed Stats =====")
    log_info("\nD2 (FULL) Confusion:\n" + json.dumps(d2_confusion, indent=2))

    # 保存详细结果为 JSON
    os.makedirs("examples/eval/eval_depth/results", exist_ok=True)
    with open("examples/eval/eval_depth/results/refinement_predictions.json", "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # 保存混淆矩阵（CSV 格式）
    save_confusion_matrix_csv(d2_confusion, "examples/eval/eval_depth/results/refinement_confusion_matrix.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_stage", type=int, default=3, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=True, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=True, help="Enable FlashAtten2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Models
    parser.add_argument("--pretrain", type=str, default="checkpoint/SciERC/Actor/TEST/V0", help="HF pretrain model name or path")

    # Custom dataset
    parser.add_argument("--full_data_path", type=str, default="examples/eval/eval_depth/results/merged_eval_predictions.json")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=4096, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)

    # For logger
    parser.add_argument("--save_dir", type=str, default="examples/eval/eval_depth/logs/refinement-on-test.log")

    args = parser.parse_args()

    logger = setup_logger(args.save_dir)
    merged_results = get_data(args.full_data_path)
    actor, tokenizer = prepare_model(args)
    run_relation_correct(merged_results, actor, tokenizer, args)

# deepspeed examples/eval/eval_depth/eval_refinement.py