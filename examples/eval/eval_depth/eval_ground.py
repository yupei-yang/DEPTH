import logging
import json

from openrlhf.models import Actor
from utils import get_predicted_rel, extract_real_data_info
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


def get_predictions(eval_data, actor, tokenizer, args, collect_details=False):
    results = []
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, 12, batch_size)):
        batch_end = min(batch_start + batch_size, len(eval_data))
        batch = eval_data[batch_start:batch_end]
        questions = [ex["question"] for ex in batch]
        true_rels = [ex["correct"] for ex in batch]
        raw_sentences = [ex["raw_sentence"] for ex in batch]
        sentence_llm_infos = [ex["sentence_llm_dp_info_4_refinement"] for ex in batch]

        sentence_entity_pairs = [extract_real_data_info(q) for q in questions]
        sentences, entity1s, entity2s = zip(*sentence_entity_pairs)

        inputs = tokenizer(
            questions,
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
        predicted_rels = [get_predicted_rel(ans) for ans in decoded_answers]

        for idx in range(len(batch)):
            log_info("=" * 50)
            log_info(f"[Sample {batch_start + idx}]")
            log_info(f"Question:\n{questions[idx]}")
            log_info(f"True Relation: {true_rels[idx]}")
            log_info(f"Predicted Relation: {predicted_rels[idx]}")

            if collect_details:
                results.append({
                    "raw_sentence": raw_sentences[idx],
                    "sentence": sentences[idx],
                    "entity1": entity1s[idx],
                    "entity2": entity2s[idx],
                    "true_relation": true_rels[idx],
                    "predicted_relation": predicted_rels[idx],
                    "sentence_llm_dp_info_4_refinement": sentence_llm_infos[idx]
                })
            else:
                results.append((true_rels[idx], predicted_rels[idx]))
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


def merge_results_by_raw_sentence(full_results):
    """根据 raw_sentence 合并 full_results 列表"""
    merged_results = defaultdict(lambda: {
        "raw_sentence": "",
        "true_rels": [],
        "predicted_rels": [],
        "sentence_llm_dp_info_4_refinement": []
    })
    
    for result in full_results:
        raw_sentence = result["raw_sentence"]
        entity1 = result["entity1"]
        entity2 = result["entity2"]
        predicted_rel = result["predicted_relation"]
        true_rel = result["true_relation"]
        sentence_llm_dp_info_4_refinement = result["sentence_llm_dp_info_4_refinement"]
        
        # 合并数据
        merged_results[raw_sentence]["raw_sentence"] = raw_sentence
        merged_results[raw_sentence]["predicted_rels"].append((entity1, entity2, predicted_rel))
        merged_results[raw_sentence]["true_rels"].append((entity1, entity2, true_rel))
        merged_results[raw_sentence]["sentence_llm_dp_info_4_refinement"] = sentence_llm_dp_info_4_refinement

    return list(merged_results.values())


def run_relation_eval(full_eval_data, actor, tokenizer, args):   
    # ===== 实验：全样本评估 =====
    log_info("\n===== Experiment: FULL (POS + NO-RELATION) evaluation =====")
    full_results = get_predictions(full_eval_data, actor, tokenizer, args, collect_details=True)

    full_true = [item["true_relation"] for item in full_results]
    full_pred = [item["predicted_relation"] for item in full_results]
    
    # 计算 D2 的混淆矩阵
    d2_confusion = get_confusion_counts(full_true, full_pred, FULL_RELATION_TYPES)
    
    # 计算 D2 的全局 Micro 指标
    d2_global_metrics = compute_micro_metrics(d2_confusion)
    log_info(
        f"[D2 Global Metrics] Micro-P: {d2_global_metrics['Micro-P']:.4f}, "
        f"Micro-R: {d2_global_metrics['Micro-R']:.4f}, "
        f"Micro-F1: {d2_global_metrics['Micro-F1']:.4f}"
    )

    # 计算 D2 的正样本（POS_RELATION_TYPES）的 Micro 指标
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
    with open("examples/eval/eval_depth/results/full_eval_predictions.json", "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # 合并结果根据 raw_sentence
    merged_results = merge_results_by_raw_sentence(full_results)
    
    # 保存合并后的结果为 JSON
    with open("examples/eval/eval_depth/results/merged_eval_predictions.json", "w") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    # 保存 D1 和 D2 的混淆矩阵（CSV 格式）
    save_confusion_matrix_csv(d2_confusion, "examples/eval/eval_depth/results/confusion_matrix.csv")


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
    parser.add_argument("--full_data_path", type=str, default="dataset/SciERC/test_data/json_files/processed/SciERC-eval-test.json")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=4096, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)

    # For logger
    parser.add_argument("--save_dir", type=str, default="examples/eval/eval_depth/logs/eval-on-test.log")
    
    args = parser.parse_args()

    logger = setup_logger(args.save_dir)
    full_eval_data = get_data(args.full_data_path)
    actor, tokenizer = prepare_model(args)
    run_relation_eval(full_eval_data, actor, tokenizer, args)

# deepspeed examples/eval/eval_depth/eval_ground.py