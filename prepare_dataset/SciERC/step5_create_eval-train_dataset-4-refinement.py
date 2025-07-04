import os
from dotenv import load_dotenv
from prompt import (
    MY_Prompt_GIVEN_NE_ALL_with_MERGED_DP
)
from langchain_core.messages import HumanMessage
from tools.logger import get_logger, setup_logging_config
from utils import (
    parse_relationship
)

from langchain_community.llms import Tongyi
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from time import sleep
from tenacity import retry, wait_exponential, stop_after_attempt
import json
import random


# 环境变量配置
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key:
    os.environ["DASHSCOPE_API_KEY"] = api_key
else:
    raise ValueError("DASHSCOPE_API_KEY 环境变量未设置，请先配置环境变量。")

# 日志设置
logger = get_logger("util")
setup_logging_config()

DEFAULT_RELATION_TYPES = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]


def create_dataset(train_data, dev_data, train_data_ref, dev_data_ref):
    logger.info("==================================================")
    logger.info("Number of samples: %d", len(train_data) + len(dev_data))
    logger.info("==================================================")
    logger.info("********** Start **********")

    num_tests = 0

    # 创建 raw_sentence 到 dp_info 的映射字典，加速查找
    sentence_to_dpinfo_train = {
        item["raw_sentence"]: item["raw_sentence_llm_dp_info"]
        for item in train_data_ref
    }

    sentence_to_dpinfo_dev = {
        item["raw_sentence"]: item["raw_sentence_llm_dp_info"]
        for item in dev_data_ref
    }

    rm_data = []

    for item in train_data:
        sentence = item["sentence"]
        ner_pair = item["ner_pair"]
        raw_sentence = item["raw_sentence"]
        raw_sentence_llm_dp_info = item["raw_sentence_llm_dp_info"]

        # 查找对应的dp信息
        if raw_sentence not in sentence_to_dpinfo_train:
            logger.warning(f"raw_sentence not found in ref_data: {raw_sentence}")
            continue

        sentence_llm_dp_info_4_refinement = sentence_to_dpinfo_train[raw_sentence]

        context_base = {
            'relations': DEFAULT_RELATION_TYPES,
            'source_entity': ner_pair[0][0],
            'target_entity': ner_pair[1][0],
            'sentence_llm_dp_info': raw_sentence_llm_dp_info,
        }
        
        question = MY_Prompt_GIVEN_NE_ALL_with_MERGED_DP.format(**context_base, input_text=sentence)

        rm_data.append({
            "raw_sentence": item["raw_sentence"],
            "question": question,
            'source_entity': ner_pair[0][0],
            'target_entity': ner_pair[1][0],    
            "correct": ner_pair[-1],
            "sentence_llm_dp_info_4_refinement": sentence_llm_dp_info_4_refinement,
        })     

    
    for item in dev_data:
        sentence = item["sentence"]
        ner_pair = item["ner_pair"]
        raw_sentence = item["raw_sentence"]
        raw_sentence_llm_dp_info = item["raw_sentence_llm_dp_info"]

        # 查找对应的dp信息
        if raw_sentence not in sentence_to_dpinfo_dev:
            logger.warning(f"raw_sentence not found in ref_data: {raw_sentence}")
            continue

        sentence_llm_dp_info_4_refinement = sentence_to_dpinfo_dev[raw_sentence]

        context_base = {
            'relations': DEFAULT_RELATION_TYPES,
            'source_entity': ner_pair[0][0],
            'target_entity': ner_pair[1][0],
            'sentence_llm_dp_info': raw_sentence_llm_dp_info,
        }
        
        question = MY_Prompt_GIVEN_NE_ALL_with_MERGED_DP.format(**context_base, input_text=sentence)

        rm_data.append({
            "raw_sentence": item["raw_sentence"],
            "question": question,
            'source_entity': ner_pair[0][0],
            'target_entity': ner_pair[1][0],    
            "correct": ner_pair[-1],
            "sentence_llm_dp_info_4_refinement": sentence_llm_dp_info_4_refinement,
        })  
    
    with open("dataset/SciERC/train_data/json_files/processed/SciERC-eval-train.json", "w", encoding="utf-8") as f:
        json.dump(rm_data, f, ensure_ascii=False, indent=4)
    logger.info("Dataset for reward model evaluation has been saved to SciERC-eval-train.json.")
    logger.info("********** Done **********")


if __name__ == "__main__":
    with open("dataset/SciERC/train_data/json_files/raw/train_dataset_merged.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("dataset/SciERC/train_data/json_files/raw/dev_dataset.json", "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    with open("dataset/SciERC/train_data/json_files/raw/SciERC-eval_train_sentence_llm_dp.json", "r", encoding="utf-8") as f:
        train_data_ref = json.load(f)
    with open("dataset/SciERC/train_data/json_files/raw/SciERC-eval_dev_sentence_llm_dp.json", "r", encoding="utf-8") as f:
        dev_data_ref = json.load(f)
        
    create_dataset(train_data, dev_data, train_data_ref, dev_data_ref)
