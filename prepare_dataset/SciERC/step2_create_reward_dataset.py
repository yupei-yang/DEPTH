import os
from dotenv import load_dotenv
from prompt import (
    MYPROPMT_4_DATASET
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


def create_dataset(train_data, dev_data):
    logger.info("==================================================")
    logger.info("Number of samples: %d", len(train_data) + len(dev_data))
    logger.info("==================================================")
    logger.info("********** Start **********")

    num_tests = 0
    rm_data = []

    for item in train_data:
        sentence = item["sentence"]
        ner_pair = item["ner_pair"]
        raw_sentence_llm_dp_info = item["raw_sentence_llm_dp_info"]

        context_base = {
            'relations': DEFAULT_RELATION_TYPES,
            'source_entity': ner_pair[0][0],
            'target_entity': ner_pair[1][0],
            'sentence_llm_dp_info': raw_sentence_llm_dp_info,
        }
        
        question = MYPROPMT_4_DATASET.format(**context_base, input_text=sentence)

        correct = f'"relationship": "{ner_pair[-1]}"'
        filtered_relations = [rt for rt in DEFAULT_RELATION_TYPES if rt != ner_pair[-1]]        

        for wrong_rel in filtered_relations:
            rm_data.append({
                "question": question,                    
                "correct": correct,
                "incorrect": f'"relationship": "{wrong_rel}"',
            })

    for item in dev_data:
        sentence = item["sentence"]
        ner_pair = item["ner_pair"]
        raw_sentence_llm_dp_info = item["raw_sentence_llm_dp_info"]

        context_base = {
            'relations': DEFAULT_RELATION_TYPES,
            'source_entity': ner_pair[0][0],
            'target_entity': ner_pair[1][0],
            'sentence_llm_dp_info': raw_sentence_llm_dp_info,
        }
        
        question = MYPROPMT_4_DATASET.format(**context_base, input_text=sentence)

        correct = f'"relationship": "{ner_pair[-1]}"'
        filtered_relations = [rt for rt in DEFAULT_RELATION_TYPES if rt != ner_pair[-1]]        

        for wrong_rel in filtered_relations:
            rm_data.append({
                "question": question,                    
                "correct": correct,
                "incorrect": f'"relationship": "{wrong_rel}"',
            })

    with open("dataset/SciERC/train_data/json_files/processed/SciERC-RM-full.json", "w", encoding="utf-8") as f:
        json.dump(rm_data, f, ensure_ascii=False, indent=4)
    logger.info("Dataset for reward model evaluation has been saved to SciERC-RM-full.json.")
    logger.info("********** Done **********")


if __name__ == "__main__":
    with open("dataset/SciERC/train_data/json_files/raw/train_dataset_merged.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("dataset/SciERC/train_data/json_files/raw/dev_dataset.json", "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    create_dataset(train_data, dev_data)
