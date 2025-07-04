import os
import spacy
from dotenv import load_dotenv
from prompt import VanillaPrompt_CONVERT_SPACY_TO_LANGUAGE
from DataLoader import process_SciERC_file
from langchain_core.messages import HumanMessage
from tools.logger import get_logger, setup_logging_config
from langchain_community.llms import Tongyi
from tqdm import tqdm
import json

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

# spaCy加载
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")


class Data_Simplifier:
    def __init__(self):
        self.llm_4_dp = Tongyi(model="qwen2.5-72b-instruct")
        logger.info("Initialized VanillaDP with model: %s", "qwen2.5-72b-instruct")

    def spacy_to_language(self, raw_sentence, spacy_dp_info):
        # 准备prompt内容
        context_base = {
            'spacy_dp_info': spacy_dp_info,
        }

        # 使用格式化的prompt来调用LLM进行依存信息转化
        extract_prompt = VanillaPrompt_CONVERT_SPACY_TO_LANGUAGE.format(**context_base, input_text=raw_sentence)
        final_result = self.llm_4_dp.invoke([HumanMessage(content=extract_prompt)])
        logger.info(f"final_result: {final_result}")
        return final_result

    def extract_spacy_dp_info(self, doc):
        """
        Extract dependency information from the spacy Doc object.
        """
        dep_info = []
        for token in doc:
            dep_info.append({
                "word": token.text,
                "dep": token.dep_,
                "head": token.head.text
            })
        return dep_info

def pre_processing(llm_simplifier, raw_data, nlp, output_file="dataset/SciERC/train_data/json_files/raw/SciERC-eval_train_sentence_llm_dp.json"):
    logger.info("==================================================")
    logger.info("********** PreProcessing Start **********")

    result_data = []  # 用于存储最终的 (sentence, ner_pair)

    for item in tqdm(raw_data):
        for text_list in item["full_sentence"]:
            raw_sentence = text_list[0]
            doc = nlp(raw_sentence)
            spacy_dp_info = llm_simplifier.extract_spacy_dp_info(doc)
            
            # 使用LLM将spacy_dp_info转换为语言描述
            raw_sentence_llm_dp_info = llm_simplifier.spacy_to_language(raw_sentence, spacy_dp_info)

            result_data.append({
                "raw_sentence": raw_sentence,
                "raw_sentence_llm_dp_info": raw_sentence_llm_dp_info,
            })

    # 将收集到的数据写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    logger.info("********** PreProcessing Done **********")
    logger.info(f"Processed data has been saved to {output_file}")

if __name__ == "__main__":
    # 假设你的 raw_data 来自 SciERC 数据集
    raw_data = process_SciERC_file(filepath='dataset/SciERC/train_data/json_files/raw/train_dataset_merged.json', para_level=False)
    
    llm_simplifier = Data_Simplifier()    
    pre_processing(llm_simplifier, raw_data, nlp)
