import os
import spacy
spacy.prefer_gpu()

from dotenv import load_dotenv
from prompt import (
    VanillaPrompt_DP,
    VanillaPrompt_CONVERT_SENTENCE_USING_DP,
)
from DataLoader import process_SciERC_file
from langchain_core.messages import HumanMessage
from tools.logger import get_logger, setup_logging_config
from utils import (
    get_sdp,
    pair_entities,
    parse_simplified_sentences
)

from langchain_community.llms import Tongyi
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


class Data_Simplifier:
    def __init__(self):
        self.llm_4_dp = Tongyi(model="qwen2.5-72b-instruct")
        logger.info("Initialized VanillaDP with model: %s", "qwen2.5-72b-instruct")

    def sdp_to_sentence(self, content, ground_rel, path_text):
        context_base = {
            'source_entity': ground_rel[0][0],
            'target_entity': ground_rel[1][0],
            'sdp_info': path_text,
        }
        
        extract_prompt = VanillaPrompt_CONVERT_SENTENCE_USING_DP.format(**context_base, input_text=content)
        final_result = self.llm_4_dp.invoke([HumanMessage(content=extract_prompt)])
        sentence = parse_simplified_sentences(final_result)
        return sentence

    def get_llm_dp_info(self, content, ner_pair):
        context_base = {
            'source_entity': ner_pair[0][0],
            'target_entity': ner_pair[1][0],
        }        
        extract_prompt = VanillaPrompt_DP.format(**context_base, input_text=content)        
        llm_dp_info = self.llm_4_dp.invoke([HumanMessage(content=extract_prompt)])
        return llm_dp_info


def pre_processing(llm_simplifier, raw_data, nlp, output_file="dataset/SciERC/test_data/json_files/raw"):
    logger.info("==================================================")
    logger.info("********** PreProcessing Start **********")

    result_data = []  # 用于存储最终的 (sentence, ner_pair)

    num_tests = 0  # 计数器
    for item in raw_data:
        for text_list, ground_ner_list, ground_rels_list in zip(item["full_sentence"], 
                                                                item["ner"], 
                                                                item["relations"]):
            raw_sentence = text_list[0]
            paired_ners_list = pair_entities(ground_ner_list, ground_rels_list)

            for ner_pair in paired_ners_list:
                num_tests += 1
                logger.info("*********************************")
                logger.info("Processing Data #%d", num_tests)
                logger.info("Sentence:%s", raw_sentence)
                logger.info("Ground-truth Relation:%s", ner_pair)

                raw_sentence_llm_dp_info = llm_simplifier.get_llm_dp_info(raw_sentence, ner_pair)
                logger.info("Raw Sentence LLM DP Info:%s", raw_sentence_llm_dp_info)

                try:
                    doc = nlp(raw_sentence)
                    sdp = get_sdp(nlp, doc, ner_pair[0][0], ner_pair[1][0])
                    if sdp:
                        path_text = " → ".join([token.text for token in sdp])
                        logger.info(f"SDP: {path_text}")
                        logger.info("Dependency path: " + str([(token.text, token.dep_, token.head.text) for token in sdp]))
                        sentence = llm_simplifier.sdp_to_sentence(raw_sentence, ner_pair, path_text)
                        logger.info(f"Simplified sentence: {sentence}")
                        sentence_llm_dp_info = llm_simplifier.get_llm_dp_info(sentence, ner_pair)
                        logger.info("Simplified Sentence LLM DP Info:%s", sentence_llm_dp_info)
                    else:
                        logger.info("No path between entities")
                        sentence = raw_sentence
                        sentence_llm_dp_info = raw_sentence_llm_dp_info

                except Exception as e:
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
                    sentence = raw_sentence
                    sentence_llm_dp_info = raw_sentence_llm_dp_info

                # 保存 (sentence, ner_pair) 到 result_data
                result_data.append({
                    "raw_sentence": raw_sentence,                    
                    "ner_pair": ner_pair,
                    "raw_sentence_llm_dp_info": raw_sentence_llm_dp_info,
                    "sdp_path_text": path_text,
                    "sentence": sentence,
                    "sentence_llm_dp_info": sentence_llm_dp_info
                })

    # 将收集到的数据写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    logger.info("********** PreProcessing Done **********")
    logger.info(f"Processed data has been saved to {output_file}")


if __name__ == "__main__":
    raw_data = process_SciERC_file(filepath='data/RE/SciERC/json/test.json', para_level=False)
    llm_simplifier = Data_Simplifier()    
    nlp = spacy.load("en_core_web_trf")
    pre_processing(llm_simplifier, raw_data, nlp)
