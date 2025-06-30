import json
from pathlib import Path
from typing import List, Dict
from tools.logger import get_logger, setup_logging_config

# 配置日志
logger = get_logger("util")
setup_logging_config()

# 配置常量
POS_RELATION_TYPES = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION"]
FULL_RELATION_TYPES = POS_RELATION_TYPES + ["NO-RELATION"]

PROMPT_TEMPLATE = """
Determine which relationship can be inferred from the given sentence and two entities.

Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
All possible relations are: {relations}

Provide the answer in the following JSON format:
{{
  "relationship": ""
}}

Answer:
"""

OUTPUT_DIR = Path("dataset/SciERC/test_data/json_files/processed")
INPUT_FILE = Path("dataset/SciERC/test_data/json_files/raw/test_dataset.json")


def make_dataset(test_data: List[Dict]) -> None:
    logger.info("========== [MAKE DATASET] Start ==========")
    logger.info(f"Total samples: {len(test_data)}")

    data_full, data_pos = [], []

    for item in test_data:
        sentence = item["raw_sentence"]
        source_entity, target_entity = item["ner_pair"][0][0], item["ner_pair"][1][0]
        relation = item["ner_pair"][-1]

        full_question = PROMPT_TEMPLATE.format(
            relations=FULL_RELATION_TYPES,
            input_text=sentence,
            source_entity=source_entity,
            target_entity=target_entity
        )        

        entry = {
            "question": full_question,
            "correct": relation
        }
        data_full.append(entry)

        if relation != "NO-RELATION":
            pos_question = PROMPT_TEMPLATE.format(
                relations=POS_RELATION_TYPES,
                input_text=sentence,
                source_entity=source_entity,
                target_entity=target_entity
            )

            pos_entry = {
                "question": pos_question,
                "correct": relation
            }

            data_pos.append(pos_entry)

    logger.info(f"Data FULL: {len(data_full)} | HAS-RELATION: {len(data_pos)}")
    logger.info("========== [MAKE DATASET] Done ==========")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "SciERC_FULL.json", data_full)
    save_json(OUTPUT_DIR / "SciERC_POS.json", data_pos)


def save_json(path: Path, data: List[Dict]) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved: {path} ({len(data)} samples)")
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")


if __name__ == "__main__":
    try:
        with INPUT_FILE.open("r", encoding="utf-8") as f:
            test_data = json.load(f)
        make_dataset(test_data)
    except FileNotFoundError:
        logger.error(f"Input file not found: {INPUT_FILE}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
