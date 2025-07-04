import re
import json


def extract_corrected_relation(model_output, true_rel, cand_rel, fallback_on_failure=True):
    predicted_label = get_predicted_rel(model_output)

    if predicted_label == "Fail" and fallback_on_failure:
        return [(true_rel[2], cand_rel[2])]
    else:
        return [(true_rel[2], predicted_label)]


def extract_real_data_info(question: str):
    """
    Extract the Real Data part from the prompt.
    Ensures we are not extracting from the example section.
    """
    # 提取 Real Data 部分
    real_data_match = re.search(
        r"-Real Data-\s*#+\s*Sentence:\s*(.*?)\nEntity 1:\s*(.*?)\nEntity 2:\s*(.*?)\nDependency Parsing Information:",
        question,
        re.DOTALL
    )

    if real_data_match:
        sentence = real_data_match.group(1).strip()
        entity1 = real_data_match.group(2).strip()
        entity2 = real_data_match.group(3).strip()
    else:
        sentence, entity1, entity2 = "", "", ""

    return sentence, entity1, entity2


def get_predicted_rel(full_response: str) -> str:
    header_split = re.split(
        r'^#+\s*-Relation\s*Definitions-\s*#+',
        full_response, 
        flags=re.MULTILINE | re.IGNORECASE,
        maxsplit=1
    )
    processed = header_split[0].strip()  # 保留任务定义头
    
    real_data_split = re.split(
        r'^#+\s*-Real\s*Data-\s*#+',  # 匹配Real Data标题
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
    return parsed_relation


def parse_relationship(answer_content: str) -> str:
    try:
        json_str = answer_content.strip()
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)

        if json_match:
            json_str = json_match.group()
        
        data = json.loads(json_str)
        relationship = data.get("relationship", None)

        if relationship and isinstance(relationship, str):
            return relationship

    except json.JSONDecodeError as e:
        pass
  
    try:
        relationship_match = re.search(r'"relationship"\s*:\s*"([^"]+)"', answer_content)

        if relationship_match:
            relationship = relationship_match.group(1)
            return relationship
    except Exception as e:
        pass
  
    try:
        relationship_match = re.search(r'\"relationship\":\s*\"(\w+)\"', answer_content)

        if relationship_match:
            relationship = relationship_match.group(1)
            return relationship
    except Exception as e:
        pass
  
    return "Fail"