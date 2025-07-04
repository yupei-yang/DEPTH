import json
import re


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