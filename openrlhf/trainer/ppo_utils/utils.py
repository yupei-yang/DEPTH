import json
import re


def extract_reward_content(full_response: str) -> str:
    """
    Extract structured content from full response in four phases.
    """
    # Phase 1: Remove the middle part between "Relation Definitions" and "Real Data"
    # Split into task definition header and the remaining part
    header_split = re.split(
        r'^#+\s*-Relation\s*Definitions-\s*#+',  # Match the "Relation Definitions" section
        full_response,
        flags=re.MULTILINE | re.IGNORECASE,
        maxsplit=1
    )
    processed = header_split[0].strip()  # Keep the task definition header

    # Phase 2: Extract content after "Real Data"
    real_data_split = re.split(
        r'^#+\s*-Real\s*Data-\s*#+',  # Match the "Real Data" section
        header_split[-1],
        flags=re.MULTILINE | re.IGNORECASE,
        maxsplit=1
    )
    if len(real_data_split) > 1:
        # Combine task definition header and "Real Data" section
        processed += '\n\n' + real_data_split[-1].strip()

    # Phase 3: Split content before and after "Answer"
    answer_split = re.split(
        r'\nAnswer:\s*\n',  # Match the "Answer" declaration
        processed,
        flags=re.IGNORECASE,
        maxsplit=1
    )
    front_part = answer_split[0].strip()
    answer_content = answer_split[1].strip() if len(answer_split) > 1 else ""

    # Phase 4: Parse and reconstruct the "Answer"
    parsed_relation = parse_relationship(answer_content)
    structured_answer = f'{{\"relationship\": \"{parsed_relation}\"}}'

    # Final combination
    return f"\n{front_part}\n\nAnswer:\n{structured_answer}"


def parse_relationship(answer_content: str) -> str:
    """
    Parse `answer_content` and extract the "relationship" field.
    If parsing fails, return "Fail" as the default value.
    """
    # Try parsing as JSON
    try:
        json_str = answer_content.strip()
        # Sometimes additional content may be present, attempt to locate the first JSON object
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        data = json.loads(json_str)
        relationship = data.get("relationship", None)
        if relationship and isinstance(relationship, str):
            return relationship
    except json.JSONDecodeError:
        pass

    # If JSON parsing fails, attempt regex-based extraction of "relationship"
    try:
        # Assume "relationship" is a string enclosed in double quotes
        relationship_match = re.search(r'"relationship"\s*:\s*"([^"]+)"', answer_content)
        if relationship_match:
            return relationship_match.group(1)
    except Exception:
        pass

    # As a last resort, return "Fail"
    return "Fail"