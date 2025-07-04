from collections import Counter
from Levenshtein import distance
from itertools import combinations
import json
import re
import networkx as nx
from tools.logger import get_logger, setup_logging_config
logger = get_logger("util")


def pair_entities(ground_ner_list, ground_rels_list):
    # 1. 构建一个字典，用于将实体名称映射到它的标签
    entity2label = { entity: label for (entity, label) in ground_ner_list }
    
    # 2. 构建一个字典，只记录原方向出现的关系
    #   比如 ground_rels_list 有 (A, B, r1) 就只存 rel_map[(A,B)] = r1
    rel_map = {}
    for e1, e2, rel in ground_rels_list:
        rel_map[(e1, e2)] = rel

    # 3. 取出全部实体（保持在 ground_ner_list 中的顺序），进行两两组合
    entities = [item[0] for item in ground_ner_list]
    
    valid_pairs = []
    for e1, e2 in combinations(entities, 2):
        # 跳过实体名称互相包含的情况
        if e1 in e2 or e2 in e1:
            continue
        
        has_e1e2 = (e1, e2) in rel_map
        has_e2e1 = (e2, e1) in rel_map
        
        # 如果 (e1,e2) 在 rel_map 中
        if has_e1e2:
            valid_pairs.append(
                ((e1, entity2label[e1]), (e2, entity2label[e2]), rel_map[(e1, e2)])
            )
        
        # 如果 (e2,e1) 在 rel_map 中
        if has_e2e1:
            valid_pairs.append(
                ((e2, entity2label[e2]), (e1, entity2label[e1]), rel_map[(e2, e1)])
            )
        
        # 如果两个方向都没有，则视作 "NO-RELATION"
        if not has_e1e2 and not has_e2e1:
            valid_pairs.append(
                ((e1, entity2label[e1]), (e2, entity2label[e2]), "NO-RELATION")
            )
    
    return valid_pairs


# 对 VanillaPrompt 在 SciERC 上后处理
def parse_relations(final_result: str) -> list:
    # 初始化空列表
    relations_list = []

    # 按行分割final_result
    lines = final_result.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 每行格式类似: (LCS, USED-FOR, human interaction with data sources)
        # 去掉前后的括号
        line = line.strip('()')
        
        # 用逗号分割
        parts = [p.strip() for p in line.split(',')]
        
        # parts应当有三个元素：source, relation, target
        if len(parts) == 3:
            source, relation, target = parts
            relations_list.append([source, target, relation])
    
    return relations_list


def calc_metrics(correct, predicted_count, gold_count):
    if predicted_count == 0 or gold_count == 0:
        return 0.0, 0.0, 0.0
    precision = correct / predicted_count if predicted_count else 0.0
    recall = correct / gold_count if gold_count else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def cal_re_score_strict(ground_rels: list, extracted_rels: list, include_relation: bool = False):
    """
    ground_rels: list of [source, target, relation]
    extracted_rels: list of [source, target, relation]
    include_relation: 如果为True，那么匹配时需要关系类型相同；否则只匹配无序的(source, target)对。
    """
    # 准备一个可重复计数的结构来记录ground_rels
    # 对于严格匹配，用与之前相同的表示方法来计数
    if include_relation:
        # (frozenset({s,t}), r)做键
        ground_counter = Counter((frozenset({s,t}), r) for s,t,r in ground_rels)
        def match_func(s, t, r):
            key = (frozenset({s,t}), r)
            if ground_counter[key] > 0:
                ground_counter[key] -= 1
                return True
            return False
    else:
        # frozenset({s,t})做键
        ground_counter = Counter(frozenset({s,t}) for s,t,r in ground_rels)
        def match_func(s, t, r):
            key = frozenset({s,t})
            if ground_counter[key] > 0:
                ground_counter[key] -= 1
                return True
            return False

    correct_count = 0
    for s, t, r in extracted_rels:
        if match_func(s, t, r):
            correct_count += 1

    gold_count = len(ground_rels)
    predicted_count = len(extracted_rels)
    return calc_metrics(correct_count, predicted_count, gold_count)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算s1和s2之间的Levenshtein编辑距离。
    """
    s1, s2 = s1.lower(), s2.lower()
    m, n = len(s1), len(s2)
    # 使用动态规划求解编辑距离
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,    # 删除
                dp[i][j-1] + 1,    # 插入
                dp[i-1][j-1] + cost # 替换
            )
    return dp[m][n]


def edit_distance_match(a: str, b: str, threshold: int = 2) -> bool:
    """
    当两个字符串的编辑距离小于等于给定阈值（threshold）时，认为匹配成功。
    可根据实际需要调整threshold，比如2或3。
    """
    dist = levenshtein_distance(a, b)
    dist2 = distance(a, b)
    print("dist among {} and {} is {} and {}.".format(a, b, dist, dist2))
    return dist <= threshold


def cal_re_score_loose(ground_rels: list, extracted_rels: list, include_relation: bool = False, threshold: int = 2):
    """
    使用编辑距离作为宽松匹配标准计算P,R,F1:
    ground_rels: [[s, t, r], ...]
    extracted_rels: [[s, t, r], ...]
    include_relation: True则需要relation匹配；False则不需要。
    threshold: 编辑距离匹配阈值, 小于或等于该值即认为匹配。

    宽松匹配标准：
    - 使用编辑距离判断(source, target)是否近似匹配
    - (source, target) 与 (target, source)不区分顺序
    """
    # 拷贝一份ground_rels用于标记使用情况
    unused_ground = [(g_s, g_t, g_r) for g_s, g_t, g_r in ground_rels]

    correct_count = 0

    for ex_s, ex_t, ex_r in extracted_rels:
        matched = False
        for i, (g_s, g_t, g_r) in enumerate(unused_ground):
            if g_s is None:  # 已匹配过的跳过
                continue
            
            pair_match = False
            # 无序对匹配 + 编辑距离
            # 检查无序匹配, ex_s匹配g_s且ex_t匹配g_t，或ex_s匹配g_t且ex_t匹配g_s
            if edit_distance_match(ex_s, g_s, threshold) and edit_distance_match(ex_t, g_t, threshold):
                pair_match = True
            elif edit_distance_match(ex_s, g_t, threshold) and edit_distance_match(ex_t, g_s, threshold):
                pair_match = True

            if not pair_match:
                continue

            # 如果不需要关系匹配，pair_match成立即可
            # 如果需要关系匹配，则还需检查relation是否一致（严格大小写匹配或需根据实际要求宽松匹配）
            if not include_relation or (include_relation and ex_r == g_r):
                # 找到匹配关系后，将该ground_rel标记为已使用
                unused_ground[i] = (None, None, None)
                matched = True
                break

        if matched:
            correct_count += 1

    gold_count = len(ground_rels)
    predicted_count = len(extracted_rels)
    return calc_metrics(correct_count, predicted_count, gold_count)


# FUNCTION TO PRINT the number of trainable paraemters
def print_number_of_trainable_model_parameters(model, logger):
    trainable_model_params=0
    all_model_params=0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    logger.info(f"trainable params:{trainable_model_params} || all params: {all_model_params} || trainable% {trainable_model_params/all_model_params}")


def parse_relationship(final_result):
    """
    解析 final_result 并提取 relationship 字段。
    如果解析失败，返回 None 或者一个默认值。
    """
    # 尝试直接解析为 JSON
    try:
        # 如果 final_result 是一个字符串，确保它是有效的 JSON
        json_str = final_result.strip()
        # 有时候模型可能会多输出一些内容，尝试找到第一个 JSON 对象
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        data = json.loads(json_str)
        relationship = data.get("relationship", None)
        if relationship and isinstance(relationship, str):
            return relationship
        else:
            logger.warning("JSON解析成功，但缺少 'relationship' 字段或字段类型不正确。")
    except json.JSONDecodeError as e:
        logger.warning("JSON解析失败: %s", e)

    # 如果直接解析失败，尝试使用正则表达式提取 relationship
    try:
        # 假设 relationship 是一个双引号包裹的字符串
        relationship_match = re.search(r'"relationship"\s*:\s*"([^"]+)"', final_result)
        if relationship_match:
            relationship = relationship_match.group(1)
            return relationship
    except Exception as e:
        logger.warning("使用正则表达式提取 'relationship' 失败: %s", e)

    # 如果仍然失败，尝试提取第一行包含关系的词
    try:
        # 假设关系是引号中的第一个词
        relationship_match = re.search(r'\"relationship\":\s*\"(\w+)\"', final_result)
        if relationship_match:
            relationship = relationship_match.group(1)
            return relationship
    except Exception as e:
        logger.warning("尝试提取 'relationship' 时出错: %s", e)

    # 最后，返回 None 或者一个默认关系
    logger.error("无法从 final_result 中提取 'relationship'。")
    return "Fail"


def parse_simplified_sentences(final_result: str) -> str:
    """
    将大模型返回的 JSON 格式字符串 final_result 中的 
    'Simplified sentence' 字段提取为字符串并返回。

    :param final_result: 包含 JSON 格式内容的字符串
    :return: 从 final_result 中提取的 'Simplified sentence' 字段内容
    """
    try:
        # 解析 JSON 格式字符串
        data = json.loads(final_result)
        # 返回字段 "Simplified sentence" 中的内容
        return data["Simplified sentence"]
    except (json.JSONDecodeError, KeyError) as e:
        # 如果解析失败或者找不到 "Simplified sentence" 字段，
        # 可以根据需求处理，比如返回原始字符串或返回空字符串
        logger.error(f"Error parsing final_result: {e}")
        return final_result


def build_dep_graph(doc):
    """
    构建依存关系无向图

    参数:
    doc (spacy.tokens.Doc): 经过 spaCy 处理的文本对象

    返回:
    G (networkx.Graph): 包含句子依存关系的无向图
    """
    G = nx.Graph()
    for token in doc:
        # 给每个 (token.i, token.head.i) 添加无向边
        # 注意 token.head == token 本身时表示根 (ROOT)；无论如何加一次不会产生冲突
        G.add_edge(token.i, token.head.i)
    return G

def find_span_head(doc, entity_str):
    """
    在 doc.text 中查找字符串 entity_str 对应的 spaCy Span 并返回其 root 的索引。
    如果无法找到或无法对齐，则返回 None。

    参数:
    doc (spacy.tokens.Doc): 经过 spaCy 处理的文本对象
    entity_str (str): 实体字符串，可包含多个单词

    返回:
    int 或 None: 若找到相应 Span，则返回 Span.root 的 token 索引；否则返回 None
    """
    # 在原句中查找 entity_str 的起始位置（不区分大小写）
    start_idx = doc.text.lower().find(entity_str.lower())
    if start_idx == -1:
        return None  # 未找到匹配

    end_idx = start_idx + len(entity_str)

    # 尝试使用 char_span 获得对应的 Span
    # 注意 alignment_mode 的选用，对于多词实体常用 'expand' 以允许边界扩展
    span = doc.char_span(start_idx, end_idx, alignment_mode='expand')
    if span is None:
        return None  # 无法对齐为 Span

    return span.root.i  # 返回该 Span 在依存树中的头部 token 的索引

def get_sdp(nlp, doc, entity1, entity2):
    """
    获取两个多词实体间的最短依存路径（SDP）。

    参数:
    nlp: spaCy 模型
    doc (spacy.tokens.Doc): 经过 spaCy 处理的文本对象
    entity1 (str): 实体1（可能包含多个词）
    entity2 (str): 实体2（可能包含多个词）

    返回:
    list[spacy.tokens.Token]: 从 entity1 到 entity2 的依存路径上的 token 列表；
                              如果无法找到或没有路径，则返回空列表
    """
    # 找到实体1、实体2在依存树中的头部索引
    e1_head_idx = find_span_head(doc, entity1)
    e2_head_idx = find_span_head(doc, entity2)

    # 如果任意一方没找到，直接返回空
    if e1_head_idx is None or e2_head_idx is None:
        raise ValueError("Empty Head.")

    # 构建无向依存图
    G = build_dep_graph(doc)

    # 使用 networkx 查找最短路径
    try:
        path_indices = nx.shortest_path(G, e1_head_idx, e2_head_idx)
    except nx.NetworkXNoPath:
        raise ValueError("Fail to find SDP.")

    # 最终返回路径上对应的 token 对象
    return [doc[i] for i in path_indices]
