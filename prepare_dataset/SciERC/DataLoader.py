import json
import string


def concatenate_sentences(sentences, para_level=False):
    """
    将句子列表（每个句子是单词列表）连接成一个完整的字符串。
    如果handle_punctuation为True，则移除标点符号前的空格。

    参数:
        sentences (list of list of str): 要连接的句子列表。
        handle_punctuation (bool): 是否处理标点符号，将标点前的空格移除。

    返回:
        str: 连接后的完整句子字符串。
        list of str: 合并后的单词列表。
    """
    processed_sentences = []
    if para_level:
        for sentence in sentences:
            processed_sentences += sentence
        sentence_str = ' '.join(processed_sentences)
    else:
        sentence_str = []
        for sentence in sentences:
            processed_sentences += sentence
            sentence_str.append([' '.join(sentence)])
    return sentence_str, processed_sentences


def get_entity_text(processed_sentences, start, end):
    """
    根据起始和结束索引，从processed_sentences中提取实体文本。

    参数:
        processed_sentences (list of str): 合并后的单词列表。
        start (int): 实体的起始索引。
        end (int): 实体的结束索引。

    返回:
        str: 实体文本。
    """
    return ' '.join(processed_sentences[start:end+1])


def process_SciERC_file(filepath='data/RE/SciERC/json/dev.json', para_level=False):
    """
    处理 JSON 文件，打印所有文档的信息，包括连接后的完整句子。
    替换'clusters'、'ner'、'relations'中的数字索引为相应的文本。

    参数:
        filepath (str): JSON 文件的路径。

    返回:
        list of dict: 包含处理后文档信息的列表。
    """
    processed_docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {idx+1}: {e}")
                continue  # 跳过解析错误的行

            sentence_str, processed_sentences = concatenate_sentences(doc.get("sentences", []), para_level)
            
            # 处理NER
            processed_ner = []
            for ner_list in doc.get('ner', []):
                processed_ner_list = []
                for ner in ner_list:
                    start, end, label = ner
                    entity_text = get_entity_text(processed_sentences, start, end)
                    if para_level:
                        processed_ner.append([entity_text, label])
                    else:
                        processed_ner_list.append([entity_text, label])
                if not para_level:
                    processed_ner.append(processed_ner_list)
                                    
            # 处理Clusters
            processed_clusters = []
            for cluster in doc.get('clusters', []):
                processed_cluster = []
                for span in cluster:
                    start, end = span
                    entity_text = get_entity_text(processed_sentences, start, end)
                    processed_cluster.append(entity_text)
                processed_clusters.append(processed_cluster)
            
            # 处理Relations
            processed_relations = []
            for relation_list in doc.get('relations', []):
                processed_relations_list = []
                for relation in relation_list:
                    if len(relation) != 5:
                        print(f"Invalid relation format in doc {doc.get('doc_key', '')}: {relation}")
                        continue
                    start1, end1, start2, end2, label = relation
                    entity1 = get_entity_text(processed_sentences, start1, end1)
                    entity2 = get_entity_text(processed_sentences, start2, end2)
                    if para_level:
                        processed_relations.append([entity1, entity2, label])
                    else:
                        processed_relations_list.append([entity1, entity2, label])
                if not para_level:
                    processed_relations.append(processed_relations_list)
            
            processed_doc = {
                'clusters': processed_clusters,
                'sentences': processed_sentences,
                'ner': processed_ner,
                'relations': processed_relations,
                'doc_key': doc.get('doc_key', ''),
                'full_sentence': sentence_str
            }
            processed_docs.append(processed_doc)
                    
    return processed_docs
