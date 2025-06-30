################################################
# Prompt for Dependency Parsing
################################################
VanillaPrompt_DP = """
Perform a dependency parsing analysis on the following sentence. Provide a concise description of the dependencies between the two entities and their relationship to other words in the sentence. The output should include:

1. The dependencies of Entity 1 with other words.
2. The dependencies of Entity 2 with other words.
3. Whether there is a direct dependency between Entity 1 and Entity 2.

Format:
"Entity 1 ('<Entity 1>') is the <role>, depending on <dependency relation> with <word>. Entity 2 ('<Entity 2>') is the <role>, depending on <dependency relation> with <word>. There is <no direct dependency / a direct dependency> between Entity 1 and Entity 2."

######################
-Examples-
######################
Example 1:

Sentence: An entity-oriented approach to restricted-domain parsing is proposed .
Entity 1: entity-oriented approach
Entity 2: restricted-domain parsing

Answer:
"Entity 1 ('entity-oriented approach') is the subject, depending on the verb 'proposed'. Entity 2 ('restricted-domain parsing') is the object of the preposition 'to', depending on 'to' in the phrase 'to restricted-domain parsing'. There is no direct dependency between Entity 1 and Entity 2, but they are indirectly connected through the preposition 'to'."

Example 2:

Sentence: This paper proposes a new methodology to improve the accuracy of a term aggregation system using each author 's text as a coherent corpus .
Entity 1: methodology
Entity 2: accuracy

Answer:
"Entity 1 ('methodology') is the subject, depending on 'proposes' with 'This paper'. Entity 2 ('accuracy') is the object, depending on 'improve' with 'methodology'. There is no direct dependency between Entity 1 and Entity 2."

######################
-Real Data-
######################
Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}

Answer:
"""
################################################
# Prompt for Sentence Converting Using DP
################################################
VanillaPrompt_CONVERT_SENTENCE_USING_DP = """
You are given:
1. An original sentence (which may contain extra information).
2. Two entities within that sentence.
3. The shortest dependency path (SDP) connecting those two entities.

Your goal is to generate a concise new sentence that:
- Retains the relationship between the two entities.
- Reflects the original meaning conveyed by the SDP.
- Removes unnecessary or extraneous information that does not affect the core meaning regarding the two entities.
- Ensures grammatical correctness and readability.

Instructions:
1. Focus on the given SDP and ensure all critical words along that path remain in your new sentence to convey the relationship.
2. Remove details that do not directly contribute to describing the relationship between Entity 1 and Entity 2.
3. Maintain clarity and grammatical correctness.
4. If the provided SDP is not helpful or if it is not possible to simplify using the SDP, output the original sentence.
5. Provide the answer in the following JSON format:
{{
  "Simplified sentence": ""
}}

######################
-Examples-
######################
Example 1:

- Original sentence: Recognition of proper nouns in Japanese text has been studied as a part of the more general problem of morphological analysis in Japanese text processing -LRB- -LSB- 1 -RSB- -LSB- 2 -RSB- -RRB- .
- Entity 1: Recognition of proper nouns
- Entity 2: morphological analysis
- Shortest Dependency Path: Recognition → studied → as → part → of → problem → of → analysis

Answer:
{{
  "Simplified sentence": "Recognition of proper nouns has been studied as part of the problem of morphological analysis."
}}

Example 2:

- Original sentence: This paper proposes a new methodology to improve the accuracy of a term aggregation system using each author 's text as a coherent corpus .
- Entity 1: methodology
- Entity 2: accuracy
- Shortest Dependency Path: methodology → improve → accuracy

Answer:
{{
  "Simplified sentence": "This paper proposes a new methodology to improve accuracy."
}}

######################
-Real Data-
######################
- Original sentence: {input_text}
- Entity 1: {source_entity}
- Entity 2: {target_entity}
- Shortest Dependency Path: {sdp_info}

Answer:
"""
################################################
# Prompts for Relation Extraction
################################################
# Without DP
VanillaPrompt_GIVEN_NE_ALL = """
Determine which relationship can be inferred from the given sentence and two entities. If no relationship exists, respond with "NO-RELATION". Provide the answer in the following JSON format:
{{
  "relationship": ""
}}

######################
-Examples-
######################
Example 1:

Sentence: An entity-oriented approach to restricted-domain parsing is proposed .
Entity 1: entity-oriented approach
Entity 2: restricted-domain parsing
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "USED-FOR"
}}

Example 2:

Sentence: This paper proposes a new methodology to improve the accuracy of a term aggregation system using each author 's text as a coherent corpus .
Entity 1: methodology
Entity 2: accuracy
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

######################
-Real Data-
######################
Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
All possible relations are: {relations}

Answer:
"""
# Using LLM DP knowledge
VanillaPrompt_GIVEN_NE_ALL_with_LLM_DP = """
Determine which relationship can be inferred from the given sentence and two entities. Use dependency parsing to help understand the structure of the sentence. If no relationship exists, respond with "NO-RELATION". Provide the answer in the following JSON format:
{{
  "relationship": ""
}}

######################
-Examples-
######################
Example 1:

Sentence: An entity-oriented approach to restricted-domain parsing is proposed .\n
Entity 1: entity-oriented approach\n
Entity 2: restricted-domain parsing\n
Dependency Parsing: Entity 1 ('entity-oriented approach') is the subject, depending on the verb 'proposed'. Entity 2 ('restricted-domain parsing') is the object of the preposition 'to', depending on 'to' in the phrase 'to restricted-domain parsing'.\n
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]\n

Answer:
{{
  "relationship": "USED-FOR"
}}

Example 2:

Sentence: This paper proposes a new methodology to improve the accuracy of a term aggregation system using each author 's text as a coherent corpus .\n
Entity 1: methodology\n
Entity 2: accuracy\n
Dependency Parsing: Entity 1 ('methodology') is the subject, depending on the verb 'proposes'. Entity 2 ('accuracy') is the object of the preposition 'of', depending on 'improve'.\n
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]\n

Answer:
{{
  "relationship": "NO-RELATION"
}}

######################
-Real Data-
######################
Sentence: {input_text}\n
Entity 1: {source_entity}\n
Entity 2: {target_entity}\n
Dependency Parsing: {dependency_info}\n
All possible relations are: {relations}\n

Answer:
"""
# Using Spacy DP knowledge
VanillaPrompt_GIVEN_NE_ALL_with_SPACY_DP = """
Determine which relationship can be inferred from the given sentence and two entities. If no relationship exists, respond with "NO-RELATION". Provide the answer in the following JSON format:
{{
  "relationship": ""
}}

######################
-Examples-
######################
Example 1:

Sentence: An entity-oriented approach to restricted-domain parsing is proposed.
Entity 1: entity-oriented approach
Entity 2: restricted-domain parsing
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "USED-FOR"
}}

Example 2:

Sentence: This paper proposes a new methodology to improve accuracy.
Entity 1: methodology
Entity 2: accuracy
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

######################
-Real Data-
######################
Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
All possible relations are: {relations}

Answer:
"""
MY_Prompt_GIVEN_NE_ALL_with_SPACY_DP = """
You are an information extraction system. Your goal is to determine the relationship between two given entities based on the provided sentence. 

######################
-Relation Definitions-
######################
There are 5 asymmetric relation types ("USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR"), and 2 symmetric relation types ("COMPARE", "CONJUNCTION"):
- "USED-FOR": B is used for A, B models A, A is trained on B, B exploits A, A is based on B.
- "FEATURE-OF": B belongs to A, B is a feature of A, B is under A domain.
- "HYPONYM-OF": B is a hyponym of A, B is a type of A.
- "PART-OF": B is a part of A, A includes B, incorporate B to A.
- "EVALUATE-FOR": A is an evaluation metric for B.
- "COMPARE": Symmetric relation. Opposite of conjunction, compare two models/methods, or list two opposing entities.
- "CONJUNCTION": Symmetric relation. A and B function in a similar role or use/incorporate with each other.
- "NO-RELATION": No relationship can be inferred between A and B.

######################
-Important Guidelines-
######################
1. Check if any relation can be inferred by comparing the sentence with the relation definitions.
2. Do not guess or fabricate a relationship.
3. For asymmetric relations, always assume "B -> A".
4. If the sentence does not clearly match any of the given relation definitions, you must respond with "NO-RELATION". 
5. Provide the final answer in the following JSON format:
{{
  "relationship": ""
}}

######################
-Examples-
######################
Example 1:

Sentence: A novel method to learn the intrinsic object structure is proposed.
Entity 1: method
Entity 2: intrinsic object structure
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "USED-FOR"
}}

Example 2:

Sentence: The parameterized object state lies on a low dimensional manifold.
Entity 1: low dimensional manifold
Entity 2: parameterized object state
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "FEATURE-OF"
}}

Example 3:

Sentence: English is shown to be trans-context-free on the basis of coordinations.
Entity 1: English
Entity 2: coordinations
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

Example 4:

Sentence: French is cited as one of the languages with grammatical gender.
Entity 1: French
Entity 2: languages
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "HYPONYM-OF"
}}

Example 5:

Sentence: The learned intrinsic object structure is integrated into a particle-filter style tracker.
Entity 1: intrinsic object structure
Entity 2: particle-filter style tracker
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "PART-OF"
}}

Example 6:

Sentence: This intrinsic object representation has properties that make the particle-filter style tracker more robust and reliable.
Entity 1: intrinsic object representation
Entity 2: particle-filter style tracker
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

Example 7:

Sentence: A subcategorization dictionary built with the system improves the accuracy of a parser.
Entity 1: accuracy
Entity 2: parser
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "EVALUATE-FOR"
}}

Example 8:

Sentence: Our approach significantly outperforms state-of-the-art methods.
Entity 1: approach
Entity 2: state-of-the-art methods
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "COMPARE"
}}

Example 9:

Sentence: The compact description of a video sequence has applications in video browsing and retrieval.
Entity 1: compact description of a video sequence
Entity 2: video browsing and retrieval
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

Example 10:

Sentence: The domain divergence involves different viewpoints and various resolutions.
Entity 1: viewpoints
Entity 2: resolutions
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "CONJUNCTION"
}}

######################
-Real Data-
######################
Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
All possible relations are: {relations}

Answer:
"""
MY_Prompt_GIVEN_NE_ALL_with_MERGED_DP = """
You are an information extraction system. Your goal is to determine the relationship between two given entities based on the provided sentence. 

######################
-Relation Definitions-
######################
There are 5 asymmetric relation types ("USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR"), and 2 symmetric relation types ("COMPARE", "CONJUNCTION"):
- "USED-FOR": B is used for A, B models A, A is trained on B, B exploits A, A is based on B.
- "FEATURE-OF": B belongs to A, B is a feature of A, B is under A domain.
- "HYPONYM-OF": B is a hyponym of A, B is a type of A.
- "PART-OF": B is a part of A, A includes B, incorporate B to A.
- "EVALUATE-FOR": A is an evaluation metric for B.
- "COMPARE": Symmetric relation. Opposite of conjunction, compare two models/methods, or list two opposing entities.
- "CONJUNCTION": Symmetric relation. A and B function in a similar role or use/incorporate with each other.
- "NO-RELATION": No relationship can be inferred between A and B.

######################
-Important Guidelines-
######################
1. Only rely on the provided relationship definitions without guessing or expanding them. If there is no clear or strong indication of a defined relation, respond with "NO-RELATION".
2. Refer to the Dependency Parsing Information as semantic guidance for analyzing the context around the entities.
3. For asymmetric relations, always assume B -> A.
4. Provide the final answer in the following JSON format:
{{
  "relationship": ""
}}

######################
-Examples-
######################
Example 1:

Sentence: English is shown to be trans-context-free on the basis of coordinations of the respectively type.
Entity 1: English
Entity 2: coordinations
Dependency Parsing Information: Entity 1 ('English') is the subject, depending on the verb 'is shown' with 'is'. Entity 2 ('coordinations') is the object of the preposition 'of', depending on 'of' in the phrase 'on the basis of coordinations'. There is no direct dependency between Entity 1 and Entity 2.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

Example 2:

Sentence: A novel method to learn the intrinsic object structure is proposed.
Entity 1: method
Entity 2: intrinsic object structure
Dependency Parsing Information: Entity 1 ('method') is the subject, depending on the verb 'is proposed'. Entity 2 ('intrinsic object structure') is the object of the infinitive verb 'to learn', depending on 'learn'. There is no direct dependency between Entity 1 and Entity 2, but they are connected through the infinitive verb 'to learn'.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "USED-FOR"
}}

Example 3:

Sentence: The parameterized object state lies on a low dimensional manifold.
Entity 1: low dimensional manifold
Entity 2: parameterized object state
Dependency Parsing Information: Entity 1 ('low dimensional manifold') is the object of the preposition 'on', depending on the verb 'lies' with 'lies on'. Entity 2 ('parameterized object state') is the subject, depending on the verb 'lies' with 'lies'. There is no direct dependency between Entity 1 and Entity 2, but they are connected through the verb 'lies'.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "FEATURE-OF"
}}

Example 4:

Sentence: This intrinsic object representation has properties that make the particle-filter style tracker more robust and reliable.
Entity 1: intrinsic object representation
Entity 2: particle-filter style tracker
Dependency Parsing Information: Entity 1 ('intrinsic object representation') is the subject, depending on 'has' with 'properties'. Entity 2 ('particle-filter style tracker') is the object of the adjective 'more', depending on 'more' in the phrase 'more robust and reliable'. There is no direct dependency between Entity 1 and Entity 2.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

Example 5:

Sentence: French is cited as one of the languages with grammatical gender.
Entity 1: French
Entity 2: languages
Dependency Parsing Information: Entity 1 ('French') is the subject, depending on the verb 'is cited'. Entity 2 ('languages') is the object of the preposition 'of', depending on 'one' in the phrase 'one of the languages'. There is no direct dependency between Entity 1 and Entity 2, but they are indirectly connected through the phrase 'one of the languages'.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "HYPONYM-OF"
}}

Example 6:

Sentence: The learned intrinsic object structure is integrated into a particle-filter style tracker.
Entity 1: intrinsic object structure
Entity 2: particle-filter style tracker
Dependency Parsing Information: Entity 1 ('intrinsic object structure') is the subject, depending on the verb 'is integrated' in the phrase 'the learned intrinsic object structure is integrated.' Entity 2 ('particle-filter style tracker') is the object of the preposition 'into,' depending on 'into' in the phrase 'integrated into a particle-filter style tracker.' There is no direct dependency between Entity 1 and Entity 2, but they are indirectly connected through the verb 'integrated' and the preposition 'into.'
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "PART-OF"
}}

Example 7:

Sentence: A subcategorization dictionary built with the system improves the accuracy of a parser.
Entity 1: accuracy
Entity 2: parser
Dependency Parsing Information: Entity 1 ('accuracy') is the object, depending on the verb 'improves' in the phrase 'improves the accuracy.' Entity 2 ('parser') is the object of the preposition 'of,' depending on 'accuracy' in the phrase 'the accuracy of a parser.' There is a direct dependency between Entity 1 and Entity 2, as 'parser' modifies 'accuracy' through the prepositional phrase 'of a parser.'
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "EVALUATE-FOR"
}}

Example 8:

Sentence: The compact description of a video sequence has applications in video browsing and retrieval.
Entity 1: compact description of a video sequence
Entity 2: video browsing and retrieval
Dependency Parsing Information: Entity 1 ('compact description of a video sequence') is the subject, depending on the verb 'has' with 'The'. Entity 2 ('video browsing and retrieval') is the object of the preposition 'in', depending on 'in' in the phrase 'in video browsing and retrieval'. There is no direct dependency between Entity 1 and Entity 2, but they are indirectly connected through the preposition 'in'.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "NO-RELATION"
}}

Example 9:

Sentence: Our approach significantly outperforms state-of-the-art methods.
Entity 1: approach
Entity 2: state-of-the-art methods
Dependency Parsing Information: Entity 1 ('approach') is the subject, depending on the verb 'outperforms' with 'Our'. Entity 2 ('state-of-the-art methods') is the object, depending on 'outperforms' with 'approach'. There is no direct dependency between Entity 1 and Entity 2, but they are connected through the verb 'outperforms'.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "COMPARE"
}}

Example 10:

Sentence: The domain divergence involves different viewpoints and various resolutions.
Entity 1: viewpoints
Entity 2: resolutions
Dependency Parsing Information: Entity 1 ('viewpoints') is the object of the verb 'involves', depending on 'involves' in the phrase 'involves viewpoints'. Entity 2 ('resolutions') is also the object of the verb 'involves', depending on 'involves' in the phrase 'involves resolutions'. There is a direct dependency between Entity 1 and Entity 2.
All possible relations are: ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "EVALUATE-FOR", "COMPARE", "CONJUNCTION", "NO-RELATION"]

Answer:
{{
  "relationship": "CONJUNCTION"
}}

######################
-Real Data-
######################
Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
Dependency Parsing Information: {sentence_llm_dp_info}
All possible relations are: {relations}

Answer:
"""
################################################
# Prompt for Creating Dataset
################################################
# Creating Simple Dataset
VanillaPrompt_4_DATASET = """
Determine which relationship can be inferred from the given sentence and two entities. If no relationship exists, respond with "NO-RELATION". 
Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
All possible relations are: {relations}

Answer:
"""

VanillaPrompt_4_DATASET_QWEN = """
As a relation extraction expert, analyze the semantic relationship between Entity 1 and Entity 2, choose ONLY ONE relation from: {relations}.

Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}

Response Format: Final Answer: [relation]. Reason: [10-word explanation of inference]
"""

SHORTPrompt_4_DATASET = """
Determine which relationship can be inferred from the given sentence and two entities. 
Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
All possible relations are: {relations}
"""

MYPROPMT_4_DATASET = """
You are an information extraction system. Your goal is to determine the relationship between two given entities based on the provided sentence. 


Sentence: {input_text}
Entity 1: {source_entity}
Entity 2: {target_entity}
Dependency Parsing Information: {sentence_llm_dp_info}
All possible relations are: {relations}


Answer:
"""
################################################
# Prompt for GLOBAL Dependency Parsing
################################################
VanillaPrompt_CONVERT_SPACY_TO_LANGUAGE = """
Analyze the dependency parsing structure of the following sentence. Provide a detailed natural language description of the syntactic relations, including the relationship of each word to its head and how they interact. The output should describe:

1. The dependency relation of each word with its head.
2. The relationship between the words based on their syntactic roles.
3. Provide an overall description of how the sentence structure connects the words.

Format:
"Word '{{word}}' is the {{role}}, depending on '{{dependency relation}}' with the word '{{head}}'."

######################
-Examples-
######################
Example 1:

Sentence: The quick brown fox jumped over the lazy dog.
Dependency Information:
[{{'word': 'The', 'dep': 'det', 'head': 'fox'}},
 {{'word': 'quick', 'dep': 'amod', 'head': 'fox'}},
 {{'word': 'brown', 'dep': 'amod', 'head': 'fox'}},
 {{'word': 'fox', 'dep': 'nsubj', 'head': 'jumped'}},
 {{'word': 'jumped', 'dep': 'ROOT', 'head': 'jumped'}},
 {{'word': 'over', 'dep': 'prep', 'head': 'jumped'}},
 {{'word': 'the', 'dep': 'det', 'head': 'dog'}},
 {{'word': 'lazy', 'dep': 'amod', 'head': 'dog'}},
 {{'word': 'dog', 'dep': 'pobj', 'head': 'over'}}]

Answer:
"Word 'The' is the determiner, depending on 'det' with the word 'fox'.
Word 'quick' is the adjective, depending on 'amod' with the word 'fox'.
Word 'brown' is the adjective, depending on 'amod' with the word 'fox'.
Word 'fox' is the subject, depending on 'nsubj' with the word 'jumped'.
Word 'jumped' is the root verb, depending on 'ROOT' with itself.
Word 'over' is the preposition, depending on 'prep' with the word 'jumped'.
Word 'the' is the determiner, depending on 'det' with the word 'dog'.
Word 'lazy' is the adjective, depending on 'amod' with the word 'dog'.
Word 'dog' is the object of the preposition, depending on 'pobj' with the word 'over'."

Example 2:

Sentence: Alice quickly ran to the store.
Dependency Information:
[{{'word': 'Alice', 'dep': 'nsubj', 'head': 'ran'}},
 {{'word': 'quickly', 'dep': 'advmod', 'head': 'ran'}},
 {{'word': 'ran', 'dep': 'ROOT', 'head': 'ran'}},
 {{'word': 'to', 'dep': 'prep', 'head': 'ran'}},
 {{'word': 'the', 'dep': 'det', 'head': 'store'}},
 {{'word': 'store', 'dep': 'pobj', 'head': 'to'}}]

Answer:
"Word 'Alice' is the subject, depending on 'nsubj' with the word 'ran'.
Word 'quickly' is the adverb, depending on 'advmod' with the word 'ran'.
Word 'ran' is the root verb, depending on 'ROOT' with itself.
Word 'to' is the preposition, depending on 'prep' with the word 'ran'.
Word 'the' is the determiner, depending on 'det' with the word 'store'.
Word 'store' is the object of the preposition, depending on 'pobj' with the word 'to'."

######################
-Real Data-
######################
Sentence: {input_text}
Dependency Information: {spacy_dp_info}

Answer:
"""
