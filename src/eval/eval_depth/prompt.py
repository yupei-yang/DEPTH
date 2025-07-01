MY_Prompt_4_Refinement = """
You are a scientific information extraction assistant specialized in refining relation predictions on the SciERC dataset.

Your task is to **validate and correct the relation type for a specific pair of entities**, using global sentence-level information such as the dependency parse and the model's full set of predicted relations. This is **not a relation extraction task from scratch**, but a **post-processing correction stage** after an initial prediction round.

######################
-Task Description-
######################
For each input, you will:
1. Review the **candidate relation** predicted between two entities.
2. Refer to the **full sentence**, **its dependency parse**, and the **model’s other predicted relations** to reason about correctness.
3. Decide whether to:
   - **CONFIRM** the predicted relation.
   - **REPLACE** it with a better-fitting relation.
   - **REMOVE** it if no valid relation exists (by outputting `"NO-RELATION"`).
   - **ADD** the correct relation if the candidate is missing (in this case, the Candidate Relation will be `None`, `null`, or `Fail`).

You should focus **only** on the given entity pair, but you **can and should** use sentence-level context and other predicted relations for support.

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
-Output Format-
######################
Return the corrected relation in **strict JSON** format:

{{
  "relationship": "[ONE OF THE 8 TYPES]"
}}

######################
-Examples-
######################

Example 1: Fixing Incorrect Relation  

Sentence: LSTM networks outperform CNN models in sequence prediction tasks.
Entity 1: LSTM networks
Entity 2: CNN models
Dependency Parsing Information of the whole sentence: Word 'LSTM' is a compound modifier (compound) of 'networks'  
Word 'networks' is the subject (nsubj) of 'outperform'  
Word 'outperform' is the root verb (ROOT)  
Word 'CNN' is a compound modifier (compound) of 'models'  
Word 'models' is the direct object (dobj) of 'outperform'  
Word 'in' introduces a prepositional phrase (prep_in) modifying 'outperform'  
Word 'sequence' is an adjective modifier (amod) of 'tasks'  
Word 'prediction' is a compound modifier (compound) of 'tasks'  
Word 'tasks' is the object of preposition (pobj) for 'in' 
All Candidate Relations for the sentence:
[["LSTM networks", "CNN models", "PART-OF"], ["LSTM networks", "sequence prediction tasks", "USED-FOR"]]
Candidate Relation for Entity 1 and Entity 2: "PART-OF"

Answer:
{{
  "relationship": "COMPARE"
}}

Example 2: Adding Missing Relation

Sentence: The transformer architecture includes both encoder and decoder components.
Entity 1: encoder  
Entity 2: transformer architecture
Dependency Parsing Information of the whole sentence: Word 'The' is the determiner (det) of 'architecture'  
Word 'transformer' is a compound modifier (compound) of 'architecture'  
Word 'architecture' is the subject (nsubj) of 'includes'  
Word 'includes' is the root verb (ROOT)  
Word 'both' is a coordinating determiner (predeterminer) of 'encoder'  
Word 'encoder' and 'decoder' are conjuncts (conj) connected by 'and'  
Word 'components' is the head noun modified by both 'encoder' and 'decoder' (conjunct noun modifiers)  
Word 'encoder' modifies 'components' (compound)  
Word 'decoder' modifies 'components' (compound)
All Candidate Relations for the sentence:
[["decoder", "transformer architecture", "PART-OF"]]
Candidate Relation for Entity 1 and Entity 2: None

Answer:
{{
  "relationship": "PART-OF"
}}

Example 3: Removing Incorrect Relation

Sentence: Our framework does not use attention mechanisms for this task.
Entity 1: framework
Entity 2: attention mechanisms
Dependency Parsing Information of the whole sentence: Word 'Our' is the possessive determiner (poss) of 'framework'  
Word 'framework' is the subject (nsubj) of 'use'  
Word 'does' is the auxiliary verb (aux) of 'use'  
Word 'not' is the negation marker (neg) of 'use'  
Word 'use' is the root verb (ROOT)  
Word 'attention' is a compound modifier (compound) of 'mechanisms'  
Word 'mechanisms' is the direct object (dobj) of 'use'  
Word 'for' introduces a prepositional phrase (prep) modifying 'use'  
Word 'this' is the determiner (det) of 'task'  
Word 'task' is the object of preposition (pobj) for 'for'
All Candidate Relations for the sentence:
[["framework", "attention mechanisms", "USED-FOR"]]
Candidate Relation for Entity 1 and Entity 2: "USED-FOR"

Answer:
{{
  "relationship": "NO-RELATION"
}}

Example 4: Confirming Correct Relation

Sentence: BERT is evaluated on the GLUE benchmark using accuracy metrics.
Entity 1: BERT 
Entity 2: GLUE benchmark
Dependency Parsing Information of the whole sentence: Word 'BERT' is the passive subject (nsubjpass) of 'evaluated'  
Word 'is' is the passive auxiliary (auxpass) for 'evaluated'  
Word 'evaluated' is the root verb (ROOT)  
Word 'on' introduces a prepositional phrase (prep) modifying 'evaluated'  
Word 'the' is the determiner (det) of 'benchmark'  
Word 'GLUE' is a compound modifier (compound) of 'benchmark'  
Word 'benchmark' is the object of preposition (pobj) for 'on'  
Word 'using' introduces a prepositional phrase (prep) modifying 'evaluated'  
Word 'accuracy' is a compound modifier (compound) of 'metrics'  
Word 'metrics' is the object of preposition (pobj) for 'using'
All Candidate Relations for the sentence:
[["BERT", "GLUE benchmark", "EVALUATE-FOR"], ["accuracy metrics", "GLUE benchmark", "USED-FOR"]]
Candidate Relation for Entity 1 and Entity 2: "EVALUATE-FOR"

Answer:
{{
  "relationship": "EVALUATE-FOR"
}}

######################
-Real Data-
######################
Sentence: {raw_sentence}  
Entity 1: {source_entity}  
Entity 2: {target_entity}  
Dependency Parsing Information of the whole sentence: {dp_description}
All possible relations are: {relation_types}  
All Candidate Relations for the sentence: {predicted_rels}  
Candidate Relation for Entity 1 and Entity 2: {cand_rel}

Answer:
"""
