# Section 03: Multi-Format Representations & Semantic Importance (Stages 11–12)

This document details the generation of 5 multi-format corpus representations and the 9-feature semantic importance scoring engine.

---

## Stage 11: Multi-Format Corpus Representations
- **Script**: [scripts/11_corpus_representations.py](file:///c:/--Files--/Programming/pipeline/scripts/11_corpus_representations.py)
- **Core Logic**: Renders each occurrence record into **5 distinct text representations**:
  1. **Narrative**: Flowing natural language prose paragraphs.
  2. **Key-Value**: Structured `Field: Value` formatted lines.
  3. **Template**: Standardized slot-filled sentences.
  4. **JSON**: Serialized JSON strings.
  5. **Mixed**: Hybrid prose paired with key-value metadata headers.
- **Output Directory**: [outputs/corpus_representations/](file:///c:/--Files--/Programming/pipeline/outputs/corpus_representations)
  - `narrative.jsonl`
  - `key_value.jsonl`
  - `template.jsonl`
  - `json.jsonl`
  - `mixed.jsonl`

---

## Stage 12: Semantic Importance Assessment & Knowledge Classification
- **Script**: [scripts/12_semantic_importance.py](file:///c:/--Files--/Programming/pipeline/scripts/12_semantic_importance.py)

### 1. 9-Feature Weighted Scoring Formula

Every document in `clean_documents.jsonl` is scored across 9 domain features:

$$\text{Importance Score} = \text{Clip}\left( 100 \times \sum_{i=1}^9 w_i f_i - 0.10 \times \text{RedundancyPenalty}, \; 0, \; 100 \right)$$

| Feature ($f_i$) | Weight ($w_i$) | Description |
| :--- | :--- | :--- |
| **Maritime Term Density** | $0.30$ | Ratio of extracted maritime vocabulary terms to total raw words. |
| **Rare Term Count** | $0.20$ | Frequency of low-corpus-frequency domain terms (e.g. `gyrocompass`, `epirb`, `transom`). |
| **Concept Diversity** | $0.15$ | Number of covered subdomains (Navigation, Weather, Safety, Machinery, Vessel, Casualty). |
| **Entity Diversity** | $0.10$ | Distinct vessel names, GT values, flags, and registered countries. |
| **Event Complexity** | $0.10$ | Sequence length and multi-incident occurrence indicators. |
| **Information Density** | $0.10$ | Non-stopword lexical density ratio. |
| **Metadata Completeness** | $0.05$ | Percentage of non-null attributes in the underlying record. |
| **Linguistic Diversity** | $0.05$ | Unigram entropy and Type-Token Ratio (TTR). |
| **Domain Novelty** | $0.05$ | Inverse Document Frequency (IDF) novel term score. |

---

### 2. Knowledge Subset Classifier & Quantile Extraction

Documents are classified into knowledge tiers and exported to [outputs/subsets/](file:///c:/--Files--/Programming/pipeline/outputs/subsets):

- **High Knowledge**: Top 1,000 highest-scoring documents (`high_knowledge.jsonl`).
- **Medium Knowledge**: 1,000 median-scoring documents (`medium_knowledge.jsonl`).
- **Low Knowledge**: 1,000 lower-scoring non-boilerplate documents (`low_knowledge.jsonl`).
- **Balanced Knowledge**: 1,000 documents evenly sampled across High, Med, Low (`balanced_knowledge.jsonl`).
- **Random Baseline**: 1,000 randomly sampled documents (`random_baseline.jsonl`).
- **General English Baseline**: 10 general English sentences (`general_english_baseline.jsonl`).

---

### Output Artifacts
- [outputs/document_importance.jsonl](file:///c:/--Files--/Programming/pipeline/outputs/document_importance.jsonl)
- [outputs/importance_statistics.json](file:///c:/--Files--/Programming/pipeline/outputs/importance_statistics.json)
- [outputs/importance_distribution.png](file:///c:/--Files--/Programming/pipeline/outputs/importance_distribution.png)
- [outputs/subsets/](file:///c:/--Files--/Programming/pipeline/outputs/subsets)
