# Section 05: Masked Language Model (MLM) Evaluation Matrix (Stage 14)

This document details Stage 14: Multi-Model Masked Language Model Benchmark Matrix execution mechanics.

---

## Stage 14: Multi-Model MLM Evaluation Matrix
- **Script**: [scripts/14_mlm_evaluation.py](file:///c:/--Files--/Programming/pipeline/scripts/14_mlm_evaluation.py)
- **Grid Dimensions**: 7 Representative Models × 5 Corpus Representations × 5 Knowledge Subsets = **175 Independent Evaluation Runs**.

---

## Evaluation Grid Dimensions

1. **Representative Models (7)**:
   - `bert-base-uncased`
   - `dmis-lab/biobert-base-cased-v1.2`
   - `nlpaueb/legal-bert-base-uncased`
   - `allenai/scibert_scivocab_uncased`
   - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
   - `answerdotai/ModernBERT-base`
   - `roberta-base`

2. **Corpus Representations (5)**:
   - `narrative`, `key_value`, `template`, `json`, `mixed`

3. **Knowledge Subsets (5)**:
   - `high_knowledge`, `medium_knowledge`, `low_knowledge`, `balanced_knowledge`, `random_baseline`

---

## Core Masking & Evaluation Algorithm

For each of the 175 evaluation runs:
1. **Occurrence-ID Document Matching**:
   Representation text documents are filtered strictly by `occurrence_id` present in `outputs/subsets/<subset>.jsonl`:
   ```python
   target_docs = [rec["document"] for rec in rep_records if rec["occurrence_id"] in sub_occ_ids][:200]
   ```
2. **Token Masking**:
   15% of non-special tokens are masked with the model's `[MASK]` token.
3. **Loss Computation**:
   Cross-entropy loss $\mathcal{L}_{\text{CE}}$ is calculated over masked token predictions:
   $$\mathcal{L}_{\text{MLM}} = -\frac{1}{N_{\text{masked}}} \sum_{i=1}^{N_{\text{masked}}} \log P(w_i \mid \text{Context})$$
4. **Accuracy Recall**:
   Computes Top-1, Top-5, and Top-10 recall accuracy for general tokens, maritime domain tokens, rare terms, and 6 subdomain categories (Navigation, Weather, Safety, Machinery, Vessel, Casualties).

---

## Output Artifacts
- Cached JSON evaluation files in [outputs/evaluations/cache/](file:///c:/--Files--/Programming/pipeline/outputs/evaluations/cache) (`<model>__<rep>__<subset>.json`)
- Model summary JSON reports in [outputs/evaluations/](file:///c:/--Files--/Programming/pipeline/outputs/evaluations)
