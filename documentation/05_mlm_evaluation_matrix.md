# Section 05: Masked Language Model (MLM) Evaluation Matrix (Stage 14)

This document provides complete technical specifications for Stage 14: Multi-Model Masked Language Model Benchmark Matrix.

---

## Stage 14: Multi-Model MLM Evaluation Matrix
- **Script**: [`scripts/14_mlm_evaluation.py`](file:///c:/--Files--/Programming/pipeline/scripts/14_mlm_evaluation.py)
- **Execution Command**: `python run_pipeline.py --stage 14`

---

## 1. 175-Run Evaluation Grid Architecture

Stage 14 executes a systematic **175-run evaluation grid**:
$$\text{Total Grid Runs} = 7 \text{ Representative Models} \times 5 \text{ Multi-Format Representations} \times 5 \text{ Knowledge Subsets} = 175 \text{ Runs}$$

### 7 Representative Model Families
1. `bert-base-uncased`: Standard WordPiece baseline.
2. `dmis-lab/biobert-base-cased-v1.2`: Medical/Bio cased WordPiece.
3. `nlpaueb/legal-bert-base-uncased`: Domain WordPiece for legal syntax.
4. `allenai/scibert_scivocab_uncased`: Domain WordPiece for scientific vocabulary.
5. `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`: PubMed specialized WordPiece.
6. `roberta-base`: Byte-Level BPE baseline.
7. `answerdotai/ModernBERT-base`: Modern extended BPE.

### 5 Representations Evaluated
- Narrative Prose (`narrative`)
- Key-Value Pairs (`key_value`)
- Slot-Filled Templates (`template`)
- Compact JSON (`json`)
- Hybrid Prose/Metadata (`mixed`)

### 5 Knowledge Subsets Evaluated
- High Knowledge (`high_knowledge.jsonl`)
- Medium Knowledge (`medium_knowledge.jsonl`)
- Low Knowledge (`low_knowledge.jsonl`)
- Balanced Knowledge (`balanced_knowledge.jsonl`)
- Random Baseline (`random_baseline.jsonl`)

---

## 2. Masking Protocol & Category Recall Subdomains

- **Masking Protocol**: Tokens are randomly masked at a **15% rate** following standard BERT pretraining practice. Cross-Entropy Loss is calculated strictly over masked token positions.
- **6 Subdomain Categories Evaluated**:
  1. `vessel_terminology`: Ship types, hull specs, deck attributes.
  2. `navigation`: GPS, AIS, radar, VHF, sonar, compass, VDR.
  3. `machinery_propulsion`: Main engines, boilers, steering gear, windlasses.
  4. `casualty_incident`: Collision, grounding, stranding, flooding, fatalities.
  5. `weather_environment`: Wind, sea states, visibility, fog, temperature.
  6. `safety_lifesaving`: Lifeboats, liferafts, EPIRBs, SARTs, lifejackets.

---

## Output Artifacts
- Evaluation run JSON cache files in [`outputs/evaluations/cache/*.json`](file:///c:/--Files--/Programming/pipeline/outputs/evaluations/cache)
- Summarized model reports in [`outputs/evaluations/`](file:///c:/--Files--/Programming/pipeline/outputs/evaluations)
