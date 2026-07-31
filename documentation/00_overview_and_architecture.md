# Section 00: Overview & Pipeline Architecture

## 1. High-Level Overview

The **Maritime NLP Corpus Generation & Evaluation Pipeline** is a modular, end-to-end research framework designed to ingest raw relational databases (TSB MARSIS), process natural language accident reports, construct multi-format text representations, score semantic importance, benchmark multi-architecture tokenizers, run a 175-evaluation Masked Language Model (MLM) benchmark matrix across representative model families, perform statistical significance & feature ablation testing, and programmatically select the optimal domain adaptation strategy.

The pipeline answers the fundamental research question:
> *Does continuing Domain-Adaptive Pretraining (DAPT) on existing general language models suffice for maritime accident data, or is training a specialized domain model (**MaritimeBERT**) from scratch required?*

---

## 2. End-to-End Pipeline Architecture Diagram

```mermaid
flowchart TD
    subgraph Data Ingestion & Preprocessing
        S01[01_parse_dictionary.py]
        S02[02_profile_dataset.py]
        S03[03_discover_relationships.py]
        S04[04_select_semantic_columns.py]
        S05[05_merge_tables.py]
        S05a[05a_validate_records.py]
    end

    subgraph Document Synthesis & Corpus Export
        S06[06_generate_documents.py]
        S07[07_clean_documents.py]
        S08[08_export_corpus.py]
        S09[09_statistics.py]
        S10[10_extract_vocabulary.py]
    end

    subgraph Representations & Importance Assessment
        S11[11_corpus_representations.py]
        S12[12_semantic_importance.py]
    end

    subgraph Benchmarking & Decision Engine
        S13[13_tokenizer_analysis.py]
        S14[14_mlm_evaluation.py]
        S15[15_cross_model_benchmarking.py]
        S16[16_statistical_analysis.py]
        S17[17_decision_engine.py]
        S18[18_lint_corpus.py]
    end

    S01 & S02 & S03 --> S04 --> S05 --> S05a & S06
    S06 --> S07 --> S08 & S09 & S10
    S05 --> S11
    S07 & S10 --> S12 & S13
    S11 & S12 & S13 --> S14 --> S15 --> S16 --> S17
    S07 --> S18
```

---

## 3. Modular Directory Structure

```text
pipeline/
├── config/
│   └── config.json                  # Master configuration settings & threshold parameters
├── data/                            # Raw MARSIS CSV files (git-ignored)
├── scripts/                         # 18 Modular execution stage scripts & helpers
│   ├── pipeline_utils.py            # Shared logging, path resolution, encoding helpers
│   ├── text_sanitizer.py            # Text normalization & PII scrubbing regex engine
│   ├── 01_parse_dictionary.py       # Data dictionary parser
│   ├── 02_profile_dataset.py        # Statistical profiling of raw CSVs
│   ├── 03_discover_relationships.py # Foreign key discovery & cardinality mapping
│   ├── 04_select_semantic_columns.py# Semantic attribute selection
│   ├── 05_merge_tables.py           # Hierarchical relational join
│   ├── 05a_validate_records.py      # Record constraint validation
│   ├── 06_generate_documents.py     # Template narrative generation
│   ├── 07_clean_documents.py        # Header leakage stripping & normalization
│   ├── 08_export_corpus.py          # Plain text/JSONL export & SHA-256 manifest
│   ├── 09_statistics.py             # Corpus token & vocabulary statistics
│   ├── 10_extract_vocabulary.py     # Maritime TF-IDF term extraction
│   ├── 11_corpus_representations.py# 5 Multi-format representations generator
│   ├── 12_semantic_importance.py    # 9-feature semantic importance scoring engine
│   ├── 13_tokenizer_analysis.py     # Tokenizer benchmarking (fertility, coverage, speed)
│   ├── 14_mlm_evaluation.py         # 175-run MLM evaluation matrix grid
│   ├── 15_cross_model_benchmarking.py # MUI composite scoring, leaderboard, & plots
│   ├── 16_statistical_analysis.py   # Bootstrap CIs, paired t-tests, Cohen's d, ablation
│   ├── 17_decision_engine.py       # Threshold decision rules & report generation
│   └── 18_lint_corpus.py            # Corpus regex quality linting
├── outputs/                         # Exported JSON, JSONL, CSV, PNG, and MD artifacts
├── documentation/                   # Dedicated modular technical documentation
├── run_pipeline.py                  # Master CLI orchestrator
├── README.md                        # Quick start guide
└── DOCUMENTATION.md                 # Single-file master documentation
```

---

## 4. Master Orchestrator Usage (`run_pipeline.py`)

The pipeline can be run in its entirety or stage-by-stage using `run_pipeline.py`:

```bash
# Execute all 18 stages sequentially
python run_pipeline.py

# Execute a specific stage (e.g. Stage 15 Cross-Model Benchmarking)
python run_pipeline.py --stage 15
```
