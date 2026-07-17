# Maritime NLP Corpus Generation Pipeline

A production-grade, memory-efficient data engineering and Natural Language Processing (NLP) pipeline designed to convert relational maritime accident databases (TSB MARSIS views) into a clean, high-quality, domain-specific text corpus suitable for training domain-specific language models like **MaritimeBERT** using Masked Language Modeling (MLM).

## Project Goal
Generate a clean, high-quality, research-grade maritime text corpus from structured maritime accident databases while preserving structured metadata for future NLP tasks (such as instruction tuning or relation extraction).

---

## Directory Structure

```
pipeline/
├── config/
│   └── config.json          # Pipeline and validation configuration
├── data/                    # Raw maritime accident CSV files
├── templates/               # Randomized template families
│   ├── vessel_templates.json
│   ├── injury_templates.json
│   └── equipment_templates.json
├── scripts/                 # Core pipeline Python scripts
│   ├── pipeline_utils.py    # Common helper utilities and logging
│   ├── 01_parse_dictionary.py
│   ├── 02_profile_dataset.py
│   ├── 03_discover_relationships.py
│   ├── 04_select_semantic_columns.py
│   ├── 05_merge_tables.py
│   ├── 05a_validate_records.py
│   ├── 06_generate_documents.py
│   ├── 07_clean_documents.py
│   ├── 08_export_corpus.py
│   ├── 09_statistics.py
│   ├── 10_extract_vocabulary.py
│   └── 11_tokenizer_analysis.py
├── outputs/                 # Pipeline output files and logs
│   ├── merged_records.jsonl
│   ├── raw_documents.jsonl
│   ├── clean_documents.jsonl
│   ├── maritime_corpus.txt  # Final plain-text corpus (BERT pretraining)
│   ├── maritime_corpus.jsonl # Schema-preserving final corpus
│   ├── maritime_vocabulary.txt # Extracted domain vocabulary terms
│   ├── statistics.json
│   ├── tokenizer_analysis.json # BERT subwords split analysis
│   ├── manifest.json        # Corpus version metadata
│   ├── corpus_quality_report.md # Human-readable corpus summary report
│   └── logs/                # Executions log folder
├── run_pipeline.py          # Master orchestrator script
├── requirements.txt         # Package dependencies
└── README.md                # Documentation
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.12+
- The dataset CSV files placed in the `data/` directory (e.g. `Occurrence.csv`, `Occurrence_Vessel.csv`, etc.).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Pipeline Execution

The pipeline is fully modular and supports both running the entire sequence and running individual stages.

### Run the Full Pipeline
To run all 11 stages sequentially:
```bash
python run_pipeline.py
```

### Run a Specific Stage
If you are adjusting templates or changing text cleaning thresholds, you can rerun individual stages without reprocessing the entire dataset:
```bash
# Run only the merging stage
python run_pipeline.py --stage 05

# Run validation checks
python run_pipeline.py --stage 05a

# Regenerate natural language documents
python run_pipeline.py --stage 06

# Recalculate statistics and quality reports
python run_pipeline.py --stage 09
```

---

## Detailed Pipeline Stages

1. **`01_parse_dictionary.py`**: Parses the inventory data dictionary, auto-pairs numeric code ID/Enum columns with their English translations (`DisplayEng`), and generates column metadata.
2. **`02_profile_dataset.py`**: Analyzes the CSV datasets, computing row/column counts, missing percentage, unique counts, cardinality, and inferring primary/foreign keys.
3. **`03_discover_relationships.py`**: Builds a schema relationship graph using `networkx` to map joins.
4. **`04_select_semantic_columns.py`**: Discards system/admin columns and French duplicates, preserving only NLP semantic fields.
5. **`05_merge_tables.py`**: Merges occurrences, vessels, and child tables into a nested schema, saved to `merged_records.jsonl`. Guarantees 100% occurrence data retention via Left Outer Join semantics and aggregates orphan child records into a single synthetic placeholder vessel.
6. **`05a_validate_records.py`**: Assesses join integrity, orphan rows, missing primary keys, impossible dates, and implausible numeric values, gracefully handling placeholder occurrences.
7. **`06_generate_documents.py`**: Synthesizes structured occurrence and vessel records into multiple semantically focused documents per occurrence (such as profiles, characteristics, weather environment, cargo, voyage activity, equipment, injuries, and integrated operational contexts) to maximize MLM training diversity. Enforces adaptive granularity, information density thresholds, and local deduplication.
8. **`07_clean_documents.py`**: Normalizes whitespace, cleans encoding artifacts, deduplicates duplicate sentences, and filters short documents.
9. **`08_export_corpus.py`**: Exports the final plain text `maritime_corpus.txt` (BERT MLM format) and metadata-preserving `maritime_corpus.jsonl`, writing `manifest.json`.
10. **`09_statistics.py`**: Computes Shannon entropy, Type-Token Ratio, common bigrams/trigrams, and generates `corpus_quality_report.md`.
11. **`10_extract_vocabulary.py`**: Compiles a sorted domain-specific maritime vocabulary list (`maritime_vocabulary.txt`).
12. **`11_tokenizer_analysis.py`**: Tests `bert-base-uncased` on the corpus to measure OOV rates, subword splitting, and subwords-per-token ratios for maritime terms.
