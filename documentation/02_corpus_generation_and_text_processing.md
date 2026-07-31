# Section 02: Document Generation & Corpus Processing (Stages 06–10)

This document provides complete technical specifications for natural language document synthesis, text cleaning and normalization, multi-format corpus exporting, statistical profiling, and domain vocabulary extraction.

---

## Stage 06: Generate Natural Language Documents
- **Script**: [`scripts/06_generate_documents.py`](file:///c:/--Files--/Programming/pipeline/scripts/06_generate_documents.py)
- **Execution Command**: `python run_pipeline.py --stage 06`
- **Algorithmic Logic**:
  1. Ingests nested occurrence records from `merged_records.jsonl`.
  2. Applies template rules (`templates/vessel_templates.json`, `injury_templates.json`, `equipment_templates.json`) to render discrete codes into natural English sentences.
  3. Synthesizes documents covering occurrence background, environmental conditions, vessel specs, activity phase, equipment status, and casualties.
- **Output File**: [`outputs/raw_documents.jsonl`](file:///c:/--Files--/Programming/pipeline/outputs/raw_documents.jsonl) (891 MB, 96,714 records)

---

## Stage 07: Clean and Normalize Documents
- **Script**: [`scripts/07_clean_documents.py`](file:///c:/--Files--/Programming/pipeline/scripts/07_clean_documents.py)
- **Execution Command**: `python run_pipeline.py --stage 07`
- **Algorithmic Logic**:
  1. Ingests `raw_documents.jsonl` line-by-line.
  2. Invokes `strip_administrative_noise()` from `text_sanitizer.py` to remove header tags (`RECORD ID: 12345`, `formerly occno: X`).
  3. Normalizes whitespace, hyphens, and quotes; removes non-ASCII artifacts.
  4. Filters out short or non-informative documents below `min_doc_length` (50 characters).
- **Output File**: [`outputs/clean_documents.jsonl`](file:///c:/--Files--/Programming/pipeline/outputs/clean_documents.jsonl) (807 MB, 96,714 records)

---

## Stage 08: Export Maritime Corpus & Manifest
- **Script**: [`scripts/08_export_corpus.py`](file:///c:/--Files--/Programming/pipeline/scripts/08_export_corpus.py)
- **Execution Command**: `python run_pipeline.py --stage 08`
- **Algorithmic Logic**:
  1. Exports `clean_documents.jsonl` into plain text line-by-line format (`maritime_corpus.txt`) for direct use in language model pretraining.
  2. Exports schema-preserving JSONL format (`maritime_corpus.jsonl`).
  3. Computes SHA-256 checksums and file sizes to write dataset distribution metadata to `manifest.json`.
- **Output Artifacts**:
  - [`outputs/maritime_corpus.txt`](file:///c:/--Files--/Programming/pipeline/outputs/maritime_corpus.txt) (21 MB text line export)
  - [`outputs/maritime_corpus.jsonl`](file:///c:/--Files--/Programming/pipeline/outputs/maritime_corpus.jsonl) (796 MB)
  - [`outputs/manifest.json`](file:///c:/--Files--/Programming/pipeline/outputs/manifest.json)

---

## Stage 09: Calculate Corpus Statistics & Report
- **Script**: [`scripts/09_statistics.py`](file:///c:/--Files--/Programming/pipeline/scripts/09_statistics.py)
- **Execution Command**: `python run_pipeline.py --stage 09`
- **Algorithmic Logic**:
  1. Analyzes token and character length distributions across all documents.
  2. Computes total token counts, unique vocabulary size, Type-Token Ratio ($\text{TTR} = |V| / N$), and unigram Shannon entropy.
  3. Calculates top 20 frequent bigrams and trigrams.
  4. Formats executive statistical summary into [`outputs/corpus_quality_report.md`](file:///c:/--Files--/Programming/pipeline/outputs/corpus_quality_report.md).
- **Output Artifacts**:
  - [`outputs/statistics.json`](file:///c:/--Files--/Programming/pipeline/outputs/statistics.json)
  - [`outputs/corpus_quality_report.md`](file:///c:/--Files--/Programming/pipeline/outputs/corpus_quality_report.md)

---

## Stage 10: Extract Maritime Vocabulary
- **Script**: [`scripts/10_extract_vocabulary.py`](file:///c:/--Files--/Programming/pipeline/scripts/10_extract_vocabulary.py)
- **Execution Command**: `python run_pipeline.py --stage 10`
- **Algorithmic Logic**:
  1. Fits a TF-IDF vectorizer over `clean_documents.jsonl`.
  2. Applies domain keyword frequency analysis and filters out general English stopwords.
  3. Isolates 334 domain-specific maritime terms across vessel types, navigation equipment, weather phenomena, and casualty classes.
- **Output File**: [`outputs/maritime_vocabulary.txt`](file:///c:/--Files--/Programming/pipeline/outputs/maritime_vocabulary.txt) (334 domain terms)
