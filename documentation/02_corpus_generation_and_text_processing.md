# Section 02: Document Generation & Corpus Processing (Stages 06–10)

This document details document synthesis, text cleaning, corpus exporting, statistical profiling, and maritime vocabulary extraction.

---

## Stage 06: Generate Natural Language Documents
- **Script**: [scripts/06_generate_documents.py](file:///c:/--Files--/Programming/pipeline/scripts/06_generate_documents.py)
- **Core Logic**: Applies structured text template rules to nested occurrence records from `merged_records.jsonl`. Translates discrete numerical codes and key-value fields into natural English sentences.
- **Output File**: [outputs/raw_documents.jsonl](file:///c:/--Files--/Programming/pipeline/outputs/raw_documents.jsonl) (891 MB, 96,714 records)

---

## Stage 07: Clean and Normalize Documents
- **Script**: [scripts/07_clean_documents.py](file:///c:/--Files--/Programming/pipeline/scripts/07_clean_documents.py)
- **Core Logic**: Normalizes raw documents using `text_sanitizer.py`:
  - Strips administrative header tags (e.g. `RECORD ID: 12345`).
  - Normalizes punctuation, quotes, and hyphens.
  - Sanitizes non-ASCII characters.
  - Removes empty or non-informative text sentences.
- **Output File**: [outputs/clean_documents.jsonl](file:///c:/--Files--/Programming/pipeline/outputs/clean_documents.jsonl) (807 MB, 96,714 records)

---

## Stage 08: Export Maritime Corpus & Manifest
- **Script**: [scripts/08_export_corpus.py](file:///c:/--Files--/Programming/pipeline/scripts/08_export_corpus.py)
- **Core Logic**: Exports the corpus into distribution formats:
  - Plain text file (`maritime_corpus.txt`) with one clean document per line.
  - JSONL file (`maritime_corpus.jsonl`).
  - SHA-256 integrity manifest (`manifest.json`).
- **Output Files**:
  - [outputs/maritime_corpus.txt](file:///c:/--Files--/Programming/pipeline/outputs/maritime_corpus.txt) (21 MB text line export)
  - [outputs/maritime_corpus.jsonl](file:///c:/--Files--/Programming/pipeline/outputs/maritime_corpus.jsonl) (796 MB)
  - [outputs/manifest.json](file:///c:/--Files--/Programming/pipeline/outputs/manifest.json)

---

## Stage 09: Calculate Corpus Statistics & Report
- **Script**: [scripts/09_statistics.py](file:///c:/--Files--/Programming/pipeline/scripts/09_statistics.py)
- **Core Logic**: Computes corpus-wide metrics:
  - Total tokens, unique vocabulary size, Type-Token Ratio (TTR).
  - Sentence length distribution (mean, median, std, quartiles).
  - Document character and word length distributions.
- **Output Files**:
  - [outputs/statistics.json](file:///c:/--Files--/Programming/pipeline/outputs/statistics.json)
  - [outputs/corpus_quality_report.md](file:///c:/--Files--/Programming/pipeline/outputs/corpus_quality_report.md)

---

## Stage 10: Extract Maritime Vocabulary
- **Script**: [scripts/10_extract_vocabulary.py](file:///c:/--Files--/Programming/pipeline/scripts/10_extract_vocabulary.py)
- **Core Logic**: Runs TF-IDF keyword extraction and frequency analysis over `clean_documents.jsonl`. Filters out general English stopwords to isolate domain-specific maritime terms (vessels, navigation gear, casualty classes, weather states).
- **Output File**: [outputs/maritime_vocabulary.txt](file:///c:/--Files--/Programming/pipeline/outputs/maritime_vocabulary.txt) (334 domain terms)
