# Section 04: Multi-Model Tokenizer Benchmarking (Stage 13)

This document details Stage 13: Multi-Model Tokenizer Benchmark Analysis and the tokenizer-driven model filtering methodology.

---

## Stage 13: Multi-Model Tokenizer Benchmark Analysis
- **Script**: [scripts/13_tokenizer_analysis.py](file:///c:/--Files--/Programming/pipeline/scripts/13_tokenizer_analysis.py)
- **Target Tokenizer Candidates**: 14 Hugging Face encoder models (BERT Uncased, BioBERT, LegalBERT, SciBERT, PubMedBERT, ModernBERT, RoBERTa, DeBERTa, ELECTRA, FinBERT, etc.).

---

## Benchmark Metrics Evaluated

1. **Vocabulary Size ($V$)**: Total subword vocabulary capacity.
2. **Single-Token Coverage (%)**: Percentage of 334 maritime terms tokenized as a single token.
3. **Subword Fragmentation Rate (%)**: Percentage of domain terms split into 2+ subword pieces.
4. **Subword Fertility**: Average number of subwords generated per raw word:
   $$\text{Fertility} = \frac{\sum N_{\text{subwords}}}{\sum N_{\text{words}}}$$
5. **Out-of-Vocabulary (OOV) Rate (%)**: Proportion of unk tokens (`[UNK]`).
6. **Tokenizer Speed (tok/sec)**: Subword tokenization throughput profiling.

---

## Empirical Benchmark Ranking Table

| Rank | Model Name | Vocab Size | Coverage (%) | Fragmentation (%) | Fertility | OOV Rate (%) | Speed (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `bert-base-uncased` | 30,522 | **74.85%** | **25.15%** | **1.3416** | 0.00% | 241,993 |
| **1** | `bert-large-uncased` | 30,522 | **74.85%** | **25.15%** | **1.3416** | 0.00% | 248,067 |
| **1** | `distilbert-base-uncased` | 30,522 | **74.85%** | **25.15%** | **1.3416** | 0.00% | 248,338 |
| **1** | `google/electra-base-discriminator` | 30,522 | **74.85%** | **25.15%** | **1.3416** | 0.00% | 252,753 |
| **1** | `ProsusAI/finbert` | 30,522 | **74.85%** | **25.15%** | **1.3416** | 0.00% | 255,646 |
| **2** | `dmis-lab/biobert-base-cased-v1.2` | 28,996 | 65.87% | 34.13% | 1.4235 | 0.00% | 284,969 |
| **2** | `emilyalsentzer/Bio_ClinicalBERT` | 28,996 | 65.87% | 34.13% | 1.4235 | 0.00% | 252,405 |
| **3** | `nlpaueb/legal-bert-base-uncased` | 30,522 | 63.77% | 36.23% | 1.4198 | 0.04% | 255,161 |
| **4** | `allenai/scibert_scivocab_uncased` | 31,090 | 58.08% | 41.92% | 1.4107 | 0.00% | 266,700 |
| **5** | `microsoft/BiomedNLP-PubMedBERT...` | 30,522 | 57.19% | 42.81% | 1.4050 | 0.00% | 286,098 |
| **6** | `answerdotai/ModernBERT-base` | 50,280 | 36.53% | 63.47% | 1.5060 | 0.00% | 328,791 |
| **7** | `roberta-base` | 50,265 | 35.33% | 64.67% | 1.5124 | 0.00% | **337,205** |

---

## Tokenizer-Driven Model Selection

To prevent evaluation redundancy, models sharing the **exact same tokenizer binary** were grouped together, selecting 1 representative model per family for downstream MLM benchmarking:

1. 🥇 **`bert-base-uncased`** (Standard WordPiece 30k representative)
2. 🥈 **`dmis-lab/biobert-base-cased-v1.2`** (Bio WordPiece 29k representative)
3. 🥉 **`nlpaueb/legal-bert-base-uncased`** (Legal WordPiece 30k)
4. **`allenai/scibert_scivocab_uncased`** (SciVocab WordPiece 31k)
5. **`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`** (PubMed WordPiece 30k)
6. **`answerdotai/ModernBERT-base`** (Modern Extended BPE 50k)
7. **`roberta-base`** (Byte-Level BPE 50k)

### Computational Savings
- Full Grid: 14 models × 5 reps × 5 subsets = **350 runs**
- Filtered Representative Grid: 7 models × 5 reps × 5 subsets = **175 runs** (50% reduction, ~2 hours saved)

---

### Output Artifacts
- [outputs/tokenizer_analysis/tokenizer_comparison.csv](file:///c:/--Files--/Programming/pipeline/outputs/tokenizer_analysis/tokenizer_comparison.csv)
- Per-model JSON reports in [outputs/tokenizer_analysis/](file:///c:/--Files--/Programming/pipeline/outputs/tokenizer_analysis)
