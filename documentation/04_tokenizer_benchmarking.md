# Section 04: Multi-Model Tokenizer Benchmarking (Stage 13)

This document provides complete technical specifications for Stage 13: Multi-Model Tokenizer Benchmark Analysis.

---

## Stage 13: Multi-Model Tokenizer Benchmark Analysis
- **Script**: [`scripts/13_tokenizer_analysis.py`](file:///c:/--Files--/Programming/pipeline/scripts/13_tokenizer_analysis.py)
- **Execution Command**: `python run_pipeline.py --stage 13`

---

## 1. Benchmarked Tokenizer Models (14 Tokenizer Architectures)

Stage 13 evaluates 14 target Hugging Face tokenizers spanning WordPiece, Byte-Pair Encoding (BPE), SentencePiece BPE, and Byte-Level BPE families:

| Model ID | Tokenizer Family | Vocabulary Size | Target Domain |
| :--- | :--- | :--- | :--- |
| `bert-base-uncased` | Standard WordPiece | 30,522 | General English |
| `bert-large-uncased` | Standard WordPiece | 30,522 | General English |
| `roberta-base` | Byte-Level BPE | 50,265 | General English |
| `microsoft/deberta-v3-base` | SentencePiece BPE | 128,000 | General English |
| `answerdotai/ModernBERT-base` | Extended BPE | 50,280 | Modern General English |
| `allenai/scibert_scivocab_uncased` | SciVocab WordPiece | 31,090 | Scientific Literature |
| `dmis-lab/biobert-base-cased-v1.2` | Bio WordPiece | 28,996 | Biomedical |
| `microsoft/BiomedNLP-PubMedBERT...` | PubMed WordPiece | 30,522 | PubMed Abstracts |
| `emilyalsentzer/Bio_ClinicalBERT` | Clinical WordPiece | 30,522 | Clinical Records |
| `nlpaueb/legal-bert-base-uncased` | Legal WordPiece | 30,522 | Legal Contracts |
| `ProsusAI/finbert` | Financial WordPiece | 30,522 | Financial Reports |
| `anferico/bert-for-patents` | Patent WordPiece | 30,522 | Patent Specifications |
| `google/electra-base-discriminator` | WordPiece | 30,522 | General English |
| `distilbert-base-uncased` | WordPiece | 30,522 | Distilled English |

---

## 2. Evaluation Metrics Definitions

1. **Single-Token Vocabulary Coverage (%)**:
   Percentage of the 334 extracted maritime vocabulary terms that exist as single, atomic, un-fragmented tokens in the tokenizer dictionary:
   $$\text{Coverage} = \frac{\sum \mathbf{1}(|T(w)| = 1)}{|V_{\text{maritime}}|} \times 100$$

2. **Subword Fragmentation Rate (%)**:
   Percentage of maritime terms split into 2 or more subwords:
   $$\text{Fragmentation Rate} = \frac{\sum \mathbf{1}(|T(w)| > 1)}{|V_{\text{maritime}}|} \times 100$$

3. **Subwords-per-Word Fertility Ratio**:
   Mean number of subword units produced per domain word:
   $$\text{Fertility} = \frac{1}{|V_{\text{maritime}}|} \sum_{w \in V_{\text{maritime}}} |T(w)|$$

4. **Out-Of-Vocabulary (OOV) Rate (%)**:
   Percentage of tokens mapped to `[UNK]` or unmapped character sequences.

5. **Tokenization Throughput (tokens/sec)**:
   Speed of tokenizing raw text streams on CPU/GPU hardware.

---

## Output Artifacts
- [`outputs/tokenizer_analysis/tokenizer_comparison.csv`](file:///c:/--Files--/Programming/pipeline/outputs/tokenizer_analysis/tokenizer_comparison.csv)
- Detailed per-model JSON reports in [`outputs/tokenizer_analysis/*.json`](file:///c:/--Files--/Programming/pipeline/outputs/tokenizer_analysis)
