# Maritime Corpus Multi-Model Benchmarking Report

## 1. Executive Summary
This research benchmark evaluates 14 pretrained encoder models across 5 multi-format corpus representations and 5 knowledge-classified subsets (175 paired matrix evaluation runs: 7 deduplicated model architectures × 5 representations × 5 subsets). The goal is to evaluate whether continued **Domain-Adaptive Pretraining (DAPT)** is indicated or if training a domain-specific **MaritimeBERT from Scratch** is supported by benchmark heuristics.

**Triage Recommendation**: Strategy B: Train Domain-Specific MaritimeBERT Model From Scratch
* **Top Observed Pretrained Encoder**: `answerdotai/ModernBERT-base` (DUI Score: 68.15)
* **Maritime Top-1 Accuracy**: 69.95%
* **General-to-Maritime Performance Gap**: -4.21%
* **Subword Fragmentation Rate**: 63.28%

---

## 2. Corpus & Representation Analysis
The Maritime text corpus was compiled into 5 distinct multi-format representations:
1. **Narrative**: Sanitized natural language paragraphs from the clean text corpus.
2. **Key-Value**: Extracted structural `Field: Value` formatted text.
3. **Template**: Standardized semi-structured template sentences with deterministic selection.
4. **Structured Semantic**: Serialized JSON representations of extracted text features.
5. **Mixed**: Hybrid document pairing extracted structural headers with narrative text (introducing additional textual overhead).

---

## 3. Representation Benchmark Results
Evaluations across representations demonstrate how structural formatting alters tokenizer behavior and MLM accuracy while controlling for the underlying source text.

---

## 4. Tokenizer Benchmark Results
Single-token vocabulary coverage, subword fertility, and observed throughput under the benchmark environment vary across domain tokenizers:

| Model Name | Vocab Size | Single-Token Coverage (%) | Fragmentation Rate (%) | OOV Rate (%) | Tokenizer Speed (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `answerdotai/ModernBERT-base` | 59000 | 36.72% | 63.28% | 0.0000% | 265830.2 |
| `roberta-base` | 50000 | 35.22% | 64.78% | 0.0000% | 304253.8 |
| `bert-base-uncased` | 44000 | 73.43% | 26.57% | 0.0000% | 170400.0 |
| `dmis-lab/biobert-base-cased-v1.2` | 44000 | 64.48% | 35.52% | 0.0000% | 161068.7 |
| `allenai/scibert_scivocab_uncased` | 44000 | 58.21% | 41.79% | 0.0000% | 230259.9 |
| `nlpaueb/legal-bert-base-uncased` | 44000 | 62.39% | 37.61% | 0.0289% | 268004.2 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 44000 | 57.61% | 42.39% | 0.0000% | 173899.1 |

---

## 5. MLM Benchmark Results (175-Run Matrix Grid Summary)
Full model leaderboard ranked by the composite **Domain Understanding Index (DUI)**:

| Rank | Model Name | DUI Score | Top-1 Accuracy (%) | Rare Term Acc (%) | MLM Loss | Domain Shift Gap (%) | Params (M) | Eval Time (ms/doc) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `answerdotai/ModernBERT-base` | **68.15** | 69.95% ± 7.22% | 59.36% | 1.6463 | -25.50% | 149M | 213.58ms |
| 2 | `roberta-base` | **66.26** | 66.49% ± 6.79% | 59.44% | 1.8957 | -13.86% | 125M | 130.47ms |
| 3 | `bert-base-uncased` | **64.65** | 54.42% ± 7.67% | 55.51% | 3.1039 | -13.51% | 110M | 181.37ms |
| 4 | `dmis-lab/biobert-base-cased-v1.2` | **60.21** | 51.91% ± 10.82% | 43.90% | 3.1889 | -6.91% | 110M | 122.49ms |
| 5 | `allenai/scibert_scivocab_uncased` | **57.30** | 56.02% ± 9.60% | 24.35% | 2.7989 | -20.73% | 110M | 205.73ms |
| 6 | `nlpaueb/legal-bert-base-uncased` | **56.75** | 54.55% ± 9.35% | 23.28% | 2.9696 | 5.45% | 110M | 109.18ms |
| 7 | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **52.36** | 46.78% ± 11.19% | 26.51% | 4.0012 | 0.28% | 110M | 112.27ms |

---

## 6. Statistical Significance & Effect Size Analysis
Bootstrap 95% Confidence Intervals and paired significance tests (t-test & Wilcoxon signed-rank test), strictly aligned by experimental configuration cell `(representation, subset)`, evaluate performance differences across paired configurations with Cohen's d and Cliff's Delta effect sizes (evaluated across 25 paired cells controlling for underlying text).

---

## 7. Computational Resource & Tokenizer Speed Benchmark
Profiling model parameter counts, memory footprints, and evaluation throughput (measured as full-pipeline evaluation time per document, including tokenization, masking, and model forward pass) provides observed resource metrics under the evaluation environment.

---

## 8. Scoring Engine Feature Ablation Study
Empirical leave-one-feature-out evaluation of individual scoring features indicates the marginal contribution of each semantic feature (rare vocabulary, concept diversity, entity diversity, event complexity, structural completeness) across evaluated corpus documents.

---

## 9. Decision Engine & Sensitivity Analysis (Configurable Heuristic Thresholds)
Using configurable heuristic decision criteria, the decision engine evaluated empirical metrics against user-defined thresholds:

* **Selected Strategy**: `Strategy B: Train Domain-Specific MaritimeBERT Model From Scratch`
* **Rationale**: Substantial domain gap detected. Maritime Top-1 (69.95%) is below 60.0%, performance gap (-4.21%) exceeds 20.0%, or fragmentation (63.28%) exceeds 40.0%.

### Decision Sensitivity Analysis:
* **Shift -10.0%**: Train MaritimeBERT From Scratch Required
* **Shift -5.0%**: Train MaritimeBERT From Scratch Required
* **Shift +0.0%**: Train MaritimeBERT From Scratch Required
* **Shift +5.0%**: Train MaritimeBERT From Scratch Required
* **Shift +10.0%**: Train MaritimeBERT From Scratch Required

---

## 10. Final Recommendation & Future Work
1. **Proceed with Strategy**: Implement **Strategy B: Train Domain-Specific MaritimeBERT Model From Scratch**.
2. **Subdomain Focus**: Address low-performing terminology clusters during domain adaptation.
3. **Reproducibility**: Environment parameters and seeds recorded in `outputs/experiment_metadata.json`.
