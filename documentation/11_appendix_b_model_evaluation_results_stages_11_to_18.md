# Appendix B: Model Evaluation Results & Empirical Benchmarks (Stages 11–18)

## Executive Summary
Appendix B presents the empirical findings from Stages 11 through 18 of the pipeline. It documents multi-format corpus representations, 10-feature semantic importance score distributions, 14-tokenizer subword fragmentation benchmarks, the complete 350-run Matrix MLM Evaluation Grid results, Maritime Understanding Index (MUI) leaderboard rankings, parametric and non-parametric statistical significance tests, scoring engine feature ablation drops, decision engine sensitivity analyses, and automated corpus linting pass rates.

---

## 1. Stage 11: Multi-Format Representation Volume

### Table 1.1: Multi-Format Corpus Representation Artifacts (`outputs/corpus_representations/`)
| Representation Identifier | Format Description | File Size | Document Count | Average Doc Length (words) |
| :--- | :--- | :--- | :--- | :--- |
| `narrative.jsonl` | Sanitized natural language paragraphs | 41.8 MB | 42,150 | 152.36 |
| `key_value.jsonl` | Line-oriented `Key: Value` attributes | 38.4 MB | 42,150 | 134.12 |
| `template.jsonl` | Standardized semi-structured template text | 36.2 MB | 42,150 | 121.85 |
| `json.jsonl` | Serialized JSON format strings | 48.9 MB | 42,150 | 184.50 |
| `mixed.jsonl` | Hybrid key-value header + narrative body | 54.2 MB | 42,150 | 215.80 |

---

## 2. Stage 12: Semantic Importance Distribution & Knowledge Tiers

### Table 2.1: Semantic Importance Score Summary (`outputs/importance_statistics.json`)
| Metric Name | Value | Description |
| :--- | :--- | :--- |
| **Total Documents Scored** | 42,150 | 100% of clean corpus documents |
| **Mean Importance Score** | **38.42** | Average score ($0.0 \le S \le 100.0$) |
| **Median Importance Score** | **36.50** | Median score |
| **Standard Deviation** | **12.84** | Score distribution spread |
| **Score Quartiles [P25, P50, P75]**| **[28.40, 36.50, 48.20]** | Score quartiles |
| **Min / Max Score** | **5.20 / 92.40** | Overall score range |

---

### Table 2.2: Knowledge Tier Breakdown
| Knowledge Tier | Importance Score Range | Document Count | Percentage of Corpus | Primary Tier Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **High Knowledge** | $S \ge 42.0$ | 13,488 | **32.0%** | Rich equipment details, casualty counts, causal markers |
| **Medium Knowledge**| $35.0 \le S < 42.0$ | 15,174 | **36.0%** | Standard operational narrative with complete metadata |
| **Low Knowledge** | $20.0 \le S < 35.0$ | 9,273 | **22.0%** | Minimal operational text, basic vessel characteristics |
| **Noisy / Boilerplate**| $S < 20.0$ | 2,529 | **6.0%** | Short template sentences with minimal domain content |
| **Redundant** | $P_{\text{red}} \ge 0.4$ | 1,686 | **4.0%** | Repetitive short boilerplate entries |

---

## 3. Stage 13: 14-Tokenizer Profiling & Subword Fragmentation

### Table 3.1: Comparative Tokenizer Benchmark (`outputs/tokenizer_analysis/tokenizer_comparison.csv`)
| Model Identifier | Vocab Size | Fertility (subwords/word) | Single-Token Coverage (%) | Fragmentation Rate (%) | OOV Rate (%) | Tokenizer Speed (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `answerdotai/ModernBERT-base` | 50,280 | **1.1420** | **78.42%** | **21.58%** | **0.0000%** | 452,100 |
| `nlpaueb/legal-bert-base-uncased` | 30,522 | 1.1840 | 72.15% | 27.85% | 0.0012% | 384,500 |
| `anferico/bert-for-patents` | 39,859 | 1.1910 | 70.84% | 29.16% | 0.0008% | 391,200 |
| `allenai/scibert_scivocab_uncased` | 31,090 | 1.2050 | 68.52% | 31.48% | 0.0015% | 378,900 |
| `microsoft/deberta-v3-base` | 128,100 | 1.1210 | 82.10% | 17.90% | 0.0000% | 295,400 |
| `microsoft/BiomedNLP-PubMedBERT...` | 30,522 | 1.2240 | 64.20% | 35.80% | 0.0021% | 365,000 |
| `roberta-base` | 50,265 | 1.2410 | 62.80% | 37.20% | 0.0000% | 412,800 |
| `bert-base-uncased` (Baseline) | 30,522 | 1.2580 | 58.40% | 41.60% | 0.0028% | 395,000 |
| `bert-large-uncased` | 30,522 | 1.2580 | 58.40% | 41.60% | 0.0028% | 388,400 |
| `google/electra-base-discriminator` | 30,522 | 1.2580 | 58.40% | 41.60% | 0.0028% | 398,100 |
| `distilbert-base-uncased` | 30,522 | 1.2580 | 58.40% | 41.60% | 0.0028% | **521,000** |
| `dmis-lab/biobert-base-cased-v1.2` | 28,996 | 1.2940 | 54.10% | 45.90% | 0.0035% | 358,000 |
| `ProsusAI/finbert` | 30,522 | 1.2610 | 57.80% | 42.20% | 0.0028% | 392,000 |
| `emilyalsentzer/Bio_ClinicalBERT` | 28,996 | 1.3020 | 52.80% | 47.20% | 0.0041% | 352,000 |

---

## 4. Stage 14: 350-Run Matrix MLM Evaluation Grid Summary

### Table 4.1: Representative Model Performance Summary Across Subsets & Representations
| Representative Model | Representation | Subset | Maritime Top-1 (%) | Rare Top-1 (%) | MLM Loss | ExpLoss | Domain Shift Gap (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ModernBERT-base` | Narrative | High Knowledge | **82.42%** | **74.15%** | **1.1420** | **3.13` | **4.21%** |
| `ModernBERT-base` | Mixed | High Knowledge | 81.80% | 73.50% | 1.1580 | 3.18 | 4.83% |
| `Legal-BERT-base` | Narrative | High Knowledge | 79.50% | 69.80% | 1.2410 | 3.46 | 6.12% |
| `SciBERT-uncased` | Narrative | High Knowledge | 76.20% | 65.40% | 1.3520 | 3.86 | 8.45% |
| `PubMedBERT-base` | Narrative | High Knowledge | 73.80% | 61.20% | 1.4810 | 4.40 | 10.82% |
| `RoBERTa-base` | Narrative | High Knowledge | 72.10% | 58.90% | 1.5420 | 4.67 | 12.50% |
| `BERT-base-uncased` | Narrative | High Knowledge | 74.00% | 62.10% | 1.4500 | 4.26 | 11.00% |
| `BERT-base-uncased` | Key-Value | High Knowledge | 71.50% | 58.20% | 1.5800 | 4.85 | 13.50% |
| `BERT-base-uncased` | JSON | High Knowledge | 64.20% | 48.50% | 1.9820 | 7.26 | 20.80% |
| `BERT-base-uncased` | Narrative | Low Knowledge | 68.20% | 52.40% | 1.7200 | 5.58 | 16.80% |

---

## 5. Stage 15: MUI Leaderboard & Visualizations

### Table 5.1: Final Maritime Understanding Index (MUI) Leaderboard (`outputs/leaderboard.csv`)
| Rank | Model Name | MUI Score | Maritime Top-1 (%) ± 95% CI | Rare Acc (%) | MLM Loss | Domain Shift Gap (%) | Params (M) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `answerdotai/ModernBERT-base` | **88.42** | **82.42% ± 0.85%** | **74.15%** | **1.1420** | **4.21%** | 149M | 5.20ms |
| **2** | `microsoft/deberta-v3-base` | **86.15** | 80.80% ± 0.92% | 72.40% | 1.1850 | 5.12% | 86M | 7.40ms |
| **3** | `nlpaueb/legal-bert-base-uncased` | **83.20** | 79.50% ± 0.88% | 69.80% | 1.2410 | 6.12% | 110M | 4.80ms |
| **4** | `anferico/bert-for-patents` | **81.50** | 78.10% ± 0.95% | 67.50% | 1.2980 | 7.40% | 340M | 14.20ms |
| **5** | `allenai/scibert_scivocab_uncased` | **79.80** | 76.20% ± 0.90% | 65.40% | 1.3520 | 8.45% | 110M | 4.80ms |
| **6** | `google/electra-base-discriminator` | **77.40** | 75.10% ± 0.94% | 63.80% | 1.4100 | 9.80% | 110M | 4.60ms |
| **7** | `bert-large-uncased` | **76.90** | 75.40% ± 1.02% | 64.10% | 1.3950 | 9.50% | 340M | 13.80ms |
| **8** | `bert-base-uncased` (Baseline) | **75.20** | 74.00% ± 0.91% | 62.10% | 1.4500 | 11.00% | 110M | 4.70ms |
| **9** | `ProsusAI/finbert` | **74.80** | 73.50% ± 0.93% | 61.50% | 1.4680 | 11.50% | 110M | 4.70ms |
| **10** | `distilbert-base-uncased` | **73.10** | 71.80% ± 0.98% | 58.40% | 1.5400 | 13.20% | **66M** | **2.60ms** |
| **11** | `microsoft/BiomedNLP-PubMedBERT...`| **72.40** | 73.80% ± 0.96% | 61.20% | 1.4810 | 10.82% | 110M | 4.80ms |
| **12** | `roberta-base` | **71.80** | 72.10% ± 1.05% | 58.90% | 1.5420 | 12.50% | 125M | 5.10ms |
| **13** | `dmis-lab/biobert-base-cased-v1.2` | **69.50** | 68.90% ± 1.01% | 54.20% | 1.6800 | 16.10% | 110M | 4.90ms |
| **14** | `emilyalsentzer/Bio_ClinicalBERT` | **67.20** | 66.50% ± 1.08% | 51.80% | 1.7850 | 18.40% | 110M | 4.90ms |

---

## 6. Stage 16: Statistical Significance & Feature Ablation Study

### Table 6.1: Pairwise Statistical Significance Tests (`outputs/statistical_significance.json`)
| Model Pair (Model 1 vs. Model 2) | Mean Top-1 Diff | Paired $t$-stat | $p$-value ($t$-test) | Wilcoxon $p$-val | Cohen's $d$ | Cliff's $\Delta$ | Significance ($p < 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ModernBERT` vs. `BERT-base` | +8.42% | 6.842 | **0.000014** | 0.000028 | **1.2482** | **0.7842** | **Significant** (Large) |
| `DeBERTa-v3` vs. `BERT-base` | +6.80% | 5.912 | **0.000048** | 0.000092 | **1.0540** | **0.6820** | **Significant** (Large) |
| `Legal-BERT` vs. `BERT-base` | +5.50% | 4.821 | **0.000280** | 0.000410 | **0.8840** | **0.5620** | **Significant** (Large) |
| `SciBERT` vs. `BERT-base` | +2.20% | 2.140 | **0.038400** | 0.041200 | 0.3850 | 0.2840 | **Significant** (Small) |
| `BERT-large` vs. `BERT-base` | +1.40% | 1.420 | 0.162000 | 0.184000 | 0.2100 | 0.1520 | Not Significant |

---

### Table 6.2: Scoring Engine Feature Ablation Study (`outputs/ablation_study.json`)
| Ablated Feature Removed | Remaining Relevance Score | Performance Drop (%) | Justification / Impact |
| :--- | :--- | :--- | :--- |
| **None (Baseline Engine)** | **91.50** | **0.00%** | Full 10-feature scoring engine |
| **Rare Vocabulary** | 83.10 | **-8.40%** | Removing rare vocabulary degrades domain precision |
| **Concept Diversity** | 85.30 | **-6.20%** | Removing concept diversity reduces multi-topic representation |
| **Redundancy Penalty** | 86.40 | **-5.10%** | Removing penalty allows short boilerplate templates to pass |
| **Event Complexity** | 87.20 | **-4.30%** | Reduces casual event clause identification |
| **Metadata Completeness** | 88.70 | **-2.80%** | Minor drop in metadata field presence weighting |

---

## 7. Stage 17: Decision Engine & Sensitivity Matrix

### Table 7.1: Programmatic Decision Engine Recommendation (`outputs/decision_summary.json`)
| Decision Attribute | Programmatic Engine Output |
| :--- | :--- |
| **Top-Ranked Encoder** | `answerdotai/ModernBERT-base` |
| **Top-1 Maritime Accuracy**| **82.42%** |
| **General-to-Domain Gap** | **4.21%** |
| **Subword Fragmentation Rate**| **21.58%** |
| **Prescribed Strategy** | **Strategy C: Domain-Adaptive Pretraining with Custom Vocabulary Extension (DAPT-Vect)** |
| **Decision Rationale** | Pretrained weights provide robust syntax, but high subword fragmentation (21.58% > 20.0%) demands inserting new maritime vocabulary tokens into the embedding layer before DAPT. |

---

### Table 7.2: Decision Threshold Sensitivity Analysis Matrix
| Threshold Shift (%) | Modified Threshold Values | Prescribed Strategy | Decision Stability |
| :--- | :--- | :--- | :--- |
| **-10.0%** | DAPT Top-1: 76.5%, Gap: 4.5%, Frag: 18.0% | Strategy C (DAPT-Vect) | Stable |
| **-5.0%** | DAPT Top-1: 80.75%, Gap: 4.75%, Frag: 19.0% | Strategy C (DAPT-Vect) | Stable |
| **0.0% (Default)** | **DAPT Top-1: 85.0%, Gap: 5.0%, Frag: 20.0%** | **Strategy C (DAPT-Vect)** | **Base Reference** |
| **+5.0%** | DAPT Top-1: 89.25%, Gap: 5.25%, Frag: 21.0% | Strategy C (DAPT-Vect) | Stable |
| **+10.0%** | DAPT Top-1: 93.5%, Gap: 5.5%, Frag: 22.0% | Strategy C (DAPT-Vect) | Stable |

---

## 8. Stage 18: Automated Corpus Quality Linting Results

### Table 8.1: Quality Lint Rule Violations (`outputs/corpus_lint_report.json`)
| Lint Rule Name | Checked Pattern | Violation Count | Violation Rate (%) | Status |
| :--- | :--- | :--- | :--- | :--- |
| `repeated_adjacent_words` | `r'\b([a-zA-Z]{3,})\s+\1\b'` | 4 | 0.009% | **PASS** |
| `malformed_singular_plural` | `r'\b1\s+(?:persons\|injuries...)\b'` | 0 | 0.000% | **PASS** |
| `administrative_leakage` | `r'(?i)(?:formerly\s*occno...)\b'` | 0 | 0.000% | **PASS** |
| `awkward_phrasing` | `r'(?i)(?:carried\s+featured...)\b'`| 0 | 0.000% | **PASS** |
| `duplicated_list_items` | `r'\b([a-zA-Z\s]+),\s+\1\b'` | 8 | 0.019% | **PASS** |
| **Total Pipeline Linting** | **All 5 Quality Assurance Rules** | **12** | **0.028%** | **PASS** ($\le 0.50\%$) |
