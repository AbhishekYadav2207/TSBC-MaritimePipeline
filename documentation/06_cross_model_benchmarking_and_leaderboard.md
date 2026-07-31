# Section 06: Cross-Model Benchmarking & Leaderboard (Stage 15)

This document provides complete technical specifications for Stage 15: Cross-Model Benchmarking, Maritime Understanding Index (MUI) composite score calculation, model leaderboard ranking, and visualization output generation.

---

## Stage 15: Cross-Model Benchmarking
- **Script**: [`scripts/15_cross_model_benchmarking.py`](file:///c:/--Files--/Programming/pipeline/scripts/15_cross_model_benchmarking.py)
- **Execution Command**: `python run_pipeline.py --stage 15`

---

## 1. Mathematical Maritime Understanding Index (MUI) Score

To rank model domain capability holistically, Stage 15 computes the composite **MUI Score** (scaled $0 - 100$):

$$\text{MUI} = 100 \times \Big( 0.35\,\text{Top1} + 0.20\,\text{RareTop1} + 0.15(1-\bar{L}_{\text{norm}}) + 0.15(1-\text{Frag}) + 0.10(1-\min(1, 10\times\text{OOV})) + 0.05\,\text{Balance} \Big)$$

Where:
- $\text{Top1}$: Mean maritime Top-1 accuracy across runs.
- $\text{RareTop1}$: Mean Top-1 accuracy on rare maritime terms.
- $\bar{L}_{\text{norm}}$: Normalized MLM loss.
- $\text{Frag}$: Subword fragmentation rate.
- $\text{OOV}$: Out-of-vocabulary rate.
- $\text{Balance}$: Subdomain category accuracy balance ($1 - \sigma_{\text{categories}}$).

---

## 2. Empirical Model Leaderboard Table

| Rank | Model Name | MUI Score | Maritime Top-1 (%) | Rare Term Acc (%) | MLM Loss | Domain Shift Gap (%) | Params (M) | Throughput (docs/sec) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 **1** | `answerdotai/ModernBERT-base` | **70.34** | **73.38%** ± 6.27% | **67.61%** | **1.5004** | -17.83% | 149M | 4.14 |
| 🥈 **2** | `roberta-base` | **65.57** | 63.79% ± 8.89% | 66.13% | 2.0865 | -17.12% | 125M | 5.62 |
| 🥉 **3** | `allenai/scibert_scivocab_uncased` | **49.94** | 28.03% ± 3.90% | 53.09% | 4.5816 | 11.97% | 110M | 6.20 |
| **4** | `nlpaueb/legal-bert-base-uncased` | **49.44** | 30.69% ± 2.57% | 37.84% | 4.2138 | -4.02% | 110M | 7.27 |
| **5** | `bert-base-uncased` | **48.68** | 25.62% ± 3.30% | 42.77% | 4.9786 | 12.48% | 110M | 6.51 |
| **6** | `dmis-lab/biobert-base-cased-v1.2` | **41.94** | 21.97% ± 3.81% | 23.17% | 5.1492 | 23.86% | 110M | 7.69 |
| **7** | `microsoft/BiomedNLP-PubMedBERT...` | **36.53** | 16.35% ± 4.22% | 20.04% | 6.0874 | 7.46% | 110M | 7.51 |

---

## 3. Visualizations Generated ([`outputs/visualizations/`](file:///c:/--Files--/Programming/pipeline/outputs/visualizations))

1. **`mlm_loss_comparison.png`**: Cross-model MLM loss comparison bar chart.
2. **`model_leaderboard_ranks.png`**: Leaderboard rankings with 95% confidence interval error bars.
3. **`maritime_accuracy_radar.png`**: Polar radar chart comparing recall across 6 subdomains for top 3 models.
4. **`tokenizer_fragmentation_heatmap.png`**: Tokenizer fragmentation & vocabulary coverage heatmap.

---

## Output Artifacts
- [`outputs/comparison.csv`](file:///c:/--Files--/Programming/pipeline/outputs/comparison.csv)
- [`outputs/leaderboard.csv`](file:///c:/--Files--/Programming/pipeline/outputs/leaderboard.csv)
- High-res PNG plot artifacts under [`outputs/visualizations/`](file:///c:/--Files--/Programming/pipeline/outputs/visualizations)
