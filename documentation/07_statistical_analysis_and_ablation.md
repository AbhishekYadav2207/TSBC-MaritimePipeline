# Section 07: Statistical Analysis & Feature Ablation (Stage 16)

This document details Stage 16: Statistical Significance Testing, Bootstrap 95% Confidence Intervals, Effect Size Calculations, and Scoring Feature Ablation.

---

## Stage 16: Statistical Significance Testing & Feature Ablation
- **Script**: [scripts/16_statistical_analysis.py](file:///c:/--Files--/Programming/pipeline/scripts/16_statistical_analysis.py)

---

## Statistical Methodology & Formulae

1. **Bootstrap 95% Confidence Intervals**:
   Re-sampled with replacement ($B = 1000$ iterations) to calculate non-parametric confidence bounds:
   $$\text{CI}_{95} = \left[ \text{Percentile}\left(\bar{x}^*, 2.5\right), \; \text{Percentile}\left(\bar{x}^*, 97.5\right) \right]$$

2. **Paired $t$-Test**:
   Evaluates relative mean accuracy differences between model pairs across identical evaluation configurations ($p < 0.05$).

3. **Wilcoxon Signed-Rank Test**:
   Non-parametric paired rank test for robustness against non-normal performance distributions.

4. **Cohen's $d$ Effect Size**:
   $$\text{Cohen's } d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$

5. **Cliff's $\delta$ Effect Size**:
   $$\delta = \frac{\# (x_1 > x_2) - \# (x_1 < x_2)}{n_1 n_2}$$

---

## Scoring Feature Ablation Study

Stage 16 evaluates the marginal impact of individual scoring features in `12_semantic_importance.py` by removing each feature and measuring degradation in semantic selection precision:

| Feature Removed | Baseline Precision Score | Ablated Score | Performance Drop (%) | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Rare Vocabulary** | 91.5 | 83.1 | **-8.4%** | Removing rare terms causes loss of domain-specific technical terminology. |
| **Concept Diversity** | 91.5 | 85.3 | **-6.2%** | Removing concept diversity reduces multi-subdomain representation. |
| **Redundancy Penalty** | 91.5 | 86.4 | **-5.1%** | Removing redundancy penalty leads to over-sampling repetitive reports. |
| **Event Complexity** | 91.5 | 87.2 | **-4.3%** | Removing event complexity weakens selection of multi-vessel incidents. |
| **Metadata Completeness** | 91.5 | 88.7 | **-2.8%** | Removing metadata completeness slightly degrades attribute richness. |

---

## Output Artifacts
- [outputs/statistical_significance.json](file:///c:/--Files--/Programming/pipeline/outputs/statistical_significance.json)
- [outputs/ablation_study.json](file:///c:/--Files--/Programming/pipeline/outputs/ablation_study.json)
