# Section 08: Decision Engine, Research Report & Quality Linting (Stages 17–18)

This document details Stage 17: Objective Threshold Decision Engine, Strategy Selection, Sensitivity Analysis, Publication Benchmark Report, and Stage 18: Automated Corpus Quality Linting.

---

## Stage 17: Objective Threshold Decision Engine
- **Script**: [scripts/17_decision_engine.py](file:///c:/--Files--/Programming/pipeline/scripts/17_decision_engine.py)

---

## Programmatic Decision Rules

The engine evaluates model metrics against configured decision thresholds:

```python
if top1_acc >= 85.0 and perf_gap <= 5.0 and frag_rate <= 20.0:
    # Strategy A: Continued Domain-Adaptive Pretraining (DAPT)
elif top1_acc < 60.0 or perf_gap > 20.0 or frag_rate > 40.0:
    # Strategy B: Train Domain-Specific MaritimeBERT Model From Scratch
else:
    # Strategy C: Targeted DAPT + Custom Vocabulary Extension
```

### Execution Findings & Final Strategy Selection

- **Selected Strategy**: **`Strategy B: Train Domain-Specific MaritimeBERT Model From Scratch`**
- **Decision Rationale**: High subword fragmentation on BPE models (**63.47%**) paired with a significant performance gap on general pre-trained models indicates a substantial domain gap best resolved by scratch pretraining.
- **Sensitivity Analysis**: Decision stability confirmed across threshold shifts of $\pm 10\%$.

---

## 10-Section Benchmark Report

Stage 17 automatically generates [outputs/benchmark_report.md](file:///c:/--Files--/Programming/pipeline/outputs/benchmark_report.md) containing:
1. Executive Summary & Strategy Recommendation
2. Corpus & Representation Analysis
3. Representation Benchmark Results
4. Tokenizer Benchmark Ranking
5. 175-Run MLM Matrix Benchmark Results
6. Statistical Significance & Effect Size Metrics
7. Computational Resource & Tokenizer Speed Profiling
8. Scoring Engine Feature Ablation Results
9. Objective Decision Engine & Sensitivity Matrix
10. Final Recommendations & Pretraining Roadmap

---

## Stage 18: Automated Corpus Quality Linting
- **Script**: [scripts/18_lint_corpus.py](file:///c:/--Files--/Programming/pipeline/scripts/18_lint_corpus.py)
- **Core Logic**: Executes regex quality linting across all 96,714 clean documents:
  - `repeated_adjacent_words`: `\b([a-zA-Z]{3,})\s+\1\b`
  - `malformed_singular_plural`: `\b1\s+(?:persons|injuries|fatalities|deaths)\b`
  - `administrative_leakage`: `(?i)(?:formerly\s*occno|extraction\s+status)`
  - `awkward_phrasing`: `(?i)(?:sustained\s+damaged|damaged\s+damage)`
  - `duplicated_list_items`: `\b([a-zA-Z\s]+),\s+\1\b`
- **Output Status**: `PASS` (Violation rate: `0.044%` < `0.50%` threshold)

---

## Output Artifacts
- [outputs/experiment_metadata.json](file:///c:/--Files--/Programming/pipeline/outputs/experiment_metadata.json)
- [outputs/decision_summary.json](file:///c:/--Files--/Programming/pipeline/outputs/decision_summary.json)
- [outputs/benchmark_report.md](file:///c:/--Files--/Programming/pipeline/outputs/benchmark_report.md)
- [outputs/corpus_lint_report.json](file:///c:/--Files--/Programming/pipeline/outputs/corpus_lint_report.json)
