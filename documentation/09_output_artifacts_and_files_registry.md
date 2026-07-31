# Section 09: Output Artifacts & Files Registry

This document provides a comprehensive reference of every generated file in the `outputs/` directory.

---

## Output Files Catalog

| Output File Path | File Description | Key Structure / Schema | Downstream Usage |
| :--- | :--- | :--- | :--- |
| `outputs/dictionary_metadata.json` | Data dictionary column specs & enum translations | Dict mapping table names to column specifications | Input for Stage 04 attribute selection |
| `outputs/profiling_report.json` | Raw CSV table statistics & missingness profiling | JSON object with row counts, missing %, data types | Input for Stage 03 schema discovery |
| `outputs/relationships.json` | Foreign key schema relationship graph | Parent-child join keys and cardinalities | Input for Stage 04 & Stage 05 merging |
| `outputs/selected_semantic_columns.json` | Descriptive column selection metadata | Selected semantic attributes per table | Input for Stage 05 table merging |
| `outputs/merged_records.jsonl` | Nested relational occurrence JSONL (346 MB) | Nested JSON records grouped by `OccID` | Input for Stage 05a, 06, and 11 |
| `outputs/validation_report.json` | Data integrity validation report | Key check counts, error counts, orphan count | Input for Stage 09 corpus reporting |
| `outputs/raw_documents.jsonl` | Template-generated natural text documents (891 MB) | `{"occurrence_id": occ_id, "document": text}` | Input for Stage 07 cleaning |
| `outputs/clean_documents.jsonl` | Cleaned & normalized text documents (807 MB) | `{"occurrence_id": occ_id, "document": text}` | Input for Stage 08, 09, 10, 12, 13, 18 |
| `outputs/maritime_corpus.txt` | Plain text line-by-line corpus (21 MB) | One clean document string per line | Pretraining text dataset export |
| `outputs/maritime_corpus.jsonl` | Final corpus export in JSONL (796 MB) | `{"occurrence_id": occ_id, "text": doc}` | Corpus distribution format |
| `outputs/manifest.json` | Checksums & manifest for corpus distribution | SHA-256 hashes, file sizes, document count | Dataset publication verification |
| `outputs/statistics.json` | Corpus token, vocabulary, & sentence statistics | Token count, vocabulary size, TTR, sentence lengths | Input for Stage 09 Markdown report |
| `outputs/corpus_quality_report.md` | Executive Markdown summary of corpus stats | Formatted tables & statistics summary | Corpus documentation report |
| `outputs/maritime_vocabulary.txt` | Top domain-specific maritime terms (TF-IDF) | List of extracted domain keywords | Input for Stage 12 and Stage 13 |
| `outputs/corpus_representations/*.jsonl` | 5 multi-format corpus representations | JSONL files (`narrative`, `key_value`, etc.) | Input for Stage 14 MLM evaluation grid |
| `outputs/document_importance.jsonl` | 9-feature semantic importance scores (43 MB) | `{"occurrence_id": occ_id, "importance_score": s}` | Input for Stage 12 subset extraction |
| `outputs/importance_statistics.json` | Score distribution quartiles & tier counts | Mean, median, std, min, max, quartiles | Scoring engine analytics report |
| `outputs/importance_distribution.png` | Histogram plot of document importance scores | High-res PNG plot image | Visualization for research reports |
| `outputs/subsets/*.jsonl` | 6 knowledge-classified evaluation subsets | JSONL subset files (`high`, `med`, `low`, etc.) | Input for Stage 14 MLM evaluation grid |
| `outputs/tokenizer_analysis/tokenizer_comparison.csv` | Benchmarked tokenizer metrics across models | CSV table (fertility, coverage %, fragmentation %) | Input for Stage 15 benchmarking |
| `outputs/tokenizer_analysis/*.json` | Per-model detailed tokenizer analysis reports | JSON report containing tokenization splits | Tokenizer research reports |
| `outputs/evaluations/cache/*.json` | 175-run MLM matrix evaluation cached outputs | Individual JSON evaluation cache records | Input for Stage 15 cross-model benchmarking |
| `outputs/comparison.csv` | Full 175-run MLM evaluation matrix results | Detailed metrics per run (model, rep, subset) | Input for Stage 15 & 16 statistical analysis |
| `outputs/leaderboard.csv` | Model leaderboard ranked by MUI Composite Score | Aggregated metrics, MUI score, latency, CIs | Input for Stage 17 Decision Engine |
| `outputs/visualizations/*.png` | 4 publication-grade benchmark plots | PNG plots (`mlm_loss`, `ranks`, `radar`, `heatmap`) | Benchmark report figures |
| `outputs/statistical_significance.json` | Bootstrap CIs, t-test, Wilcoxon, Cohen's d, Cliff's delta | Statistical test results and effect size metrics | Input for Stage 17 Decision Engine |
| `outputs/ablation_study.json` | Scoring engine feature ablation impact | Performance drop % per removed feature | Input for Stage 17 Benchmark Report |
| `outputs/experiment_metadata.json` | Hardware, software, & seed environment params | System specs, PyTorch/Transformers versions | Reproducibility metadata |
| `outputs/decision_summary.json` | Objective decision engine strategy selection | Selected strategy, rationale, threshold values | Decision output |
| `outputs/benchmark_report.md` | 10-Section publication-grade benchmark report | Comprehensive research report in Markdown | Master research report |
| `outputs/corpus_lint_report.json` | Quality regex linting results (`PASS`/`WARN`) | Violation counts, percentages, and snippet samples | Quality assurance report |
