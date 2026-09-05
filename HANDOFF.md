# HANDOFF: Universal Benchmark Refactor (Stages 11–18)

## PROJECT STATE
Complete. The benchmarking pipeline from Stage 11 onward has been universalized. The canonical interface contract is a cleaned plain-text corpus (`*_corpus.txt`), separated by double newlines (`\n\n`). Stages 11–18 have zero dependencies on `clean_documents.jsonl`, MARSIS schemas, `OccID`, `VesselID`, or hardcoded domain fields. All tests and static dependency audits pass with zero violations.

## LAST COMPLETED STAGE
Robustness & Research-Grade Hardening Pass (Stages 11–18):
- Span-aware domain term detection implemented in Stage 14 (character spans mapped via Fast Tokenizer offset mappings), preventing global subword contamination.
- Fake general-English 0.85 fallback eliminated in Stage 14 (explicitly reports `baseline_available=False`, `domain_shift_gap=None` when missing).
- Hidden domain lexicons removed from Stage 12 generic core; optional domain profiles externalized to `config.json`.
- Domain lint patterns externalized to `config.json`; Stage 18 generic core operates cleanly with empty pattern lists.
- Latency metric accurately renamed to `evaluation_time_per_document_ms` in Stage 15/17.
- Silent auto-discovery replaced with deterministic `FileNotFoundError` unless explicitly configured.
- Explicit duplicate detection vs. deduplication semantics implemented in `load_corpus_documents()`.
- Decoupled `information_density` (content word ratio) from `domain_density` in Stage 12, preventing double-counting.
- Statistical unit strictly documented as $N = 25$ paired experimental cells.
- Real empirical leave-one-feature-out contribution clarified in Stage 16.

## CURRENT TASK
Robustness hardening complete. All 22 regression and unit tests pass.

## FILES CHANGED
- `config/config.json`: Added `domain_semantic_lexicons`, `domain_lint_patterns`, discovery and deduplication flags.
- `scripts/pipeline_utils.py`: Deterministic corpus loading, explicit duplicate flags, strict missing-file errors.
- `scripts/11_corpus_representations.py`: Strict corpus resolution without silent fallback.
- `scripts/12_semantic_importance.py`: Decoupled information density, generic linguistic heuristics, subset overlap tracking.
- `scripts/13_tokenizer_analysis.py`: Deterministic corpus loading without silent fallback.
- `scripts/14_mlm_evaluation.py`: Span-aware domain token identification via offset mappings; removed fake 0.85 fallback.
- `scripts/15_cross_model_benchmarking.py`: Renamed latency to `evaluation_time_per_document_ms`, isolated legacy compatibility.
- `scripts/16_statistical_analysis.py`: Statistical unit phrased as $N=25$ paired cells; clarified empirical ablation contribution.
- `scripts/17_decision_engine.py`: Updated benchmark report table with accurate timing terminology (`Eval Time (ms/doc)`).
- `scripts/18_lint_corpus.py`: Generic core rules separated from configurable domain lint patterns.
- `documentation/15_universal_benchmark_refactor.md`: Added `ROBUSTNESS HARDENING` section.

## FILES CREATED
- `HANDOFF.md`: Repository root handoff file.
- `tests/domain_test_corpus.txt`: Synthetic non-maritime corpus.
- `tests/test_universal_benchmark.py`: Comprehensive test suite (22 unit & regression tests + static AST audit).
- `documentation/15_universal_benchmark_refactor.md`: Architecture and refactor reference document.

## TESTS PASSED
- `tests/test_universal_benchmark.py`: 22/22 PASSED (including 10 new regression tests for subword span contamination, missing baseline, empty lexicons, non-maritime scoring, non-maritime linting, missing corpus error, dynamic category scoring, duplicate semantics, density independence, deterministic sampling, and static AST dependency audit).
- `tests/test_quality_optimizations.py`: 7/7 PASSED.
- `tests/verify_pipeline.py`: ALL 25 REQUIRED OUTPUTS VERIFIED.
- Static AST & token dependency audit: ZERO violations across `scripts/11*` through `scripts/18*`.

## KNOWN FAILURES
None.

## NEXT ACTIONS
- Benchmark is hardened and ready for final runs or execution on any external domain corpus.

## IMPORTANT DESIGN DECISIONS
- The canonical interface is plain-text `*_corpus.txt` with `\n\n` document boundary delimiters.
- Document identifier is `doc_id` with `occurrence_id` retained as a backwards-compatible alias.
- Stage 14 masks and evaluates actual domain-term character spans mapped to token offsets, never global token IDs.
- Stage 16 pairs observations strictly on `(representation, subset)` keys ($N=25$ paired cells per model comparison).
- Feature ablation in Stage 16 is calculated empirically over document feature vectors.
- Decision engine thresholds in `config.json` support both decimal (0.85) and percentage (85.0) representations.

## DO NOT CHANGE
- `dapt/` directory (untouched).
- `maritimebert_validation.ipynb` (untouched).
- Stages 01–10 (domain-specific corpus construction).
- `clean_documents.jsonl` (preserved for upstream provenance, but ignored by Stages 11–18).
