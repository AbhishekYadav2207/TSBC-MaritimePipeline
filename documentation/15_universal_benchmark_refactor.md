# Universal Benchmark Refactor (Stages 11–18)

## A. What Changed
1. **Canonical Benchmark Interface Boundary**: Established plain-text `*_corpus.txt` (separated by double newlines `\n\n`) as the sole formal interface between domain corpus construction (Stages 01–10) and universal benchmarking (Stages 11–18).
2. **Decoupled Stages 11–18 from `clean_documents.jsonl`**: Removed all references, imports, and file accesses to `clean_documents.jsonl` and database schemas from Stages 11–18.
3. **Stage 11 (Representations)**: Refactored to consume the plain-text corpus directly. All 5 representations (Narrative, Key-Value, Template, Structured Semantic / JSON, Mixed) are now extracted deterministically from document text. Replaced process-randomized `hash()` with stable SHA-256 hashing.
4. **Stage 12 (Semantic Importance)**: Replaced database column dependencies with text-derived metrics (entity extraction, syntactic/event complexity, structural completeness across 5 semantic facets, linguistic diversity, domain novelty).
5. **Stage 13 (Tokenizer Analysis)**: Refactored to read plain-text corpus and domain vocabulary without structured metadata.
6. **Stage 14 (MLM Evaluation)**: Generalized domain categories and vocabulary loading. Updated documentation and logs to reflect the exact 175-run evaluation matrix (7 models × 5 representations × 5 subsets).
7. **Stage 15 (Cross-Model Benchmarking)**: Generalized metric to Domain Understanding Index (DUI / MUI) with dynamic subdomain category recall.
8. **Stage 16 (Statistical Analysis & Ablation)**:
   - Fixed pairing logic: strictly aligned pairwise comparisons on `(representation, subset)` experimental conditions ($N=25$).
   - Implemented real empirical feature ablation calculations across corpus documents, removing hardcoded drop constants.
9. **Stage 17 (Decision Engine)**: Externalized decision thresholds and domain names into `config/config.json`, supporting decimal or percentage formats.
10. **Stage 18 (Corpus Linter)**: Refactored to lint the plain-text corpus directly using document boundary splitting.

---

## B. Why It Changed
In the original implementation, Stages 11–18 directly consumed `clean_documents.jsonl` and accessed MARSIS-specific relational fields (e.g., `OccID`, `VesselID`, `GrossTonnage`, `NearestLocationDescription`, `NavigationAidTypeDisplayEng`). This tightly coupled the benchmarking suite to the Canadian maritime accident database, preventing researchers from evaluating other domains (e.g. clinical, legal, finance, materials) without modifying core benchmarking logic.

By enforcing `domain_corpus.txt` as the formal interface, the benchmarking suite (Stages 11–18) becomes 100% domain-agnostic and universally reusable on any clean `.txt` corpus.

---

## C. Before Architecture
```text
Stages 01–06: Table ingestion, joining, text generation
     ↓
Stage 07: clean_documents.py → clean_documents.jsonl (with MARSIS structured fields)
     ↓
     ├──────────────────────────────────────┬───────────────────────────────────┐
     ↓                                      ↓                                   ↓
Stage 08 (maritime_corpus.txt)   Stage 11 (clean_documents.jsonl)   Stage 12 (clean_documents.jsonl)
                                            ↓                                   ↓
                                 Stages 13–18 coupled to MARSIS schemas & OccID
```

---

## D. After Architecture
```text
Stages 01–10: Domain-Specific Corpus Construction
Raw Domain Tables / Data
     ↓
clean_documents.jsonl (Preserved as upstream provenance artifact)
     ↓
maritime_corpus.txt / domain_corpus.txt (Canonical Boundary)
     ↓
========================================================================
UNIVERSAL BENCHMARK CORE (Stages 11–18)
Config: config/config.json (domain_name, categories, thresholds)
     ↓
Stage 11: Multi-Format Representations (Narrative, Key-Value, Template, Structured Semantic, Mixed)
Stage 12: Text-Derived Semantic Importance Scoring & Knowledge Classification
Stage 13: Domain-Agnostic Tokenizer Analysis
Stage 14: MLM Evaluation Matrix (175 runs: 7 models × 5 reps × 5 subsets)
Stage 15: Cross-Model Benchmarking & Domain Understanding Index (DUI/MUI)
Stage 16: Paired Statistical Significance (Aligned Cells) & Real Feature Ablation
Stage 17: Objective Threshold Decision Engine & Publication Report
Stage 18: Plain-Text Corpus Quality Linting
========================================================================
```

---

## E. Files Modified

| Path | Change | Reason | Status |
| :--- | :--- | :--- | :--- |
| `config/config.json` | Added `benchmark` and `decision` sections | Configurable domain parameters, categories, and decision thresholds | Completed |
| `scripts/pipeline_utils.py` | Added `load_corpus_documents()` and `get_benchmark_config()` | Canonical text loader and benchmark config loader | Completed |
| `scripts/11_corpus_representations.py` | Consumes `.txt` corpus; deterministic text extraction; SHA-256 template hashing | Decouple from JSONL and structured database records | Completed |
| `scripts/12_semantic_importance.py` | Text-derived feature calculations (entity diversity, event complexity, structural completeness) | Eliminate dependency on MARSIS fields | Completed |
| `scripts/13_tokenizer_analysis.py` | Reads `.txt` corpus directly; domain-agnostic metric names | Decouple from `clean_documents.jsonl` | Completed |
| `scripts/14_mlm_evaluation.py` | Configurable categories; doc matching by `doc_id`; corrected matrix size (175 runs) | Domain-agnostic evaluation grid | Completed |
| `scripts/15_cross_model_benchmarking.py` | Configurable DUI/MUI metrics; dynamic subdomain recall | Decouple from hardcoded maritime categories | Completed |
| `scripts/16_statistical_analysis.py` | Strictly aligned pairing on `(representation, subset)`; real empirical feature ablation | Fix statistical pairing bug and eliminate fake hardcoded ablation numbers | Completed |
| `scripts/17_decision_engine.py` | Configurable thresholds from config; dynamic domain names; corrected 175-run text | Parameterized decision engine | Completed |
| `scripts/18_lint_corpus.py` | Lints `.txt` corpus directly via `load_corpus_documents()` | Decouple from `clean_documents.jsonl` | Completed |

---

## F. Files Created

| Path | Purpose | Status |
| :--- | :--- | :--- |
| `HANDOFF.md` | Human and machine handoff instructions at repository root | Completed |
| `tests/domain_test_corpus.txt` | Synthetic non-maritime corpus (biomedical/clinical text) for domain-agnostic testing | Completed |
| `tests/test_universal_benchmark.py` | Comprehensive test suite (12 unit tests + static AST dependency audit) | Completed |
| `documentation/15_universal_benchmark_refactor.md` | Dedicated architecture and refactor reference document | Completed |

---

## G. Files Intentionally NOT Changed
1. **Stages 01–10 (`scripts/01_*.py` through `scripts/10_*.py`)**:
   - These scripts constitute the domain-specific raw data ingestion, table merging, and corpus construction. Keeping them unchanged guarantees that the Canadian MARSIS maritime dataset pipeline continues to produce valid maritime corpora without regression.
2. **`dapt/` directory**:
   - Explicitly protected by project constraints. Contains domain-adaptive pretraining code.
3. **`maritimebert_validation.ipynb`**:
   - Explicitly protected by project constraints. Contains notebook evaluation workflows.
4. **`outputs/clean_documents.jsonl`**:
   - Preserved as an upstream audit and provenance artifact from Stage 07.

---

## H. Remaining Work
- Optional: Run full 175-run MLM matrix evaluation if GPU compute is provisioned (cached results currently active for development/testing).

---

## I. Known Issues
- None. All static dependency checks and unit tests pass with zero violations.

---

## J. Test Results
1. **Unit Test Suite (`tests/test_universal_benchmark.py`)**:
   - `test_01_txt_corpus_loading`: PASSED
   - `test_02_document_boundary_detection`: PASSED
   - `test_03_empty_and_invalid_documents`: PASSED
   - `test_04_duplicate_documents`: PASSED
   - `test_05_vocabulary_extraction`: PASSED
   - `test_06_representation_generation`: PASSED
   - `test_07_semantic_scoring`: PASSED
   - `test_08_real_feature_ablation_calculation`: PASSED
   - `test_09_paired_statistical_alignment`: PASSED
   - `test_10_decision_engine_configuration`: PASSED
   - `test_11_non_maritime_corpus_pipeline_flow`: PASSED
   - `test_12_static_dependency_audit_scripts_11_to_18`: PASSED
   - **Summary**: 12/12 tests passed.
2. **Optimization Tests (`tests/test_quality_optimizations.py`)**:
   - 7/7 tests passed.
3. **Pipeline Output Verification (`tests/verify_pipeline.py`)**:
   - All 25 required artifacts verified. All schemas valid.
4. **Static AST & Token Dependency Audit**:
   - Zero occurrences of forbidden tokens (`clean_documents.jsonl`, `OccID`, `VesselID`, `MARSIS`, or relational fields) across `scripts/11*` through `scripts/18*`.

---

## K. Current Pipeline State
The repository is in a clean, working state. Another developer can execute any stage independently or provide an alternative domain corpus by:
1. Setting `"corpus_file": "your_domain_corpus.txt"` in `config/config.json`.
2. Running Stages 11 through 18 using `python run_pipeline.py --stage <XX>`.

---

## ROBUSTNESS HARDENING

### Fixes Made
1. **Span-Aware Domain Term Identification (Stage 14)**:
   - Eliminated global subword token ID contamination (e.g. `dynamic` from `hemodynamic`).
   - Implemented exact character span matching on document text (`extract_domain_spans()`) and mapped to token spans via Fast Tokenizer `offset_mapping` with a deterministic string fallback.
2. **Elimination of Fake General-English Fallback (Stage 14)**:
   - Removed all hard-coded fallback scores (such as `0.85`).
   - If the baseline is unavailable or empty, explicitly sets `baseline_available = False` and `domain_shift_gap = None` to prevent misleading metric inflation.
3. **Decoupling Hidden Domain Lexicons (Stage 12)**:
   - Removed hard-coded domain word lists (`vessel`, `ship`, `boat`, `patient`, `occurred`, `damage`, etc.) from generic core scoring.
   - Core scoring uses generic linguistic/syntactic heuristics (proper nouns, participles, causal markers, numeric metrics) plus optional domain-specific lexicons passed via `config/config.json`.
4. **Decoupling Domain Lint Rules (Stage 18)**:
   - Core linter now enforces only generic linguistic rules (adjacent duplicate words, singular/plural agreement, phrasing).
   - Domain-specific artifact patterns (e.g. `OccNo`, `marine occurrence`) moved to `domain_lint_patterns` in `config/config.json`.
5. **Accurate Latency Terminology (Stage 15 & 17)**:
   - Renamed misleading metric `inference_latency_ms` to `evaluation_time_per_document_ms` (retaining compatibility aliases).
6. **Deterministic Corpus Resolution & Safe Missing File Handling**:
   - Removed silent auto-discovery `list(glob("*_corpus.txt"))[0]`. A missing configured corpus now raises a deterministic `FileNotFoundError` unless `allow_corpus_auto_discovery: true` is explicitly configured.
7. **Explicit Duplicate Handling (`pipeline_utils.load_corpus_documents`)**:
   - Default: `deduplicate=False`, preserving all documents and assigning unique `doc_id` while flagging `is_duplicate=True`.
   - Optional: `deduplicate=True` skips exact text boundary duplicates.
8. **Decoupled Information Density vs. Domain Density (Stage 12)**:
   - Redefined `information_density` as the ratio of lexical content words (non-functional stopwords) to total tokens, eliminating double-counting of domain tokens.
9. **Clarified Rare Domain Term Novelty (Stage 12)**:
   - Documented and exposed `rare_domain_term_novelty` as IDF-weighted rare term frequency.

### Methodological Decisions
- **Statistical Unit (Stage 16)**: Explicitly defined as $N = 25$ paired experimental cells across identical `(representation, subset)` conditions (not independent document samples).
- **Ablation Interpretation (Stage 16)**: Leave-one-feature-out measures the empirical marginal score contribution of each feature vector weight across evaluated documents without model retraining.
- **Subset Overlap Tracking (Stage 12)**: Exported `outputs/subset_overlap_statistics.json` detailing sample sizes, pairwise overlap counts, and composition methodology.

### Tests Added
- `test_13_subword_domain_span_contamination`: Proves isolated subwords (e.g. `dynamic`) are not contaminated by domain terms (`hemodynamic`).
- `test_14_missing_general_english_baseline`: Confirms no fake 0.85 fallback is injected.
- `test_15_empty_domain_lexicons`: Validates that Stage 12 runs with empty domain lexicons using generic heuristics.
- `test_16_non_maritime_semantic_scoring`: Validates scoring on non-maritime (legal) documents.
- `test_17_non_maritime_linting`: Validates Stage 18 linter with empty domain patterns.
- `test_18_missing_configured_corpus_error`: Verifies deterministic `FileNotFoundError` on missing corpus.
- `test_19_dynamic_category_scoring`: Validates dynamic category balance without maritime fields.
- `test_20_duplicate_detection_semantics`: Validates explicit duplicate flagging vs. deduplication.
- `test_21_density_and_information_density_independence`: Demonstrates information density is positive even when domain density is zero.
- `test_22_stable_deterministic_sampling`: Verifies sampling repeatability.
- Total test count: 22/22 unit & regression tests passing.

### Remaining Limitations
- MLM matrix runs rely on cached checkpoint evaluations unless full GPU recomputation is explicitly requested.

---

## FINAL UNIVERSALITY & RESEARCH-GRADE HARDENING

### Architectural & Methodological Guarantees
1. **TXT Boundary**: The sole formal input to Stages 11–18 is the plain-text corpus (`*_corpus.txt`) where documents are separated by `\n\n+`. Stages 11–18 have zero dependencies on `clean_documents.jsonl` or relational schemas.
2. **Configuration-Driven Domain Semantics**: All domain concepts (domain name, corpus/vocabulary files, categories, rare domain terms, semantic lexicons, measurement units, lint patterns, decision thresholds, random seed) are governed by `config/config.json`. Core Python code contains zero hardcoded maritime, medical, or other single-domain terms.
3. **Five Representation Design**: The five representations (Narrative, Key-Value, Template, Structured Semantic, Mixed) are deterministically extracted from the same underlying text. The benchmark measures the impact of surface formatting on tokenizer fragmentation and MLM accuracy while strictly controlling for underlying semantic content.
4. **Mixed Representation Overhead**: The Mixed representation pairs an extracted structural header with the full narrative text, deliberately introducing textual overhead and resulting in higher whitespace token counts.
5. **Span-Aware Domain Matching**: Stage 14 locates character-level spans of configured domain terms in the text and maps them via Fast Tokenizer `offset_mapping`. Subtokens (e.g. `dynamic` in `dynamic routing`) are never contaminated by unreferenced domain terms (`hemodynamic`).
6. **Honest Baseline Handling**: No fake 0.85 general-English baseline is fabricated. If unavailable, `baseline_available = False` and `domain_shift_gap = None`.
7. **Accurate Evaluation Timing Definition**: Metric is `evaluation_time_per_document_ms` (not bare inference latency). Divisor is the actual `evaluated_doc_count`, and the system fails safely if document counts are missing.
8. **Paired Experimental Cell Interpretation**: The 25 units in Stage 16 represent paired experimental configuration cells across 5 representations and 5 score-partitioned subsets controlling for source content, not independent document observations.
9. **Subset Overlap Tracking**: Subsets are non-independent partitions of the corpus. Pairwise overlap counts are tracked and exported to `subset_overlap_statistics.json`.
10. **Configurable Heuristic Decision Thresholds**: Stage 17 rules triage candidate models using user-configurable heuristic tolerance thresholds rather than an objectively validated or causal boundary. Safely handles `None` and `NaN` metrics without fabricating decisions.

### Methodological Limitations
1. **Rule-Based Entity Extraction**: Stage 11 entity extraction uses deterministic lexical patterns (quotes, acronyms, capitalized sequences), not a trained statistical Named Entity Recognition (NER) model.
2. **Tokenizer OOV Semantic Discrepancy**: Byte-level BPE tokenizers represent any byte sequence (0.0% OOV), while WordPiece emits `[UNK]`. OOV rates cannot be compared across different tokenizer families.
3. **Observed Throughput**: Reported throughput reflects observed processing speed in the specific local benchmarking runtime environment.
4. **Non-Retrained Ablation**: Feature ablation reflects leave-one-feature-out marginal contribution of scoring weights over evaluated corpus vectors without retraining.


