# HANDOFF: Universal Domain-Agnostic Benchmark Core (Stages 11–18)

## PROJECT STATE
Complete and Hardened. Stages 11–18 constitute a fully domain-independent, deterministic, reproducible benchmarking suite for evaluating text representations and language models on arbitrary text corpora. The canonical interface boundary is a clean plain-text corpus (`*_corpus.txt`), separated by double newlines (`\n\n+`). Stages 11–18 have zero dependencies on upstream corpus construction (`clean_documents.jsonl`), relational databases, or hardcoded maritime schemas.

## FINAL UNIVERSALITY & RESEARCH-GRADE HARDENING

### Core Architectural Guarantees
1. **TXT Interface Contract**: Stages 11–18 consume only plain text from `corpus_file` (documents delimited by `\n\n+`), assigning deterministic sequential identifiers (`doc_id`). `clean_documents.jsonl` is strictly isolated as an upstream provenance artifact.
2. **Zero Domain-Specific Hardcoding**:
   - Stage 11: Removed domain measurement units (`GT`, `knots`, `tonnes`, `fatalities`, etc.) and replaced with generic quantity extraction (`20 mg`, `15%`, `120 ms`, `500 kg`, `$500`, `€200`, `3.5 GHz`, `20km`, `120ms`) with optional config units.
   - Stage 11: Replaced event/operational ontology with neutral template structures (`Record`, `Subject`, `Context`, `Details`, `Statement`). Removed invented fallbacks (`general operational context` replaced with `Not specified`).
   - Removed `occurrence_id` from all output schemas across Stages 11, 12, 17, and 18; only `doc_id` is generated.
   - Token count in Stage 11 structured output explicitly named `whitespace_token_count` to distinguish from model-token counts.
   - Stage 12: Core semantic importance scoring operates on generic lexical/syntactic heuristics and works identically with empty domain lexicons.
   - Stage 13: Vocabulary analysis handles arbitrary tokenizer families (WordPiece, BPE, Byte-Level BPE) with documented OOV semantic differences.
   - Stage 14: Uses span-aware character matching on actual document text mapped via Fast Tokenizer `offset_mapping`, completely preventing subword contamination (e.g. `dynamic` is never marked as `hemodynamic`).
   - Stage 14: Never fabricates a baseline. When general baseline is unavailable, `baseline_available = False` and `domain_shift_gap = None`.
   - Stage 15: Per-document timing divisor uses actual `evaluated_doc_count` (never assumes `/200`) and reports `evaluation_time_per_document_ms` (not bare inference latency). Dynamic category scoring has zero hardcoded legacy categories.
   - Stage 16: Statistical comparisons strictly align on `(representation, subset)` keys ($N=25$ paired experimental cells controlling for underlying text, not 25 independent document draws). Feature ablation is documented as empirical leave-one-feature-out marginal contribution without retraining.
   - Stage 17: Operates using configurable heuristic decision thresholds, safely handling `None` and `NaN` metrics without fabricating decisions.
   - Stage 18: Core lint rules are generic linguistic heuristics; domain-specific artifact patterns come strictly from configuration.

### Representation Methodology & Overhead
The five representations (Narrative, Key-Value, Template, Structured Semantic, Mixed) originate from the identical underlying document. The experiment evaluates the effect of surface format while controlling for underlying semantic content. The Mixed representation deliberately pairs structured headers with narrative body, introducing intentional textual overhead and higher whitespace token counts.

### Known Methodological Limitations
1. **Paired Experimental Unit Scope**: The 25 units in Stage 16 represent paired experimental configuration cells across 5 representations and 5 score-partitioned subsets. They are not independent random samples of documents.
2. **Subset Overlap**: High, Medium, and Low subsets are score distribution partitions; Balanced intentionally samples from each tier, and Random is drawn uniformly without replacement. Subsets overlap and are tracked in `subset_overlap_statistics.json`.
3. **Lexical/Pattern Extraction**: Stage 11 entity extraction is deterministic rule-based pattern matching (quotes, acronyms, proper nouns), not a trained statistical Named Entity Recognition (NER) model.
4. **Tokenizer OOV Incomparability**: Byte-level BPE tokenizers (RoBERTa, ModernBERT) represent any byte sequence and thus report 0.0% OOV, whereas WordPiece tokenizers emit `[UNK]`. OOV rates cannot be compared across different tokenizer families.
5. **Observed Throughput Context**: Throughput metrics represent observed pipeline processing speed under the local hardware/software benchmarking environment.
6. **Heuristic Decision Engine**: Stage 17 rules triage architectures using user-configurable heuristic tolerance thresholds rather than objectively validated causal boundaries.

## TEST SUITE STATUS
- `tests/test_universal_benchmark.py`: **34/34 PASSED** (100% pass rate covering all 34 required test areas).
- `tests/test_quality_optimizations.py`: **7/7 PASSED**.
- Static AST & token dependency audit: **ZERO violations** across `scripts/11*` through `scripts/18*`.
- Semantic domain-hardcoding audit: **ZERO violations** across `scripts/11*` through `scripts/18*`.

## HOW TO RUN ON ANY DOMAIN CORPUS
1. Place plain text corpus into `outputs/<domain>_corpus.txt` (documents separated by `\n\n`).
2. Update `config/config.json`:
   ```json
   "benchmark": {
     "domain_name": "<domain>",
     "corpus_file": "<domain>_corpus.txt",
     "vocabulary_file": "<domain>_vocabulary.txt",
     "categories": { ... },
     "rare_domain_terms": [ ... ]
   }
   ```
3. Run Stages 11 through 18 sequentially. No Python source modifications are required.
