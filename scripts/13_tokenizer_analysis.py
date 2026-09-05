import os
import json
import time
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from pipeline_utils import (
    setup_logging,
    load_config,
    get_project_root,
    get_benchmark_config,
    load_corpus_documents
)

logger = setup_logging("13_tokenizer_analysis")

TARGET_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base",
    "allenai/scibert_scivocab_uncased",
    "dmis-lab/biobert-base-cased-v1.2",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "emilyalsentzer/Bio_ClinicalBERT",
    "nlpaueb/legal-bert-base-uncased",
    "ProsusAI/finbert",
    "anferico/bert-for-patents",
    "google/electra-base-discriminator",
    "distilbert-base-uncased"
]

def clean_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")

def analyze_tokenizer(model_name: str, vocab_terms: list, corpus_docs: list) -> dict:
    logger.info(f"Analyzing tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer '{model_name}': {e}. Skipping.")
        return None

    # 1. Domain Vocabulary Tokenization & Piece Statistics
    vocab_splits = []
    piece_counts = []
    single_token_count = 0

    for term in vocab_terms:
        tokens = tokenizer.tokenize(term)
        num_pieces = len(tokens)
        piece_counts.append(num_pieces)
        if num_pieces == 1:
            single_token_count += 1
        vocab_splits.append({
            "term": term,
            "tokens": tokens,
            "num_pieces": num_pieces
        })

    vocab_splits.sort(key=lambda x: x["num_pieces"], reverse=True)
    total_terms = len(vocab_terms) if vocab_terms else 1

    single_token_coverage = single_token_count / total_terms
    domain_frag_rate = (total_terms - single_token_count) / total_terms
    avg_pieces = float(np.mean(piece_counts)) if piece_counts else 1.0
    median_pieces = float(np.median(piece_counts)) if piece_counts else 1.0
    p95_pieces = float(np.percentile(piece_counts, 95)) if piece_counts else 1.0
    max_pieces = int(np.max(piece_counts)) if piece_counts else 1

    # 2. Corpus Subword Fertility, OOV Rate & Tokenizer Speed Profiling
    total_raw_words = 0
    total_subword_tokens = 0
    total_unk_tokens = 0
    seq_length_dist = {"under_128": 0, "under_256": 0, "under_512": 0, "over_512": 0}

    t0 = time.time()
    for doc_text in corpus_docs:
        raw_words = len(doc_text.split())
        if raw_words == 0:
            continue
        tokens = tokenizer.tokenize(doc_text)
        num_tokens = len(tokens)
        total_raw_words += raw_words
        total_subword_tokens += num_tokens

        unk_tok = getattr(tokenizer, "unk_token", "[UNK]")
        if unk_tok and unk_tok in tokens:
            total_unk_tokens += tokens.count(unk_tok)

        if num_tokens <= 128: seq_length_dist["under_128"] += 1
        elif num_tokens <= 256: seq_length_dist["under_256"] += 1
        elif num_tokens <= 512: seq_length_dist["under_512"] += 1
        else: seq_length_dist["over_512"] += 1

    tok_elapsed = time.time() - t0
    tokenizer_speed = total_subword_tokens / tok_elapsed if tok_elapsed > 0 else 0.0
    fertility = total_subword_tokens / total_raw_words if total_raw_words > 0 else 0.0
    oov_rate = total_unk_tokens / total_subword_tokens if total_subword_tokens > 0 else 0.0

    return {
        "model_name": model_name,
        "clean_model_name": clean_model_filename(model_name),
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer),
        "sampled_documents": len(corpus_docs),
        "total_raw_words_analyzed": total_raw_words,
        "total_subword_tokens_analyzed": total_subword_tokens,
        "average_subwords_per_word": fertility,
        "domain_fragmentation_rate": domain_frag_rate,
        "maritime_fragmentation_rate": domain_frag_rate,  # Compatibility alias
        "single_token_vocabulary_coverage": single_token_coverage,
        "single_token_count": single_token_count,
        "total_domain_terms": total_terms,
        "total_maritime_terms": total_terms,  # Compatibility alias
        "avg_pieces_per_term": avg_pieces,
        "median_pieces_per_term": median_pieces,
        "p95_pieces_per_term": p95_pieces,
        "max_pieces_per_term": max_pieces,
        "oov_rate": oov_rate,
        "tokenizer_speed_tokens_per_sec": tokenizer_speed,
        "sequence_length_distribution": seq_length_dist,
        "worst_fragmented_terms": vocab_splits[:15],
        "domain_vocabulary_splits": vocab_splits[:50],
        "maritime_vocabulary_splits": vocab_splits[:50]  # Compatibility alias
    }

def main():
    root = get_project_root()
    config = load_config()
    bench_cfg = get_benchmark_config()
    output_dir = root / config.get("output_dir", "outputs")

    # Load vocabulary from benchmark config or file
    vocab_filename = bench_cfg.get("vocabulary_file", "maritime_vocabulary.txt")
    vocab_path = output_dir / vocab_filename
    if not vocab_path.exists():
        fallback_vocabs = list(output_dir.glob("*_vocabulary.txt"))
        if fallback_vocabs:
            vocab_path = fallback_vocabs[0]

    vocab_terms = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fv:
            vocab_terms = [line.strip() for line in fv if line.strip()]

    # Load corpus text directly from plain-text corpus (deterministic, no silent fallback)
    corpus_filename = bench_cfg.get("corpus_file", "maritime_corpus.txt")
    corpus_path = output_dir / corpus_filename

    allow_discovery = bench_cfg.get("allow_corpus_auto_discovery", False)
    dedup = bench_cfg.get("deduplicate_corpus", False)

    docs = load_corpus_documents(corpus_path, deduplicate=dedup, allow_auto_discovery=allow_discovery)
    corpus_docs = [d["document"] for d in docs[:1500]]

    tok_dir = output_dir / "tokenizer_analysis"
    tok_dir.mkdir(parents=True, exist_ok=True)

    summary_reports = []

    for model_name in TARGET_MODELS:
        report = analyze_tokenizer(model_name, vocab_terms, corpus_docs)
        if report:
            summary_reports.append(report)
            clean_name = report["clean_model_name"]
            with open(tok_dir / f"{clean_name}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

    # Copy BERT baseline to outputs/tokenizer_analysis.json for backwards compatibility
    bert_report = next((r for r in summary_reports if r["model_name"] == "bert-base-uncased"), summary_reports[0] if summary_reports else {})
    with open(output_dir / "tokenizer_analysis.json", "w", encoding="utf-8") as f:
        json.dump(bert_report, f, indent=2)

    # Generate Tokenizer Comparison CSV
    csv_rows = []
    for r in summary_reports:
        csv_rows.append({
            "model_name": r["model_name"],
            "vocab_size": r["vocab_size"],
            "subwords_per_word_fertility": round(r["average_subwords_per_word"], 4),
            "single_token_coverage_pct": round(r["single_token_vocabulary_coverage"] * 100, 2),
            "fragmentation_rate_pct": round(r["domain_fragmentation_rate"] * 100, 2),
            "oov_rate_pct": round(r["oov_rate"] * 100, 4),
            "avg_pieces_per_term": round(r["avg_pieces_per_term"], 2),
            "median_pieces": r["median_pieces_per_term"],
            "p95_pieces": r["p95_pieces_per_term"],
            "max_pieces": r["max_pieces_per_term"],
            "tokenizer_speed_tok_sec": round(r["tokenizer_speed_tokens_per_sec"], 2)
        })

    df = pd.DataFrame(csv_rows)
    df.sort_values(by="single_token_coverage_pct", ascending=False, inplace=True)
    df.to_csv(tok_dir / "tokenizer_comparison.csv", index=False)

    logger.info(f"Stage 13 completed. Analyzed {len(summary_reports)} tokenizers. Reports saved to {tok_dir}")

if __name__ == "__main__":
    main()
