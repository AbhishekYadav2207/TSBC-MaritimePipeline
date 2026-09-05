import os
import json
import math
import random
import time
import re
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pipeline_utils import (
    setup_logging,
    load_config,
    get_project_root,
    get_benchmark_config
)

logger = setup_logging("14_mlm_evaluation")

# Tokenizer-Deduplicated Representative Subset (7 Distinct Tokenizer Families)
TARGET_MODELS = [
    "bert-base-uncased",                                            # Standard WordPiece (30,522) - Represents BERT-Base, BERT-Large, DistilBERT, ELECTRA, FinBERT
    "dmis-lab/biobert-base-cased-v1.2",                            # Bio/Clinical WordPiece (28,996 Cased) - Represents BioBERT, Bio_ClinicalBERT
    "nlpaueb/legal-bert-base-uncased",                              # Legal WordPiece (30,522)
    "allenai/scibert_scivocab_uncased",                             # SciVocab WordPiece (31,090)
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",# PubMed Domain WordPiece (30,522)
    "roberta-base",                                                 # Standard Byte-Level BPE (50,265)
    "answerdotai/ModernBERT-base"                                   # Modern Extended BPE (50,280)
]

def clean_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")

def get_term_category(term: str, categories: dict) -> str:
    term_lower = term.lower()
    for cat, stems in categories.items():
        if any(stem in term_lower for stem in stems):
            return cat
    return list(categories.keys())[0] if categories else "domain_terminology"

def extract_domain_spans(text: str, vocab_terms: list, rare_terms: set, categories: dict) -> list:
    """
    Finds exact character-level spans of domain terms in the document text.
    Ensures domain-token classification is based on actual textual occurrences,
    preventing subwords (e.g. 'dynamic' in 'hemodynamic') from being globally contaminated.
    """
    spans = []
    text_lower = text.lower()

    # Prioritize longer multi-word phrases over single-word tokens
    all_terms = sorted(list(set(vocab_terms).union(rare_terms)), key=lambda x: len(x), reverse=True)

    for term in all_terms:
        term_clean = term.strip().lower()
        if not term_clean or len(term_clean) < 2:
            continue
        # Strict word-boundary matching
        pattern = r'\b' + re.escape(term_clean) + r'\b'
        for m in re.finditer(pattern, text_lower):
            spans.append({
                "start": m.start(),
                "end": m.end(),
                "term": term_clean,
                "is_rare": term_clean in rare_terms,
                "category": get_term_category(term_clean, categories)
            })
    return spans

def evaluate_model_on_docs(
    model,
    tokenizer,
    docs: list,
    vocab_terms: list,
    categories: dict,
    rare_terms: list,
    device: torch.device
) -> dict:
    if not docs:
        return {}

    rare_terms_set = set(r.lower() for r in rare_terms)

    general_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    domain_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    rare_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    cat_stats = {cat: {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0} for cat in categories}

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    batch_size = 16
    eval_docs = docs[:200]  # Efficient evaluation sample size

    t_start = time.time()

    with torch.no_grad():
        for i in range(0, len(eval_docs), batch_size):
            batch_texts = eval_docs[i:i+batch_size]

            # Use fast tokenizer offset_mapping if available
            try:
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_offsets_mapping=True,
                    return_tensors="pt"
                )
                offset_mapping = inputs.pop("offset_mapping", None)
            except Exception:
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                offset_mapping = None

            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            # Construct Bernoulli masking matrix (15% standard MLM rate, excluding special tokens)
            probability_matrix = torch.full(labels.shape, 0.15, device=device)
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.cpu().tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=device), value=0.0)

            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100

            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_indices] = mask_token_id

            try:
                outputs = model(input_ids=masked_input_ids, attention_mask=inputs.get("attention_mask"))
                logits = outputs.logits
            except Exception as e:
                logger.warning(f"Error during model forward pass: {e}")
                continue

            for b in range(labels.shape[0]):
                doc_text = batch_texts[b]
                # Extract character spans for actual domain term occurrences in this document
                doc_spans = extract_domain_spans(doc_text, vocab_terms, rare_terms_set, categories)
                b_offsets = offset_mapping[b].cpu().tolist() if offset_mapping is not None else None

                mask_positions = torch.where(masked_indices[b])[0]
                for pos in mask_positions:
                    p_idx = pos.item()
                    target_id = labels[b, pos].item()
                    token_logits = logits[b, pos]

                    token_loss = criterion(token_logits.unsqueeze(0), torch.tensor([target_id], device=device)).item()
                    top_k_indices = torch.topk(token_logits, 10).indices.tolist()

                    is_top1 = 1 if target_id == top_k_indices[0] else 0
                    is_top5 = 1 if target_id in top_k_indices[:5] else 0
                    is_top10 = 1 if target_id in top_k_indices[:10] else 0

                    # SPAN-AWARE DETECTION: check if token overlaps with an actual domain term occurrence in text
                    is_domain = False
                    is_rare = False
                    matched_category = None

                    if b_offsets is not None and p_idx < len(b_offsets):
                        tok_start, tok_end = b_offsets[p_idx]
                        if tok_start < tok_end:  # Non-special token
                            for span in doc_spans:
                                # Overlap check between token character span and domain term character span
                                if max(tok_start, span["start"]) < min(tok_end, span["end"]):
                                    is_domain = True
                                    if span["is_rare"]:
                                        is_rare = True
                                    matched_category = span["category"]
                                    break
                    else:
                        # Fallback for tokenizers without offset_mapping: check decoded token string against doc spans
                        tok_str = tokenizer.decode([target_id]).strip().lower().replace("##", "")
                        if tok_str and len(tok_str) >= 3:
                            for span in doc_spans:
                                if tok_str in span["term"]:
                                    is_domain = True
                                    if span["is_rare"]:
                                        is_rare = True
                                    matched_category = span["category"]
                                    break

                    if is_rare:
                        rare_stats["loss"] += token_loss
                        rare_stats["top1"] += is_top1
                        rare_stats["top5"] += is_top5
                        rare_stats["top10"] += is_top10
                        rare_stats["count"] += 1

                    if is_domain:
                        domain_stats["loss"] += token_loss
                        domain_stats["top1"] += is_top1
                        domain_stats["top5"] += is_top5
                        domain_stats["top10"] += is_top10
                        domain_stats["count"] += 1

                        if matched_category and matched_category in cat_stats:
                            cat_stats[matched_category]["loss"] += token_loss
                            cat_stats[matched_category]["top1"] += is_top1
                            cat_stats[matched_category]["top5"] += is_top5
                            cat_stats[matched_category]["top10"] += is_top10
                            cat_stats[matched_category]["count"] += 1
                    else:
                        general_stats["loss"] += token_loss
                        general_stats["top1"] += is_top1
                        general_stats["top5"] += is_top5
                        general_stats["top10"] += is_top10
                        general_stats["count"] += 1

    eval_time = time.time() - t_start

    def summarize(st):
        cnt = max(st["count"], 1)
        avg_loss = st["loss"] / cnt
        loss_exp = math.exp(avg_loss) if avg_loss < 20 else 99999.0
        return {
            "masked_sample_count": st["count"],
            "mlm_loss": float(avg_loss) if st["count"] > 0 else 0.0,
            "mlm_loss_derived_exponential": float(loss_exp) if st["count"] > 0 else 0.0,
            "top1_accuracy": float(st["top1"] / cnt) if st["count"] > 0 else 0.0,
            "top5_accuracy": float(st["top5"] / cnt) if st["count"] > 0 else 0.0,
            "top10_accuracy": float(st["top10"] / cnt) if st["count"] > 0 else 0.0
        }

    gen_summary = summarize(general_stats)
    dom_summary = summarize(domain_stats)
    rare_summary = summarize(rare_stats)
    cat_summaries = {cat: summarize(st) for cat, st in cat_stats.items()}

    performance_gap = gen_summary["top1_accuracy"] - dom_summary["top1_accuracy"]

    return {
        "evaluated_documents": len(eval_docs),
        "evaluation_time_sec": float(eval_time),
        "general_tokens_summary": gen_summary,
        "domain_tokens_summary": dom_summary,
        "rare_domain_tokens_summary": rare_summary,
        "category_recall": {cat: round(st["top1_accuracy"], 4) for cat, st in cat_summaries.items()},
        "category_breakdown": cat_summaries,
        "performance_gap_top1": float(performance_gap)
    }

def main():
    random.seed(42)
    torch.manual_seed(42)

    root = get_project_root()
    config = load_config()
    bench_cfg = get_benchmark_config()
    output_dir = root / config.get("output_dir", "outputs")

    categories = bench_cfg.get("categories", {})
    rare_terms = bench_cfg.get("rare_domain_terms", [])

    vocab_filename = bench_cfg.get("vocabulary_file", "vocabulary.txt")
    vocab_path = output_dir / vocab_filename
    if not vocab_path.exists():
        if bench_cfg.get("allow_corpus_auto_discovery", False):
            fallback_vocabs = list(output_dir.glob("*_vocabulary.txt"))
            vocab_path = fallback_vocabs[0] if fallback_vocabs else vocab_path
        else:
            logger.warning(f"Vocabulary file not found at {vocab_path}.")

    vocab_terms = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fv:
            vocab_terms = [line.strip() for line in fv if line.strip()]

    # Representations & Subsets
    reps_dir = output_dir / "corpus_representations"
    subsets_dir = output_dir / "subsets"

    representations = ["narrative", "key_value", "template", "json", "mixed"]
    subsets = ["high_knowledge", "medium_knowledge", "low_knowledge", "balanced_knowledge", "random_baseline"]

    # Load General English Baseline Subset
    gen_eng_docs = []
    gen_eng_path = subsets_dir / "general_english_baseline.jsonl"
    if gen_eng_path.exists():
        with open(gen_eng_path, "r", encoding="utf-8") as f:
            gen_eng_docs = [json.loads(l)["document"] for l in f]

    eval_out_dir = output_dir / "evaluations"
    cache_dir = eval_out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    total_runs = len(TARGET_MODELS) * len(representations) * len(subsets)
    logger.info(f"Starting 175-Run Matrix Evaluation Grid ({total_runs} independent runs: {len(TARGET_MODELS)} models × {len(representations)} reps × {len(subsets)} subsets)...")

    run_count = 0

    for model_name in TARGET_MODELS:
        clean_model = clean_model_filename(model_name)
        logger.info(f"Loading model & tokenizer: {model_name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model.to(device)
            model.eval()
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face model '{model_name}': {e}. Skipping model.")
            continue

        # Evaluate General English Baseline once for Domain Shift calculation
        # Methodological Hardening: NO FAKE FALLBACKS. If baseline cannot be evaluated, explicitly mark unavailable.
        gen_eng_top1 = None
        baseline_available = False

        if gen_eng_docs:
            gen_eng_eval = evaluate_model_on_docs(model, tokenizer, gen_eng_docs, vocab_terms, categories, rare_terms, device)
            gen_sum = gen_eng_eval.get("general_tokens_summary")
            if gen_sum and gen_sum.get("masked_sample_count", 0) > 0:
                gen_eng_top1 = float(gen_sum["top1_accuracy"])
                baseline_available = True
            else:
                logger.warning(f"General English baseline evaluation returned 0 masked tokens for {model_name}.")
        else:
            logger.warning(f"General English baseline documents not found at {gen_eng_path}. Baseline will be marked unavailable.")

        eval_record = None
        for rep in representations:
            rep_path = reps_dir / f"{rep}.jsonl"
            if not rep_path.exists():
                continue
            with open(rep_path, "r", encoding="utf-8") as f:
                rep_records = [json.loads(l) for l in f]

            for sub in subsets:
                cache_key = f"{clean_model}__{rep}__{sub}.json"
                cache_path = cache_dir / cache_key

                # Check Resumable Cache
                if cache_path.exists():
                    run_count += 1
                    with open(cache_path, "r", encoding="utf-8") as f_c:
                        eval_record = json.load(f_c)
                    continue

                sub_path = subsets_dir / f"{sub}.jsonl"
                sub_doc_ids = set()
                if sub_path.exists():
                    with open(sub_path, "r", encoding="utf-8") as f:
                        for l in f:
                            r_json = json.loads(l)
                            if r_json.get("doc_id"):
                                sub_doc_ids.add(r_json["doc_id"])

                if not sub_doc_ids:
                    logger.warning(f"Subset '{sub}' contains 0 document IDs. Skipping.")
                    continue

                # Filter representation documents strictly by subset document identifier
                target_docs = [
                    rec["document"]
                    for rec in rep_records
                    if rec.get("doc_id") in sub_doc_ids
                ][:200]

                if not target_docs:
                    logger.warning(f"Representation '{rep}' has 0 matching documents for subset '{sub}'. Skipping.")
                    continue

                logger.info(f"[{run_count + 1}/{total_runs}] Evaluating {clean_model} | Rep: {rep} | Subset: {sub} | Matched Documents: {len(target_docs)}")

                eval_res = evaluate_model_on_docs(model, tokenizer, target_docs, vocab_terms, categories, rare_terms, device)

                dom_top1 = eval_res.get("domain_tokens_summary", {}).get("top1_accuracy", 0.0)

                # Strict Domain Gap calculation: Null/None if baseline unavailable
                if baseline_available and gen_eng_top1 is not None:
                    domain_shift_gap = float(gen_eng_top1 - dom_top1)
                else:
                    domain_shift_gap = None

                eval_record = {
                    "model_name": model_name,
                    "clean_model_name": clean_model,
                    "representation": rep,
                    "subset": sub,
                    "evaluated_doc_count": len(target_docs),
                    "baseline_available": baseline_available,
                    "general_english_baseline_top1": gen_eng_top1,
                    "domain_shift_gap": domain_shift_gap,
                    "evaluation_metrics": eval_res
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(eval_record, f, indent=2)

                run_count += 1
                logger.info(f"[{run_count}/{total_runs}] Completed: {clean_model} | Rep: {rep} | Subset: {sub} | Top1: {dom_top1:.4f}")

        # Also store model summary under outputs/evaluations/<clean_model>.json
        if eval_record is not None:
            with open(eval_out_dir / f"{clean_model}.json", "w", encoding="utf-8") as f:
                json.dump(eval_record, f, indent=2)

    # Copy BERT baseline to outputs/bert_mlm_evaluation.json for backwards compatibility
    bert_clean = clean_model_filename("bert-base-uncased")
    bert_cache = list(cache_dir.glob(f"{bert_clean}__*.json"))
    if bert_cache:
        with open(bert_cache[0], "r", encoding="utf-8") as f_in, open(output_dir / "bert_mlm_evaluation.json", "w", encoding="utf-8") as f_out:
            json.dump(json.load(f_in), f_out, indent=2)

    logger.info(f"Stage 14 completed. Evaluated {run_count} matrix runs. Saved cache files to {cache_dir}")

if __name__ == "__main__":
    main()
