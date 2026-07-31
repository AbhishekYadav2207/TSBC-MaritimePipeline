import os
import json
import math
import random
import time
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pipeline_utils import setup_logging, load_config, get_project_root

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

CATEGORIES = {
    "vessel_terminology": ["vess", "ship", "boat", "barge", "tug", "tanker", "trawler", "carrier", "hull", "deck", "keel", "tonnage", "transom", "freeboard", "gunwale", "bilge"],
    "navigation": ["navig", "gps", "ais", "vhf", "radar", "sonar", "compass", "gyro", "sounder", "chart", "vdr", "fathometer"],
    "machinery_propulsion": ["engine", "propel", "machinery", "motor", "shaft", "boiler", "fuel", "steering", "windlass", "hawser"],
    "casualty_incident": ["collision", "grounding", "stranding", "flooding", "leak", "capsiz", "sink", "injury", "death", "fatality", "missing", "damage"],
    "weather_environment": ["weather", "wind", "sea", "wave", "swell", "temp", "ice", "visibility", "fog", "clear", "windward", "leeward"],
    "safety_lifesaving": ["lifeboat", "liferaft", "lifejack", "lsa", "epirb", "sart", "buoy", "flare", "safety", "davit", "coxswain"]
}

RARE_MARITIME_TERMS = [
    "gyrocompass", "fathometer", "forepeak", "bulwark", "stempost", "windlass",
    "epirb", "sart", "hawser", "freeboard", "coxswain", "transom", "gunwale",
    "bilge", "fairlead", "windward", "leeward", "davit", "bitts", "bollard"
]

def clean_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")

def get_term_category(term: str) -> str:
    term_lower = term.lower()
    for cat, stems in CATEGORIES.items():
        if any(stem in term_lower for stem in stems):
            return cat
    return "vessel_terminology"

def evaluate_model_on_docs(model, tokenizer, docs: list, vocab_terms: list, device: torch.device) -> dict:
    if not docs:
        return {}

    maritime_token_ids = set()
    category_token_ids = {cat: set() for cat in CATEGORIES}
    rare_token_ids = set()

    for term in vocab_terms:
        sub_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(term))
        maritime_token_ids.update(sub_ids)
        cat = get_term_category(term)
        category_token_ids[cat].update(sub_ids)

    for r_term in RARE_MARITIME_TERMS:
        r_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r_term))
        rare_token_ids.update(r_ids)
        maritime_token_ids.update(r_ids)
        category_token_ids["navigation"].update(r_ids)

    general_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    maritime_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    rare_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    cat_stats = {cat: {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0} for cat in CATEGORIES}

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    batch_size = 16
    eval_docs = docs[:200]  # Efficient CPU evaluation sample size

    t_start = time.time()

    with torch.no_grad():
        for i in range(0, len(eval_docs), batch_size):
            batch_texts = eval_docs[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

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
                mask_positions = torch.where(masked_indices[b])[0]
                for pos in mask_positions:
                    target_id = labels[b, pos].item()
                    token_logits = logits[b, pos]

                    token_loss = criterion(token_logits.unsqueeze(0), torch.tensor([target_id], device=device)).item()
                    top_k_indices = torch.topk(token_logits, 10).indices.tolist()

                    is_top1 = 1 if target_id == top_k_indices[0] else 0
                    is_top5 = 1 if target_id in top_k_indices[:5] else 0
                    is_top10 = 1 if target_id in top_k_indices[:10] else 0

                    is_rare = target_id in rare_token_ids
                    is_maritime = target_id in maritime_token_ids

                    if is_rare:
                        rare_stats["loss"] += token_loss
                        rare_stats["top1"] += is_top1
                        rare_stats["top5"] += is_top5
                        rare_stats["top10"] += is_top10
                        rare_stats["count"] += 1

                    if is_maritime:
                        maritime_stats["loss"] += token_loss
                        maritime_stats["top1"] += is_top1
                        maritime_stats["top5"] += is_top5
                        maritime_stats["top10"] += is_top10
                        maritime_stats["count"] += 1

                        for cat, cat_ids in category_token_ids.items():
                            if target_id in cat_ids:
                                cat_stats[cat]["loss"] += token_loss
                                cat_stats[cat]["top1"] += is_top1
                                cat_stats[cat]["top5"] += is_top5
                                cat_stats[cat]["top10"] += is_top10
                                cat_stats[cat]["count"] += 1
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
            "mlm_loss": float(avg_loss),
            "mlm_loss_derived_exponential": float(loss_exp),
            "top1_accuracy": float(st["top1"] / cnt),
            "top5_accuracy": float(st["top5"] / cnt),
            "top10_accuracy": float(st["top10"] / cnt)
        }

    gen_summary = summarize(general_stats)
    mar_summary = summarize(maritime_stats)
    rare_summary = summarize(rare_stats)
    cat_summaries = {cat: summarize(st) for cat, st in cat_stats.items()}

    performance_gap = gen_summary["top1_accuracy"] - mar_summary["top1_accuracy"]

    return {
        "evaluated_documents": len(eval_docs),
        "evaluation_time_sec": float(eval_time),
        "general_tokens_summary": gen_summary,
        "maritime_tokens_summary": mar_summary,
        "rare_maritime_tokens_summary": rare_summary,
        "category_recall": {cat: round(st["top1_accuracy"], 4) for cat, st in cat_summaries.items()},
        "category_breakdown": cat_summaries,
        "performance_gap_top1": float(performance_gap)
    }

def main():
    random.seed(42)
    torch.manual_seed(42)

    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")

    vocab_path = output_dir / "maritime_vocabulary.txt"
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
    logger.info(f"Starting 350-Run Matrix Evaluation Grid ({total_runs} independent runs) across {len(TARGET_MODELS)} models...")

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
        gen_eng_eval = evaluate_model_on_docs(model, tokenizer, gen_eng_docs, vocab_terms, device)
        gen_eng_top1 = gen_eng_eval.get("general_tokens_summary", {}).get("top1_accuracy", 0.85)

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
                sub_occ_ids = set()
                if sub_path.exists():
                    with open(sub_path, "r", encoding="utf-8") as f:
                        sub_occ_ids = {json.loads(l)["occurrence_id"] for l in f}

                if not sub_occ_ids:
                    logger.warning(f"Subset '{sub}' contains 0 occurrence IDs. Skipping.")
                    continue

                # Filter representation documents strictly by subset occurrence_id
                target_docs = [
                    rec["document"]
                    for rec in rep_records
                    if rec.get("occurrence_id") in sub_occ_ids
                ][:200]

                if not target_docs:
                    logger.warning(f"Representation '{rep}' has 0 matching documents for subset '{sub}'. Skipping.")
                    continue

                logger.info(f"[{run_count + 1}/{total_runs}] Evaluating {clean_model} | Rep: {rep} | Subset: {sub} | Matched Documents: {len(target_docs)}")

                eval_res = evaluate_model_on_docs(model, tokenizer, target_docs, vocab_terms, device)

                maritime_top1 = eval_res.get("maritime_tokens_summary", {}).get("top1_accuracy", 0.0)
                domain_shift_gap = float(gen_eng_top1 - maritime_top1)

                eval_record = {
                    "model_name": model_name,
                    "clean_model_name": clean_model,
                    "representation": rep,
                    "subset": sub,
                    "evaluated_doc_count": len(target_docs),
                    "general_english_baseline_top1": float(gen_eng_top1),
                    "domain_shift_gap": domain_shift_gap,
                    "evaluation_metrics": eval_res
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(eval_record, f, indent=2)

                run_count += 1
                logger.info(f"[{run_count}/{total_runs}] Completed: {clean_model} | Rep: {rep} | Subset: {sub} | Top1: {maritime_top1:.4f}")

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
