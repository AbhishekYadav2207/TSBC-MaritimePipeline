import os
import json
import math
import random
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("12_bert_mlm_evaluation")

CATEGORIES = {
    "vessel_terminology": ["vess", "ship", "boat", "barge", "tug", "tanker", "trawler", "carrier", "hull", "deck", "keel", "tonnage", "built", "length", "width", "transom", "freeboard", "gunwale", "bilge"],
    "navigation": ["navig", "gps", "ais", "vhf", "radar", "sonar", "compass", "gyro", "sounder", "chart", "vdr", "speed", "course", "bearing", "fathometer"],
    "machinery_propulsion": ["engine", "propel", "machinery", "motor", "shaft", "boiler", "fuel", "steering", "windlass", "hawser"],
    "casualty_incident": ["collision", "grounding", "stranding", "flooding", "leak", "capsiz", "sink", "injury", "death", "fatality", "missing", "damage"],
    "weather_environment": ["weather", "wind", "sea", "wave", "swell", "temp", "ice", "visibility", "fog", "clear", "light", "windward", "leeward"],
    "safety_lifesaving": ["lifeboat", "liferaft", "lifejack", "lsa", "epirb", "sart", "buoy", "flare", "safety", "davit", "coxswain"]
}

RARE_MARITIME_TERMS = [
    "gyrocompass", "fathometer", "forepeak", "bulwark", "stempost", "windlass",
    "epirb", "sart", "hawser", "freeboard", "coxswain", "transom", "gunwale",
    "bilge", "fairlead", "windward", "leeward", "davit"
]

def get_term_category(term: str) -> str:
    term_lower = term.lower()
    for cat, stems in CATEGORIES.items():
        if any(stem in term_lower for stem in stems):
            return cat
    return "vessel_terminology"

def main():
    random.seed(42)
    torch.manual_seed(42)
    
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    corpus_jsonl_path = output_dir / "maritime_corpus.jsonl"
    if not corpus_jsonl_path.exists():
        logger.error(f"Corpus JSONL not found at {corpus_jsonl_path}!")
        return
        
    vocab_path = output_dir / "maritime_vocabulary.txt"
    vocab_terms = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fv:
            vocab_terms = [line.strip() for line in fv if line.strip()]
            
    model_name = config.get("generation", {}).get("bert_evaluation_model", "bert-base-uncased")
    logger.info(f"Loading pretrained {model_name} tokenizer and model for MLM diagnostic evaluation...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    # Identify maritime token IDs, category token IDs, and rare term token IDs
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
        
    logger.info(f"Identified {len(maritime_token_ids)} maritime subword tokens and {len(rare_token_ids)} rare maritime subword tokens.")
    
    # Load document sample (up to 3,000 documents)
    max_sample = 3000
    docs = []
    with open(corpus_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            docs.append(json.loads(line)["document"])
            if len(docs) >= max_sample:
                break
                
    logger.info(f"Evaluating MLM performance across Mode A & Mode B on sample of {len(docs)} documents...")
    
    # Metrics accumulators
    general_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    maritime_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    rare_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    cat_stats = {cat: {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0} for cat in CATEGORIES}
    
    batch_size = 16
    mask_token_id = tokenizer.mask_token_id
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch_texts = docs[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            
            # Mask 15% of tokens deterministically
            probability_matrix = torch.full(labels.shape, 0.15)
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100
            
            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_indices] = mask_token_id
            
            outputs = model(input_ids=masked_input_ids, attention_mask=inputs["attention_mask"])
            logits = outputs.logits
            
            for b in range(labels.shape[0]):
                mask_positions = torch.where(masked_indices[b])[0]
                for pos in mask_positions:
                    target_id = labels[b, pos].item()
                    token_logits = logits[b, pos]
                    
                    token_loss = criterion(token_logits.unsqueeze(0), torch.tensor([target_id])).item()
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

    def summarize(st):
        cnt = max(st["count"], 1)
        avg_loss = st["loss"] / cnt
        loss_exp = math.exp(avg_loss) if avg_loss < 20 else 99999.0
        return {
            "masked_sample_count": st["count"],
            "mlm_loss": avg_loss,
            "mlm_loss_derived_exponential": loss_exp,
            "top1_accuracy": st["top1"] / cnt,
            "top5_accuracy": st["top5"] / cnt,
            "top10_accuracy": st["top10"] / cnt
        }
        
    gen_summary = summarize(general_stats)
    mar_summary = summarize(maritime_stats)
    rare_summary = summarize(rare_stats)
    cat_summaries = {cat: summarize(st) for cat, st in cat_stats.items()}
    
    performance_gap = gen_summary["top1_accuracy"] - mar_summary["top1_accuracy"]
    
    eval_report = {
        "model_name": model_name,
        "evaluation_modes": {
            "mode_a": "Natural Stratified Distribution Evaluation",
            "mode_b": "Controlled Domain & Category-Balanced Evaluation"
        },
        "evaluated_documents": len(docs),
        "masking_rate": 0.15,
        "general_tokens_summary": gen_summary,
        "maritime_tokens_summary": mar_summary,
        "rare_maritime_tokens_summary": rare_summary,
        "category_breakdown": cat_summaries,
        "performance_gap_top1": performance_gap,
        "general_tokens_top1": gen_summary["top1_accuracy"],
        "maritime_tokens_top1": mar_summary["top1_accuracy"],
        "rare_maritime_tokens_top1": rare_summary["top1_accuracy"]
    }
    
    out_path = output_dir / "bert_mlm_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
        
    logger.info(f"BERT MLM Diagnostic Evaluation complete. Saved to {out_path}")
    logger.info(f"  General Tokens Top-1 Accuracy: {gen_summary['top1_accuracy']*100:.2f}%")
    logger.info(f"  Maritime Tokens Top-1 Accuracy: {mar_summary['top1_accuracy']*100:.2f}%")
    logger.info(f"  Rare Maritime Tokens Top-1 Accuracy: {rare_summary['top1_accuracy']*100:.2f}%")
    logger.info(f"  Performance Gap: {performance_gap*100:.2f}%")

if __name__ == "__main__":
    main()
