import os
import json
import re
import math
import random
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("12_semantic_importance")

CATEGORIES = {
    "vessel_terminology": ["vess", "ship", "boat", "barge", "tug", "tanker", "trawler", "carrier", "hull", "deck", "keel", "tonnage", "transom", "freeboard", "gunwale", "bilge"],
    "navigation": ["navig", "gps", "ais", "vhf", "radar", "sonar", "compass", "gyro", "sounder", "chart", "vdr", "fathometer"],
    "machinery_propulsion": ["engine", "propel", "machinery", "motor", "shaft", "boiler", "fuel", "steering", "windlass", "hawser"],
    "casualty_incident": ["collision", "grounding", "stranding", "flooding", "leak", "capsiz", "sink", "injury", "death", "fatality", "missing", "damage"],
    "weather_environment": ["weather", "wind", "sea", "wave", "swell", "temp", "ice", "visibility", "fog", "clear", "windward", "leeward"],
    "safety_lifesaving": ["lifeboat", "liferaft", "lifejack", "lsa", "epirb", "sart", "buoy", "flare", "safety", "davit", "coxswain"]
}

RARE_MARITIME_TERMS = {
    "gyrocompass", "fathometer", "forepeak", "bulwark", "stempost", "windlass",
    "epirb", "sart", "hawser", "freeboard", "coxswain", "transom", "gunwale",
    "bilge", "fairlead", "windward", "leeward", "davit", "bitts", "bollard", "focsle"
}

GENERAL_ENGLISH_SENTENCES = [
    "The library offers a wide variety of books and digital media for public access.",
    "Engineers designed a new solar panel system to increase energy efficiency in urban homes.",
    "Scientists conducted an extensive field study on migratory birds across northern lakes.",
    "The financial market experienced significant fluctuation following quarterly earnings announcements.",
    "Students participated in a regional mathematics competition held at the city convention center.",
    "Local park authorities launched an initiative to plant native trees and preserve wetland habitats.",
    "Software developers released a major software update addressing security vulnerabilities.",
    "The museum featured an exhibit highlighting ancient architectural techniques and pottery.",
    "Researchers analyzed statistical trends in public transportation usage across major cities.",
    "A team of doctors published findings on early diagnostic methods for cardiovascular health."
]

def compute_document_features(doc_text: str, structured: dict, term_freq_map: Counter, total_docs: int) -> dict:
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', doc_text.lower())
    num_tokens = len(tokens)
    if num_tokens == 0:
        return {
            "maritime_density": 0.0, "rare_term_count": 0, "concept_diversity": 0.0,
            "entity_diversity": 0.0, "event_complexity": 0.0, "information_density": 0.0,
            "redundancy_penalty": 1.0, "metadata_completeness": 0.0, "linguistic_diversity": 0.0,
            "domain_novelty": 0.0, "concepts": []
        }
        
    # 1. Maritime Terminology Density
    maritime_tokens = 0
    detected_concepts = set()
    for tok in tokens:
        for cat, stems in CATEGORIES.items():
            if any(stem in tok for stem in stems):
                maritime_tokens += 1
                detected_concepts.add(cat)
                break
    maritime_density = maritime_tokens / num_tokens
    
    # 2. Rare Maritime Vocabulary
    rare_count = sum(1 for tok in tokens if tok in RARE_MARITIME_TERMS)
    rare_score = min(1.0, rare_count / 3.0)
    
    # 3. Concept Diversity
    concept_diversity = len(detected_concepts) / len(CATEGORIES)
    
    # 4. Entity Diversity
    occ = structured.get("occurrence", {})
    vessels = structured.get("vessels", [])
    distinct_entities = set()
    if occ.get("NearestLocationDescription"): distinct_entities.add(occ.get("NearestLocationDescription"))
    if occ.get("WeatherConditionDisplayEng"): distinct_entities.add(occ.get("WeatherConditionDisplayEng"))
    for v in vessels:
        if v.get("VesselName"): distinct_entities.add(v.get("VesselName"))
        if v.get("VesselTypeDisplayEng"): distinct_entities.add(v.get("VesselTypeDisplayEng"))
        if v.get("HullMaterialDisplayEng"): distinct_entities.add(v.get("HullMaterialDisplayEng"))
    entity_diversity = min(1.0, len(distinct_entities) / 6.0)
    
    # 5. Event Complexity
    causal_markers = len(re.findall(r'\b(caused|during|while|resulted|following|underway|sustained)\b', doc_text.lower()))
    num_clauses = len(re.split(r'[,;.]', doc_text))
    event_complexity = min(1.0, (causal_markers * 0.3 + num_clauses * 0.1))
    
    # 6. Information Density
    info_density = min(1.0, maritime_tokens / (num_tokens * 0.5)) if num_tokens > 0 else 0.0
    
    # 7. Redundancy Penalty (based on boilerplate text patterns)
    is_boilerplate = 1.0 if "resulting in a marine occurrence" in doc_text and len(doc_text) < 120 else 0.0
    redundancy_penalty = is_boilerplate * 0.5
    
    # 8. Metadata Completeness
    meta_fields = [
        occ.get("NearestLocationDescription"), occ.get("WeatherConditionDisplayEng"),
        occ.get("AccIncTypeDisplayEng"), vessels[0].get("VesselName") if vessels else None,
        vessels[0].get("GrossTonnage") if vessels else None
    ]
    meta_completeness = sum(1 for f in meta_fields if f) / len(meta_fields)
    
    # 9. Linguistic Diversity (TTR)
    ttr = len(set(tokens)) / num_tokens if num_tokens > 0 else 0.0
    
    # 10. Domain Novelty (IDF sum of maritime terms)
    novelty_sum = sum(math.log((total_docs + 1) / (term_freq_map.get(tok, 1) + 1)) for tok in tokens if tok in RARE_MARITIME_TERMS)
    domain_novelty = min(1.0, novelty_sum / 10.0)
    
    return {
        "maritime_density": maritime_density,
        "rare_term_count": rare_count,
        "rare_score": rare_score,
        "concept_diversity": concept_diversity,
        "entity_diversity": entity_diversity,
        "event_complexity": event_complexity,
        "information_density": info_density,
        "redundancy_penalty": redundancy_penalty,
        "metadata_completeness": meta_completeness,
        "linguistic_diversity": ttr,
        "domain_novelty": domain_novelty,
        "concepts": list(detected_concepts)
    }

def main():
    random.seed(42)
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents not found at {clean_path}! Run Step 7 first.")
        return
        
    logger.info("First pass: Computing corpus-wide vocabulary document frequencies for Domain Novelty...")
    term_freq = Counter()
    total_docs = 0
    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in fin:
            doc = json.loads(line)["document"].lower()
            words = set(re.findall(r'\b[a-zA-Z]{3,}\b', doc))
            term_freq.update(words)
            total_docs += 1
            
    logger.info(f"Second pass: Computing Semantic Importance Scores and Knowledge Classification for {total_docs} documents...")
    scored_records = []
    
    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total_docs, desc="Scoring Documents"):
            record = json.loads(line)
            occ_id = record.get("occurrence_id")
            doc_text = record.get("document", "")
            structured = record.get("structured", {})
            
            feats = compute_document_features(doc_text, structured, term_freq, total_docs)
            
            raw_score = (
                0.30 * feats["maritime_density"] +
                0.20 * feats["rare_score"] +
                0.15 * feats["concept_diversity"] +
                0.10 * feats["entity_diversity"] +
                0.10 * feats["event_complexity"] +
                0.10 * feats["information_density"] +
                0.05 * feats["metadata_completeness"] +
                0.05 * feats["linguistic_diversity"] +
                0.05 * feats["domain_novelty"] -
                0.10 * feats["redundancy_penalty"]
            )
            
            importance_score = float(np.clip(raw_score * 100.0, 0.0, 100.0))
            
            # Knowledge Classifier based on relative score thresholds
            if feats["redundancy_penalty"] >= 0.4:
                knowledge_tier = "Redundant"
            elif importance_score >= 42.0:
                knowledge_tier = "High Knowledge"
            elif importance_score >= 35.0:
                knowledge_tier = "Medium Knowledge"
            elif importance_score >= 20.0:
                knowledge_tier = "Low Knowledge"
            else:
                knowledge_tier = "Noisy / Boilerplate"
                
            scored_records.append({
                "occurrence_id": occ_id,
                "importance_score": round(importance_score, 2),
                "knowledge_tier": knowledge_tier,
                "maritime_density": round(feats["maritime_density"], 4),
                "rare_term_count": feats["rare_term_count"],
                "concept_diversity": round(feats["concept_diversity"], 4),
                "concepts": feats["concepts"],
                "document": doc_text
            })
            
    # Export document_importance.jsonl
    imp_path = output_dir / "document_importance.jsonl"
    with open(imp_path, "w", encoding="utf-8") as fout:
        for rec in scored_records:
            fout.write(json.dumps(rec) + "\n")
            
    # Compute Statistics
    scores = [r["importance_score"] for r in scored_records]
    tier_counts = Counter(r["knowledge_tier"] for r in scored_records)
    
    stats = {
        "total_documents": len(scores),
        "mean_importance_score": float(np.mean(scores)),
        "median_importance_score": float(np.median(scores)),
        "std_importance_score": float(np.std(scores)),
        "min_importance_score": float(np.min(scores)),
        "max_importance_score": float(np.max(scores)),
        "quartiles": [float(q) for q in np.percentile(scores, [25, 50, 75])],
        "knowledge_tier_breakdown": dict(tier_counts)
    }
    
    stat_path = output_dir / "importance_statistics.json"
    with open(stat_path, "w", encoding="utf-8") as fstat:
        json.dump(stats, fstat, indent=2)
        
    # Plot Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(scores), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {np.mean(scores):.2f}")
    plt.axvline(np.median(scores), color="green", linestyle="dotted", linewidth=2, label=f"Median: {np.median(scores):.2f}")
    plt.title("Corpus Semantic Importance Score Distribution")
    plt.xlabel("Semantic Importance Score (0 - 100)")
    plt.ylabel("Document Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plot_path = output_dir / "importance_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create Evaluation Subsets under outputs/subsets/
    subsets_dir = output_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort scored records by importance score descending for quantile-based subset extraction
    sorted_records = sorted(scored_records, key=lambda x: x["importance_score"], reverse=True)
    
    target_sample_size = min(1000, len(scored_records))
    
    # High docs: Top scoring documents
    high_docs = sorted_records[:target_sample_size]
    
    # Low docs: Bottom non-redundant/non-boilerplate documents
    valid_low = [r for r in sorted_records if r["knowledge_tier"] in ("Low Knowledge", "Noisy / Boilerplate")]
    low_docs = valid_low[-target_sample_size:] if len(valid_low) >= target_sample_size else sorted_records[-target_sample_size:]
    
    # Med docs: Middle range documents
    mid_start = (len(sorted_records) - target_sample_size) // 2
    med_docs = sorted_records[mid_start:mid_start + target_sample_size]
    
    # 1. High Knowledge Subset
    with open(subsets_dir / "high_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in high_docs:
            f.write(json.dumps(r) + "\n")
            
    # 2. Medium Knowledge Subset
    with open(subsets_dir / "medium_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in med_docs:
            f.write(json.dumps(r) + "\n")
            
    # 3. Low Knowledge Subset
    with open(subsets_dir / "low_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in low_docs:
            f.write(json.dumps(r) + "\n")
            
    # 4. Balanced Knowledge Subset (Equal blend of High, Medium, Low)
    n_per_tier = target_sample_size // 3
    balanced_pool = high_docs[:n_per_tier] + med_docs[:n_per_tier] + low_docs[:n_per_tier]
    random.shuffle(balanced_pool)
    with open(subsets_dir / "balanced_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in balanced_pool:
            f.write(json.dumps(r) + "\n")
            
    # 5. Random Baseline Subset
    random_pool = random.sample(scored_records, min(target_sample_size, len(scored_records)))
    with open(subsets_dir / "random_baseline.jsonl", "w", encoding="utf-8") as f:
        for r in random_pool:
            f.write(json.dumps(r) + "\n")
            
    # 6. General English Baseline Subset
    with open(subsets_dir / "general_english_baseline.jsonl", "w", encoding="utf-8") as f:
        for idx, sent in enumerate(GENERAL_ENGLISH_SENTENCES):
            f.write(json.dumps({
                "occurrence_id": 900000 + idx,
                "importance_score": 10.0,
                "knowledge_tier": "General English",
                "document": sent
            }) + "\n")
            
    logger.info(f"Stage 12 completed successfully. Saved importance scores and generated 6 evaluation subsets in {subsets_dir}")

if __name__ == "__main__":
    main()
