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
from pipeline_utils import (
    setup_logging,
    load_config,
    get_project_root,
    get_benchmark_config,
    load_corpus_documents
)

logger = setup_logging("12_semantic_importance")

# General functional stopwords used to evaluate lexical information density (independent of domain tokens)
FUNCTIONAL_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "of", "in", "on", "off", "over", "under",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "should", "could", "ought", "i", "you",
    "he", "she", "it", "we", "they", "them", "their", "theirs", "this", "that", "these",
    "those", "am", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don"
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

FEATURE_WEIGHTS = {
    "domain_density": 0.30,
    "rare_score": 0.20,
    "concept_diversity": 0.15,
    "entity_diversity": 0.10,
    "event_complexity": 0.10,
    "information_density": 0.10,
    "structural_completeness": 0.05,
    "linguistic_diversity": 0.05,
    "rare_domain_term_novelty": 0.05,
    "redundancy_penalty": -0.10
}

def compute_document_features(
    doc_text: str,
    categories: dict,
    rare_terms: set,
    term_freq_map: Counter,
    total_docs: int,
    domain_lexicons: dict = None
) -> dict:
    """
    Computes text-derived semantic importance features for any domain corpus document.
    Generic core: Uses generic syntactic and lexical heuristics.
    Configurable domain profile: Optionally incorporates domain semantic lexicons if provided.
    Works fully if domain lexicons are empty or omitted.
    """
    lexicons = domain_lexicons or {}
    subject_kw = set(lexicons.get("subject_entity_keywords", []))
    context_kw = set(lexicons.get("context_keywords", []))
    action_kw = set(lexicons.get("action_keywords", lexicons.get("incident_keywords", [])))
    impact_kw = set(lexicons.get("impact_keywords", lexicons.get("consequence_keywords", [])))

    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', doc_text.lower())
    num_tokens = len(tokens)
    if num_tokens == 0:
        return {
            "domain_density": 0.0, "rare_term_count": 0, "rare_score": 0.0,
            "concept_diversity": 0.0, "entity_diversity": 0.0, "event_complexity": 0.0,
            "information_density": 0.0, "redundancy_penalty": 1.0, "structural_completeness": 0.0,
            "linguistic_diversity": 0.0,
            "rare_domain_term_novelty": 0.0, "domain_novelty": 0.0,
            "concepts": []
        }

    # 1. Domain Terminology Density & Detected Concepts
    domain_tokens = 0
    detected_concepts = set()
    for tok in tokens:
        for cat, stems in categories.items():
            if any(stem in tok for stem in stems):
                domain_tokens += 1
                detected_concepts.add(cat)
                break
    domain_density = domain_tokens / num_tokens

    # 2. Rare Domain Vocabulary
    rare_count = sum(1 for tok in tokens if tok in rare_terms)
    rare_score = min(1.0, rare_count / 3.0)

    # 3. Concept Diversity (Coverage of domain subcategories)
    concept_diversity = (len(detected_concepts) / len(categories)) if categories else 0.0

    # 4. Text-Derived Entity Diversity (Generic quoted phrases, acronyms, capitalized proper nouns)
    quoted = re.findall(r"['\"]([A-Za-z0-9\s\-]+)['\"]", doc_text)
    capitalized = re.findall(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b", doc_text)
    distinct_entities = set([e.strip() for e in quoted + capitalized if len(e.strip()) > 1])
    entity_diversity = min(1.0, len(distinct_entities) / 5.0)

    # 5. Text-Derived Event/Syntactic Complexity (Generic causal conjunctions and clause structures)
    causal_markers = len(re.findall(r'\b(caused|during|while|resulted|following|underway|sustained|due to|after|because|leading to|consequently)\b', doc_text.lower()))
    num_clauses = len(re.split(r'[,;.]', doc_text))
    event_complexity = min(1.0, (causal_markers * 0.25 + num_clauses * 0.08))

    # 6. Information Density (Lexical Content Word Ratio, distinct from domain_density)
    # Measures the ratio of non-functional content words to total tokens, avoiding double-counting domain tokens
    content_tokens = [t for t in tokens if t not in FUNCTIONAL_STOPWORDS and len(t) >= 3]
    info_density = len(content_tokens) / num_tokens if num_tokens > 0 else 0.0

    # 7. Redundancy Penalty (Generic boilerplate and repetition detection)
    ttr = len(set(tokens)) / num_tokens if num_tokens > 0 else 0.0
    is_boilerplate = 1.0 if ttr < 0.35 or (len(doc_text.split()) < 15 and num_clauses <= 1) else 0.0
    redundancy_penalty = is_boilerplate * 0.5

    # 8. Structural Completeness (Generic 5-facet semantic coverage)
    # Checks 5 core informational dimensions via generic linguistics + optional domain lexicons
    # (a) Subject entity: detected entity or presence of subject keywords
    has_subject = 1 if len(distinct_entities) > 0 or any(k in tokens for k in subject_kw) else 0
    # (b) Context / environmental conditions: temporal/spatial prepositions or context keywords
    has_context = 1 if re.search(r'\b(under|during|while|at|in|near|between|conditions|phase)\b', doc_text.lower()) or any(k in tokens for k in context_kw) else 0
    # (c) Incident / action verbs: verbal participles or action keywords
    has_action = 1 if re.search(r'\b([a-z]{3,}(?:ed|ing|ied))\b', doc_text.lower()) or any(k in tokens for k in action_kw) else 0
    # (d) Numeric measurement or metric indicator: numeric specs with optional units/symbols
    has_metric = 1 if re.search(r'\b\d+(?:\.\d+)?(?:\s*[a-zA-Z%$/\^]+)?\b', doc_text) else 0
    # (e) Consequence / impact description: evaluative outcomes or impact keywords
    has_impact = 1 if re.search(r'\b(damage|loss|injury|death|failure|severe|minor|major|status|result|impact|outcome|effect)\b', doc_text.lower()) or any(k in tokens for k in impact_kw) else 0

    structural_completeness = (has_subject + has_context + has_action + has_metric + has_impact) / 5.0

    # 9. Linguistic Diversity (Type-Token Ratio)
    linguistic_diversity = ttr

    # 10. Rare Domain Term Novelty (IDF sum of configured rare domain terms)
    novelty_sum = sum(math.log((total_docs + 1) / (term_freq_map.get(tok, 1) + 1)) for tok in tokens if tok in rare_terms)
    rare_domain_term_novelty = min(1.0, novelty_sum / 10.0)

    return {
        "domain_density": domain_density,
        "rare_term_count": rare_count,
        "rare_score": rare_score,
        "concept_diversity": concept_diversity,
        "entity_diversity": entity_diversity,
        "event_complexity": event_complexity,
        "information_density": info_density,
        "redundancy_penalty": redundancy_penalty,
        "structural_completeness": structural_completeness,
        "linguistic_diversity": linguistic_diversity,
        "rare_domain_term_novelty": rare_domain_term_novelty,
        "domain_novelty": rare_domain_term_novelty,
        "concepts": list(detected_concepts)
    }

def compute_raw_score(feats: dict, weights: dict = None) -> float:
    """Computes weighted linear combination score clipped to [0, 100]."""
    w = weights if weights is not None else FEATURE_WEIGHTS
    novelty_key = "rare_domain_term_novelty" if "rare_domain_term_novelty" in feats else "domain_novelty"
    raw = (
        w.get("domain_density", 0.30) * feats["domain_density"] +
        w.get("rare_score", 0.20) * feats["rare_score"] +
        w.get("concept_diversity", 0.15) * feats["concept_diversity"] +
        w.get("entity_diversity", 0.10) * feats["entity_diversity"] +
        w.get("event_complexity", 0.10) * feats["event_complexity"] +
        w.get("information_density", 0.10) * feats["information_density"] +
        w.get("structural_completeness", 0.05) * feats["structural_completeness"] +
        w.get("linguistic_diversity", 0.05) * feats["linguistic_diversity"] +
        w.get("rare_domain_term_novelty", 0.05) * feats[novelty_key] +
        w.get("redundancy_penalty", -0.10) * feats["redundancy_penalty"]
    )
    return float(np.clip(raw * 100.0, 0.0, 100.0))

def main():
    random.seed(42)
    np.random.seed(42)
    root = get_project_root()
    config = load_config()
    bench_cfg = get_benchmark_config()
    output_dir = root / config.get("output_dir", "outputs")

    categories = bench_cfg.get("categories", {})
    rare_terms = set(bench_cfg.get("rare_domain_terms", []))
    domain_lexicons = bench_cfg.get("domain_semantic_lexicons", {})

    # Resolve TXT corpus path (deterministic, no silent fallback)
    corpus_filename = bench_cfg.get("corpus_file", "corpus.txt")
    corpus_path = output_dir / corpus_filename

    allow_discovery = bench_cfg.get("allow_corpus_auto_discovery", False)
    dedup = bench_cfg.get("deduplicate_corpus", False)

    logger.info(f"Loading documents from text corpus: {corpus_path}...")
    documents = load_corpus_documents(corpus_path, deduplicate=dedup, allow_auto_discovery=allow_discovery)
    total_docs = len(documents)
    logger.info(f"Loaded {total_docs} documents.")

    logger.info("Pass 1: Computing vocabulary document frequencies for Rare Domain Term Novelty IDF...")
    term_freq = Counter()
    for doc_item in documents:
        doc_text = doc_item["document"].lower()
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', doc_text))
        term_freq.update(words)

    logger.info("Pass 2: Computing text-derived semantic importance scores...")
    scored_records = []

    for doc_item in tqdm(documents, desc="Scoring Documents"):
        doc_id = doc_item["doc_id"]
        doc_text = doc_item["document"]

        feats = compute_document_features(doc_text, categories, rare_terms, term_freq, total_docs, domain_lexicons)
        importance_score = compute_raw_score(feats)

        # Knowledge tier classification based on score thresholds
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
            "doc_id": doc_id,
            "importance_score": round(importance_score, 2),
            "knowledge_tier": knowledge_tier,
            "domain_density": round(feats["domain_density"], 4),
            "rare_term_count": feats["rare_term_count"],
            "concept_diversity": round(feats["concept_diversity"], 4),
            "information_density": round(feats["information_density"], 4),
            "structural_completeness": round(feats["structural_completeness"], 4),
            "rare_domain_term_novelty": round(feats["rare_domain_term_novelty"], 4),
            "concepts": feats["concepts"],
            "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items() if k != "concepts"},
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
    plt.title(f"Corpus Semantic Importance Score Distribution ({bench_cfg.get('domain_name', 'Domain').capitalize()})")
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

    sorted_records = sorted(scored_records, key=lambda x: x["importance_score"], reverse=True)
    target_sample_size = min(1000, len(scored_records))

    high_docs = sorted_records[:target_sample_size]
    valid_low = [r for r in sorted_records if r["knowledge_tier"] in ("Low Knowledge", "Noisy / Boilerplate")]
    low_docs = valid_low[-target_sample_size:] if len(valid_low) >= target_sample_size else sorted_records[-target_sample_size:]
    mid_start = (len(sorted_records) - target_sample_size) // 2
    med_docs = sorted_records[mid_start:mid_start + target_sample_size]

    # Deterministic balanced blending
    n_per_tier = target_sample_size // 3
    balanced_pool = high_docs[:n_per_tier] + med_docs[:n_per_tier] + low_docs[:n_per_tier]
    random.seed(42)
    random.shuffle(balanced_pool)

    # Deterministic random baseline
    random.seed(42)
    random_pool = random.sample(scored_records, min(target_sample_size, len(scored_records)))

    subset_map = {
        "high_knowledge": high_docs,
        "medium_knowledge": med_docs,
        "low_knowledge": low_docs,
        "balanced_knowledge": balanced_pool,
        "random_baseline": random_pool
    }

    # 1-5. Export Evaluation Subsets
    for name, docs in subset_map.items():
        with open(subsets_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in docs:
                f.write(json.dumps(r) + "\n")

    # 6. General English Baseline Subset
    with open(subsets_dir / "general_english_baseline.jsonl", "w", encoding="utf-8") as f:
        for idx, sent in enumerate(GENERAL_ENGLISH_SENTENCES):
            f.write(json.dumps({
                "doc_id": f"gen_eng_{idx+1:04d}",
                "importance_score": 10.0,
                "knowledge_tier": "General English",
                "document": sent
            }) + "\n")

    # Methodological Hardening: Subset Overlap Analysis
    subset_ids = {name: set(r["doc_id"] for r in docs) for name, docs in subset_map.items()}
    overlap_report = {
        "target_sample_size": target_sample_size,
        "subsets_document_counts": {name: len(docs) for name, docs in subset_map.items()},
        "pairwise_overlap_counts": {},
        "methodological_notes": (
            "High, Medium, and Low subsets are constructed from score distribution partitions. "
            "Balanced Knowledge intentionally samples 1/3 from each tier. "
            "Random Baseline is sampled uniformly without replacement from the full corpus. "
            "Document IDs are tracked deterministically across all subsets."
        )
    }

    names = list(subset_map.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            overlap = len(subset_ids[n1].intersection(subset_ids[n2]))
            overlap_report["pairwise_overlap_counts"][f"{n1}__vs__{n2}"] = {
                "overlap_count": overlap,
                "overlap_pct_of_first": round(overlap / len(subset_ids[n1]) * 100, 2) if subset_ids[n1] else 0.0
            }

    with open(output_dir / "subset_overlap_statistics.json", "w", encoding="utf-8") as f:
        json.dump(overlap_report, f, indent=2)

    logger.info(f"Stage 12 completed successfully. Saved importance scores, subsets, and subset overlap analysis in {output_dir}")

if __name__ == "__main__":
    main()
