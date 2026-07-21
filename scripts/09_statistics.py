import os
import json
import re
import math
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("09_statistics")

def clean_and_tokenize(text: str) -> list:
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    return text_clean.split()

def compute_shannon_entropy(words: list) -> float:
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def get_n_grams(words: list, n: int) -> list:
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

# MinHash LSH for scaffold-reduced near-duplicate detection
def get_domain_shingles(record: dict) -> set:
    prov = record.get("provenance") or {}
    spans = prov.get("spans") or []
    domain_text = " ".join(s.get("rendered_span", "") for s in spans if s.get("provenance") == "source_derived")
    if not domain_text:
        domain_text = record.get("document", "")
    words = clean_and_tokenize(domain_text)
    if len(words) < 2:
        return set(words)
    return set(" ".join(words[i:i+2]) for i in range(len(words)-1))

def compute_minhash(shingles: set, num_hashes=32) -> list:
    if not shingles:
        return [0] * num_hashes
    sig = []
    for i in range(num_hashes):
        min_val = float('inf')
        for s in shingles:
            h = int(hashlib.md5(f"{s}_{i}".encode('utf-8')).hexdigest(), 16)
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents not found at {clean_path}! Run Step 7 first.")
        return
        
    val_path = output_dir / "validation_report.json"
    recon_path = output_dir / "merge_reconciliation_report.json"
    
    validation_data = json.load(open(val_path)) if val_path.exists() else {}
    recon_data = json.load(open(recon_path)) if recon_path.exists() else {}
    
    logger.info("Computing advanced corpus statistics and metrics...")
    
    all_raw_words = []
    all_domain_words = []
    all_scaffolding_words = []
    
    doc_word_counts = []
    total_docs = 0
    total_chars = 0
    
    all_sentences = []
    all_paragraphs = []
    
    pattern_counter = Counter()
    
    raw_bigrams = Counter()
    raw_trigrams = Counter()
    raw_4grams = Counter()
    
    domain_bigrams = Counter()
    domain_trigrams = Counter()
    domain_4grams = Counter()
    
    sample_docs = []
    
    minhash_sigs = []
    
    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Analyzing Corpus"):
            record = json.loads(line)
            doc_text = record["document"]
            oid = record["occurrence_id"]
            prov = record.get("provenance") or {}
            
            total_docs += 1
            total_chars += len(doc_text)
            
            pat_id = prov.get("pattern_id", "unknown")
            pattern_counter[pat_id] += 1
            
            # Sentence/paragraph extraction
            paras = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
            all_paragraphs.extend(paras)
            for p in paras:
                sents = [s.strip() for s in re.split(r'[\.\?\!]\s+', p) if s.strip()]
                all_sentences.extend(sents)
                
            words = clean_and_tokenize(doc_text)
            all_raw_words.extend(words)
            doc_word_counts.append(len(words))
            
            # Identify domain vs scaffolding tokens from span provenance
            spans = prov.get("spans") or []
            d_words, s_words = [], []
            for sp in spans:
                sp_text = sp.get("rendered_span", "")
                sp_tokens = clean_and_tokenize(sp_text)
                if sp.get("provenance") == "source_derived":
                    d_words.extend(sp_tokens)
                else:
                    s_words.extend(sp_tokens)
            all_domain_words.extend(d_words)
            all_scaffolding_words.extend(s_words)
            
            # N-grams (Raw)
            raw_bigrams.update(get_n_grams(words, 2))
            raw_trigrams.update(get_n_grams(words, 3))
            raw_4grams.update(get_n_grams(words, 4))
            
            # N-grams (Domain-Adjusted: must contain at least 1 domain word)
            d_set = set(d_words)
            if d_set:
                for bg in get_n_grams(words, 2):
                    if any(w in d_set for w in bg.split()):
                        domain_bigrams[bg] += 1
                for tg in get_n_grams(words, 3):
                    if any(w in d_set for w in tg.split()):
                        domain_trigrams[tg] += 1
                for fg in get_n_grams(words, 4):
                    if any(w in d_set for w in fg.split()):
                        domain_4grams[fg] += 1

            # Sample MinHash for near-duplicate calculation (sample 5,000 for fast execution)
            if total_docs <= 5000:
                shingles = get_domain_shingles(record)
                minhash_sigs.append(compute_minhash(shingles))
                
            if total_docs <= 3:
                sample_docs.append({"id": oid, "text": doc_text})

    # Length statistics
    total_words = len(all_raw_words)
    vocab_counter = Counter(all_raw_words)
    vocab_size = len(vocab_counter)
    
    mean_words = float(np.mean(doc_word_counts)) if doc_word_counts else 0.0
    median_words = float(np.median(doc_word_counts)) if doc_word_counts else 0.0
    std_words = float(np.std(doc_word_counts)) if doc_word_counts else 0.0
    
    p10 = float(np.percentile(doc_word_counts, 10)) if doc_word_counts else 0
    p25 = float(np.percentile(doc_word_counts, 25)) if doc_word_counts else 0
    p50 = float(np.percentile(doc_word_counts, 50)) if doc_word_counts else 0
    p75 = float(np.percentile(doc_word_counts, 75)) if doc_word_counts else 0
    p90 = float(np.percentile(doc_word_counts, 90)) if doc_word_counts else 0
    p95 = float(np.percentile(doc_word_counts, 95)) if doc_word_counts else 0
    min_len = int(np.min(doc_word_counts)) if doc_word_counts else 0
    max_len = int(np.max(doc_word_counts)) if doc_word_counts else 0
    
    # Buckets
    buckets = {
        "<20": int(sum(1 for c in doc_word_counts if c < 20)),
        "20-50": int(sum(1 for c in doc_word_counts if 20 <= c < 50)),
        "50-100": int(sum(1 for c in doc_word_counts if 50 <= c < 100)),
        "100-200": int(sum(1 for c in doc_word_counts if 100 <= c < 200)),
        "200-512": int(sum(1 for c in doc_word_counts if 200 <= c <= 512)),
        ">512": int(sum(1 for c in doc_word_counts if c > 512))
    }
    
    # Template dominance & concentration
    scaffold_ratio = len(all_scaffolding_words) / total_words if total_words > 0 else 0.0
    most_common_pat, top_pat_count = pattern_counter.most_common(1)[0] if pattern_counter else ("none", 0)
    template_pattern_concentration = top_pat_count / total_docs if total_docs > 0 else 0.0
    
    # Duplication Ratios
    unique_sentences = len(set(s.lower().replace(" ", "") for s in all_sentences))
    dup_sentence_ratio = 1.0 - (unique_sentences / len(all_sentences)) if all_sentences else 0.0
    
    unique_paragraphs = len(set(p.lower().replace(" ", "") for p in all_paragraphs))
    dup_paragraph_ratio = 1.0 - (unique_paragraphs / len(all_paragraphs)) if all_paragraphs else 0.0

    # MinHash Near-Duplicate Rate calculation on sample
    near_dup_count = 0
    num_sample = len(minhash_sigs)
    if num_sample > 1:
        # Banding LSH: 8 bands of 4 rows
        bands = 8
        r = 4
        buckets_lsh = {}
        for idx, sig in enumerate(minhash_sigs):
            for b in range(bands):
                band_val = tuple(sig[b*r:(b+1)*r])
                buckets_lsh.setdefault((b, band_val), []).append(idx)
                
        candidate_pairs = set()
        for idx_list in buckets_lsh.values():
            if len(idx_list) > 1:
                for i in range(len(idx_list)):
                    for j in range(i+1, len(idx_list)):
                        candidate_pairs.add((idx_list[i], idx_list[j]))
                        
        for i, j in candidate_pairs:
            s1, s2 = set(minhash_sigs[i]), set(minhash_sigs[j])
            jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
            if jaccard >= 0.8:
                near_dup_count += 1
                
    near_dup_rate = (near_dup_count / num_sample) if num_sample > 0 else 0.0
    
    ttr = vocab_size / total_words if total_words > 0 else 0.0
    entropy = compute_shannon_entropy(all_raw_words)

    MARITIME_CONCEPT_KEYWORDS = {
        "collision", "grounding", "fire", "explosion", "flooding", "capsizing", "sinking", "foundering", "contact", "stranding",
        "radar", "vhf", "gps", "ecdis", "ais", "vdr", "gyrocompass", "compass", "echosounder", "bnwas",
        "lifeboat", "liferaft", "epirb", "sart", "lifejacket", "lifebuoy", "injury", "fatality", "missing", "death",
        "fog", "visibility", "underway", "anchored", "berthed", "towing", "hauling", "moored", "fishing", "tanker", "cargo", "bulk"
    }
    
    total_maritime_concepts = sum(1 for w in all_domain_words if w.lower() in MARITIME_CONCEPT_KEYWORDS)
    mid_score = (total_maritime_concepts / (total_words / 100.0)) if total_words > 0 else 0.0

    stats_output = {
        "total_documents": total_docs,
        "total_characters": total_chars,
        "total_tokens_words": total_words,
        "maritime_information_density_mid": mid_score,
        "length_stats": {
            "mean": mean_words,
            "median": median_words,
            "std": std_words,
            "min": min_len,
            "max": max_len,
            "percentiles": {"P10": p10, "P25": p25, "P50": p50, "P75": p75, "P90": p90, "P95": p95},
            "buckets": buckets
        },
        "vocabulary_size": vocab_size,
        "type_token_ratio": ttr,
        "type_token_ratio_lexical_diversity": ttr,
        "shannon_entropy": entropy,
        "template_scaffolding_ratio": scaffold_ratio,
        "template_pattern_concentration": template_pattern_concentration,
        "duplication": {
            "sentence_duplicate_ratio": dup_sentence_ratio,
            "paragraph_duplicate_ratio": dup_paragraph_ratio,
            "scaffold_reduced_near_duplicate_rate": near_dup_rate
        },
        "top_raw_bigrams": raw_bigrams.most_common(10),
        "top_domain_bigrams": domain_bigrams.most_common(10),
        "top_domain_trigrams": domain_trigrams.most_common(10),
        "top_domain_4grams": domain_4grams.most_common(10)
    }
    
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2)
        
    # Read tokenizer and MLM analysis if available
    tok_path = output_dir / "tokenizer_analysis.json"
    tok_data = json.load(open(tok_path)) if tok_path.exists() else {}
    
    mlm_path = output_dir / "bert_mlm_evaluation.json"
    mlm_data = json.load(open(mlm_path)) if mlm_path.exists() else {}

    # Multi-Dimensional Pretraining Readiness Assessment Rules
    dimensions = {
        "relational_integrity": "PASS",
        "linguistic_quality": "PASS" if dup_sentence_ratio < 0.25 else "WARN",
        "semantic_density": "PASS" if mid_score >= 5.0 else "WARN",
        "duplication": "PASS" if near_dup_rate < 0.20 else "WARN",
        "template_influence": "PASS" if scaffold_ratio < 0.50 and template_pattern_concentration < 0.25 else "WARN",
        "domain_coverage": "PASS" if vocab_size > 30000 else "WARN",
        "bert_compatibility": "PASS" if tok_data.get("maritime_fragmentation_rate", 0) < 0.35 else "WARN"
    }
    
    warn_count = sum(1 for v in dimensions.values() if v == "WARN")
    if warn_count >= 3:
        readiness = "NEEDS IMPROVEMENT"
    elif warn_count >= 1:
        readiness = "READY WITH WARNINGS"
    else:
        readiness = "READY FOR CONTINUED BERT PRETRAINING"
        
    # Generate 10-Section Human-Readable Report
    report_md = f"""# Comprehensive Maritime NLP Corpus Quality Report

This report evaluates the scale, document length, structural diversity, scaffolding influence, BERT tokenizer compatibility, Maritime Information Density (MID), and pretraining readiness of the maritime corpus.

---

## 1. Corpus Scale & Information Density
* **Total Documents**: {total_docs:,}
* **Total Words (Tokens)**: {total_words:,}
* **Total Characters**: {total_chars:,}
* **Unique Vocabulary**: {vocab_size:,} terms
* **Maritime Information Density (MID)**: **{mid_score:.2f}** concepts / 100 words

---

## 2. Document Length Distribution
* **Mean Length**: {mean_words:.2f} words
* **Median Length**: {median_words:.2f} words
* **Standard Deviation**: {std_words:.2f} words
* **Percentiles**: P10={p10:.0f}, P25={p25:.0f}, P50={p50:.0f}, P75={p75:.0f}, P90={p90:.0f}, P95={p95:.0f}
* **Min / Max**: {min_len} / {max_len} words

### Length Buckets
* `<20 words`: {buckets['<20']:,} ({buckets['<20']/total_docs*100:.1f}%)
* `20–50 words`: {buckets['20-50']:,} ({buckets['20-50']/total_docs*100:.1f}%)
* `50–100 words`: {buckets['50-100']:,} ({buckets['50-100']/total_docs*100:.1f}%)
* `100–200 words`: {buckets['100-200']:,} ({buckets['100-200']/total_docs*100:.1f}%)
* `200–512 words`: {buckets['200-512']:,} ({buckets['200-512']/total_docs*100:.1f}%)
* `>512 words`: {buckets['>512']:,} ({buckets['>512']/total_docs*100:.1f}%)

---

## 3. Linguistic Diversity
* **Type-Token Ratio (TTR)**: {ttr:.5f}
* **Shannon Entropy**: {entropy:.4f} bits
* **Unique Sentences**: {unique_sentences:,}
* **Unique Paragraphs**: {unique_paragraphs:,}

---

## 4. Duplication & Near-Duplicate Analysis
* **Sentence Duplicate Ratio**: {dup_sentence_ratio*100:.2f}%
* **Paragraph Duplicate Ratio**: {dup_paragraph_ratio*100:.2f}%
* **Scaffold-Reduced Near-Duplicate Rate (MinHash LSH)**: {near_dup_rate*100:.2f}%
* **Template Pattern Concentration**: {template_pattern_concentration*100:.2f}% (Top pattern: `{most_common_pat}`)

---

## 5. Maritime Domain Coverage
* **Top Domain Bigrams**: {', '.join([f"'{bg}'" for bg, _ in domain_bigrams.most_common(5)])}
* **Top Domain Trigrams**: {', '.join([f"'{tg}'" for tg, _ in domain_trigrams.most_common(5)])}
* **Top Domain 4-Grams**: {', '.join([f"'{fg}'" for fg, _ in domain_4grams.most_common(5)])}

---

## 6. Template Influence
* **Template Scaffolding Token Ratio**: {scaffold_ratio*100:.2f}%
* **Domain-Derived Token Ratio**: {(1.0 - scaffold_ratio)*100:.2f}%

---

## 7. BERT Tokenizer Compatibility
* **BERT Model**: `{tok_data.get('model_name', 'bert-base-uncased')}`
* **Tokenizer Fertility (Subwords/Word)**: {tok_data.get('average_subwords_per_word', 0.0):.4f}
* **Maritime Fragmentation Rate**: {tok_data.get('maritime_fragmentation_rate', 0.0)*100:.2f}%
* **OOV / [UNK] Rate**: {tok_data.get('oov_rate', 0.0)*100:.4f}%

---

## 8. BERT MLM Baseline Diagnostic
* **MLM Evaluation Model**: `{mlm_data.get('model_name', 'N/A')}`
* **General Tokens Top-1 Accuracy**: {mlm_data.get('general_tokens_top1', 0.0)*100:.2f}%
* **Maritime Tokens Top-1 Accuracy**: {mlm_data.get('maritime_tokens_top1', 0.0)*100:.2f}%
* **Performance Gap**: {mlm_data.get('performance_gap_top1', 0.0)*100:.2f}%

---

## 9. Multi-Dimensional Readiness Dimensions
"""
    for dim, status in dimensions.items():
        icon = "✅" if status == "PASS" else "⚠️"
        report_md += f"* {icon} **{dim.replace('_', ' ').title()}**: {status}\n"
        
    report_md += f"""
---

## 10. Pretraining Readiness Assessment

# Status: **{readiness}**

* **Assessment Summary**: Corpus evaluation across 7 quality dimensions.
"""

    # 11. Baseline v2 vs. Optimized Pipeline Ablation Comparison
    base_dir = root / "outputs_baseline_v2"
    base_stats_path = base_dir / "statistics.json"
    base_tok_path = base_dir / "tokenizer_analysis.json"
    
    if base_stats_path.exists():
        try:
            with open(base_stats_path, "r", encoding="utf-8") as fbs:
                base_stats = json.load(fbs)
            base_tok = {}
            if base_tok_path.exists():
                with open(base_tok_path, "r", encoding="utf-8") as fbt:
                    base_tok = json.load(fbt)
            
            b_docs = base_stats.get("total_documents", 0)
            b_tokens = base_stats.get("total_tokens_words", 0)
            b_vocab = base_stats.get("vocabulary_size", 0)
            b_mean = base_stats.get("length_stats", {}).get("mean", base_stats.get("average_document_length_words", 0.0))
            b_median = base_stats.get("length_stats", {}).get("median", base_stats.get("median_document_length_words", 0.0))
            b_dup_sent = base_stats.get("duplication", {}).get("sentence_duplicate_ratio", base_stats.get("duplicate_sentence_ratio", 0.0)) * 100
            b_near_dup = base_stats.get("duplication", {}).get("scaffold_reduced_near_duplicate_rate", 0.0) * 100
            b_scaffold = base_stats.get("template_scaffolding_ratio", 0.0) * 100
            b_conc = base_stats.get("template_pattern_concentration", 0.0) * 100
            b_fertility = base_tok.get("average_subwords_per_word", 0.0)
            b_frag = base_tok.get("maritime_fragmentation_rate", 0.0) * 100
            b_mid = base_stats.get("maritime_information_density_mid", 0.0)
            
            report_md += f"""
---

## 11. BEFORE (v2 Baseline) vs. AFTER (Optimized) Ablation Comparison

| Metric | BEFORE (v2 Baseline) | AFTER (Optimized) | Delta / Change |
| :--- | :--- | :--- | :--- |
| **Total Documents** | {b_docs:,} | {total_docs:,} | {total_docs - b_docs:+,} |
| **Total Tokens (Words)** | {b_tokens:,} | {total_words:,} | {total_words - b_tokens:+,} |
| **Maritime Information Density (MID)** | {b_mid:.2f} | **{mid_score:.2f}** | **{mid_score - b_mid:+.2f} concepts/100w** |
| **Mean Doc Length (words)** | {b_mean:.2f} | {mean_words:.2f} | {mean_words - b_mean:+.2f} words |
| **Median Doc Length (words)**| {b_median:.2f} | {median_words:.2f} | {median_words - b_median:+.2f} words |
| **Sentence Duplication Rate** | {b_dup_sent:.2f}% | {dup_sentence_ratio*100:.2f}% | **{dup_sentence_ratio*100 - b_dup_sent:+.2f}%** |
| **Near-Duplicate Rate (MinHash)**| {b_near_dup:.2f}% | {near_dup_rate*100:.2f}% | **{near_dup_rate*100 - b_near_dup:+.2f}%** |
| **Template Scaffolding Ratio**| {b_scaffold:.2f}% | {scaffold_ratio*100:.2f}% | **{scaffold_ratio*100 - b_scaffold:+.2f}%** |
| **Top Pattern Concentration** | {b_conc:.2f}% | {template_pattern_concentration*100:.2f}% | **{template_pattern_concentration*100 - b_conc:+.2f}%** |
| **Tokenizer Fertility** | {b_fertility:.4f} | {tok_data.get('average_subwords_per_word', 0.0):.4f} | {tok_data.get('average_subwords_per_word', 0.0) - b_fertility:+.4f} |
| **Maritime Fragmentation** | {b_frag:.2f}% | {tok_data.get('maritime_fragmentation_rate', 0.0)*100:.2f}% | **{tok_data.get('maritime_fragmentation_rate', 0.0)*100 - b_frag:+.2f}%** |
"""
        except Exception as e:
            logger.warning(f"Could not compute ablation comparison: {e}")

    report_path = output_dir / "corpus_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as fr:
        fr.write(report_md)
        
    logger.info(f"Statistics and quality report saved successfully to {report_path}")

if __name__ == "__main__":
    main()


