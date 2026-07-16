import os
import json
import re
import math
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("09_statistics")

def clean_and_tokenize(text: str) -> list:
    """Tokenizes text into lowercase words, stripping punctuation."""
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    return text_clean.split()

def compute_shannon_entropy(words: list) -> float:
    """Computes the Shannon entropy of a list of words."""
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
    """Generates contiguous n-grams from a list of words."""
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents not found at {clean_path}! Run Step 7 first.")
        return
        
    val_path = output_dir / "validation_report.json"
    validation_data = {}
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as fv:
            validation_data = json.load(fv)
            
    logger.info("Computing corpus statistics...")
    
    all_words = []
    doc_word_counts = []
    total_docs = 0
    total_chars = 0
    
    all_sentences = []
    all_paragraphs = []
    
    sample_docs = []
    
    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Analyzing Corpus"):
            record = json.loads(line)
            doc_text = record["document"]
            oid = record["occurrence_id"]
            
            total_docs += 1
            total_chars += len(doc_text)
            
            # Sentence/paragraph metrics
            paras = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
            all_paragraphs.extend(paras)
            
            # Simple sentence split for statistics
            for p in paras:
                # split by . ? ! followed by space
                sents = [s.strip() for s in re.split(r'[\.\?\!]\s+', p) if s.strip()]
                all_sentences.extend(sents)
                
            # Tokenization for vocab
            words = clean_and_tokenize(doc_text)
            all_words.extend(words)
            doc_word_counts.append(len(words))
            
            # Save a few sample documents for the markdown report
            if total_docs <= 3:
                sample_docs.append({
                    "id": oid,
                    "text": doc_text
                })
                
    total_words = len(all_words)
    vocab_counter = Counter(all_words)
    vocab_size = len(vocab_counter)
    
    avg_words = np.mean(doc_word_counts) if doc_word_counts else 0
    median_words = np.median(doc_word_counts) if doc_word_counts else 0
    
    # Bigrams & Trigrams
    logger.info("Generating n-grams...")
    bigrams = []
    trigrams = []
    
    # Process ngrams document by document to avoid joining boundaries
    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in fin:
            doc = json.loads(line)["document"]
            words = clean_and_tokenize(doc)
            bigrams.extend(get_n_grams(words, 2))
            trigrams.extend(get_n_grams(words, 3))
            
    bigram_counter = Counter(bigrams)
    trigram_counter = Counter(trigrams)
    
    # Duplicate sentence & paragraph ratios
    unique_sentences = len(set(s.lower().replace(" ", "") for s in all_sentences))
    duplicate_sentence_ratio = 1.0 - (unique_sentences / len(all_sentences)) if all_sentences else 0.0
    
    unique_paragraphs = len(set(p.lower().replace(" ", "") for p in all_paragraphs))
    duplicate_paragraph_ratio = 1.0 - (unique_paragraphs / len(all_paragraphs)) if all_paragraphs else 0.0
    
    # Lexical Diversity & Entropy
    ttr = vocab_size / total_words if total_words > 0 else 0.0
    entropy = compute_shannon_entropy(all_words)
    
    # Document length distribution (histogram bins)
    hist_bins = [0, 50, 100, 200, 500, 1000, 10000]
    hist_counts, _ = np.histogram(doc_word_counts, bins=hist_bins)
    hist_distribution = {
        f"{hist_bins[i]}-{hist_bins[i+1]} words": int(hist_counts[i]) for i in range(len(hist_bins)-1)
    }
    
    stats_output = {
        "total_documents": total_docs,
        "total_characters": total_chars,
        "total_tokens_words": total_words,
        "average_document_length_words": float(avg_words),
        "median_document_length_words": float(median_words),
        "vocabulary_size": vocab_size,
        "type_token_ratio_lexical_diversity": ttr,
        "shannon_entropy": entropy,
        "duplicate_sentence_ratio": duplicate_sentence_ratio,
        "duplicate_paragraph_ratio": duplicate_paragraph_ratio,
        "document_length_distribution": hist_distribution,
        "top_50_terms": vocab_counter.most_common(50),
        "top_10_bigrams": bigram_counter.most_common(10),
        "top_10_trigrams": trigram_counter.most_common(10)
    }
    
    # Save statistics.json
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2)
        
    logger.info(f"Statistics saved successfully to {stats_path}")
    
    # Generate human-readable corpus_quality_report.md
    logger.info("Generating corpus quality report...")
    
    # Format top terms, bigrams, trigrams as tables
    top_terms_table = "| Rank | Term | Frequency |\n|---|---|---|\n"
    for i, (term, freq) in enumerate(vocab_counter.most_common(15), 1):
        top_terms_table += f"| {i} | {term} | {freq} |\n"
        
    top_bigrams_table = "| Rank | Bigram | Frequency |\n|---|---|---|\n"
    for i, (bg, freq) in enumerate(bigram_counter.most_common(10), 1):
        top_bigrams_table += f"| {i} | {bg} | {freq} |\n"
        
    top_trigrams_table = "| Rank | Trigram | Frequency |\n|---|---|---|\n"
    for i, (tg, freq) in enumerate(trigram_counter.most_common(10), 1):
        top_trigrams_table += f"| {i} | {tg} | {freq} |\n"
        
    # Read validation counts if available
    status = validation_data.get("status", "N/A")
    val_warnings_count = validation_data.get("total_value_warnings", 0)
    orphan_counts = validation_data.get("orphan_counts", {})
    
    orphan_table = "| Relationship | Count |\n|---|---|\n"
    for rel, count in orphan_counts.items():
        orphan_table += f"| {rel.replace('_', ' ').title()} | {count} |\n"
        
    # Formatting samples
    samples_md = ""
    for sd in sample_docs:
        samples_md += f"### Occurrence ID: {sd['id']}\n```text\n{sd['text']}\n```\n\n"
        
    report_md = f"""# Maritime Accident Corpus Quality Report

This report summarizes the size, linguistic quality, and validation health of the generated maritime NLP pretraining corpus.

## 1. Corpus Summary Metrics

* **Total Documents (Occurrences)**: {total_docs}
* **Total Words (Tokens)**: {total_words}
* **Total Characters**: {total_chars}
* **Average Document Length**: {avg_words:.2f} words
* **Median Document Length**: {median_words:.2f} words
* **Vocabulary Size**: {vocab_size} unique words
* **Type-Token Ratio (Lexical Diversity)**: {ttr:.5f}
* **Shannon Entropy**: {entropy:.4f} bits

### Document Length Distribution
* **0-50 words**: {hist_distribution.get('0-50 words', 0)} documents
* **50-100 words**: {hist_distribution.get('50-100 words', 0)} documents
* **100-200 words**: {hist_distribution.get('100-200 words', 0)} documents
* **200-500 words**: {hist_distribution.get('200-500 words', 0)} documents
* **500-1000 words**: {hist_distribution.get('500-1000 words', 0)} documents
* **1000+ words**: {hist_distribution.get('1000-10000 words', 0)} documents

---

## 2. Redundancy & Deduplication Metrics

* **Duplicate Sentence Ratio**: {duplicate_sentence_ratio * 100:.2f}%
* **Duplicate Paragraph Ratio**: {duplicate_paragraph_ratio * 100:.2f}%

---

## 3. Top N-Grams

### Top 15 Terms
{top_terms_table}

### Top 10 Bigrams
{top_bigrams_table}

### Top 10 Trigrams
{top_trigrams_table}

---

## 4. Data Validation and Join Integrity

* **Validation Status**: **{status}**
* **Value Consistency Warnings**: {val_warnings_count}

### Raw Table Joining Orphan Statistics
{orphan_table}

---

## 5. Sample Generated Documents

{samples_md}
"""

    report_path = output_dir / "corpus_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as fr:
        fr.write(report_md)
        
    logger.info(f"Human-readable quality report written to {report_path}")

if __name__ == "__main__":
    main()
