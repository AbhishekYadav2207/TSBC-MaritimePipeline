import os
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("11_tokenizer_analysis")

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    corpus_jsonl_path = output_dir / "maritime_corpus.jsonl"
    if not corpus_jsonl_path.exists():
        logger.error(f"Corpus JSONL not found at {corpus_jsonl_path}! Run Step 8 first.")
        return
        
    vocab_path = output_dir / "maritime_vocabulary.txt"
    if not vocab_path.exists():
        logger.error(f"Maritime vocabulary not found at {vocab_path}! Run Step 10 first.")
        return
        
    # Read top vocab words
    vocab_words = []
    with open(vocab_path, "r", encoding="utf-8") as fv:
        # Read first 50 words
        vocab_words = [line.strip() for line in fv if line.strip()][:50]
        
    logger.info("Loading bert-base-uncased tokenizer from HuggingFace/Cache...")
    # This might download the tokenizer files if not already cached
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 1. Analyze maritime terms splits
    logger.info("Analyzing how BERT splits maritime vocabulary terms...")
    vocab_splits = []
    for word in vocab_words:
        tokens = tokenizer.tokenize(word)
        vocab_splits.append({
            "term": word,
            "tokens": tokens,
            "num_pieces": len(tokens)
        })
        
    # 2. Process sample of corpus to analyze subword ratio and OOV rate
    logger.info("Analyzing subwords-per-word ratio and OOV (unknown token) rate on a sample of the corpus...")
    
    total_words = 0
    total_subwords = 0
    total_unk_tokens = 0
    num_sampled_docs = 0
    
    # Analyze up to 1000 documents to be fast and efficient
    max_sampled_docs = 1000
    
    with open(corpus_jsonl_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=max_sampled_docs, desc="Analyzing Tokenization"):
            if num_sampled_docs >= max_sampled_docs:
                break
                
            record = json.loads(line)
            doc_text = record["document"]
            
            # Count raw whitespace words
            raw_words_count = len(doc_text.split())
            if raw_words_count == 0:
                continue
                
            # Tokenize using BERT
            bert_tokens = tokenizer.tokenize(doc_text)
            
            total_words += raw_words_count
            total_subwords += len(bert_tokens)
            total_unk_tokens += bert_tokens.count(tokenizer.unk_token)
            num_sampled_docs += 1
            
    avg_subwords_per_word = total_subwords / total_words if total_words > 0 else 0.0
    oov_rate = total_unk_tokens / total_subwords if total_subwords > 0 else 0.0
    
    analysis_output = {
        "model_name": "bert-base-uncased",
        "sampled_documents": num_sampled_docs,
        "total_raw_words_analyzed": total_words,
        "total_subword_tokens_analyzed": total_subwords,
        "average_subwords_per_word": avg_subwords_per_word,
        "unk_token_count": total_unk_tokens,
        "oov_rate": oov_rate,
        "maritime_vocabulary_splits": vocab_splits
    }
    
    out_path = output_dir / "tokenizer_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis_output, f, indent=2)
        
    logger.info(f"Tokenizer analysis completed successfully. Saved to {out_path}")
    logger.info(f"  BERT Subwords/Word Ratio: {avg_subwords_per_word:.4f}")
    logger.info(f"  BERT UNK/OOV Rate: {oov_rate:.6f}")

if __name__ == "__main__":
    main()
