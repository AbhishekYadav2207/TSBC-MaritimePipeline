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
        
    vocab_terms = []
    with open(vocab_path, "r", encoding="utf-8") as fv:
        vocab_terms = [line.strip() for line in fv if line.strip()]
        
    logger.info("Loading bert-base-uncased tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 1. Analyze maritime terms splits
    logger.info("Analyzing BERT tokenization splits on maritime vocabulary terms...")
    vocab_splits = []
    split_count = 0
    total_pieces = 0
    
    for term in vocab_terms:
        tokens = tokenizer.tokenize(term)
        num_pieces = len(tokens)
        total_pieces += num_pieces
        if num_pieces > 1:
            split_count += 1
            
        vocab_splits.append({
            "term": term,
            "tokens": tokens,
            "num_pieces": num_pieces
        })
        
    vocab_splits.sort(key=lambda x: x["num_pieces"], reverse=True)
    maritime_frag_rate = split_count / len(vocab_terms) if vocab_terms else 0.0
    
    # 2. Process sample of corpus for subword fertility and sequence lengths
    logger.info("Analyzing subwords-per-word ratio and sequence length distribution...")
    
    total_raw_words = 0
    total_subword_tokens = 0
    total_unk_tokens = 0
    sampled_docs = 0
    max_sampled_docs = 2000
    
    seq_under_128 = 0
    seq_under_256 = 0
    seq_under_512 = 0
    seq_over_512 = 0
    
    with open(corpus_jsonl_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=max_sampled_docs, desc="Analyzing Tokenization"):
            if sampled_docs >= max_sampled_docs:
                break
                
            record = json.loads(line)
            doc_text = record["document"]
            
            raw_words = len(doc_text.split())
            if raw_words == 0:
                continue
                
            bert_tokens = tokenizer.tokenize(doc_text)
            num_tokens = len(bert_tokens)
            
            total_raw_words += raw_words
            total_subword_tokens += num_tokens
            total_unk_tokens += bert_tokens.count(tokenizer.unk_token)
            
            if num_tokens <= 128: seq_under_128 += 1
            if num_tokens <= 256: seq_under_256 += 1
            if num_tokens <= 512: seq_under_512 += 1
            else: seq_over_512 += 1
            
            sampled_docs += 1
            
    fertility = total_subword_tokens / total_raw_words if total_raw_words > 0 else 0.0
    oov_rate = total_unk_tokens / total_subword_tokens if total_subword_tokens > 0 else 0.0
    
    analysis_output = {
        "model_name": "bert-base-uncased",
        "sampled_documents": sampled_docs,
        "total_raw_words_analyzed": total_raw_words,
        "total_subword_tokens_analyzed": total_subword_tokens,
        "average_subwords_per_word": fertility,
        "maritime_fragmentation_rate": maritime_frag_rate,
        "oov_rate": oov_rate,
        "sequence_length_distribution": {
            "under_128": seq_under_128,
            "under_256": seq_under_256,
            "under_512": seq_under_512,
            "over_512": seq_over_512
        },
        "worst_fragmented_terms": vocab_splits[:15],
        "maritime_vocabulary_splits": vocab_splits[:50]
    }
    
    out_path = output_dir / "tokenizer_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis_output, f, indent=2)
        
    logger.info(f"Tokenizer analysis completed successfully. Saved to {out_path}")
    logger.info(f"  BERT Subwords/Word Fertility: {fertility:.4f}")
    logger.info(f"  Maritime Fragmentation Rate: {maritime_frag_rate*100:.2f}%")

if __name__ == "__main__":
    main()
