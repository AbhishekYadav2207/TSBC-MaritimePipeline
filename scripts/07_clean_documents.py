import os
import json
import re
import hashlib
from pathlib import Path
from tqdm import tqdm
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("07_clean_documents")

def split_sentences(text: str) -> list:
    """Splits text into sentences using a regex pattern that ignores common abbreviations."""
    # Matches sentence endings (., ?, !) followed by space, avoiding splits on common abbreviations
    sentence_end = re.compile(r'(?<!\bCapt)(?<!\bCapt\.)(?<!\bMr)(?<!\bMr\.)(?<!\bU\.S\.)(?<!\bco\.)(?<!\bco)(?<!\bLtd\.)(?<!\bLtd)(?<!\bpe\.)(?<!\bpe)(?<=[\.\?\!])\s+')
    # Python re backreference issue fix: let's simplify lookbehind pattern to be safe
    # We can split by standard punctuation followed by space, and merge elements if they end with abbreviations.
    raw_splits = re.split(r'([\.\?\!]\s+)', text)
    sentences = []
    
    # Reassemble splits
    i = 0
    temp_sent = ""
    abbreviations = {"capt.", "capt", "mr.", "mr", "u.s.", "co.", "ltd.", "pe.", "p.e.", "no.", "id.", "i.m.o."}
    
    while i < len(raw_splits):
        chunk = raw_splits[i]
        if i % 2 == 1:
            # This is the punctuation + space
            temp_sent += chunk
            # Check if temp_sent ends with an abbreviation before appending
            last_word = temp_sent.split()[-1].lower() if temp_sent.split() else ""
            if last_word in abbreviations:
                # Keep accumulating
                pass
            else:
                sentences.append(temp_sent.strip())
                temp_sent = ""
        else:
            temp_sent += chunk
        i += 1
        
    if temp_sent.strip():
        sentences.append(temp_sent.strip())
        
    return sentences

def clean_text(text: str) -> str:
    """Cleans up whitespaces, repeated punctuation, and common encoding artifacts."""
    if not text:
        return ""
        
    # Replace unicode replacement characters (from encoding issues)
    text = text.replace("\uFFFD", " ")
    
    # Fix repeated punctuation (e.g., "...", ",,,", "??")
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\?+', '?', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,\.\?\!])', r'\1', text)
    
    # Normalize spaces
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def deduplicate_sentences(text: str) -> str:
    """Removes duplicate sentences within a document while preserving paragraph structure."""
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    seen_sentences = set()
    
    for para in paragraphs:
        sentences = split_sentences(para)
        para_sentences = []
        for sent in sentences:
            sent_clean = sent.strip().lower().rstrip(".").replace(" ", "")
            if not sent_clean:
                continue
            # Deduplicate
            if sent_clean not in seen_sentences:
                seen_sentences.add(sent_clean)
                para_sentences.append(sent.strip())
        if para_sentences:
            cleaned_paragraphs.append(" ".join(para_sentences))
            
    return "\n\n".join(cleaned_paragraphs)

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    raw_path = output_dir / "raw_documents.jsonl"
    if not raw_path.exists():
        logger.error(f"Raw documents not found at {raw_path}! Run Step 6 first.")
        return
        
    min_len = config.get("text_cleaning", {}).get("min_doc_length", 50)
    
    clean_path = output_dir / "clean_documents.jsonl"
    logger.info(f"Cleaning documents and exporting to {clean_path}...")
    
    seen_document_hashes = set()
    num_input = 0
    num_cleaned = 0
    num_duplicates = 0
    num_too_short = 0
    
    with open(raw_path, "r", encoding="utf-8") as fin, open(clean_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Cleaning Documents"):
            num_input += 1
            record = json.loads(line)
            doc_text = record["document"]
            
            # Clean text
            doc_clean = clean_text(doc_text)
            doc_clean = deduplicate_sentences(doc_clean)
            
            # Check length constraint
            if len(doc_clean) < min_len:
                num_too_short += 1
                continue
                
            # Document deduplication check using MD5
            doc_hash = hashlib.md5(doc_clean.encode("utf-8")).hexdigest()
            if doc_hash in seen_document_hashes:
                num_duplicates += 1
                continue
            seen_document_hashes.add(doc_hash)
            
            # Save clean record
            record["document"] = doc_clean
            fout.write(json.dumps(record) + "\n")
            num_cleaned += 1
            
    logger.info(f"Text cleaning completed:")
    logger.info(f"  Total occurrences processed: {num_input}")
    logger.info(f"  Cleaned documents exported: {num_cleaned}")
    logger.info(f"  Duplicates removed: {num_duplicates}")
    logger.info(f"  Too short documents filtered: {num_too_short}")

if __name__ == "__main__":
    main()
