import os
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("08_export_corpus")

def get_git_commit() -> str:
    """Gets the current git commit hash if available, otherwise returns 'unknown'."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents not found at {clean_path}! Run Step 7 first.")
        return
        
    corpus_txt_path = output_dir / "maritime_corpus.txt"
    corpus_jsonl_path = output_dir / "maritime_corpus.jsonl"
    
    logger.info("Exporting clean documents to final corpus files...")
    
    num_docs = 0
    
    with open(clean_path, "r", encoding="utf-8") as fin, \
         open(corpus_txt_path, "w", encoding="utf-8") as ftxt, \
         open(corpus_jsonl_path, "w", encoding="utf-8") as fjsonl:
         
        for line in tqdm(fin, desc="Exporting Corpus"):
            record = json.loads(line)
            doc_text = record["document"]
            oid = record["occurrence_id"]
            structured = record["structured"]
            
            # 1. Export plain text corpus (separated by a blank line)
            # Normalize carriage returns and replace any sequence of internal newlines with a single newline
            doc_text_clean = re.sub(r'\n+', '\n', doc_text.replace("\r\n", "\n").replace("\r", "\n"))
            ftxt.write(doc_text_clean + "\n\n")
            
            # 2. Export metadata-preserving JSONL
            output_obj = {
                "occurrence_id": oid,
                "document": doc_text,
                "structured": structured
            }
            fjsonl.write(json.dumps(output_obj) + "\n")
            
            num_docs += 1
            
    # Create manifest
    manifest = {
        "version": "1.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "documents": num_docs,
        "source": "MARSIS",
        "language": "English",
        "pipeline_version": "1.0",
        "git_commit": get_git_commit()
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, indent=2)
        
    logger.info(f"Corpus exported successfully:")
    logger.info(f"  Final text file: {corpus_txt_path}")
    logger.info(f"  Final JSONL file: {corpus_jsonl_path}")
    logger.info(f"  Manifest written: {manifest_path}")
    logger.info(f"  Total documents: {num_docs}")

if __name__ == "__main__":
    main()
