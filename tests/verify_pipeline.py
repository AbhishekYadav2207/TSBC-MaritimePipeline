import os
import json
import re
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def check_file_exists_and_nonempty(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"[ERROR] File does not exist: {file_path}")
        return False
    if file_path.stat().st_size == 0:
        print(f"[ERROR] File is empty: {file_path}")
        return False
    print(f"[OK] File exists and is non-empty: {file_path.name}")
    return True

def test_pipeline_outputs():
    root = get_project_root()
    output_dir = root / "outputs"
    
    print("\n==================================================")
    # List of required pipeline outputs
    required_files = [
        "dictionary_metadata.json",
        "profiling_report.json",
        "relationships.json",
        "selected_semantic_columns.json",
        "merged_records.jsonl",
        "validation_report.json",
        "raw_documents.jsonl",
        "clean_documents.jsonl",
        "maritime_corpus.txt",
        "maritime_corpus.jsonl",
        "maritime_vocabulary.txt",
        "statistics.json",
        "tokenizer_analysis.json",
        "manifest.json",
        "corpus_quality_report.md"
    ]
    
    all_exist = True
    for f_name in required_files:
        path = output_dir / f_name
        if not check_file_exists_and_nonempty(path):
            all_exist = False
            
    if not all_exist:
        print("[FAIL] Pipeline Verification Failed: Missing or empty output files.")
        return False
        
    print("\n==================================================")
    print("Validating file schemas and formats...")
    
    # 1. Validate manifest.json
    try:
        with open(output_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        required_manifest_keys = {"version", "created", "documents", "source", "language", "pipeline_version", "git_commit"}
        assert required_manifest_keys.issubset(manifest.keys()), "Manifest keys missing"
        print("[OK] manifest.json schema is valid.")
    except Exception as e:
        print(f"[FAIL] manifest.json validation failed: {e}")
        return False
        
    # 2. Validate clean_documents.jsonl matches manifest documents count
    try:
        clean_count = 0
        with open(output_dir / "clean_documents.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                clean_count += 1
                record = json.loads(line)
                assert "occurrence_id" in record
                assert "document" in record
                assert "structured" in record
        assert clean_count == manifest["documents"], f"Documents count mismatch: JSONL has {clean_count}, manifest has {manifest['documents']}"
        print(f"[OK] clean_documents.jsonl count matches manifest: {clean_count} documents.")
    except Exception as e:
        print(f"[FAIL] clean_documents.jsonl validation failed: {e}")
        return False
        
    # 3. Validate plain text corpus format (Document separator should be blank line)
    try:
        with open(output_dir / "maritime_corpus.txt", "r", encoding="utf-8") as f:
            content = f.read()
        # Splits should be double newlines
        docs = [d for d in content.split("\n\n") if d.strip()]
        assert len(docs) == clean_count, f"Corpus text document count mismatch: txt split has {len(docs)}, expected {clean_count}"
        print("[OK] maritime_corpus.txt is properly formatted with blank lines.")
    except Exception as e:
        print(f"[FAIL] maritime_corpus.txt validation failed: {e}")
        return False
        
    # 4. Validate statistics.json
    try:
        with open(output_dir / "statistics.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
        assert stats["total_documents"] == clean_count, "Stats total documents mismatch"
        assert "vocabulary_size" in stats
        assert "shannon_entropy" in stats
        assert "type_token_ratio_lexical_diversity" in stats
        print("[OK] statistics.json schema is valid.")
    except Exception as e:
        print(f"[FAIL] statistics.json validation failed: {e}")
        return False
        
    # 5. Validate vocabulary file
    try:
        with open(output_dir / "maritime_vocabulary.txt", "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
        assert len(vocab) > 0, "Vocabulary file is empty"
        print(f"[OK] maritime_vocabulary.txt loaded successfully with {len(vocab)} words.")
    except Exception as e:
        print(f"[FAIL] maritime_vocabulary.txt validation failed: {e}")
        return False
        
    print("\n==================================================")
    print("[SUCCESS] ALL PIPELINE OUTPUTS VERIFIED SUCCESSFULLY!")
    print("==================================================")
    return True

if __name__ == "__main__":
    test_pipeline_outputs()
