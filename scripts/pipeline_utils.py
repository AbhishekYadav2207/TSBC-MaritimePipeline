import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # Since this file is in scripts/, the project root is its parent
    return Path(__file__).resolve().parent.parent

def load_config() -> dict:
    """Loads configuration from config/config.json."""
    root = get_project_root()
    config_path = root / "config" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_logging(stage_name: str) -> logging.Logger:
    """Sets up a logger that outputs to both console and a log file."""
    config = load_config()
    root = get_project_root()
    
    log_dir = root / config.get("log_dir", "outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / config.get("log_file", "pipeline.log")
    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger(stage_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def read_csv_safe(file_path: Path, **kwargs) -> pd.DataFrame:
    """Reads a CSV file with automatic encoding detection, strips BOM, and filters usecols if specified."""
    # Handle usecols filtering to prevent crashes if columns are missing or misspelled
    usecols = kwargs.get("usecols", None)
    
    # Try reading headers first to validate columns
    encoding = 'utf-8-sig'
    try:
        header_df = pd.read_csv(file_path, encoding=encoding, nrows=0)
        actual_cols = [c.strip() for c in header_df.columns]
    except Exception:
        encoding = 'latin-1'
        try:
            header_df = pd.read_csv(file_path, encoding=encoding, nrows=0)
            actual_cols = [c.strip() for c in header_df.columns]
        except Exception as e:
            raise IOError(f"Could not read headers of CSV file {file_path}: {e}")
            
    if usecols is not None:
        # Keep only columns that exist in the CSV (case-sensitive check)
        valid_usecols = [c for c in usecols if c in actual_cols]
        # If case-sensitive failed, check case-insensitive
        actual_cols_lower = {c.lower(): c for c in actual_cols}
        for c in usecols:
            if c not in valid_usecols and c.lower() in actual_cols_lower:
                valid_usecols.append(actual_cols_lower[c.lower()])
                
        if not valid_usecols:
            # Fallback to no usecols if none are found, or keep join keys
            logger = logging.getLogger("pipeline_utils")
            logger.warning(f"No selected columns found in {file_path.name}. Reading all columns.")
            kwargs.pop("usecols")
        else:
            kwargs["usecols"] = valid_usecols
            
    # Set default low_memory to False to prevent parser dtype warnings and IndexError bugs
    kwargs.setdefault("low_memory", False)
            
    try:
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        # fallback to latin-1 if we were trying utf-8-sig
        if encoding == 'utf-8-sig':
            try:
                df = pd.read_csv(file_path, encoding='latin-1', **kwargs)
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception as e2:
                raise IOError(f"Could not read CSV file {file_path} with utf-8-sig or latin-1: {e2}")
        raise IOError(f"Could not read CSV file {file_path}: {e}")

def detect_datasets() -> dict:
    """Auto-detects CSV datasets in the data directory and maps them to table names."""
    config = load_config()
    root = get_project_root()
    data_dir = root / config.get("data_dir", "data")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
    csv_files = list(data_dir.glob("*.csv"))
    mapping = {}
    
    # Standard table name stems from the data dictionary
    dictionary_stems = [
        "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC",
        "MDOTW_VW_OCCURRENCE_PUBLIC",
        "MDOTW_VW_OCCURRENCE_VESSEL_REC_EQUIPMENT_PUBLIC",
        "MDOTW_VW_INJURIES_PUBLIC",
        "MDOTW_VW_OCCURRENCE_VESSEL_LSA_EQUIPMENT_PUBLIC",
        "MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC"
    ]
    
    dictionary_file = None
    
    for f in csv_files:
        name = f.name
        # Check if this is the dictionary
        if "dictionary" in name.lower() or "inventory" in name.lower():
            dictionary_file = f
            continue
            
        # Match table stems
        matched = False
        for stem in dictionary_stems:
            # Table name appears somewhere inside the file name (case-insensitive)
            if stem.lower() in name.lower() or name.lower().replace("_", "").endswith(stem.lower().replace("_", "") + ".csv") or stem.replace("MDOTW_VW_", "") in name:
                mapping[stem] = f
                matched = True
                break
                
        # Fallback heuristic mapping if exact stem match fails
        if not matched:
            if "nav" in name.lower():
                mapping["MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC"] = f
            elif "rec" in name.lower():
                mapping["MDOTW_VW_OCCURRENCE_VESSEL_REC_EQUIPMENT_PUBLIC"] = f
            elif "lsa" in name.lower():
                mapping["MDOTW_VW_OCCURRENCE_VESSEL_LSA_EQUIPMENT_PUBLIC"] = f
            elif "injur" in name.lower():
                mapping["MDOTW_VW_INJURIES_PUBLIC"] = f
            elif "vessel" in name.lower():
                mapping["MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC"] = f
            elif "occurrence" in name.lower():
                mapping["MDOTW_VW_OCCURRENCE_PUBLIC"] = f
                
    if dictionary_file is None:
        # Search for any remaining file that might be the dictionary
        for f in csv_files:
            if f not in mapping.values():
                dictionary_file = f
                break
                
    return {
        "datasets": mapping,
        "dictionary": dictionary_file
    }

def get_benchmark_config() -> dict:
    """Returns the benchmark configuration section from config/config.json with safe defaults."""
    config = load_config()
    benchmark_cfg = config.get("benchmark", {})
    return {
        "domain_name": benchmark_cfg.get("domain_name", "generic"),
        "corpus_file": benchmark_cfg.get("corpus_file", "maritime_corpus.txt"),
        "vocabulary_file": benchmark_cfg.get("vocabulary_file", "maritime_vocabulary.txt"),
        "metric_name": benchmark_cfg.get("metric_name", "DUI"),
        "metric_display_name": benchmark_cfg.get("metric_display_name", "Domain Understanding Index (DUI)"),
        "categories": benchmark_cfg.get("categories", {}),
        "rare_domain_terms": benchmark_cfg.get("rare_domain_terms", []),
        "allow_corpus_auto_discovery": benchmark_cfg.get("allow_corpus_auto_discovery", False),
        "deduplicate_corpus": benchmark_cfg.get("deduplicate_corpus", False),
        "domain_semantic_lexicons": benchmark_cfg.get("domain_semantic_lexicons", {}),
        "domain_lint_patterns": benchmark_cfg.get("domain_lint_patterns", [])
    }

def load_corpus_documents(
    corpus_path: Path,
    doc_delimiter: str = "\n\n",
    deduplicate: bool = False,
    allow_auto_discovery: bool = False
) -> list:
    """
    Canonical interface loader for Stage 11+ benchmarking.
    Loads documents from a plain-text corpus file (*.txt).
    Splits on double newlines by default.

    Duplicate semantics:
      - By default (deduplicate=False), all non-empty documents are preserved.
        Each document is assigned a unique, deterministic doc_id, and flagged with 'is_duplicate'.
      - If deduplicate=True, identical text boundaries are filtered out.

    Missing corpus semantics:
      - If corpus_path is missing and allow_auto_discovery=False (default), raises FileNotFoundError.
      - If allow_auto_discovery=True, attempts fallback to any discovered *_corpus.txt.
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        if allow_auto_discovery:
            parent = corpus_path.parent
            fallbacks = list(parent.glob("*_corpus.txt")) if parent.exists() else []
            if fallbacks:
                corpus_path = fallbacks[0]
            else:
                raise FileNotFoundError(f"Configured corpus file not found: {corpus_path}")
        else:
            raise FileNotFoundError(
                f"Configured corpus file not found: {corpus_path}. "
                "Automatic discovery is disabled. Set 'benchmark.allow_corpus_auto_discovery': true to enable fallback."
            )

    with open(corpus_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split documents by delimiter
    raw_docs = content.split(doc_delimiter) if doc_delimiter else content.splitlines()
    
    docs = []
    seen_hashes = set()
    import hashlib

    doc_idx = 1
    for raw in raw_docs:
        text = raw.strip()
        if not text:
            continue
        
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        is_dup = doc_hash in seen_hashes

        if is_dup and deduplicate:
            continue

        seen_hashes.add(doc_hash)
        
        doc_id = f"doc_{doc_idx:06d}"
        docs.append({
            "doc_id": doc_id,
            "document": text,
            "occurrence_id": doc_id,  # Backwards compatibility alias
            "is_duplicate": is_dup
        })
        doc_idx += 1

    return docs


