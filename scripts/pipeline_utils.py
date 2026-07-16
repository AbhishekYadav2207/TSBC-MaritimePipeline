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
