import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from pipeline_utils import setup_logging, detect_datasets, read_csv_safe, get_project_root, load_config

logger = setup_logging("02_profile_dataset")

def profile_table(file_path: Path) -> dict:
    """Profiles a single CSV dataset table."""
    logger.info(f"Profiling table: {file_path.name}")
    
    # Read in chunks to estimate row counts if large, or just read the first chunk to inspect dtypes
    # For a robust profiling, we can load the full CSV, but to prevent memory bloat, we do it in a resource-conscious way.
    # Since files are 1-100MB, pandas handles them easily.
    try:
        # Load the full file (or we could use chunks, but full load gives exact unique counts)
        df = read_csv_safe(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path.name}: {e}")
        return {}
        
    num_rows = len(df)
    num_cols = len(df.columns)
    
    col_profiles = {}
    possible_pks = []
    
    for col in df.columns:
        col_series = df[col]
        null_count = int(col_series.isnull().sum())
        nunique = int(col_series.nunique())
        dtype_str = str(col_series.dtype)
        
        cardinality_ratio = nunique / num_rows if num_rows > 0 else 0
        
        col_profiles[col] = {
            "null_count": null_count,
            "nunique": nunique,
            "cardinality_ratio": cardinality_ratio,
            "data_type": dtype_str
        }
        
        # Invariant for Primary Key candidate: 0 nulls and unique values equals total rows
        # Sometimes there's minor noise or duplicate keys, so we check if nunique == num_rows and null_count == 0
        if null_count == 0 and nunique == num_rows and num_rows > 0:
            possible_pks.append(col)
            
    # If no exact primary key matches, let's look for columns with 0 nulls and high cardinality
    if not possible_pks:
        for col, metrics in col_profiles.items():
            if metrics["null_count"] == 0 and metrics["cardinality_ratio"] > 0.95 and num_rows > 0:
                possible_pks.append(col)
                
    return {
        "file_name": file_path.name,
        "row_count": num_rows,
        "col_count": num_cols,
        "columns": col_profiles,
        "possible_pks": possible_pks
    }

def infer_foreign_keys(profile_report: dict) -> dict:
    """Infers potential foreign key relationships between tables based on column names and types."""
    # Find overlapping columns between tables
    table_names = list(profile_report.keys())
    inferred_fks = {t: [] for t in table_names}
    
    # We look for columns ending in ID, No, Key, or matching PKs of other tables
    for t1 in table_names:
        pks_other = []
        for t2 in table_names:
            if t1 == t2:
                continue
            pks_other.extend([(t2, pk) for pk in profile_report[t2]["possible_pks"]])
            
        t1_cols = profile_report[t1]["columns"]
        for col1, metrics1 in t1_cols.items():
            col1_lower = col1.lower()
            
            # Check if this column name matches a PK in another table
            for t2, pk2 in pks_other:
                if col1.lower() == pk2.lower():
                    # Check if the column is NOT a PK in t1 (indicating a FK relationship)
                    if col1 not in profile_report[t1]["possible_pks"]:
                        inferred_fks[t1].append({
                            "column": col1,
                            "referenced_table": t2,
                            "referenced_column": pk2,
                            "reason": "Column name matches primary key of referenced table"
                        })
                        break
            else:
                # If name doesn't match PK exactly, check if it contains ID/No and matches a column name in another table
                if any(k in col1_lower for k in ["id", "no", "key"]):
                    if col1 not in profile_report[t1]["possible_pks"]:
                        for t2 in table_names:
                            if t1 == t2:
                                continue
                            if col1 in profile_report[t2]["columns"]:
                                # If it's a PK in t2 or has high cardinality in t2
                                if col1 in profile_report[t2]["possible_pks"] or profile_report[t2]["columns"][col1]["cardinality_ratio"] > 0.8:
                                    inferred_fks[t1].append({
                                        "column": col1,
                                        "referenced_table": t2,
                                        "referenced_column": col1,
                                        "reason": f"Column '{col1}' exists and is unique/PK in '{t2}'"
                                    })
                                    break
                                    
    return inferred_fks

def main():
    root = get_project_root()
    detected = detect_datasets()
    datasets = detected["datasets"]
    
    if not datasets:
        logger.error("No datasets found in the data directory!")
        return
        
    logger.info(f"Found {len(datasets)} tables to profile.")
    
    report = {}
    for table_name, file_path in tqdm(datasets.items(), desc="Profiling Datasets"):
        profile = profile_table(file_path)
        if profile:
            report[table_name] = profile
            
    # Infer foreign keys
    logger.info("Inferring foreign key relationships...")
    fks = infer_foreign_keys(report)
    for table_name, list_fks in fks.items():
        report[table_name]["inferred_fks"] = list_fks
        logger.info(f"  Table '{table_name}': found {len(list_fks)} foreign key candidates.")
        
    # Write output
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = output_dir / "profiling_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Profiling report saved successfully to {out_path}")

if __name__ == "__main__":
    main()
