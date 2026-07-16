import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from pipeline_utils import setup_logging, detect_datasets, read_csv_safe, load_config, get_project_root

logger = setup_logging("05_merge_tables")

def aggregate_dataframe(df: pd.DataFrame, key_col: str, cols_meta: dict) -> pd.DataFrame:
    """Aggregates a DataFrame by a key column, handling duplicates by summing, taking max, or concatenating text."""
    agg_funcs = {}
    
    # Columns where multiple values might exist and we want to preserve all of them
    concat_cols = {
        "weatherconditiondisplayeng", 
        "reportedbydisplayeng", 
        "substantiallyinterestedstatedisplayeng",
        "activitytypedisplayeng"
    }
    
    # Identify how to aggregate each column
    for col in df.columns:
        if col == key_col:
            continue
            
        col_lower = col.lower()
        if col_lower in concat_cols:
            def custom_concat(series):
                valid_vals = [str(x).strip() for x in series.dropna().unique() if str(x).strip() not in ["", "nan", "NaN", "UNKNOWN"]]
                if not valid_vals:
                    return np.nan
                if len(valid_vals) == 1:
                    return valid_vals[0]
                return "; ".join(valid_vals)
            agg_funcs[col] = custom_concat
        else:
            # Cythonized 'first' is extremely fast
            agg_funcs[col] = "first"
            
    df_agg = df.groupby(key_col, as_index=False).agg(agg_funcs)
    return df_agg

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    # 1. Load selected columns and detected datasets
    sel_cols_path = output_dir / "selected_semantic_columns.json"
    if not sel_cols_path.exists():
        logger.error(f"Selected columns file not found at {sel_cols_path}! Run Step 4 first.")
        return
        
    with open(sel_cols_path, "r", encoding="utf-8") as f:
        selected_columns = json.load(f)
        
    detected = detect_datasets()
    datasets = detected["datasets"]
    
    # Helper to get all selected columns for a table
    def get_cols_to_read(table_name):
        meta = selected_columns[table_name]
        return list(set(
            meta["join_keys"] +
            meta["display_cols"] +
            meta["numeric_attrs"] +
            meta["boolean_attrs"] +
            meta["narrative_cols"] +
            meta["other_semantic"]
        ))
        
    # 2. Read and deduplicate parent Occurrence table
    occ_table_name = "MDOTW_VW_OCCURRENCE_PUBLIC"
    occ_cols = get_cols_to_read(occ_table_name)
    logger.info(f"Loading Occurrence table with {len(occ_cols)} columns...")
    df_occ = read_csv_safe(datasets[occ_table_name], usecols=occ_cols)
    
    logger.info(f"Deduplicating Occurrence table (original shape: {df_occ.shape})...")
    df_occ_agg = aggregate_dataframe(df_occ, "OccID", selected_columns[occ_table_name])
    logger.info(f"Occurrence table aggregated. Unique OccID count: {len(df_occ_agg)}")
    
    # 3. Read and deduplicate Vessel table
    vessel_table_name = "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC"
    vessel_cols = get_cols_to_read(vessel_table_name)
    logger.info(f"Loading Vessel table with {len(vessel_cols)} columns...")
    df_vessel = read_csv_safe(datasets[vessel_table_name], usecols=vessel_cols)
    
    logger.info(f"Deduplicating Vessel table (original shape: {df_vessel.shape})...")
    df_vessel_agg = aggregate_dataframe(df_vessel, "VesselID", selected_columns[vessel_table_name])
    logger.info(f"Vessel table aggregated. Unique VesselID count: {len(df_vessel_agg)}")
    
    # 4. Load child tables
    logger.info("Loading child tables...")
    
    # Injuries
    inj_table = "MDOTW_VW_INJURIES_PUBLIC"
    df_inj = read_csv_safe(datasets[inj_table], usecols=get_cols_to_read(inj_table))
    
    # LSA Equipment
    lsa_table = "MDOTW_VW_OCCURRENCE_VESSEL_LSA_EQUIPMENT_PUBLIC"
    df_lsa = read_csv_safe(datasets[lsa_table], usecols=get_cols_to_read(lsa_table))
    
    # Nav Equipment
    nav_table = "MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC"
    df_nav = read_csv_safe(datasets[nav_table], usecols=get_cols_to_read(nav_table))
    
    # Rec Equipment
    rec_table = "MDOTW_VW_OCCURRENCE_VESSEL_REC_EQUIPMENT_PUBLIC"
    df_rec = read_csv_safe(datasets[rec_table], usecols=get_cols_to_read(rec_table))
    
    # 5. Group child tables by VesselID for O(1) nested lookup
    # We clean nan values to keep json clean
    def clean_dict(d):
        return {k: (None if pd.isna(v) or v == "nan" else v) for k, v in d.items()}
        
    logger.info("Grouping child records by VesselID...")
    
    inj_grouped = {}
    for _, row in df_inj.iterrows():
        vid = row.get("VesselID")
        if pd.notna(vid):
            vid = int(vid)
            inj_grouped.setdefault(vid, []).append(clean_dict(row.to_dict()))
            
    lsa_grouped = {}
    for _, row in df_lsa.iterrows():
        vid = row.get("VesselID")
        if pd.notna(vid):
            vid = int(vid)
            lsa_grouped.setdefault(vid, []).append(clean_dict(row.to_dict()))
            
    nav_grouped = {}
    for _, row in df_nav.iterrows():
        vid = row.get("VesselID")
        if pd.notna(vid):
            vid = int(vid)
            nav_grouped.setdefault(vid, []).append(clean_dict(row.to_dict()))
            
    rec_grouped = {}
    for _, row in df_rec.iterrows():
        vid = row.get("VesselID")
        if pd.notna(vid):
            vid = int(vid)
            rec_grouped.setdefault(vid, []).append(clean_dict(row.to_dict()))
            
    # 6. Group Vessels by OccID
    logger.info("Grouping vessels by OccID...")
    vessels_by_occ = {}
    for _, row in df_vessel_agg.iterrows():
        oid = row.get("OccID")
        if pd.notna(oid):
            oid = int(oid)
            vid = int(row["VesselID"])
            
            # Enrich vessel record with its child equipment and injuries
            vessel_record = clean_dict(row.to_dict())
            vessel_record["injuries"] = inj_grouped.get(vid, [])
            vessel_record["lsa_equipment"] = lsa_grouped.get(vid, [])
            vessel_record["navigation_equipment"] = nav_grouped.get(vid, [])
            vessel_record["rec_equipment"] = rec_grouped.get(vid, [])
            
            vessels_by_occ.setdefault(oid, []).append(vessel_record)
            
    # 7. Merge everything into Occurrences and write to JSONL
    out_file = output_dir / "merged_records.jsonl"
    logger.info(f"Merging everything and exporting to {out_file}...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for _, row in tqdm(df_occ_agg.iterrows(), total=len(df_occ_agg), desc="Merging Records"):
            oid = int(row["OccID"])
            occurrence_record = clean_dict(row.to_dict())
            
            # Find associated vessels
            occ_vessels = vessels_by_occ.get(oid, [])
            
            # Build the nested object
            merged = {
                "occurrence_id": oid,
                "occurrence": occurrence_record,
                "vessels": occ_vessels
            }
            
            # Write to JSONL
            f.write(json.dumps(merged) + "\n")
            
    logger.info("Tables merged and saved successfully!")

if __name__ == "__main__":
    main()
