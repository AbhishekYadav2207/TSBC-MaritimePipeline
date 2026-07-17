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
    
    # 5. Group child tables and identify all unique IDs
    # We clean nan values and convert numpy types to standard python types to keep json clean and serializable
    def clean_dict(d):
        cleaned = {}
        for k, v in d.items():
            if pd.isna(v) or v == "nan":
                cleaned[k] = None
            elif hasattr(v, "item"):
                cleaned[k] = v.item()
            else:
                cleaned[k] = v
        return cleaned
        
    logger.info("Identifying unique IDs and grouping records...")
    
    # Track standard vessel IDs to distinguish orphans
    vessels_set = set(df_vessel_agg["VesselID"].dropna().unique().astype(int))
    
    # Group standard child tables by VesselID
    inj_grouped = {}
    lsa_grouped = {}
    nav_grouped = {}
    rec_grouped = {}
    
    # Group orphan child tables by OccID (for child records with missing/invalid VesselID)
    orphan_inj_by_occ = {}
    orphan_lsa_by_occ = {}
    orphan_nav_by_occ = {}
    orphan_rec_by_occ = {}
    
    # Group injuries
    for _, row in df_inj.iterrows():
        vid = row.get("VesselID")
        oid = row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and int(vid) in vessels_set:
            inj_grouped.setdefault(int(vid), []).append(cleaned_row)
        elif pd.notna(oid):
            orphan_inj_by_occ.setdefault(int(oid), []).append(cleaned_row)
            
    # Group LSA
    for _, row in df_lsa.iterrows():
        vid = row.get("VesselID")
        oid = row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and int(vid) in vessels_set:
            lsa_grouped.setdefault(int(vid), []).append(cleaned_row)
        elif pd.notna(oid):
            orphan_lsa_by_occ.setdefault(int(oid), []).append(cleaned_row)
            
    # Group Nav
    for _, row in df_nav.iterrows():
        vid = row.get("VesselID")
        oid = row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and int(vid) in vessels_set:
            nav_grouped.setdefault(int(vid), []).append(cleaned_row)
        elif pd.notna(oid):
            orphan_nav_by_occ.setdefault(int(oid), []).append(cleaned_row)
            
    # Group Rec
    for _, row in df_rec.iterrows():
        vid = row.get("VesselID")
        oid = row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and int(vid) in vessels_set:
            rec_grouped.setdefault(int(vid), []).append(cleaned_row)
        elif pd.notna(oid):
            orphan_rec_by_occ.setdefault(int(oid), []).append(cleaned_row)
            
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
            
    # 7. Merge everything into Occurrences and write to JSONL (Left Outer Join)
    # Collect all unique OccIDs from all sources to ensure 100% data retention
    occ_ids_set = set(df_occ_agg["OccID"].dropna().unique().astype(int))
    vessel_occ_ids = set(df_vessel_agg["OccID"].dropna().unique().astype(int))
    inj_occ_ids = set(df_inj["OccID"].dropna().unique().astype(int))
    lsa_occ_ids = set(df_lsa["OccID"].dropna().unique().astype(int))
    nav_occ_ids = set(df_nav["OccID"].dropna().unique().astype(int))
    rec_occ_ids = set(df_rec["OccID"].dropna().unique().astype(int))
    
    all_occ_ids = sorted([int(x) for x in (occ_ids_set | vessel_occ_ids | inj_occ_ids | lsa_occ_ids | nav_occ_ids | rec_occ_ids)])
    logger.info(f"Total unique occurrences to merge (including placeholders): {len(all_occ_ids)}")
    
    occ_by_id = {int(row["OccID"]): clean_dict(row.to_dict()) for _, row in df_occ_agg.iterrows()}
    
    out_file = output_dir / "merged_records.jsonl"
    logger.info(f"Merging everything and exporting to {out_file}...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for oid in tqdm(all_occ_ids, desc="Merging Records"):
            # Check if this is a real occurrence or a placeholder
            if oid in occ_by_id:
                occurrence_record = occ_by_id[oid]
                is_placeholder_occurrence = False
            else:
                occurrence_record = None
                is_placeholder_occurrence = True
                
            # Find associated vessels
            occ_vessels = list(vessels_by_occ.get(oid, []))
            
            # Check for any orphan child records for this OccID to synthesize a single placeholder vessel
            orph_inj = orphan_inj_by_occ.get(oid, [])
            orph_lsa = orphan_lsa_by_occ.get(oid, [])
            orph_nav = orphan_nav_by_occ.get(oid, [])
            orph_rec = orphan_rec_by_occ.get(oid, [])
            
            if orph_inj or orph_lsa or orph_nav or orph_rec:
                placeholder_vessel = {
                    "VesselID": None,
                    "VesselName": "Unknown Vessel",
                    "_placeholder": True,
                    "_reason": "orphan_child_records",
                    "injuries": orph_inj,
                    "lsa_equipment": orph_lsa,
                    "navigation_equipment": orph_nav,
                    "rec_equipment": orph_rec
                }
                occ_vessels.append(placeholder_vessel)
                
            # Build the nested object
            merged = {
                "occurrence_id": oid,
                "occurrence": occurrence_record,
                "vessels": occ_vessels
            }
            
            if is_placeholder_occurrence:
                merged["_placeholder_occurrence"] = True
                
            # Write to JSONL
            f.write(json.dumps(merged) + "\n")
            
    logger.info("Tables merged and saved successfully!")

if __name__ == "__main__":
    main()
