import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from pipeline_utils import setup_logging, detect_datasets, read_csv_safe, load_config, get_project_root

logger = setup_logging("05_merge_tables")

def aggregate_dataframe(df: pd.DataFrame, key_col, cols_meta: dict) -> pd.DataFrame:
    """Aggregates a DataFrame by key column(s), handling duplicates by concatenating specific text fields or taking first."""
    agg_funcs = {}
    
    concat_cols = {
        "weatherconditiondisplayeng", 
        "reportedbydisplayeng", 
        "substantiallyinterestedstatedisplayeng",
        "activitytypedisplayeng"
    }
    
    keys = [key_col] if isinstance(key_col, str) else list(key_col)
    
    for col in df.columns:
        if col in keys:
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
            agg_funcs[col] = "first"
            
    df_agg = df.groupby(keys, as_index=False).agg(agg_funcs)
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
        
    # 2. Read and deduplicate parent Occurrence table (Key: OccID)
    occ_table_name = "MDOTW_VW_OCCURRENCE_PUBLIC"
    occ_cols = get_cols_to_read(occ_table_name)
    logger.info(f"Loading Occurrence table with {len(occ_cols)} columns...")
    df_occ = read_csv_safe(datasets[occ_table_name], usecols=occ_cols)
    raw_occ_count = len(df_occ)
    
    logger.info(f"Deduplicating Occurrence table (original shape: {df_occ.shape})...")
    df_occ_agg = aggregate_dataframe(df_occ, "OccID", selected_columns[occ_table_name])
    logger.info(f"Occurrence table aggregated. Unique OccID count: {len(df_occ_agg)}")
    
    # 3. Read and aggregate Vessel table by Composite Key: (VesselID, OccID)
    vessel_table_name = "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC"
    vessel_cols = get_cols_to_read(vessel_table_name)
    logger.info(f"Loading Vessel table with {len(vessel_cols)} columns...")
    df_vessel = read_csv_safe(datasets[vessel_table_name], usecols=vessel_cols)
    raw_vessel_count = len(df_vessel)
    
    # Ensure VesselID and OccID are valid for grouping
    df_vessel_clean = df_vessel.dropna(subset=["VesselID", "OccID"])
    logger.info(f"Deduplicating Vessel table by composite key (VesselID, OccID) (rows: {len(df_vessel_clean)})...")
    df_vessel_agg = aggregate_dataframe(df_vessel_clean, ["VesselID", "OccID"], selected_columns[vessel_table_name])
    logger.info(f"Vessel table aggregated by (VesselID, OccID). Composite unit count: {len(df_vessel_agg)}")
    
    # 4. Load child tables
    logger.info("Loading child tables...")
    inj_table = "MDOTW_VW_INJURIES_PUBLIC"
    df_inj = read_csv_safe(datasets[inj_table], usecols=get_cols_to_read(inj_table))
    raw_inj_count = len(df_inj)
    
    lsa_table = "MDOTW_VW_OCCURRENCE_VESSEL_LSA_EQUIPMENT_PUBLIC"
    df_lsa = read_csv_safe(datasets[lsa_table], usecols=get_cols_to_read(lsa_table))
    raw_lsa_count = len(df_lsa)
    
    nav_table = "MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC"
    df_nav = read_csv_safe(datasets[nav_table], usecols=get_cols_to_read(nav_table))
    raw_nav_count = len(df_nav)
    
    rec_table = "MDOTW_VW_OCCURRENCE_VESSEL_REC_EQUIPMENT_PUBLIC"
    df_rec = read_csv_safe(datasets[rec_table], usecols=get_cols_to_read(rec_table))
    raw_rec_count = len(df_rec)
    
    # 5. Build lookup set of valid vessel-occurrence pairs
    vessel_occ_pairs = set(zip(
        df_vessel_agg["VesselID"].dropna().astype(int),
        df_vessel_agg["OccID"].dropna().astype(int)
    ))
    
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

    logger.info("Matching child records by composite key (VesselID, OccID)...")
    
    inj_grouped = {}
    lsa_grouped = {}
    nav_grouped = {}
    rec_grouped = {}
    
    orphan_inj_by_occ = {}
    orphan_lsa_by_occ = {}
    orphan_nav_by_occ = {}
    orphan_rec_by_occ = {}
    
    matched_inj, orphan_inj = 0, 0
    matched_lsa, orphan_lsa = 0, 0
    matched_nav, orphan_nav = 0, 0
    matched_rec, orphan_rec = 0, 0
    
    # Process Injuries
    for _, row in df_inj.iterrows():
        vid, oid = row.get("VesselID"), row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and pd.notna(oid) and (int(vid), int(oid)) in vessel_occ_pairs:
            inj_grouped.setdefault((int(vid), int(oid)), []).append(cleaned_row)
            matched_inj += 1
        elif pd.notna(oid):
            orphan_inj_by_occ.setdefault(int(oid), []).append(cleaned_row)
            orphan_inj += 1
            
    # Process LSA
    for _, row in df_lsa.iterrows():
        vid, oid = row.get("VesselID"), row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and pd.notna(oid) and (int(vid), int(oid)) in vessel_occ_pairs:
            lsa_grouped.setdefault((int(vid), int(oid)), []).append(cleaned_row)
            matched_lsa += 1
        elif pd.notna(oid):
            orphan_lsa_by_occ.setdefault(int(oid), []).append(cleaned_row)
            orphan_lsa += 1
            
    # Process Nav Equipment
    for _, row in df_nav.iterrows():
        vid, oid = row.get("VesselID"), row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and pd.notna(oid) and (int(vid), int(oid)) in vessel_occ_pairs:
            nav_grouped.setdefault((int(vid), int(oid)), []).append(cleaned_row)
            matched_nav += 1
        elif pd.notna(oid):
            orphan_nav_by_occ.setdefault(int(oid), []).append(cleaned_row)
            orphan_nav += 1
            
    # Process Rec Equipment
    for _, row in df_rec.iterrows():
        vid, oid = row.get("VesselID"), row.get("OccID")
        cleaned_row = clean_dict(row.to_dict())
        if pd.notna(vid) and pd.notna(oid) and (int(vid), int(oid)) in vessel_occ_pairs:
            rec_grouped.setdefault((int(vid), int(oid)), []).append(cleaned_row)
            matched_rec += 1
        elif pd.notna(oid):
            orphan_rec_by_occ.setdefault(int(oid), []).append(cleaned_row)
            orphan_rec += 1

    # 6. Group Vessels by OccID
    logger.info("Grouping composite vessels by OccID...")
    vessels_by_occ = {}
    for _, row in df_vessel_agg.iterrows():
        oid = int(row["OccID"])
        vid = int(row["VesselID"])
        
        vessel_record = clean_dict(row.to_dict())
        vessel_record["injuries"] = inj_grouped.get((vid, oid), [])
        vessel_record["lsa_equipment"] = lsa_grouped.get((vid, oid), [])
        vessel_record["navigation_equipment"] = nav_grouped.get((vid, oid), [])
        vessel_record["rec_equipment"] = rec_grouped.get((vid, oid), [])
        
        vessels_by_occ.setdefault(oid, []).append(vessel_record)

    # 7. Merge everything into Occurrences and export JSONL
    occ_ids_set = set(df_occ_agg["OccID"].dropna().unique().astype(int))
    vessel_occ_ids = set(df_vessel_agg["OccID"].dropna().unique().astype(int))
    inj_occ_ids = set(df_inj["OccID"].dropna().unique().astype(int))
    lsa_occ_ids = set(df_lsa["OccID"].dropna().unique().astype(int))
    nav_occ_ids = set(df_nav["OccID"].dropna().unique().astype(int))
    rec_occ_ids = set(df_rec["OccID"].dropna().unique().astype(int))
    
    all_occ_ids = sorted([int(x) for x in (occ_ids_set | vessel_occ_ids | inj_occ_ids | lsa_occ_ids | nav_occ_ids | rec_occ_ids)])
    logger.info(f"Total unique occurrences to merge: {len(all_occ_ids)}")
    
    occ_by_id = {int(row["OccID"]): clean_dict(row.to_dict()) for _, row in df_occ_agg.iterrows()}
    
    out_file = output_dir / "merged_records.jsonl"
    placeholder_vessels_count = 0
    placeholder_occurrences_count = 0
    
    with open(out_file, "w", encoding="utf-8") as f:
        for oid in tqdm(all_occ_ids, desc="Merging Records"):
            if oid in occ_by_id:
                occurrence_record = occ_by_id[oid]
                is_placeholder_occ = False
            else:
                occurrence_record = None
                is_placeholder_occ = True
                placeholder_occurrences_count += 1
                
            occ_vessels = list(vessels_by_occ.get(oid, []))
            
            orph_inj = orphan_inj_by_occ.get(oid, [])
            orph_lsa = orphan_lsa_by_occ.get(oid, [])
            orph_nav = orphan_nav_by_occ.get(oid, [])
            orph_rec = orphan_rec_by_occ.get(oid, [])
            
            if orph_inj or orph_lsa or orph_nav or orph_rec:
                placeholder_vessels_count += 1
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
                
            merged = {
                "occurrence_id": oid,
                "occurrence": occurrence_record,
                "vessels": occ_vessels
            }
            if is_placeholder_occ:
                merged["_placeholder_occurrence"] = True
                
            f.write(json.dumps(merged) + "\n")
            
    # Save Merge Reconciliation Report
    reconciliation_report = {
        "raw_source_rows": {
            "MDOTW_VW_OCCURRENCE_PUBLIC": raw_occ_count,
            "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC": raw_vessel_count,
            "MDOTW_VW_INJURIES_PUBLIC": raw_inj_count,
            "MDOTW_VW_OCCURRENCE_VESSEL_LSA_EQUIPMENT_PUBLIC": raw_lsa_count,
            "MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC": raw_nav_count,
            "MDOTW_VW_OCCURRENCE_VESSEL_REC_EQUIPMENT_PUBLIC": raw_rec_count
        },
        "retained_units": {
            "unique_occurrences": len(df_occ_agg),
            "merged_vessel_occurrence_units": len(df_vessel_agg),
            "total_merged_occurrences": len(all_occ_ids)
        },
        "child_table_matches": {
            "injuries": {"matched": matched_inj, "orphan": orphan_inj},
            "lsa_equipment": {"matched": matched_lsa, "orphan": orphan_lsa},
            "navigation_equipment": {"matched": matched_nav, "orphan": orphan_nav},
            "recording_equipment": {"matched": matched_rec, "orphan": orphan_rec}
        },
        "synthesized_placeholders": {
            "placeholder_vessels_created": placeholder_vessels_count,
            "placeholder_occurrences_created": placeholder_occurrences_count
        }
    }
    
    recon_path = output_dir / "merge_reconciliation_report.json"
    with open(recon_path, "w", encoding="utf-8") as fr:
        json.dump(reconciliation_report, fr, indent=2)
        
    logger.info(f"Merge Reconciliation Report exported to {recon_path}")
    logger.info("Tables merged and saved successfully!")

if __name__ == "__main__":
    main()
