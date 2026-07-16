import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from pipeline_utils import setup_logging, load_config, get_project_root, detect_datasets, read_csv_safe

logger = setup_logging("05a_validate_records")

def validate_raw_ids(datasets: dict) -> dict:
    """Checks for missing primary keys, orphan rows, and broken joins in the raw CSV tables."""
    logger.info("Performing orphan and broken join analysis on raw CSV identifiers...")
    
    # We only read the key columns to be fast and memory-efficient
    occ_df = read_csv_safe(datasets["MDOTW_VW_OCCURRENCE_PUBLIC"], usecols=["OccID", "OccNo"])
    vessel_df = read_csv_safe(datasets["MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC"], usecols=["OccID", "OccNo", "VesselID"])
    inj_df = read_csv_safe(datasets["MDOTW_VW_INJURIES_PUBLIC"], usecols=["OccID", "OccNo", "VesselID"])
    lsa_df = read_csv_safe(datasets["MDOTW_VW_OCCURRENCE_VESSEL_LSA_EQUIPMENT_PUBLIC"], usecols=["OccID", "VesselID"])
    nav_df = read_csv_safe(datasets["MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC"], usecols=["OccID", "VesselID"])
    rec_df = read_csv_safe(datasets["MDOTW_VW_OCCURRENCE_VESSEL_REC_EQUIPMENT_PUBLIC"], usecols=["OccID", "VesselID"])
    
    # Sets for membership testing
    occ_ids = set(occ_df["OccID"].dropna().unique())
    occ_nos = set(occ_df["OccNo"].dropna().unique())
    vessel_ids = set(vessel_df["VesselID"].dropna().unique())
    
    # Missing primary keys
    missing_pks = {
        "MDOTW_VW_OCCURRENCE_PUBLIC": int(occ_df["OccID"].isnull().sum()),
        "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC": int(vessel_df["VesselID"].isnull().sum()),
    }
    
    # Orphan rows (children referencing parents that don't exist)
    orphan_counts = {
        "vessels_without_occurrence": 0,
        "injuries_without_occurrence": 0,
        "injuries_without_vessel": 0,
        "lsa_without_occurrence": 0,
        "lsa_without_vessel": 0,
        "nav_without_occurrence": 0,
        "nav_without_vessel": 0,
        "rec_without_occurrence": 0,
        "rec_without_vessel": 0
    }
    
    # Vessels orphans
    vessel_occ_ids = vessel_df["OccID"].dropna().unique()
    orphan_counts["vessels_without_occurrence"] = int(sum(1 for oid in vessel_occ_ids if oid not in occ_ids))
    
    # Injuries orphans
    for _, r in inj_df.iterrows():
        oid, vid = r.get("OccID"), r.get("VesselID")
        if pd.notna(oid) and int(oid) not in occ_ids:
            orphan_counts["injuries_without_occurrence"] += 1
        if pd.notna(vid) and int(vid) not in vessel_ids:
            orphan_counts["injuries_without_vessel"] += 1
            
    # LSA Orphans
    for _, r in lsa_df.iterrows():
        oid, vid = r.get("OccID"), r.get("VesselID")
        if pd.notna(oid) and int(oid) not in occ_ids:
            orphan_counts["lsa_without_occurrence"] += 1
        if pd.notna(vid) and int(vid) not in vessel_ids:
            orphan_counts["lsa_without_vessel"] += 1
            
    # Nav Orphans
    for _, r in nav_df.iterrows():
        oid, vid = r.get("OccID"), r.get("VesselID")
        if pd.notna(oid) and int(oid) not in occ_ids:
            orphan_counts["nav_without_occurrence"] += 1
        if pd.notna(vid) and int(vid) not in vessel_ids:
            orphan_counts["nav_without_vessel"] += 1
            
    # Rec Orphans
    for _, r in rec_df.iterrows():
        oid, vid = r.get("OccID"), r.get("VesselID")
        if pd.notna(oid) and int(oid) not in occ_ids:
            orphan_counts["rec_without_occurrence"] += 1
        if pd.notna(vid) and int(vid) not in vessel_ids:
            orphan_counts["rec_without_vessel"] += 1
            
    return {
        "missing_primary_keys": missing_pks,
        "orphan_counts": orphan_counts
    }

def validate_merged_records(merged_path: Path, config: dict) -> dict:
    """Validates the structure, dates, and numeric values in the merged JSONL file."""
    logger.info("Validating merged occurrence records...")
    
    val_limits = config.get("validation", {})
    max_speed = val_limits.get("max_vessel_speed_knots", 100.0)
    max_tonnage = val_limits.get("max_tonnage", 300000.0)
    max_crew = val_limits.get("max_crew", 1000)
    
    total_processed = 0
    seen_occ_ids = set()
    duplicate_occ_ids = 0
    warnings = []
    
    current_year = datetime.now().year
    
    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            total_processed += 1
            record = json.loads(line)
            
            oid = record.get("occurrence_id")
            if oid is None:
                warnings.append("Record with missing occurrence_id")
                continue
                
            if oid in seen_occ_ids:
                duplicate_occ_ids += 1
            seen_occ_ids.add(oid)
            
            occ = record.get("occurrence", {})
            vessels = record.get("vessels", [])
            
            # 1. Date check
            occ_date_str = occ.get("OccDate")
            if occ_date_str:
                try:
                    # OccDate is typically YYYY-MM-DD HH:MM:SS or just YYYY-MM-DD
                    date_val = pd.to_datetime(occ_date_str)
                    if date_val.year > current_year:
                        warnings.append(f"OccID {oid}: Future occurrence date ({occ_date_str})")
                    if date_val.year < 1900:
                        warnings.append(f"OccID {oid}: Implausibly old occurrence date ({occ_date_str})")
                except Exception:
                    warnings.append(f"OccID {oid}: Invalid date format ({occ_date_str})")
                    
            # 2. Temperature / wind check
            wind_speed = occ.get("WindSpeed_Knots")
            if pd.notna(wind_speed) and (wind_speed < 0 or wind_speed > 150):
                warnings.append(f"OccID {oid}: Implausible wind speed ({wind_speed} knots)")
                
            air_temp = occ.get("AirTemp_Celsius")
            if pd.notna(air_temp) and (air_temp < -60 or air_temp > 60):
                warnings.append(f"OccID {oid}: Implausible air temperature ({air_temp} °C)")
                
            # 3. Vessel checks
            for v in vessels:
                vid = v.get("VesselID")
                vname = v.get("VesselName") or "Unknown"
                
                speed = v.get("Speed_Knots")
                if pd.notna(speed) and (speed < 0 or speed > max_speed):
                    warnings.append(f"OccID {oid}, Vessel {vname} ({vid}): Implausible speed ({speed} knots)")
                    
                tonnage = v.get("GrossTonnage")
                if pd.notna(tonnage) and (tonnage < 0 or tonnage > max_tonnage):
                    warnings.append(f"OccID {oid}, Vessel {vname} ({vid}): Implausible gross tonnage ({tonnage})")
                    
                crew = v.get("TotalPeopleOnBoard")
                if pd.notna(crew) and (crew < 0 or crew > max_crew):
                    warnings.append(f"OccID {oid}, Vessel {vname} ({vid}): Implausible crew/people on board count ({crew})")
                    
                # 4. Injuries inside vessel checks
                for inj in v.get("injuries", []):
                    minor = inj.get("VictimMinorInjuries")
                    serious = inj.get("VictimSeriousInjuries")
                    deaths = inj.get("VictimDeath")
                    
                    for k, val in [("minor", minor), ("serious", serious), ("deaths", deaths)]:
                        if pd.notna(val) and (val < 0 or val > max_crew):
                            warnings.append(f"OccID {oid}, Vessel {vname}: Implausible injury count ({k}: {val})")
                            
    return {
        "total_occurrences_in_merged": total_processed,
        "duplicate_occurrence_ids": duplicate_occ_ids,
        "impossible_values_warnings": warnings[:100],  # Limit warnings in report to 100
        "total_impossible_values_count": len(warnings)
    }

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    merged_path = output_dir / "merged_records.jsonl"
    if not merged_path.exists():
        logger.error(f"Merged records not found at {merged_path}! Run Step 5 first.")
        return
        
    detected = detect_datasets()
    
    # Run validations
    raw_results = validate_raw_ids(detected["datasets"])
    merged_results = validate_merged_records(merged_path, config)
    
    # Combine results
    report = {
        "status": "PASS" if merged_results["total_impossible_values_count"] == 0 else "WARNING",
        "missing_primary_keys": raw_results["missing_primary_keys"],
        "orphan_counts": raw_results["orphan_counts"],
        "total_occurrences_processed": merged_results["total_occurrences_in_merged"],
        "duplicate_occurrence_ids_count": merged_results["duplicate_occurrence_ids"],
        "total_value_warnings": merged_results["total_impossible_values_count"],
        "value_warnings_sample": merged_results["impossible_values_warnings"]
    }
    
    out_path = output_dir / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Validation report saved successfully to {out_path}")
    logger.info(f"Validation status: {report['status']} with {report['total_value_warnings']} value warnings.")

if __name__ == "__main__":
    main()
