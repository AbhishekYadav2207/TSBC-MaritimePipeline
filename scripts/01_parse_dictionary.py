import os
import json
from pathlib import Path
import pandas as pd
from pipeline_utils import setup_logging, detect_datasets, read_csv_safe, get_project_root, load_config

logger = setup_logging("01_parse_dictionary")

def map_display_columns(df_dict) -> dict:
    """Pairs ID/Enum/IND columns with their human-readable DisplayEng columns."""
    mappings = {}  # table_name -> {id_col: display_col}
    display_to_id = {} # table_name -> {display_col: id_col}
    
    table_groups = df_dict.groupby("Table name")
    
    for table_name, group in table_groups:
        cols = group["Column name"].unique().tolist()
        cols_lower = {c.lower(): c for c in cols}
        
        mappings[table_name] = {}
        display_to_id[table_name] = {}
        
        for col in cols:
            if col.endswith("DisplayEng") or col.endswith("Display"):
                suffix_len = 10 if col.endswith("DisplayEng") else 7
                stem = col[:-suffix_len].rstrip("_")
                
                # Check for candidates in the same table
                candidates = [
                    stem.lower(),
                    (stem + "id").lower(),
                    (stem + "enum").lower(),
                    (stem + "ind").lower(),
                    (stem + "code").lower(),
                ]
                
                # Special abbreviations
                if "occ" in stem.lower():
                    occ_stem = stem.lower().replace("occ", "occurrence")
                    candidates.extend([occ_stem, occ_stem + "id", occ_stem + "enum", occ_stem + "ind"])
                if "occurrence" in stem.lower():
                    occ_stem = stem.lower().replace("occurrence", "occ")
                    candidates.extend([occ_stem, occ_stem + "id", occ_stem + "enum", occ_stem + "ind"])
                if "quant" in stem.lower():
                    quant_stem = stem.lower().replace("quant", "quantity")
                    candidates.extend([quant_stem, quant_stem + "id", quant_stem + "enum", quant_stem + "ind"])
                if "quantity" in stem.lower():
                    quant_stem = stem.lower().replace("quantity", "quant")
                    candidates.extend([quant_stem, quant_stem + "id", quant_stem + "enum", quant_stem + "ind"])
                
                # Custom exceptions discovered during profiling
                if col == "LatEnum_Bearing_DisplayEng":
                    candidates.append("latinum")
                if col == "LongEnum_Bearing_DisplayEng":
                    candidates.append("longenum")
                    
                match = None
                for cand in candidates:
                    if cand in cols_lower:
                        match = cols_lower[cand]
                        break
                        
                if match:
                    mappings[table_name][match] = col
                    display_to_id[table_name][col] = match
                    logger.debug(f"Mapped display col: {table_name}.{match} -> {col}")
                else:
                    # Let's see if we can find any column starting with the stem and ending in ID/Enum
                    for c_raw in cols:
                        if c_raw.lower().startswith(stem.lower()) and any(c_raw.lower().endswith(s) for s in ["id", "enum", "ind", "code"]):
                            mappings[table_name][c_raw] = col
                            display_to_id[table_name][col] = c_raw
                            match = c_raw
                            break
                    if not match:
                        logger.warning(f"Could not map display column: {table_name}.{col}")
                        
    return mappings, display_to_id

def categorize_column(col_name: str, desc: str, table_name: str) -> str:
    """Categorizes a column into a semantic NLP category or admin metadata."""
    col_lower = col_name.lower()
    desc_lower = str(desc).lower()
    
    # Administrative or low-value metadata
    if any(k in col_lower for k in ["guid", "xrf", "version", "audit", "modified", "enteredby", "entrydate", "rowid", "created", "timestamp"]):
        return "admin"
        
    # Temporal columns
    if any(k in col_lower for k in ["date", "time", "year"]) or "date" in desc_lower or "time" in desc_lower:
        # Ignore entry/modified date as admin
        if any(k in col_lower for k in ["entry", "modified", "release"]):
            return "admin"
        return "temporal"
        
    # Spatial / Location columns
    if any(k in col_lower for k in ["latitude", "longitude", "latlong", "position", "province", "region", "port", "departure", "destination", "location", "bearing", "routing"]):
        return "spatial"
        
    # Environmental / Weather columns
    if any(k in col_lower for k in ["weather", "wind", "sea", "visib", "light", "temp", "ice", "swell", "wave", "beaufort"]):
        return "environmental"
        
    # Equipment columns
    if "equipment" in table_name.lower() or any(k in col_lower for k in ["nav", "lsa", "appliance", "radar", "compass", "vdr", "audio", "software", "recording"]):
        return "equipment"
        
    # Casualty details
    if "injuries" in table_name.lower() or any(k in col_lower for k in ["injury", "injuries", "death", "fatality", "fatalities", "missing", "casualty", "casualties", "victim", "pollution", "damage"]):
        return "casualty"
        
    # Voyage / Activity
    if any(k in col_lower for k in ["phase", "activity", "voyage", "cargo", "cargo", "towing", "towed", "towline", "fishery", "gear"]):
        return "voyage_activity"
        
    # Vessel Characteristics
    if any(k in col_lower for k in ["vesseltype", "vesselsubtype", "tonnage", "hull", "propulsion", "builder", "officialno", "imo", "mmsi", "callsign", "vesselname", "length", "width", "sms", "speed"]):
        return "vessel_profile"
        
    # Narratives and summaries
    if any(k in col_lower for k in ["summary", "narrative", "comment", "description", "note"]):
        return "narrative"
        
    return "other"

def main():
    root = get_project_root()
    detected = detect_datasets()
    dict_file = detected["dictionary"]
    
    if not dict_file:
        logger.error("No dictionary file found in the data directory!")
        return
        
    logger.info(f"Parsing dictionary: {dict_file.name}")
    
    df_dict = read_csv_safe(dict_file)
    
    # Strip whitespace from string columns
    for col in df_dict.columns:
        if df_dict[col].dtype == object:
            df_dict[col] = df_dict[col].astype(str).str.strip()
            
    # Remove null Table names or Column names
    df_dict = df_dict[df_dict["Table name"].notna() & df_dict["Column name"].notna()]
    
    # Map display columns
    id_to_display, display_to_id = map_display_columns(df_dict)
    
    # Build column metadata registry
    registry = {}
    for _, row in df_dict.iterrows():
        table = row["Table name"]
        col = row["Column name"]
        
        if table not in registry:
            registry[table] = {}
            
        desc = row["Description English"] if pd.notna(row["Description English"]) else ""
        dtype = row["Data type"] if pd.notna(row["Data type"]) else "unknown"
        desc_lower = desc.lower()
        
        # Determine semantic category
        category = categorize_column(col, desc, table)
        
        # Check if boolean/indicator
        is_bool = (
            col.endswith("IND") or 
            "indicates whether" in desc_lower or 
            "flag" in desc_lower or 
            "indicator" in desc_lower or
            (col.endswith("Enum") and "yes" in desc_lower)
        )
        if "is a boolean" in desc_lower or "yes/no" in desc_lower or "true/false" in desc_lower:
            is_bool = True
            
        # Get display partner if exists
        display_partner = id_to_display.get(table, {}).get(col, None)
        id_partner = display_to_id.get(table, {}).get(col, None)
        
        registry[table][col] = {
            "full_name": row["Full field name"] if pd.notna(row["Full field name"]) else col,
            "description": desc,
            "data_type": dtype,
            "category": category,
            "is_boolean": bool(is_bool),
            "id_to_display_map": display_partner,
            "display_to_id_map": id_partner
        }
        
    # Create outputs folder
    output_dir = root / load_config().get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_out = {
        "registry": registry,
        "id_to_display": id_to_display,
        "display_to_id": display_to_id
    }
    
    out_path = output_dir / "dictionary_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2)
        
    logger.info(f"Data dictionary metadata saved successfully to {out_path}")
    logger.info(f"Processed registry for {len(registry)} tables and {sum(len(cols) for cols in registry.values())} columns.")

if __name__ == "__main__":
    main()
