import os
import json
from pathlib import Path
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("04_select_semantic_columns")

def select_columns(metadata: dict) -> dict:
    """Filters out admin metadata and French columns, retaining only semantic NLP columns and join keys."""
    registry = metadata["registry"]
    selected = {}
    
    # Primary join keys we need to keep
    core_keys = {"occid", "occno", "vesselid", "parentvesselid"}
    
    for table_name, cols in registry.items():
        table_selected = {
            "join_keys": [],
            "display_cols": [],
            "numeric_attrs": [],
            "boolean_attrs": [],
            "narrative_cols": [],
            "other_semantic": []
        }
        
        for col_name, col_meta in cols.items():
            col_lower = col_name.lower()
            category = col_meta["category"]
            is_bool = col_meta["is_boolean"]
            
            # 1. Check if it's a join key
            if col_lower in core_keys:
                table_selected["join_keys"].append(col_name)
                continue
                
            # 2. Exclude French columns
            if col_lower.endswith("displayfre") or col_lower.endswith("fre"):
                continue
                
            # 3. Exclude admin columns
            if category == "admin":
                continue
                
            # 4. If it has a DisplayEng partner, we keep the DisplayEng and skip the raw ID/Enum/Code
            # (e.g. keep AccIncTypeDisplayEng, skip AccIncTypeID)
            # This is key to automatically selecting the translated terms!
            display_partner = col_meta.get("id_to_display_map")
            if display_partner:
                # This is an ID column that has a matching display column, so we skip the ID column
                # because we will read the display column instead!
                continue
                
            # Now, categorize the retained columns:
            if category == "narrative":
                table_selected["narrative_cols"].append(col_name)
            elif col_lower.endswith("displayeng") or col_lower.endswith("display") or col_meta.get("display_to_id_map"):
                table_selected["display_cols"].append(col_name)
            elif is_bool:
                table_selected["boolean_attrs"].append(col_name)
            elif category in ["environmental", "vessel_profile", "voyage_activity", "casualty", "spatial", "equipment"]:
                # Check data type
                dtype = col_meta["data_type"].lower()
                if any(t in dtype for t in ["int", "numeric", "smallint", "float", "double"]):
                    # Keep numeric characteristics like length, crew count, speed
                    # But ignore numerical coordinates
                    if col_lower in ["latitude", "longitude", "latlong"]:
                        continue
                    table_selected["numeric_attrs"].append(col_name)
                else:
                    table_selected["other_semantic"].append(col_name)
                    
        selected[table_name] = table_selected
        
        # Log counts for each table
        total_selected = sum(len(lst) for lst in table_selected.values())
        logger.info(f"Table '{table_name}': selected {total_selected} semantic/key columns out of {len(cols)}")
        
    return selected

def main():
    root = get_project_root()
    config = load_config()
    
    meta_path = root / config.get("output_dir", "outputs") / "dictionary_metadata.json"
    if not meta_path.exists():
        logger.error(f"Metadata file not found at {meta_path}! Run Step 1 first.")
        return
        
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    logger.info("Selecting semantic NLP columns...")
    selected_cols = select_columns(metadata)
    
    out_path = root / config.get("output_dir", "outputs") / "selected_semantic_columns.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected_cols, f, indent=2)
        
    logger.info(f"Selected semantic columns saved successfully to {out_path}")

if __name__ == "__main__":
    main()
