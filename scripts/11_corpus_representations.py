import os
import json
from pathlib import Path
from tqdm import tqdm
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("11_corpus_representations")

def build_key_value_representation(record: dict) -> str:
    """Renders structured record into a clean Key-Value text representation."""
    structured = record.get("structured", {})
    occ = structured.get("occurrence", {})
    vessels = structured.get("vessels", [])
    
    parts = []
    if occ.get("OccID"): parts.append(f"Occurrence ID: {occ.get('OccID')}")
    if occ.get("OccTypeDisplayEng"): parts.append(f"Occurrence Type: {occ.get('OccTypeDisplayEng')}")
    if occ.get("AccIncTypeDisplayEng"): parts.append(f"Incident Type: {occ.get('AccIncTypeDisplayEng')}")
    if occ.get("NearestLocationDescription"): parts.append(f"Location: {occ.get('NearestLocationDescription')}")
    if occ.get("WeatherConditionDisplayEng"): parts.append(f"Weather: {occ.get('WeatherConditionDisplayEng')}")
    if occ.get("SeaStateDisplayEng"): parts.append(f"Sea State: {occ.get('SeaStateDisplayEng')}")
    if occ.get("TotalDeaths") or occ.get("TotalMinorInjuries") or occ.get("TotalSeriousInjuries"):
        parts.append(f"Casualties: {occ.get('TotalDeaths', 0)} deaths, {occ.get('TotalSeriousInjuries', 0)} serious injuries, {occ.get('TotalMinorInjuries', 0)} minor injuries")
        
    for v in vessels:
        v_parts = []
        if v.get("VesselName"): v_parts.append(f"Vessel: {v.get('VesselName')}")
        if v.get("VesselTypeDisplayEng"): v_parts.append(f"Type: {v.get('VesselTypeDisplayEng')}")
        if v.get("VesselFlagDisplayEng"): v_parts.append(f"Flag: {v.get('VesselFlagDisplayEng')}")
        if v.get("GrossTonnage"): v_parts.append(f"Tonnage: {v.get('GrossTonnage')} GT")
        if v.get("HullMaterialDisplayEng"): v_parts.append(f"Hull: {v.get('HullMaterialDisplayEng')}")
        if v.get("VesselPhaseDisplayEng"): v_parts.append(f"Phase: {v.get('VesselPhaseDisplayEng')}")
        
        nav_eq = [e.get("NavigationAidTypeDisplayEng") for e in v.get("navigation_equipment", []) if e.get("NavigationAidTypeDisplayEng")]
        if nav_eq:
            unique_nav = list(dict.fromkeys(nav_eq))
            v_parts.append(f"Navigation Equipment: {', '.join(unique_nav)}")
            
        lsa_eq = [e.get("LsApplianceDisplayEng") for e in v.get("lsa_equipment", []) if e.get("LsApplianceDisplayEng")]
        if lsa_eq:
            unique_lsa = list(dict.fromkeys(lsa_eq))
            v_parts.append(f"LSA Equipment: {', '.join(unique_lsa)}")
            
        if v_parts:
            parts.append(" | ".join(v_parts))
            
    if occ.get("Summary"):
        parts.append(f"Summary: {occ.get('Summary')}")
        
    return " \n ".join(parts)

def build_template_representation(record: dict) -> str:
    """Renders structured record into a semi-structured template representation."""
    structured = record.get("structured", {})
    occ = structured.get("occurrence", {})
    vessels = structured.get("vessels", [])
    
    occ_type = occ.get("OccTypeDisplayEng") or "marine event"
    inc_type = occ.get("AccIncTypeDisplayEng") or "incident"
    loc = occ.get("NearestLocationDescription") or "Canadian waters"
    weather = occ.get("WeatherConditionDisplayEng") or "unspecified weather"
    
    v_descriptions = []
    for v in vessels:
        v_name = v.get("VesselName") or "unidentified vessel"
        v_type = v.get("VesselTypeDisplayEng") or "vessel"
        v_flag = v.get("VesselFlagDisplayEng") or "unknown flag"
        v_gt = f"{v.get('GrossTonnage')} GT" if v.get("GrossTonnage") else "unknown tonnage"
        v_descriptions.append(f"the {v_type} '{v_name}' (registered in {v_flag}, displacement {v_gt})")
        
    v_str = " and ".join(v_descriptions) if v_descriptions else "a vessel"
    template_doc = f"A maritime {occ_type.lower()} involving {inc_type.lower()} occurred near {loc} under {weather.lower()} conditions involving {v_str}."
    
    if occ.get("Summary"):
        template_doc += f" Official summary report: {occ.get('Summary')}."
        
    return template_doc

def build_json_representation(record: dict) -> str:
    """Renders structured record into a compact serialized JSON format string."""
    structured = record.get("structured", {})
    clean_meta = {
        "occurrence_id": record.get("occurrence_id"),
        "occurrence": structured.get("occurrence"),
        "vessels": structured.get("vessels")
    }
    return json.dumps(clean_meta, ensure_ascii=False)

def build_mixed_representation(narrative_doc: str, record: dict) -> str:
    """Combines Key-Value metadata header with natural narrative body."""
    kv_header = build_key_value_representation(record)
    return f"[METADATA]\n{kv_header}\n[NARRATIVE]\n{narrative_doc}"

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents file missing at {clean_path}! Run Step 7 first.")
        return
        
    reps_dir = output_dir / "corpus_representations"
    reps_dir.mkdir(parents=True, exist_ok=True)
    
    rep_files = {
        "narrative": open(reps_dir / "narrative.jsonl", "w", encoding="utf-8"),
        "key_value": open(reps_dir / "key_value.jsonl", "w", encoding="utf-8"),
        "template": open(reps_dir / "template.jsonl", "w", encoding="utf-8"),
        "json": open(reps_dir / "json.jsonl", "w", encoding="utf-8"),
        "mixed": open(reps_dir / "mixed.jsonl", "w", encoding="utf-8")
    }
    
    logger.info("Generating 5 multi-format corpus representations (Narrative, Key-Value, Template, JSON, Mixed)...")
    doc_count = 0
    
    try:
        with open(clean_path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc="Corpus Representations"):
                record = json.loads(line)
                occ_id = record.get("occurrence_id")
                narrative_doc = record.get("document", "")
                
                # 1. Narrative Representation
                rep_files["narrative"].write(json.dumps({
                    "occurrence_id": occ_id,
                    "representation": "narrative",
                    "document": narrative_doc
                }) + "\n")
                
                # 2. Key-Value Representation
                kv_doc = build_key_value_representation(record)
                rep_files["key_value"].write(json.dumps({
                    "occurrence_id": occ_id,
                    "representation": "key_value",
                    "document": kv_doc
                }) + "\n")
                
                # 3. Template Representation
                tmpl_doc = build_template_representation(record)
                rep_files["template"].write(json.dumps({
                    "occurrence_id": occ_id,
                    "representation": "template",
                    "document": tmpl_doc
                }) + "\n")
                
                # 4. JSON Representation
                json_doc = build_json_representation(record)
                rep_files["json"].write(json.dumps({
                    "occurrence_id": occ_id,
                    "representation": "json",
                    "document": json_doc
                }) + "\n")
                
                # 5. Mixed Representation
                mixed_doc = build_mixed_representation(narrative_doc, record)
                rep_files["mixed"].write(json.dumps({
                    "occurrence_id": occ_id,
                    "representation": "mixed",
                    "document": mixed_doc
                }) + "\n")
                
                doc_count += 1
    finally:
        for f in rep_files.values():
            f.close()
            
    logger.info(f"Stage 11 completed successfully. Generated {doc_count} documents across 5 corpus representations in {reps_dir}")

if __name__ == "__main__":
    main()
