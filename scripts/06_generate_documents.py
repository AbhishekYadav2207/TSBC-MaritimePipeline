import os
import json
import random
import re
import hashlib
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("06_generate_documents")

def load_templates(root: Path, config: dict) -> tuple:
    gen_config = config.get("generation", {})
    v_path = root / gen_config.get("template_vessel_path", "templates/vessel_templates.json")
    i_path = root / gen_config.get("template_injury_path", "templates/injury_templates.json")
    e_path = root / gen_config.get("template_equipment_path", "templates/equipment_templates.json")
    
    with open(v_path, "r", encoding="utf-8") as f:
        v_tpl = json.load(f)
    with open(i_path, "r", encoding="utf-8") as f:
        i_tpl = json.load(f)
    with open(e_path, "r", encoding="utf-8") as f:
        e_tpl = json.load(f)
        
    return v_tpl, i_tpl, e_tpl

def join_words(words: list) -> str:
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"

def clean_db_label(val: str) -> str:
    if not val:
        return ""
    val_clean = str(val).strip()
    val_clean = re.sub(r'\s*-\s*deactivated\s+\w+\.?\s+\d{4}', '', val_clean, flags=re.IGNORECASE)
    val_clean = re.sub(r'\s*-\s*active\s+\w+\.?\s+\d{4}', '', val_clean, flags=re.IGNORECASE)
    val_clean = re.sub(r'\s*-\s*.*?(19\d{2}|20\d{2})', '', val_clean, flags=re.IGNORECASE)
    val_clean = re.sub(r'\s+', ' ', val_clean)
    return val_clean.strip()

def clean_placeholder(val, default_str=None):
    if val is None or str(val).strip().upper() in ["", "NAN", "UNKNOWN", "UNSPECIFIED"]:
        return default_str
    return clean_db_label(str(val).strip())

def normalize_label(val: str) -> str:
    if not val:
        return ""
    val_clean = clean_db_label(str(val).strip())
    val_lower = val_clean.lower()
    
    fallback_map = {
        "radar1": "radar",
        "radar2": "radar",
        "radar3": "radar",
        "mf/hf": "MF/HF radio",
        "vhf": "VHF radio",
        "gps": "GPS receiver",
        "ecdis": "ECDIS",
        "ais": "AIS",
        "vdr": "VDR",
        "bnwas": "BNWAS",
        "gyro compass": "gyro compass",
        "magnetic compass": "magnetic compass",
        "direction_finder": "direction finder"
    }
    if val_lower in fallback_map:
        return fallback_map[val_lower]
        
    normalized = re.sub(r'(\w+?)\d+$', r'\1', val_clean)
    normalized = normalized.replace("_", " ").replace("-", " ")
    
    acronyms = {"gps", "ecdis", "vhf", "ais", "vdr", "bnwas", "mf", "hf", "lsa", "sart", "epirb"}
    words = normalized.split()
    cleaned_words = [w.upper() if w.lower() in acronyms else w for w in words]
    return " ".join(cleaned_words)

# ----------------------------------------------------------------------
# Span-Level Provenance Renderer
# ----------------------------------------------------------------------

def render_template(template_str: str, var_mapping: dict, pattern_id: str, perspective: str) -> dict:
    """Interpolates variables into a template string while constructing exact character span provenance.
    
    var_mapping format:
      {
        "vname": {"val": "ALEXANDRIA", "cat": "vessel_profile", "field": "VesselName"},
        ...
      }
    """
    matches = list(re.finditer(r'\{([a-zA-Z0-9_]+)\}', template_str))
    
    curr_idx = 0
    final_text = ""
    spans = []
    
    for m in matches:
        var_name = m.group(1)
        literal_part = template_str[curr_idx:m.start()]
        if literal_part:
            final_text += literal_part
            spans.append({
                "rendered_span": literal_part,
                "provenance": "template"
            })
            
        var_info = var_mapping.get(var_name, {})
        val_str = str(var_info.get("val", "")) if var_info else ""
        
        if val_str:
            final_text += val_str
            spans.append({
                "rendered_span": val_str,
                "category": var_info.get("cat", "domain"),
                "source_field": var_info.get("field", var_name),
                "provenance": "source_derived"
            })
            
        curr_idx = m.end()
        
    tail_part = template_str[curr_idx:]
    if tail_part:
        final_text += tail_part
        spans.append({
            "rendered_span": tail_part,
            "provenance": "template"
        })
        
    # Clean double spaces in final_text and adjust spans if needed
    final_text_clean = re.sub(r' +', ' ', final_text).strip()
    
    return {
        "text": final_text_clean,
        "provenance": {
            "perspective": perspective,
            "pattern_id": pattern_id,
            "spans": spans
        }
    }

# ----------------------------------------------------------------------
# Concept Extraction & Overlap Calculation
# ----------------------------------------------------------------------

CONCEPT_KEYWORDS = {
    "collision": "accident", "grounding": "accident", "fire": "accident", "explosion": "accident", "flooding": "accident",
    "radar": "equipment", "vhf": "equipment", "gps": "equipment", "ecdis": "equipment", "ais": "equipment", "vdr": "equipment",
    "lifeboat": "safety", "liferaft": "safety", "epirb": "safety", "jacket": "safety",
    "injury": "casualty", "fatality": "casualty", "missing": "casualty", "death": "casualty",
    "fog": "environment", "clear": "environment", "rough": "environment", "calm": "environment", "wind": "environment",
    "underway": "voyage", "anchored": "voyage", "berthed": "voyage", "towing": "voyage"
}

def extract_concepts(text: str) -> set:
    words = re.findall(r'\b\w+\b', text.lower())
    concepts = set()
    for w in words:
        if w in CONCEPT_KEYWORDS:
            concepts.add(f"{CONCEPT_KEYWORDS[w]}:{w}")
    return concepts

def jaccard_overlap(concepts1: set, concepts2: set) -> float:
    if not concepts1 or not concepts2:
        return 0.0
    intersection = len(concepts1 & concepts2)
    union = len(concepts1 | concepts2)
    return intersection / union if union > 0 else 0.0

# ----------------------------------------------------------------------
# Adaptive Generators
# ----------------------------------------------------------------------

def generate_occurrence_summary(record: dict) -> dict:
    occ = record.get("occurrence")
    if not occ:
        return None
    summary = occ.get("Summary")
    if summary and str(summary).strip() not in ["", "nan", "NaN"]:
        clean_sum = str(summary).strip()
        if clean_sum.startswith("Note: formerly OccNo") and len(clean_sum) < 60:
            return None
        if len(clean_sum) < 30:
            return None
        return {
            "text": clean_sum,
            "provenance": {
                "perspective": "occurrence_summary",
                "pattern_id": "raw_tsb_summary",
                "spans": [{
                    "rendered_span": clean_sum,
                    "category": "narrative",
                    "source_field": "Summary",
                    "provenance": "source_derived"
                }]
            }
        }
    return None

def generate_vessel_operational(oid: int, occ: dict, v: dict, v_tpl: dict) -> dict:
    """Primary operational narrative document combining vessel specs, phase, activity, environment, and outcomes."""
    vname = v.get("VesselName") or "An unnamed vessel"
    vtype = clean_placeholder(v.get("VesselTypeDisplayEng"), "vessel")
    vflag = clean_placeholder(v.get("VesselFlagDisplayEng"))
    hull = clean_placeholder(v.get("HullMaterialDisplayEng"))
    prop = clean_placeholder(v.get("PropulsionTypeDisplayEng"))
    tonnage = v.get("GrossTonnage")
    built = v.get("YearBuilt")
    speed = v.get("Speed_Knots")
    phase = clean_placeholder(v.get("VesselPhaseDisplayEng"))
    activity = clean_placeholder(v.get("ActivityTypeDisplayEng"))
    cargo_prod = clean_placeholder(v.get("CargoProductTypeDisplayEng"))
    cargo_qty = v.get("QuantityOnBoard")
    dmg_degree = clean_placeholder(v.get("VesselDamageDegreeDisplayEng"))
    dmg_loc = clean_placeholder(v.get("VesselDamageLocationDisplayEng"))
    pollution = clean_placeholder(v.get("SeaPollutionDegreeDisplayEng"))
    
    weather = clean_placeholder(occ.get("WeatherConditionDisplayEng")) if occ else None
    light = clean_placeholder(occ.get("LightConditionDisplayEng")) if occ else None
    sea = clean_placeholder(occ.get("SeaStateDisplayEng")) if occ else None
    occ_type = clean_placeholder(occ.get("OccurrenceTypeDisplayEng")) if occ else None
    
    ton_str = f"{float(tonnage):.0f} GT" if tonnage and pd.notna(tonnage) and float(tonnage) > 0 else None
    built_str = str(int(float(built))) if built and pd.notna(built) and int(float(built)) > 0 else None
    speed_str = f"{float(speed):.1f} knots" if speed and pd.notna(speed) and float(speed) > 0 else None

    var_map = {}
    var_map["vname"] = {"val": vname, "cat": "vessel_profile", "field": "VesselName"}
    var_map["vtype"] = {"val": vtype.lower(), "cat": "vessel_profile", "field": "VesselTypeDisplayEng"}
    
    spec_parts = []
    if built_str:
        spec_parts.append(f"built in {built_str}")
        var_map["built"] = {"val": built_str, "cat": "vessel_profile", "field": "YearBuilt"}
    if ton_str:
        spec_parts.append(f"with a gross tonnage of {ton_str}")
        var_map["tonnage"] = {"val": ton_str, "cat": "vessel_profile", "field": "GrossTonnage"}
    if vflag:
        spec_parts.append(f"registered in {vflag.title()}")
        var_map["vflag"] = {"val": vflag.title(), "cat": "vessel_profile", "field": "VesselFlagDisplayEng"}
    if hull:
        spec_parts.append(f"constructed of {hull.lower()}")
        var_map["hull"] = {"val": hull.lower(), "cat": "vessel_profile", "field": "HullMaterialDisplayEng"}
    if speed_str:
        spec_parts.append(f"operating at a speed of {speed_str}")
        var_map["speed"] = {"val": speed_str, "cat": "vessel_profile", "field": "Speed_Knots"}

    # Pattern selection based on available data
    pattern_id = "op_context_v1"
    tpl = "The {vtype} '{vname}'"
    if spec_parts:
        tpl += f", {', '.join(spec_parts)},"
        
    transit_clauses = []
    if phase and phase.upper() not in ["UNKNOWN", "UNSPECIFIED"]:
        transit_clauses.append(f"operating in the {phase.lower()} phase")
        var_map["phase"] = {"val": phase.lower(), "cat": "voyage_activity", "field": "VesselPhaseDisplayEng"}
    if activity and activity.upper() not in ["UNKNOWN", "UNSPECIFIED"]:
        transit_clauses.append(f"engaged in {activity.lower()} operations")
        var_map["activity"] = {"val": activity.lower(), "cat": "voyage_activity", "field": "ActivityTypeDisplayEng"}
    if cargo_prod:
        if cargo_qty and pd.notna(cargo_qty):
            transit_clauses.append(f"carrying {cargo_qty} of {cargo_prod.lower()}")
            var_map["cargo"] = {"val": f"{cargo_qty} of {cargo_prod.lower()}", "cat": "voyage_activity", "field": "CargoProductTypeDisplayEng"}
        else:
            transit_clauses.append(f"laden with {cargo_prod.lower()} cargo")
            var_map["cargo"] = {"val": cargo_prod.lower(), "cat": "voyage_activity", "field": "CargoProductTypeDisplayEng"}
            
    env_parts = []
    if weather and weather.upper() not in ["UNKNOWN"]:
        env_parts.append(f"{weather.lower()} weather")
        var_map["weather"] = {"val": weather.lower(), "cat": "environment", "field": "WeatherConditionDisplayEng"}
    if sea and sea.upper() not in ["UNKNOWN"]:
        env_parts.append(f"{sea.lower()} seas")
        var_map["sea"] = {"val": sea.lower(), "cat": "environment", "field": "SeaStateDisplayEng"}
    if light and light.upper() not in ["UNKNOWN"]:
        env_parts.append(f"{light.lower()} conditions")
        var_map["light"] = {"val": light.lower(), "cat": "environment", "field": "LightConditionDisplayEng"}
        
    if env_parts:
        transit_clauses.append(f"under {', '.join(env_parts)}")
        
    if transit_clauses:
        tpl += f" was {', '.join(transit_clauses)}"
    else:
        tpl += " was en route"
        
    outcome_clauses = []
    if occ_type and occ_type.upper() not in ["UNKNOWN", "UNSPECIFIED"]:
        outcome_clauses.append(f"became involved in a {occ_type.lower()} occurrence")
        var_map["event"] = {"val": occ_type.lower(), "cat": "casualty", "field": "OccurrenceTypeDisplayEng"}
    else:
        outcome_clauses.append("experienced a marine occurrence")
        
    if dmg_degree and dmg_degree.upper() not in ["NONE", "NONE APPARENT"]:
        loc_str = f" to the {dmg_loc.lower()}" if dmg_loc else " hull"
        outcome_clauses.append(f"sustaining {dmg_degree.lower()} damage{loc_str}")
        var_map["damage"] = {"val": f"{dmg_degree.lower()} damage{loc_str}", "cat": "casualty", "field": "VesselDamageDegreeDisplayEng"}
        
    if pollution and pollution.upper() not in ["NONE", "NONE APPARENT", "UNKNOWN"]:
        outcome_clauses.append(f"resulting in {pollution.lower()} sea pollution")
        var_map["pollution"] = {"val": pollution.lower(), "cat": "casualty", "field": "SeaPollutionDegreeDisplayEng"}
        
    tpl += f" when it {', '.join(outcome_clauses)}."
    
    return render_template(tpl, var_map, pattern_id, "vessel_operational")

def generate_equipment_navigation(oid: int, v: dict, e_tpl: dict) -> dict:
    """Dedicated equipment/navigation document generated ONLY when rich equipment data is available."""
    vname = v.get("VesselName") or "The vessel"
    nav_list = v.get("navigation_equipment", [])
    rec_list = v.get("rec_equipment", [])
    lsa_list = v.get("lsa_equipment", [])
    
    # Needs at least 3 active/inactive navigation aids or detailed recording/LSA items
    if len(nav_list) + len(rec_list) + len(lsa_list) < 2:
        return None
        
    on_devices = []
    off_devices = []
    for nav in nav_list:
        ntype = nav.get("NavigationAidTypeDisplayEng")
        nstatus = nav.get("OnOffEnumDisplayEng")
        if ntype:
            norm_name = normalize_label(ntype)
            if nstatus == "On":
                on_devices.append(norm_name)
            elif nstatus == "Off":
                off_devices.append(norm_name)
                
    lsa_names = []
    for lsa in lsa_list:
        ltype = lsa.get("LsApplianceDisplayEng")
        if ltype:
            lsa_names.append(normalize_label(ltype))
            
    rec_details = []
    for rec in rec_list:
        rtype = rec.get("RecordingEquipDisplayEng")
        extracted = rec.get("DataExtractedEnumDisplayEng")
        if rtype:
            rname = normalize_label(rtype)
            status = "data was extracted" if extracted == "Yes" else "data extraction status pending"
            rec_details.append(f"{rname} ({status})")
            
    if not (on_devices or off_devices or lsa_names or rec_details):
        return None
        
    var_map = {"vname": {"val": vname, "cat": "vessel_profile", "field": "VesselName"}}
    sentences = []
    
    if on_devices:
        var_map["on_list"] = {"val": join_words(on_devices), "cat": "equipment", "field": "NavigationAidTypeDisplayEng"}
        sentences.append("During transit, {vname} maintained active operational status for {on_list}.")
        
    if off_devices:
        var_map["off_list"] = {"val": join_words(off_devices), "cat": "equipment", "field": "NavigationAidTypeDisplayEng"}
        sentences.append("Navigation equipment reported as inactive or off included {off_list}.")
        
    if lsa_names:
        var_map["lsa_list"] = {"val": join_words(lsa_names), "cat": "equipment", "field": "LsApplianceDisplayEng"}
        sentences.append("Onboard lifesaving appliances carried featured {lsa_list}.")
        
    if rec_details:
        var_map["rec_list"] = {"val": join_words(rec_details), "cat": "equipment", "field": "RecordingEquipDisplayEng"}
        sentences.append("Flight and voyage recording equipment fitted on board included {rec_list}.")
        
    full_tpl = " ".join(sentences)
    return render_template(full_tpl, var_map, "equip_nav_v1", "equipment_navigation")

def generate_casualty_safety(oid: int, v: dict, i_tpl: dict) -> dict:
    """Dedicated casualty/safety document generated ONLY when casualties or water entry occurred."""
    vname = v.get("VesselName") or "The vessel"
    crew = v.get("TotalPeopleOnBoard")
    injuries = v.get("injuries", [])
    
    has_casualties = False
    minor_count, serious_count, death_count, missing_count = 0, 0, 0, 0
    water_entry_count = 0
    lse_used_name = None
    
    for inj in injuries:
        minor_count += int(inj.get("VictimMinorInjuries") or 0)
        serious_count += int(inj.get("VictimSeriousInjuries") or 0)
        death_count += int(inj.get("VictimDeath") or 0)
        missing_count += int(inj.get("VictimMissing") or 0)
        
        in_water = int(inj.get("TotalPeopleInWater") or 0)
        if in_water > 0:
            water_entry_count += in_water
            lse_type = inj.get("PeopleInWaterLseTypeDisplayEng")
            if lse_type:
                lse_used_name = clean_db_label(lse_type)

    total_inj = minor_count + serious_count + death_count + missing_count
    if total_inj == 0 and water_entry_count == 0:
        return None  # Suppress separate casualty document if no casualties or water entry
        
    var_map = {"vname": {"val": vname, "cat": "vessel_profile", "field": "VesselName"}}
    sentences = []
    
    if crew is not None and pd.notna(crew) and int(float(crew)) > 0:
        var_map["crew"] = {"val": str(int(float(crew))), "cat": "casualty", "field": "TotalPeopleOnBoard"}
        sentences.append("At the time of the occurrence, {vname} carried a total complement of {crew} persons on board.")
        
    counts_parts = []
    if minor_count > 0: counts_parts.append(f"{minor_count} minor injuries")
    if serious_count > 0: counts_parts.append(f"{serious_count} serious injuries")
    if death_count > 0: counts_parts.append(f"{death_count} fatalities")
    if missing_count > 0: counts_parts.append(f"{missing_count} missing persons")
    
    if counts_parts:
        var_map["casualty_details"] = {"val": join_words(counts_parts), "cat": "casualty", "field": "VictimInjuries"}
        sentences.append("The occurrence resulted in reported casualties comprising {casualty_details}.")
        
    if water_entry_count > 0:
        var_map["water_count"] = {"val": str(water_entry_count), "cat": "casualty", "field": "TotalPeopleInWater"}
        if lse_used_name:
            var_map["lse_type"] = {"val": lse_used_name.lower(), "cat": "safety", "field": "PeopleInWaterLseTypeDisplayEng"}
            sentences.append("During emergency response, {water_count} individuals entered the water wearing {lse_type}.")
        else:
            sentences.append("Emergency circumstances forced {water_count} personnel to enter the water.")
            
    full_tpl = " ".join(sentences)
    return render_template(full_tpl, var_map, "casualty_safety_v1", "casualty_safety")

# ----------------------------------------------------------------------
# Main Stage Orchestrator
# ----------------------------------------------------------------------

def main():
    random.seed(42)  # Guarantee reproducible generation
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    merged_path = output_dir / "merged_records.jsonl"
    if not merged_path.exists():
        logger.error(f"Merged records not found at {merged_path}! Run Step 5 first.")
        return
        
    v_tpl, i_tpl, e_tpl = load_templates(root, config)
    raw_docs_file = output_dir / "raw_documents.jsonl"
    
    logger.info("Generating documents with adaptive decomposition and span-level provenance...")
    
    total_occurrences = 0
    total_docs_generated = 0
    cross_overlap_scores = []
    
    with open(merged_path, "r", encoding="utf-8") as fin, open(raw_docs_file, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Generating Documents"):
            record = json.loads(line)
            oid = record["occurrence_id"]
            occ = record["occurrence"]
            vessels = record["vessels"]
            
            total_occurrences += 1
            generated_for_occ = []
            
            # 1. Occurrence Summary
            occ_sum = generate_occurrence_summary(record)
            if occ_sum:
                generated_for_occ.append({
                    "doc_type": "occurrence_summary",
                    "vessel_id": None,
                    "source_table": "MDOTW_VW_OCCURRENCE_PUBLIC",
                    "res": occ_sum
                })
                
            # 2. Per-Vessel Adaptive Generation
            for v in vessels:
                vid = v.get("VesselID")
                vid_val = int(vid) if vid is not None and pd.notna(vid) else None
                
                # Always consider Primary Operational document
                op_doc = generate_vessel_operational(oid, occ, v, v_tpl)
                if op_doc and len(op_doc["text"]) >= 40:
                    generated_for_occ.append({
                        "doc_type": "vessel_operational",
                        "vessel_id": vid_val,
                        "source_table": "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC",
                        "res": op_doc
                    })
                    
                # Conditionally consider Equipment/Navigation document
                eq_doc = generate_equipment_navigation(oid, v, e_tpl)
                if eq_doc and len(eq_doc["text"]) >= 40:
                    generated_for_occ.append({
                        "doc_type": "equipment_navigation",
                        "vessel_id": vid_val,
                        "source_table": "MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC",
                        "res": eq_doc
                    })
                    
                # Conditionally consider Casualty/Safety document
                cas_doc = generate_casualty_safety(oid, v, i_tpl)
                if cas_doc and len(cas_doc["text"]) >= 40:
                    generated_for_occ.append({
                        "doc_type": "casualty_safety",
                        "vessel_id": vid_val,
                        "source_table": "MDOTW_VW_INJURIES_PUBLIC",
                        "res": cas_doc
                    })
                    
            # 3. Suppress Cross-Perspective Information Overlap (> 0.7 Jaccard overlap)
            retained_docs = []
            retained_concepts = []
            
            for item in generated_for_occ:
                doc_text = item["res"]["text"]
                concepts = extract_concepts(doc_text)
                
                # Check overlap against already retained documents for this occurrence
                max_overlap = 0.0
                for prev_c in retained_concepts:
                    overlap = jaccard_overlap(concepts, prev_c)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        
                if max_overlap > 0.0:
                    cross_overlap_scores.append(max_overlap)
                    
                # Suppress if information overlap is too high (> 0.70)
                if max_overlap <= 0.70 or item["doc_type"] == "occurrence_summary":
                    retained_docs.append(item)
                    retained_concepts.append(concepts)
                    
            # 4. Write retained documents
            for item in retained_docs:
                total_docs_generated += 1
                output_obj = {
                    "occurrence_id": oid,
                    "vessel_id": item["vessel_id"],
                    "document_type": item["doc_type"],
                    "source_table": item["source_table"],
                    "document": item["res"]["text"],
                    "provenance": item["res"]["provenance"],
                    "structured": record
                }
                fout.write(json.dumps(output_obj) + "\n")
                
    avg_overlap = (sum(cross_overlap_scores) / len(cross_overlap_scores)) if cross_overlap_scores else 0.0
    logger.info(f"Document generation complete:")
    logger.info(f"  Total occurrences processed: {total_occurrences}")
    logger.info(f"  Total documents generated: {total_docs_generated} (Avg {total_docs_generated/total_occurrences:.2f} docs/occ)")
    logger.info(f"  Average cross-perspective information overlap: {avg_overlap:.4f}")

if __name__ == "__main__":
    main()
