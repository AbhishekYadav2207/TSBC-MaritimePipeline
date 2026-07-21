import os
import json
import random
import re
import hashlib
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from pipeline_utils import setup_logging, load_config, get_project_root
from text_sanitizer import (
    strip_administrative_noise,
    join_words_grammatical,
    format_cargo_description,
    format_damage_description,
    format_casualty_count
)

logger = setup_logging("06_generate_documents")

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
    if val is None or str(val).strip().upper() in ["", "NAN", "UNKNOWN", "UNSPECIFIED", "NONE"]:
        return default_str
    return clean_db_label(str(val).strip())

# ----------------------------------------------------------------------
# Concept Extractor & Concept Gain Calculator
# ----------------------------------------------------------------------

MARITIME_CONCEPT_KEYWORDS = {
    "collision": "accident", "grounding": "accident", "fire": "accident", "explosion": "accident", "flooding": "accident",
    "capsizing": "accident", "sinking": "accident", "foundering": "accident", "contact": "accident", "stranding": "accident",
    "radar": "equipment", "vhf": "equipment", "gps": "equipment", "ecdis": "equipment", "ais": "equipment", "vdr": "equipment",
    "gyrocompass": "equipment", "compass": "equipment", "echosounder": "equipment", "bnwas": "equipment",
    "lifeboat": "safety", "liferaft": "safety", "epirb": "safety", "sart": "safety", "lifejacket": "safety", "lifebuoy": "safety",
    "injury": "casualty", "fatality": "casualty", "missing": "casualty", "death": "casualty", "personnel": "casualty",
    "fog": "environment", "clear": "environment", "rough": "environment", "calm": "environment", "wind": "environment",
    "gale": "environment", "snow": "environment", "ice": "environment", "visibility": "environment",
    "underway": "voyage", "anchored": "voyage", "berthed": "voyage", "towing": "voyage", "hauling": "voyage", "moored": "voyage",
    "fishing": "vessel_type", "tanker": "vessel_type", "cargo": "vessel_type", "passenger": "vessel_type", "tug": "vessel_type",
    "bulk": "vessel_type", "container": "vessel_type", "barge": "vessel_type"
}

def extract_concepts(text: str) -> set:
    if not text:
        return set()
    words = re.findall(r'\b\w+\b', text.lower())
    concepts = set()
    for w in words:
        if w in MARITIME_CONCEPT_KEYWORDS:
            concepts.add(f"{MARITIME_CONCEPT_KEYWORDS[w]}:{w}")
    return concepts

def calculate_unique_concept_gain(existing_concepts: set, candidate_concepts: set) -> int:
    if not candidate_concepts:
        return 0
    return len(candidate_concepts - existing_concepts)

def jaccard_overlap(concepts1: set, concepts2: set) -> float:
    if not concepts1 or not concepts2:
        return 0.0
    intersection = len(concepts1 & concepts2)
    union = len(concepts1 | concepts2)
    return intersection / union if union > 0 else 0.0

# ----------------------------------------------------------------------
# Span-Level Provenance Renderer
# ----------------------------------------------------------------------

def render_template(template_str: str, var_mapping: dict, pattern_id: str, perspective: str) -> dict:
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
        
    final_text_clean = strip_administrative_noise(re.sub(r' +', ' ', final_text).strip())
    
    return {
        "text": final_text_clean,
        "provenance": {
            "perspective": perspective,
            "pattern_id": pattern_id,
            "spans": spans
        }
    }

# ----------------------------------------------------------------------
# 5 Context-Sensitive Template Families for Vessel Operational Narratives
# ----------------------------------------------------------------------

def generate_vessel_operational_narrative(oid: int, occ: dict, v: dict) -> dict:
    """Generates an operational narrative using 1 of 5 structural template families for high diversity."""
    vname = v.get("VesselName") or "An unnamed vessel"
    vtype = clean_placeholder(v.get("VesselTypeDisplayEng"), "vessel")
    vflag = clean_placeholder(v.get("VesselFlagDisplayEng"))
    hull = clean_placeholder(v.get("HullMaterialDisplayEng"))
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

    var_map = {
        "vname": {"val": vname, "cat": "vessel_profile", "field": "VesselName"},
        "vtype": {"val": vtype.lower(), "cat": "vessel_profile", "field": "VesselTypeDisplayEng"}
    }

    # Deterministically select family based on occurrence/vessel ID hash
    family_idx = abs(hash(f"{oid}_{v.get('VesselID')}")) % 5

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

    cargo_clause = format_cargo_description(cargo_prod, cargo_qty)
    if cargo_clause:
        var_map["cargo"] = {"val": cargo_clause, "cat": "voyage_activity", "field": "CargoProductTypeDisplayEng"}

    damage_clause = format_damage_description(dmg_degree, dmg_loc)
    if damage_clause:
        var_map["damage"] = {"val": damage_clause, "cat": "casualty", "field": "VesselDamageDegreeDisplayEng"}

    raw_event = occ_type.lower() if occ_type else ""
    if not raw_event or raw_event in ["occurrence", "unknown", "unspecified"]:
        event_str = "marine occurrence"
    elif raw_event.endswith("occurrence"):
        event_str = raw_event
    else:
        event_str = f"{raw_event} occurrence"
        
    var_map["event"] = {"val": event_str, "cat": "casualty", "field": "OccurrenceTypeDisplayEng"}

    env_parts = []
    if weather: env_parts.append(f"{weather.lower()} weather")
    if sea: env_parts.append(f"{sea.lower()} seas")
    if light: env_parts.append(f"{light.lower()} light conditions")
    env_str = join_words_grammatical(env_parts)
    if env_str:
        var_map["environment"] = {"val": env_str, "cat": "environment", "field": "WeatherConditionDisplayEng"}

    # Family A: Subject-First (Standard)
    if family_idx == 0:
        pattern_id = "op_family_a_subject_first"
        tpl = "The {vtype} '{vname}'"
        if spec_parts: tpl += f", {', '.join(spec_parts)},"
        if phase: tpl += f" was operating in the {phase.lower()} phase"
        elif activity: tpl += f" was engaged in {activity.lower()} operations"
        else: tpl += " was underway"
        if cargo_clause: tpl += f", {cargo_clause}"
        if env_str: tpl += f" under {env_str}"
        tpl += f" when it experienced a {event_str}"
        if damage_clause: tpl += f", {damage_clause}"
        tpl += "."

    # Family B: Operational-Context-First
    elif family_idx == 1:
        pattern_id = "op_family_b_context_first"
        tpl = ""
        if phase: tpl += f"While proceeding in the {phase.lower()} phase"
        elif activity: tpl += f"While engaged in {activity.lower()} operations"
        else: tpl += "While underway"
        if env_str: tpl += f" under {env_str}"
        tpl += f", the {vtype} '{vname}'"
        if spec_parts: tpl += f" ({', '.join(spec_parts)})"
        if cargo_clause: tpl += f", {cargo_clause},"
        tpl += f" became involved in a {event_str}"
        if damage_clause: tpl += f", {damage_clause}"
        tpl += "."

    # Family C: Incident-First
    elif family_idx == 2:
        pattern_id = "op_family_c_incident_first"
        tpl = f"A {event_str} involved the {vtype} '{vname}'"
        if spec_parts: tpl += f" ({', '.join(spec_parts)})"
        if phase: tpl += f" during the {phase.lower()} phase"
        elif activity: tpl += f" while conducting {activity.lower()} operations"
        if cargo_clause: tpl += f" while {cargo_clause}"
        if env_str: tpl += f" under {env_str}"
        if damage_clause: tpl += f", with the vessel {damage_clause}"
        tpl += "."

    # Family D: Environmental-Focus
    elif family_idx == 3:
        pattern_id = "op_family_d_env_focus"
        tpl = ""
        if env_str: tpl += f"Under {env_str}, "
        else: tpl += "During transit, "
        tpl += f"the {vtype} '{vname}'"
        if spec_parts: tpl += f", {', '.join(spec_parts)},"
        if phase: tpl += f" operated in the {phase.lower()} phase"
        if cargo_clause: tpl += f" {cargo_clause}"
        tpl += f" when a {event_str} occurred"
        if damage_clause: tpl += f", {damage_clause}"
        tpl += "."

    # Family E: Consolidated Activity Focus
    else:
        pattern_id = "op_family_e_activity_focus"
        tpl = f"During maritime operations, the {vtype} '{vname}'"
        if spec_parts: tpl += f" ({', '.join(spec_parts)})"
        if activity: tpl += f" was engaged in {activity.lower()} operations"
        if env_str: tpl += f" in {env_str}"
        if cargo_clause: tpl += f", {cargo_clause},"
        tpl += f" resulting in a {event_str}"
        if damage_clause: tpl += f" and {damage_clause}"
        tpl += "."

    return render_template(tpl, var_map, pattern_id, "vessel_operational")

def generate_equipment_clause(v: dict) -> tuple:
    """Generates clean equipment narrative clause with normalized labels and deduplicated status grouping."""
    nav_list = v.get("navigation_equipment", [])
    rec_list = v.get("rec_equipment", [])
    lsa_list = v.get("lsa_equipment", [])
    
    if not (nav_list or rec_list or lsa_list):
        return "", set()
        
    on_devices = []
    off_devices = []
    for nav in nav_list:
        norm_name = nav.get("normalized_name")
        status = nav.get("status_clean", "")
        if norm_name:
            if status == "On":
                on_devices.append(norm_name)
            elif status == "Off":
                off_devices.append(norm_name)
                
    lsa_names = [lsa.get("normalized_name") for lsa in lsa_list if lsa.get("normalized_name")]
    rec_details = []
    for rec in rec_list:
        rname = rec.get("normalized_name")
        ext = rec.get("ext_status_clean")
        if rname:
            st_str = "data was extracted" if ext == "Yes" else "fitted"
            rec_details.append(f"{rname} ({st_str})")
            
    parts = []
    eq_concepts = set()
    
    if on_devices:
        eq_str = join_words_grammatical(on_devices)
        parts.append(f"Active navigation equipment included {eq_str}")
        for dev in on_devices: eq_concepts.add(f"equipment:{dev.lower()}")
        
    if off_devices:
        eq_off_str = join_words_grammatical(off_devices)
        parts.append(f"navigation equipment reported inactive included {eq_off_str}")
        for dev in off_devices: eq_concepts.add(f"equipment:{dev.lower()}")
        
    if lsa_names:
        lsa_str = join_words_grammatical(lsa_names)
        parts.append(f"onboard lifesaving equipment included {lsa_str}")
        for lsa in lsa_names: eq_concepts.add(f"safety:{lsa.lower()}")
        
    if rec_details:
        rec_str = join_words_grammatical(rec_details)
        parts.append(f"recording equipment fitted included {rec_str}")
        
    if not parts:
        return "", set()
        
    clause_text = "; ".join(parts) + "."
    return clause_text.capitalize(), eq_concepts

def generate_casualty_clause(v: dict) -> tuple:
    """Generates clean casualty & complement clause with strict singular/plural agreement."""
    crew = v.get("TotalPeopleOnBoard")
    injuries = v.get("injuries", [])
    
    minor_count, serious_count, death_count, missing_count = 0, 0, 0, 0
    water_entry_count = 0
    
    for inj in injuries:
        minor_count += int(inj.get("VictimMinorInjuries") or 0)
        serious_count += int(inj.get("VictimSeriousInjuries") or 0)
        death_count += int(inj.get("VictimDeath") or 0)
        missing_count += int(inj.get("VictimMissing") or 0)
        water_entry_count += int(inj.get("TotalPeopleInWater") or 0)

    total_inj = minor_count + serious_count + death_count + missing_count
    if crew is None and total_inj == 0 and water_entry_count == 0:
        return "", set()
        
    parts = []
    cas_concepts = set()
    
    if crew is not None and pd.notna(crew) and int(float(crew)) > 0:
        crew_num = int(float(crew))
        parts.append(f"The vessel carried {crew_num} persons on board")
        cas_concepts.add("casualty:complement")
        
    counts_parts = []
    c_minor = format_casualty_count(minor_count, "minor injury", "minor injuries")
    c_ser = format_casualty_count(serious_count, "serious injury", "serious injuries")
    c_death = format_casualty_count(death_count, "fatality", "fatalities")
    c_miss = format_casualty_count(missing_count, "missing person", "missing persons")
    
    if c_minor: counts_parts.append(c_minor)
    if c_ser: counts_parts.append(c_ser)
    if c_death: counts_parts.append(c_death)
    if c_miss: counts_parts.append(c_miss)
    
    if counts_parts:
        cas_str = join_words_grammatical(counts_parts)
        parts.append(f"resulting in reported casualties of {cas_str}")
        cas_concepts.add("casualty:injuries")
        
    if water_entry_count > 0:
        parts.append(f"with {water_entry_count} personnel entering the water")
        cas_concepts.add("casualty:water_entry")
        
    if not parts:
        return "", set()
        
    clause_text = "; ".join(parts) + "."
    return clause_text.capitalize(), cas_concepts

# ----------------------------------------------------------------------
# Knowledge-Unit Graph & Concept-Gain Driven Document Builder
# ----------------------------------------------------------------------

def build_consolidated_documents_for_vessel(oid: int, occ: dict, v: dict) -> list:
    """Builds semantically dense, non-redundant document(s) using Knowledge Unit Graph & Concept Gain Calculator."""
    vid = v.get("VesselID")
    vid_val = int(vid) if vid is not None and pd.notna(vid) else None
    
    # 1. Primary Knowledge Unit: Operational Narrative
    op_res = generate_vessel_operational_narrative(oid, occ, v)
    base_text = op_res["text"]
    current_concepts = extract_concepts(base_text)
    
    # 2. Candidate Unit: Equipment & Navigation
    eq_clause, eq_concepts = generate_equipment_clause(v)
    eq_gain = calculate_unique_concept_gain(current_concepts, eq_concepts)
    
    # 3. Candidate Unit: Casualties & Safety
    cas_clause, cas_concepts = generate_casualty_clause(v)
    cas_gain = calculate_unique_concept_gain(current_concepts, cas_concepts)
    
    # Incrementally incorporate candidate units into the primary document if concept gain > 0
    consolidated_parts = [base_text]
    spans = list(op_res["provenance"]["spans"])
    
    if eq_clause and (eq_gain >= 1 or len(eq_clause) > 20):
        consolidated_parts.append(eq_clause)
        spans.append({"rendered_span": f" {eq_clause}", "category": "equipment", "provenance": "source_derived"})
        current_concepts.update(eq_concepts)
        
    if cas_clause and (cas_gain >= 1 or len(cas_clause) > 20):
        consolidated_parts.append(cas_clause)
        spans.append({"rendered_span": f" {cas_clause}", "category": "casualty", "provenance": "source_derived"})
        current_concepts.update(cas_concepts)
        
    final_doc_text = " ".join(consolidated_parts)
    
    return [{
        "doc_type": "vessel_consolidated_narrative",
        "vessel_id": vid_val,
        "source_table": "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC",
        "res": {
            "text": final_doc_text,
            "provenance": {
                "perspective": "vessel_consolidated_narrative",
                "pattern_id": op_res["provenance"]["pattern_id"],
                "spans": spans
            }
        }
    }]

def main():
    random.seed(42)
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    merged_path = output_dir / "merged_records.jsonl"
    if not merged_path.exists():
        logger.error(f"Merged records not found at {merged_path}! Run Step 5 first.")
        return
        
    raw_docs_file = output_dir / "raw_documents.jsonl"
    logger.info("Generating documents via Knowledge Unit Graph & Concept Gain Calculator...")
    
    total_occurrences = 0
    total_docs_generated = 0
    
    with open(merged_path, "r", encoding="utf-8") as fin, open(raw_docs_file, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Generating Documents"):
            record = json.loads(line)
            oid = record["occurrence_id"]
            occ = record["occurrence"]
            vessels = record["vessels"]
            
            total_occurrences += 1
            
            # Raw TSB Occurrence Summary (if non-trivial)
            if occ and occ.get("Summary"):
                sum_text = strip_administrative_noise(str(occ["Summary"]).strip())
                if len(sum_text) >= 40:
                    fout.write(json.dumps({
                        "occurrence_id": oid,
                        "vessel_id": None,
                        "document_type": "occurrence_summary",
                        "source_table": "MDOTW_VW_OCCURRENCE_PUBLIC",
                        "document": sum_text,
                        "provenance": {
                            "perspective": "occurrence_summary",
                            "pattern_id": "raw_tsb_summary",
                            "spans": [{"rendered_span": sum_text, "category": "narrative", "source_field": "Summary", "provenance": "source_derived"}]
                        },
                        "structured": record
                    }) + "\n")
                    total_docs_generated += 1

            # Per-Vessel Consolidated Narrative
            for v in vessels:
                docs = build_consolidated_documents_for_vessel(oid, occ, v)
                for item in docs:
                    fout.write(json.dumps({
                        "occurrence_id": oid,
                        "vessel_id": item["vessel_id"],
                        "document_type": item["doc_type"],
                        "source_table": item["source_table"],
                        "document": item["res"]["text"],
                        "provenance": item["res"]["provenance"],
                        "structured": record
                    }) + "\n")
                    total_docs_generated += 1
                    
    logger.info(f"Document generation complete:")
    logger.info(f"  Total occurrences processed: {total_occurrences}")
    logger.info(f"  Total documents generated: {total_docs_generated} (Avg {total_docs_generated/total_occurrences:.2f} docs/occ)")

if __name__ == "__main__":
    main()
