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
    """Loads text templates for vessels, injuries, and equipment from config-defined JSON files."""
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
    """Joins a list of strings with commas and 'and' naturally."""
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"

def clean_db_label(val: str) -> str:
    """Removes database-specific annotations and administrative comments from string values."""
    if not val:
        return ""
    val_clean = str(val).strip()
    # Remove database notes like " - deactivated nov. 1995" or similar
    val_clean = re.sub(r'\s*-\s*deactivated\s+\w+\.?\s+\d{4}', '', val_clean, flags=re.IGNORECASE)
    val_clean = re.sub(r'\s*-\s*active\s+\w+\.?\s+\d{4}', '', val_clean, flags=re.IGNORECASE)
    
    # General pattern for trailing comments starting with a dash and containing a 4-digit year
    val_clean = re.sub(r'\s*-\s*.*?(19\d{2}|20\d{2})', '', val_clean, flags=re.IGNORECASE)
    
    # Strip double spaces
    val_clean = re.sub(r'\s+', ' ', val_clean)
    return val_clean.strip()

def clean_placeholder(val, default_str=None):
    """Removes database placeholder strings and converts nulls to a default value."""
    if val is None or str(val).strip().upper() in ["", "NAN", "UNKNOWN", "UNSPECIFIED"]:
        return default_str
    return clean_db_label(str(val).strip())

# Core vocabulary maps for concept classification
CONCEPT_MAPPING = {
    # Entity Concepts
    "entities": {
        "equipment": ["radar", "compass", "vhf", "radio", "gps", "ecdis", "ais", "vdr", "sounder", "lifeboat", "liferaft", "suit", "jacket", "beacon", "epirb", "sart", "recorder", "device", "systems"],
        "cargo": ["cargo", "timber", "oil", "goods", "coal", "ore", "container", "wheat", "grain", "chemical", "gas", "passenger", "fish", "logs", "lumber"],
        "vessel": ["vessel", "ship", "boat", "barge", "tug", "tanker", "bulk", "cargo", "passenger", "ferry", "fishing", "carrier", "trawler"],
        "people": ["crew", "passenger", "person", "people", "crewmember", "captain", "pilot", "master", "seaman", "officer"],
        "location": ["river", "harbour", "port", "gulf", "channel", "bay", "sea", "ocean", "anchorage", "dock", "berth", "bridge"]
    },
    # Attribute Concepts
    "attributes": {
        "operational": ["active", "operational", "on", "functioning", "rely", "utilized", "deployed", "used"],
        "inactive": ["inactive", "off", "non-operational", "failure", "fail", "not used", "not deployed", "deactivated"],
        "approved": ["approved", "certified"],
        "vessel_characteristics": ["steel", "aluminum", "wood", "fiberglass", "tonnage", "gt", "gross", "built", "propulsion", "diesel", "engine", "flag", "registry", "length", "breadth"]
    },
    # Relation Concepts
    "relations": {
        "action": ["transporting", "carrying", "laden", "laden with", "encountered", "navigated", "relied", "operating", "berthed", "docked", "anchored", "damaged by", "collision with", "sunk after", "fell from", "boarded by"]
    },
    # Outcome Concepts
    "outcomes": {
        "events": ["grounded", "grounding", "collision", "fire", "sinking", "capsizing", "explosion", "flooding", "leak", "strike", "contact"],
        "casualties": ["fatality", "fatalities", "death", "injury", "injuries", "casualty", "casualties", "missing", "unharmed", "safe", "survived"]
    }
}

def normalize_equipment_list(raw_list: list, dictionary_metadata: dict = None) -> list:
    """Cleans, normalizes, trims, case-normalizes, deduplicates, and consistently sorts equipment records."""
    if not raw_list:
        return []
    normalized_set = set()
    for item in raw_list:
        if not item:
            continue
        cleaned = clean_db_label(str(item).strip())
        if not cleaned:
            continue
        normalized = normalize_label(cleaned, dictionary_metadata)
        if normalized:
            words = normalized.split()
            c_words = []
            for w in words:
                if w.isupper() and len(w) > 1:
                    c_words.append(w)
                else:
                    c_words.append(w.capitalize())
            normalized_clean = " ".join(c_words)
            if normalized_clean:
                normalized_set.add(normalized_clean)
    return sorted(list(normalized_set))

def extract_concepts(text: str) -> set:
    """Parses text and extracts a set of semantic concept strings (e.g. 'entity:equipment:radar')."""
    if not text:
        return set()
    
    words = re.findall(r'\b\w+\b', text.lower())
    extracted = set()
    
    for category, subcategories in CONCEPT_MAPPING.items():
        for subcat, keywords in subcategories.items():
            for kw in keywords:
                if kw in words or any(kw in w for w in words):
                    extracted.add(f"{category}:{subcat}:{kw}")
                    
    if re.search(r'\d+\s*(gt|gross|tons)', text.lower()):
        extracted.add("attribute:vessel_characteristics:tonnage_val")
    if re.search(r'\b(19\d{2}|20\d{2})\b', text):
        extracted.add("attribute:vessel_characteristics:built_year_val")
    if re.search(r'\d+\s*(knots|kt)', text.lower()):
        extracted.add("attribute:vessel_characteristics:speed_val")
        
    return extracted

def calculate_information_density(concepts: set, occurrence_fingerprints: list) -> int:
    """Calculates the Information Density Score for a set of concepts."""
    if not concepts:
        return 0
        
    all_accepted = set()
    for fp in occurrence_fingerprints:
        all_accepted.update(fp)
        
    score = 0
    new_concepts = concepts - all_accepted
    
    if not new_concepts:
        score -= 2
    else:
        for c in new_concepts:
            parts = c.split(":")
            if len(parts) >= 2:
                cat = parts[0]
                subcat = parts[1]
                if cat in ["entities", "outcomes"]:
                    score += 2
                elif subcat == "action":
                    score += 2
                elif subcat in ["vessel_characteristics", "operational", "inactive"]:
                    score += 1
            else:
                score += 1
                
    for c in concepts:
        if c in ["attribute:vessel_characteristics:tonnage_val", 
                 "attribute:vessel_characteristics:built_year_val", 
                 "attribute:vessel_characteristics:speed_val"]:
            if c not in all_accepted:
                score += 1
                
    intersect = concepts & all_accepted
    if len(intersect) > 3:
        score -= 2
        
    return score



def normalize_label(val: str, dictionary_metadata: dict = None) -> str:
    """Normalizes technical labels to natural English.
    Uses parsed dictionary as the primary normalization source where possible,
    and falls back to a configurable normalization map.
    """
    if not val:
        return ""
    
    val_clean = clean_db_label(str(val).strip())
    val_lower = val_clean.lower()
    
    # 1. Primary Source: Check if we can resolve it from dictionary metadata
    if dictionary_metadata and "registry" in dictionary_metadata:
        for table, cols in dictionary_metadata["registry"].items():
            for col, info in cols.items():
                if col.lower() == val_lower:
                    full_name = info.get("full_name", "")
                    if full_name:
                        # Clean up full name (remove parentheses like (English), (French), etc.)
                        full_name_clean = re.sub(r'\s*\((English|French|Eng|Fre)\)\s*', '', full_name, flags=re.IGNORECASE)
                        return clean_db_label(full_name_clean.strip())
                        
    # 2. Configurable fallback normalization map
    fallback_map = {
        "radar1": "radar",
        "radar2": "radar",
        "radar3": "radar",
        "mf/hf": "MF/HF radio",
        "vhf": "VHF radio",
        "gps": "GPS",
        "ecdis": "ECDIS",
        "ais": "AIS",
        "vdr": "VDR",
        "bnwas": "BNWAS",
        "gyro compass": "gyro compass",
        "magnetic compass": "magnetic compass",
        "direction_finder": "direction finder",
        "rt_am": "AM radio",
        "loran_c": "Loran-C",
        "inm-c": "Inmarsat-C",
        "gps_receiver": "GPS receiver"
    }
    
    if val_lower in fallback_map:
        return fallback_map[val_lower]
        
    # 3. Dynamic Regex/Heuristic Cleaning Rules
    # Remove trailing numbers from technical strings (e.g. "radar1" -> "radar")
    normalized = val_clean
    normalized = re.sub(r'(\w+?)\d+$', r'\1', normalized)
    normalized = normalized.replace("_", " ").replace("-", " ")
    
    # Capitalize acronyms
    acronyms = {"gps", "ecdis", "vhf", "ais", "vdr", "bnwas", "mf", "hf", "lsa", "sart", "epirb"}
    words = normalized.split()
    cleaned_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in acronyms:
            cleaned_words.append(w.upper())
        elif w_lower == "mf/hf":
            cleaned_words.append("MF/HF")
        else:
            cleaned_words.append(w)
    normalized = " ".join(cleaned_words)
    
    return normalized

def should_emit_no_casualty(occ: dict, vessels: list) -> bool:
    """Returns True if the occurrence involves a major incident type/keyword that warrants reporting no casualties."""
    if not occ:
        return False
        
    major_keywords = {
        "fire", "collision", "grounding", "sinking", "foundering", "sink", "capsiz", 
        "explosion", "flooding", "leak", "ingress", "striking", "allision", "abandonment"
    }
    
    # 1. Check AccIncTypeDisplayEng in occurrence
    occ_inc_type = str(occ.get("AccIncTypeDisplayEng", "")).lower()
    for kw in major_keywords:
        if kw in occ_inc_type:
            return True
            
    # 2. Check Summary in occurrence
    occ_summary = str(occ.get("Summary", "")).lower()
    for kw in major_keywords:
        if kw in occ_summary:
            return True
            
    # 3. Check AccIncTypeDisplayEng in any of the vessels
    for v in vessels:
        v_inc_type = str(v.get("AccIncTypeDisplayEng", "")).lower()
        for kw in major_keywords:
            if kw in v_inc_type:
                return True
                
    return False

def format_environmental(occ: dict, dictionary_metadata: dict = None) -> str:
    """Generates a natural-language description of environmental conditions.
    Skips the section entirely if all attributes are missing/unknown (Low-Information Suppression).
    """
    weather = occ.get("WeatherConditionDisplayEng")
    wind_dir = occ.get("WindDirection")
    wind_speed = occ.get("WindSpeed_Knots")
    sea_state = occ.get("SeaStateDisplayEng")
    visibility = occ.get("VisibilityDistance_Nm")
    light = occ.get("LightConditionDisplayEng")
    air_temp = occ.get("AirTemp_Celsius")
    
    # Normalizing helper
    def clean_val(v):
        if v is None or str(v).strip().upper() in ["", "NAN", "UNKNOWN", "UNSPECIFIED"]:
            return None
        return clean_db_label(str(v).strip())
        
    weather = clean_val(weather)
    light = clean_val(light)
    sea_state = clean_val(sea_state)
    wind_dir = clean_val(wind_dir)
    
    has_wind_speed = wind_speed is not None and str(wind_speed).strip() not in ["", "nan", "NaN"]
    has_visibility = visibility is not None and str(visibility).strip() not in ["", "nan", "NaN"]
    has_temp = air_temp is not None and str(air_temp).strip() not in ["", "nan", "NaN"]
    
    # Skip section entirely if all attributes are missing/unknown
    if not (weather or light or sea_state or wind_dir or has_wind_speed or has_visibility or has_temp):
        return ""
        
    parts = []
    
    # 1. Weather and Light
    if weather and light:
        weather_light_templates = [
            "The occurrence took place during {weather} weather in {light} conditions.",
            "Under {light} conditions, the incident occurred during {weather} weather.",
            "The vessel encountered {weather} weather under {light} conditions.",
            "At the time of the incident, conditions were {light} with {weather} weather."
        ]
        parts.append(random.choice(weather_light_templates).format(
            weather=weather.lower(),
            light=light.lower()
        ))
    elif weather:
        weather_templates = [
            "The weather at the time of the occurrence was {weather}.",
            "Conditions at the time of the incident featured {weather} weather.",
            "During the occurrence, weather was reported as {weather}.",
            "The incident occurred under {weather} weather."
        ]
        parts.append(random.choice(weather_templates).format(weather=weather.lower()))
    elif light:
        light_templates = [
            "The incident occurred during {light} conditions.",
            "At the time of the occurrence, conditions were {light}.",
            "The event took place in {light} conditions."
        ]
        parts.append(random.choice(light_templates).format(light=light.lower()))
        
    # 2. Wind
    if has_wind_speed and wind_dir:
        wind_templates = [
            "Winds were recorded at {speed} knots from the {dir}.",
            "Winds were blowing from the {dir} at {speed} knots.",
            "Wind speed was reported as {speed} knots from the {dir} direction."
        ]
        parts.append(random.choice(wind_templates).format(
            speed=int(float(wind_speed)),
            dir=wind_dir
        ))
    elif has_wind_speed:
        wind_speed_templates = [
            "Wind speed was reported as {speed} knots.",
            "Winds of {speed} knots were recorded at the time.",
            "Wind velocity was noted at {speed} knots."
        ]
        parts.append(random.choice(wind_speed_templates).format(speed=int(float(wind_speed))))
    elif wind_dir:
        wind_dir_templates = [
            "Winds were blowing from the {dir}.",
            "The wind direction was reported from the {dir}."
        ]
        parts.append(random.choice(wind_dir_templates).format(dir=wind_dir))
        
    # 3. Sea state
    if sea_state:
        sea_templates = [
            "The sea state was described as {sea}.",
            "Sea conditions were reported as {sea}.",
            "The vessel encountered sea conditions described as {sea}."
        ]
        parts.append(random.choice(sea_templates).format(sea=sea_state.lower()))
        
    # 4. Visibility
    if has_visibility:
        vis_templates = [
            "Visibility was estimated at {vis} nautical miles.",
            "Visibility was reported to be {vis} nautical miles.",
            "The visibility at the time was approximately {vis} nautical miles."
        ]
        parts.append(random.choice(vis_templates).format(vis=visibility))
        
    # 5. Temperature
    if has_temp:
        temp_templates = [
            "The ambient air temperature was {temp} degrees Celsius.",
            "Air temperature was recorded at {temp} degrees Celsius.",
            "The temperature at the time was {temp} degrees Celsius."
        ]
        parts.append(random.choice(temp_templates).format(temp=int(float(air_temp))))
        
    return " ".join(parts)

def format_vessel(v: dict, v_tpl: dict, i_tpl: dict, e_tpl: dict, dictionary_metadata: dict = None, occurrence: dict = None, other_vessels: list = None) -> str:
    """Generates natural language paragraphs describing a vessel, its voyage, damage, equipment, and injuries."""
    vname = v.get("VesselName") or "An unnamed vessel"
    vtype = v.get("VesselTypeDisplayEng") or "vessel"
    vflag = v.get("VesselFlagDisplayEng")
    hull = v.get("HullMaterialDisplayEng")
    prop = v.get("PropulsionTypeDisplayEng")
    tonnage = v.get("GrossTonnage")
    built = v.get("YearBuilt")
    phase = v.get("VesselPhaseDisplayEng")
    activity = v.get("ActivityTypeDisplayEng")
    
    vflag = clean_placeholder(vflag)
    hull = clean_placeholder(hull)
    prop = clean_placeholder(prop)
    phase = clean_placeholder(phase)
    activity = clean_placeholder(activity)
    
    # Normalize display terms to lowercase
    vtype = vtype.lower()
    if vflag:
        vflag = vflag.title()
    if hull:
        hull = hull.lower()
    if prop:
        prop = prop.lower()
    if phase:
        phase = phase.lower()
    if activity:
        activity = activity.lower()
        
    vessel_paragraphs = []
    
    is_placeholder = v.get("_placeholder", False)
    if is_placeholder:
        vessel_paragraphs.append("The following details were reported for an unnamed vessel involved in the occurrence:")
    else:
        # 1. Basic profile (using template families)
        flag_str = vflag
        hull_str = hull
        
        ton_str = None
        if tonnage is not None and str(tonnage).strip() not in ["", "nan", "NaN"] and float(tonnage) > 0:
            ton_str = f"{float(tonnage):.0f} GT"
            
        built_str = None
        if built is not None and str(built).strip() not in ["", "nan", "NaN"] and int(float(built)) > 0:
            built_str = str(int(float(built)))
            
        profile_parts = []
        
        # Build description parts dynamically, completely omitting missing fields (rather than writing boilerplate)
        if built_str and ton_str and flag_str and hull_str:
            tpl_basic = "Built in {year_built}, '{vessel_name}' is a {vessel_type} ({vessel_flag}) featuring a {hull_material} hull and a gross tonnage of {gross_tonnage}."
            profile_parts.append(tpl_basic.format(
                vessel_name=vname,
                vessel_type=vtype,
                vessel_flag=flag_str,
                hull_material=hull_str,
                gross_tonnage=ton_str,
                year_built=built_str
            ))
        elif built_str and ton_str:
            tpl_basic = random.choice(v_tpl["basic_profiles"])
            profile_parts.append(tpl_basic.format(
                vessel_name=vname,
                vessel_type=vtype,
                vessel_flag=flag_str or "unspecified flag",
                hull_material=hull_str or "unspecified material",
                gross_tonnage=ton_str,
                year_built=built_str
            ))
        else:
            # Simplified description omitting missing values entirely
            desc_pieces = [f"'{vname}' is a {vtype}"]
            if flag_str:
                desc_pieces.append(f"registered in {flag_str}")
            if hull_str:
                desc_pieces.append(f"constructed of {hull_str}")
            if ton_str:
                desc_pieces.append(f"with a gross tonnage of {ton_str}")
            if built_str:
                desc_pieces.append(f"built in {built_str}")
                
            profile_parts.append(", ".join(desc_pieces) + ".")
            
        vessel_paragraphs.append(" ".join(profile_parts))
        
        # 2. Voyage activity
        act_str = activity.lower().strip() if activity else ""
        phase_valid = phase and phase.upper() not in ["UNKNOWN", "UNSPECIFIED"]
        act_valid = activity and activity.upper() not in ["UNKNOWN", "UNSPECIFIED"]
        
        voyage_parts = []
        if phase_valid and act_valid:
            tpl_voyage = random.choice(v_tpl["voyage_activity"])
            if "{activity_type} operations" in tpl_voyage and (act_str.endswith("operations") or act_str.endswith("ops")):
                tpl_voyage = tpl_voyage.replace("{activity_type} operations", "{activity_type}")
            voyage_parts.append(tpl_voyage.format(
                vessel_name=vname,
                vessel_phase=phase,
                activity_type=act_str
            ))
        elif phase_valid:
            tpl_options = [
                "At the time of the occurrence, the vessel was {vessel_phase}.",
                "The vessel '{vessel_name}' was operating in the {vessel_phase} phase during the voyage.",
                "The ship was {vessel_phase} when the incident occurred."
            ]
            voyage_parts.append(random.choice(tpl_options).format(
                vessel_name=vname,
                vessel_phase=phase
            ))
        elif act_valid:
            tpl_options = [
                "During the voyage, the vessel was engaged in {activity_type}.",
                "The ship was performing {activity_type} operations at the time of the incident.",
                "'{vessel_name}' was engaged in {activity_type} when the occurrence took place."
            ]
            tpl_choice = random.choice(tpl_options)
            if "{activity_type} operations" in tpl_choice and (act_str.endswith("operations") or act_str.endswith("ops")):
                tpl_choice = tpl_choice.replace("{activity_type} operations", "{activity_type}")
            voyage_parts.append(tpl_choice.format(
                vessel_name=vname,
                activity_type=act_str
            ))
            
        if voyage_parts:
            vessel_paragraphs.append(" ".join(voyage_parts))
            
        # 3. Cargo info
        cargo_prod = v.get("CargoProductTypeDisplayEng")
        cargo_qty = v.get("QuantityOnBoard")
        
        cargo_prod = clean_placeholder(cargo_prod)
        if cargo_prod and cargo_qty is not None and str(cargo_qty).strip() not in ["", "nan", "NaN"]:
            tpl_cargo = random.choice(v_tpl["cargo_info"])
            vessel_paragraphs.append(tpl_cargo.format(
                quantity=f"{cargo_qty}",
                cargo_product=cargo_prod.lower()
            ))
        
    # 4. Navigation Equipment (Omitted if empty)
    nav_list = v.get("navigation_equipment", [])
    if nav_list:
        on_devices = []
        off_devices = []
        seen_nav = set()
        for nav in nav_list:
            nav_type = nav.get("NavigationAidTypeDisplayEng")
            nav_status = nav.get("OnOffEnumDisplayEng")
            if nav_type:
                nav_name = normalize_label(nav_type, dictionary_metadata)
                status_str = nav_status.lower().strip() if nav_status else "not specified"
                
                if status_str == "not specified":
                    continue
                    
                key = (nav_name.lower(), status_str)
                if key not in seen_nav:
                    seen_nav.add(key)
                    if status_str in ["on", "active"]:
                        on_devices.append(nav_name)
                    elif status_str in ["off", "inactive"]:
                        off_devices.append(nav_name)
                        
        nav_sentences = []
        if on_devices:
            tpl_active = random.choice(e_tpl.get("nav_active_group", [
                "Active navigation equipment on board included {on_list}."
            ]))
            nav_sentences.append(tpl_active.format(on_list=join_words(on_devices)))
        if off_devices:
            tpl_inactive = random.choice(e_tpl.get("nav_inactive_group", [
                "Meanwhile, the {off_list} were reported as inactive."
            ]))
            # Dynamic grammar agreement rules (Point 6)
            if len(off_devices) == 1:
                tpl_inactive = tpl_inactive.replace(" were ", " was ")
            nav_sentences.append(tpl_inactive.format(off_list=join_words(off_devices)))
            
        if nav_sentences:
            vessel_paragraphs.append(" ".join(nav_sentences))
            
    # 5. LSA Equipment (Omitted if empty)
    lsa_list = v.get("lsa_equipment", [])
    if lsa_list:
        seen_lsa = set()
        lsa_sentences = []
        for lsa in lsa_list:
            lse_type = lsa.get("LsApplianceDisplayEng")
            lse_used = lsa.get("UsedEnumDisplayEng")
            lse_approved = lsa.get("ApprovedEnumDisplayEng")
            if lse_type:
                lse_name = normalize_label(lse_type, dictionary_metadata)
                used_str = "deployed and used" if lse_used == "Yes" else "not used"
                
                # Placeholder Normalization (Point 4)
                if lse_approved == "Yes":
                    tpl_lsa = "Life-saving appliances included {lse_type}, which was approved and {lse_used}."
                else:
                    tpl_lsa = "Life-saving appliances included {lse_type}, which was {lse_used}, although its approval status was not reported."
                
                key = (lse_name.lower(), used_str, lse_approved == "Yes")
                if key not in seen_lsa:
                    seen_lsa.add(key)
                    lsa_sentences.append(tpl_lsa.format(
                        lse_type=lse_name,
                        lse_used=used_str
                    ))
        if lsa_sentences:
            vessel_paragraphs.append(" ".join(lsa_sentences))
            
    # 6. Recording Equipment (Omitted if empty)
    rec_list = v.get("rec_equipment", [])
    if rec_list:
        seen_rec = set()
        rec_sentences = []
        for rec in rec_list:
            rec_type = rec.get("RecordingEquipDisplayEng")
            extracted = rec.get("DataExtractedEnumDisplayEng")
            seized = rec.get("EquipSeizedEnumDisplayEng")
            
            if rec_type:
                rec_name = normalize_label(rec_type, dictionary_metadata)
                ext_str = "Data was successfully extracted" if extracted == "Yes" else "Data could not be extracted"
                if seized == "Yes":
                    ext_str += " and the equipment was seized by investigators"
                
                key = (rec_name.lower(), ext_str)
                if key not in seen_rec:
                    seen_rec.add(key)
                    tpl_rec = random.choice(e_tpl["rec_equipment_status"])
                    rec_sentences.append(tpl_rec.format(
                        rec_type=rec_name,
                        rec_extract_status=ext_str
                    ))
        if rec_sentences:
            vessel_paragraphs.append(" ".join(rec_sentences))
            
    # 7. Casualties (Total People on Board + Injuries)
    crew = v.get("TotalPeopleOnBoard")
    injuries = v.get("injuries", [])
    
    crew_sentences = []
    if crew is not None and str(crew).strip() not in ["", "nan", "NaN"] and int(float(crew)) >= 0:
        tpl_crew = random.choice(i_tpl["personnel_count"])
        crew_sentences.append(tpl_crew.format(crew_count=int(float(crew))))
        
    has_injuries = False
    inj_details = []
    if injuries:
        for inj in injuries:
            minor = inj.get("VictimMinorInjuries") or 0
            serious = inj.get("VictimSeriousInjuries") or 0
            deaths = inj.get("VictimDeath") or 0
            missing = inj.get("VictimMissing") or 0
            
            if minor > 0 or serious > 0 or deaths > 0 or missing > 0:
                has_injuries = True
                counts = []
                if minor > 0: counts.append(f"{minor} minor injury/injuries")
                if serious > 0: counts.append(f"{serious} serious injury/injuries")
                if deaths > 0: counts.append(f"{deaths} fatality/fatalities")
                if missing > 0: counts.append(f"{missing} person/people missing")
                inj_details.append(", ".join(counts))
                
    if has_injuries:
        tpl_inj = random.choice(i_tpl["injuries_reported"])
        crew_sentences.append(tpl_inj.format(details="; ".join(inj_details)))
    else:
        # If there are no injuries, selectively report it (Point 3 / Conditional Casualty)
        all_vessels = [v] + (other_vessels or [])
        if occurrence and should_emit_no_casualty(occurrence, all_vessels):
            tpl_no_inj = random.choice(i_tpl["no_injuries"])
            crew_sentences.append(tpl_no_inj)
            
    if crew_sentences:
        vessel_paragraphs.append(" ".join(crew_sentences))
        
    if not is_placeholder:
        # 8. Damage Assessment (Skip NONE or NONE APPARENT)
        dmg_degree = v.get("VesselDamageDegreeDisplayEng")
        dmg_loc = v.get("VesselDamageLocationDisplayEng")
        dmg_degree = clean_placeholder(dmg_degree)
        if dmg_degree and dmg_degree.upper() not in ["NONE", "NONE APPARENT"]:
            tpl_dmg = random.choice(v_tpl["damage_info"])
            loc_str = dmg_loc.lower() if dmg_loc else "hull"
            vessel_paragraphs.append(tpl_dmg.format(
                damage_degree=dmg_degree.lower(),
                damage_location=loc_str
            ))
            
        # 9. Pollution Information (Skip NONE or NONE APPARENT or UNKNOWN)
        pollution = v.get("SeaPollutionDegreeDisplayEng")
        pollution = clean_placeholder(pollution)
        if pollution and pollution.upper() not in ["NONE", "NONE APPARENT", "UNKNOWN"]:
            tpl_poll = random.choice(v_tpl["pollution_info"])
            vessel_paragraphs.append(tpl_poll.format(
                pollution_degree=pollution.lower()
            ))
        
    return " ".join(vessel_paragraphs)

# ----------------------------------------------------------------------
# Modular Document Generators
# ----------------------------------------------------------------------

def generate_occurrence_summary(oid: int, record: dict, v_tpl: dict, i_tpl: dict, e_tpl: dict, dictionary_metadata: dict = None) -> str:
    """Generates the primary occurrence-level document summary (raw TSB summary text only)."""
    occ = record.get("occurrence")
    if not occ:
        return ""
    raw_summary = occ.get("Summary")
    if raw_summary and str(raw_summary).strip() not in ["", "nan", "NaN"]:
        summary_clean = str(raw_summary).strip()
        # Suppress isolated administrative notes that contain no pretraining content
        if summary_clean.startswith("Note: formerly OccNo") and len(summary_clean) < 60:
            return ""
        return summary_clean
    return ""

def generate_environment(oid: int, record: dict, dictionary_metadata: dict = None) -> str:
    """Generates environment weather and condition document for orphan occurrences, only if they describe non-trivial conditions."""
    occ = record.get("occurrence")
    if not occ:
        return ""
    weather_raw = occ.get("WeatherConditionDisplayEng")
    weather_clean = clean_placeholder(weather_raw)
    sea_raw = occ.get("SeaStateDisplayEng")
    sea_clean = clean_placeholder(sea_raw)
    
    is_hazardous = False
    if weather_clean and weather_clean.upper() not in ["CLEAR", "CLOUDY", "SUNNY", "FAIR", "FINE", "NORMAL", "UNKNOWN"]:
        is_hazardous = True
    if sea_clean and sea_clean.upper() not in ["CALM", "SMOOTH", "UNKNOWN", "NORMAL"]:
        is_hazardous = True
        
    if not is_hazardous:
        return "" # suppress benign weather for orphan occurrences
        
    return format_environmental(occ, dictionary_metadata)

def build_integrated_context_for_vessel(oid: int, occ: dict, v: dict, v_tpl: dict, i_tpl: dict, e_tpl: dict, dictionary_metadata: dict = None) -> str:
    """Builds a single, rich integrated narrative context document for a vessel, applying relationship-first logic,

    narrative compression, weather importance, and conditional negative facts.
    """
    # 1. Basic properties
    vname = v.get("VesselName") or "An unnamed vessel"
    vtype = (v.get("VesselTypeDisplayEng") or "vessel").lower()
    vflag = clean_placeholder(v.get("VesselFlagDisplayEng"))
    hull = clean_placeholder(v.get("HullMaterialDisplayEng"))
    prop = clean_placeholder(v.get("PropulsionTypeDisplayEng"))
    tonnage = v.get("GrossTonnage")
    built = v.get("YearBuilt")
    
    is_placeholder = v.get("_placeholder", False)
    vflag_str = vflag.title() if vflag else None
    hull_str = hull.lower() if hull else None
    prop_str = prop.lower() if prop else None
    ton_str = f"{float(tonnage):.0f} GT" if tonnage and pd.notna(tonnage) and float(tonnage) > 0 else None
    built_str = str(int(float(built))) if built and pd.notna(built) and int(float(built)) > 0 else None
    
    # 2. Significant events detection
    occ_no = occ.get("OccNo") or f"ID {oid}"
    occ_type = occ.get("OccurrenceTypeDisplayEng")
    occ_type_clean = clean_placeholder(occ_type)
    
    is_significant_event = False
    event_name = None
    if occ_type_clean and occ_type_clean.upper() not in ["UNKNOWN", "UNSPECIFIED", "OTHER"]:
        is_significant_event = True
        event_name = occ_type_clean.lower()
        
    dmg_degree = clean_placeholder(v.get("VesselDamageDegreeDisplayEng"))
    dmg_loc = clean_placeholder(v.get("VesselDamageLocationDisplayEng"))
    has_damage = dmg_degree and dmg_degree.upper() not in ["NONE", "NONE APPARENT"]
    if has_damage:
        is_significant_event = True
        
    pollution = clean_placeholder(v.get("SeaPollutionDegreeDisplayEng"))
    has_pollution = pollution and pollution.upper() not in ["NONE", "NONE APPARENT", "UNKNOWN"]
    if has_pollution:
        is_significant_event = True

    # Casualties count
    injuries = v.get("injuries", [])
    minor_count = sum(int(inj.get("VictimMinorInjuries") or 0) for inj in injuries)
    serious_count = sum(int(inj.get("VictimSeriousInjuries") or 0) for inj in injuries)
    death_count = sum(int(inj.get("VictimDeath") or 0) for inj in injuries)
    missing_count = sum(int(inj.get("VictimMissing") or 0) for inj in injuries)
    total_casualties = minor_count + serious_count + death_count + missing_count
    
    if total_casualties > 0:
        is_significant_event = True

    # 3. Weather assessment
    weather_raw = occ.get("WeatherConditionDisplayEng")
    weather_clean = clean_placeholder(weather_raw)
    sea_raw = occ.get("SeaStateDisplayEng")
    sea_clean = clean_placeholder(sea_raw)
    
    is_weather_hazardous = False
    weather_desc_list = []
    
    if weather_clean:
        if weather_clean.upper() not in ["UNKNOWN"]:
            weather_desc_list.append(f"{weather_clean.lower()} weather")
        if weather_clean.upper() not in ["CLEAR", "CLOUDY", "SUNNY", "FAIR", "FINE", "NORMAL", "UNKNOWN"]:
            is_weather_hazardous = True
            
    if sea_clean:
        if sea_clean.upper() not in ["UNKNOWN"]:
            weather_desc_list.append(f"{sea_clean.lower()} seas")
        if sea_clean.upper() not in ["CALM", "SMOOTH", "UNKNOWN", "NORMAL"]:
            is_weather_hazardous = True
            
    weather_summary = " and ".join(weather_desc_list)
    
    # Weather Importance Rule: Suppress clear/benign weather unless part of an integrated narrative or contrasts with an incident
    include_weather = False
    if weather_summary:
        if is_weather_hazardous or is_significant_event:
            include_weather = True

    # 4. Voyage phase and cargo
    phase = v.get("VesselPhaseDisplayEng")
    activity = v.get("ActivityTypeDisplayEng")
    phase = clean_placeholder(phase)
    activity = clean_placeholder(activity)
    phase_valid = phase and phase.upper() not in ["UNKNOWN", "UNSPECIFIED"]
    act_valid = activity and activity.upper() not in ["UNKNOWN", "UNSPECIFIED"]
    
    activity_clause = ""
    if phase_valid and act_valid:
        p_lower = phase.lower()
        a_lower = activity.lower()
        if p_lower.startswith("underway"):
            activity_clause = f"{p_lower} while engaged in {a_lower}"
        else:
            activity_clause = f"underway {p_lower} while engaged in {a_lower}"
    elif phase_valid:
        p_lower = phase.lower()
        if p_lower.startswith("underway"):
            activity_clause = p_lower
        else:
            activity_clause = f"underway {p_lower}"
    elif act_valid:
        activity_clause = f"engaged in {activity.lower()} operations"
        
    cargo_prod = v.get("CargoProductTypeDisplayEng")
    cargo_qty = v.get("QuantityOnBoard")
    cargo_prod = clean_placeholder(cargo_prod)
    
    cargo_clause = ""
    if cargo_prod:
        if cargo_qty is not None and str(cargo_qty).strip() not in ["", "nan", "NaN"]:
            cargo_clause = f"carrying {cargo_qty} of {cargo_prod.lower()}"
        else:
            cargo_clause = f"carrying a cargo of {cargo_prod.lower()}"

    # 5. Active/failed equipment
    nav_list = v.get("navigation_equipment", [])
    on_raw = [n.get("NavigationAidTypeDisplayEng") for n in nav_list if n.get("OnOffEnumDisplayEng") == "On"]
    off_raw = [n.get("NavigationAidTypeDisplayEng") for n in nav_list if n.get("OnOffEnumDisplayEng") == "Off"]
    on_devices = normalize_equipment_list(on_raw, dictionary_metadata)
    off_devices = normalize_equipment_list(off_raw, dictionary_metadata)
    
    nav_clause = ""
    if on_devices:
        nav_clause = f"relying on active {join_words(on_devices)}"

    # 6. Build Narrative Blocks (Operational Narrative)
    # Block A: The vessel specifications
    v_noun = f"'{vname}'"
    if vtype and vtype.upper() not in ["UNKNOWN", "UNSPECIFIED"]:
        v_noun = f"the {vtype} '{vname}'"
        
    spec_details = []
    if built_str:
        spec_details.append(f"built in {built_str}")
    if ton_str:
        spec_details.append(f"with a gross tonnage of {ton_str}")
    if vflag_str:
        spec_details.append(f"registered in {vflag_str}")
        
    vessel_desc = ""
    if is_placeholder:
        if spec_details:
            vessel_desc = f"An unnamed vessel, {', '.join(spec_details)},"
        else:
            vessel_desc = "An unnamed vessel"
    else:
        if spec_details:
            vessel_desc = f"{v_noun[0].upper() + v_noun[1:]}, {', '.join(spec_details)},"
        else:
            vessel_desc = f"{v_noun[0].upper() + v_noun[1:]}"

    # Block B: Operational Transit (Narrative Compression)
    transit_parts = []
    if activity_clause:
        transit_parts.append(activity_clause)
    if cargo_clause:
        transit_parts.append(cargo_clause)
    if include_weather and weather_summary:
        transit_parts.append(f"in {weather_summary}")
    if nav_clause:
        transit_parts.append(nav_clause)
        
    transit_sentence = ""
    if transit_parts:
        transit_sentence = f"While {', '.join(transit_parts)}, {vessel_desc.lower() if vessel_desc.startswith('The') or vessel_desc.startswith('An') else vessel_desc}"
    else:
        transit_sentence = vessel_desc

    # Block C: Incident outcome
    outcome_verb = ""
    if event_name:
        if event_name in ["grounding", "grounded"]:
            outcome_verb = "grounded"
        elif event_name in ["collision", "collided"]:
            outcome_verb = random.choice([
                "was involved in a collision",
                "collided with another vessel",
                "suffered a collision"
            ])
        elif event_name in ["fire", "explosion"]:
            outcome_verb = random.choice([
                f"experienced a {event_name}",
                f"suffered a {event_name} on board"
            ])
        else:
            outcome_verb = random.choice([
                f"was involved in a {event_name} occurrence",
                f"experienced a {event_name} incident"
            ])
    else:
        default_verbs = [
            "was involved in an operational incident",
            "experienced a shipboard occurrence",
            "encountered an unspecified marine event",
            "was involved in a vessel-related occurrence"
        ]
        outcome_verb = random.choice(default_verbs)
            
    # Block D: Consequence Descriptors (Damage + Pollution + Casualties)
    consequence_parts = []
    
    if has_damage:
        loc = f" to the {dmg_loc.lower()}" if dmg_loc else "hull"
        consequence_parts.append(random.choice([
            f"sustaining {dmg_degree.lower()} damage to the {loc}",
            f"resulting in {dmg_degree.lower()} damage to the {loc}"
        ]))
        
    if has_pollution:
        consequence_parts.append(random.choice([
            f"causing {pollution.lower()} sea pollution",
            f"resulting in {pollution.lower()} sea pollution"
        ]))
        
    consequence_clause = " and ".join(consequence_parts)
    
    casualty_clause = ""
    if total_casualties > 0:
        counts = []
        if minor_count > 0: counts.append(f"{minor_count} minor injuries")
        if serious_count > 0: counts.append(f"{serious_count} serious injuries")
        if death_count > 0: counts.append(f"{death_count} fatalities")
        if missing_count > 0: counts.append(f"{missing_count} missing persons")
        casualty_clause = f"resulting in {join_words(counts)}"
    elif is_significant_event:
        # Contrast zero injuries with the severe event
        templates = [
            "without any reported crew injuries or fatalities",
            "with all crew members safely accounted for without injury",
            "but no casualties resulted from the incident",
            "although crew members escaped without injury"
        ]
        casualty_clause = random.choice(templates)
        
    # Combine outcome verb and consequence descriptors
    outcome_clause = outcome_verb
    if consequence_clause and casualty_clause:
        outcome_clause = f"{outcome_verb}, {consequence_clause}, {casualty_clause}"
    elif consequence_clause:
        outcome_clause = f"{outcome_verb}, {consequence_clause}"
    elif casualty_clause:
        outcome_clause = f"{outcome_verb}, {casualty_clause}"

    # Combine Block B and C:
    main_narrative = ""
    if transit_sentence and outcome_clause:
        main_narrative = f"{transit_sentence} {outcome_clause}."
    elif transit_sentence:
        main_narrative = f"{transit_sentence} experienced an operational event."
    elif outcome_clause:
        main_narrative = f"The vessel '{vname}' {outcome_clause}."
        
    # Block E: Inactive / recording equipment details
    additional_details = []
    if off_devices:
        off_templates = [
            f"Navigation aids reported as inactive included {join_words(off_devices)}.",
            f"The inactive navigation suite consisted of {join_words(off_devices)}.",
            f"During the voyage, {join_words(off_devices)} remained inactive.",
            f"The crew reported {join_words(off_devices)} as inactive navigation aids."
        ]
        additional_details.append(random.choice(off_templates))
        
    lsa_list = v.get("lsa_equipment", [])
    lsa_raw = [lsa.get("LsApplianceDisplayEng") for lsa in lsa_list]
    lsa_names = normalize_equipment_list(lsa_raw, dictionary_metadata)
    # Suppress low-information life-saving appliance types like "Other"
    if lsa_names and lsa_names != ["Other"]:
        lsa_templates = [
            f"Life-saving appliances carried on board consisted of {join_words(lsa_names)}.",
            f"Safety equipment on board included {join_words(lsa_names)}.",
            f"For emergency safety, the vessel was equipped with {join_words(lsa_names)}.",
            f"Life-saving equipment on the vessel included {join_words(lsa_names)}."
        ]
        additional_details.append(random.choice(lsa_templates))
        
    rec_list = v.get("rec_equipment", [])
    for rec in rec_list:
        rec_type = rec.get("RecordingEquipDisplayEng")
        extracted = rec.get("DataExtractedEnumDisplayEng")
        seized = rec.get("EquipSeizedEnumDisplayEng")
        if rec_type:
            rec_name = normalize_label(rec_type, dictionary_metadata)
            ext_str = "data was successfully extracted" if extracted == "Yes" else "data could not be extracted"
            if seized == "Yes":
                ext_str += " and the device was seized"
            
            rec_templates = [
                f"A '{rec_name}' recording device was on board; {ext_str}.",
                f"The vessel was fitted with a '{rec_name}' recorder, from which {ext_str}.",
                f"A '{rec_name}' recording system was present, and {ext_str}.",
                f"Investigators noted a '{rec_name}' device on board, from which {ext_str}."
            ]
            additional_details.append(random.choice(rec_templates))

    # Suppress check
    if not transit_parts and not event_name and not has_damage and not has_pollution and total_casualties == 0:
        if not additional_details:
            return ""

    final_sentences = [main_narrative] + additional_details
    final_sentences = [s.strip() for s in final_sentences if s.strip()]
    
    full_text = " ".join(final_sentences)
    return " ".join(full_text.strip().split())

def main():
    # Set seed for deterministic reproducibility
    random.seed(42)
    
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    merged_path = output_dir / "merged_records.jsonl"
    if not merged_path.exists():
        logger.error(f"Merged records not found at {merged_path}! Run Step 5 first.")
        return
        
    logger.info("Loading text templates...")
    v_tpl, i_tpl, e_tpl = load_templates(root, config)
    
    # Load dictionary metadata if available for technical label normalization
    dictionary_metadata = None
    meta_path = output_dir / "dictionary_metadata.json"
    if meta_path.exists():
        logger.info(f"Loading dictionary metadata from {meta_path}...")
        try:
            with open(meta_path, "r", encoding="utf-8") as fm:
                dictionary_metadata = json.load(fm)
        except Exception as e:
            logger.warning(f"Failed to load dictionary metadata: {e}")
            
    raw_docs_file = output_dir / "raw_documents.jsonl"
    logger.info(f"Generating documents and exporting to {raw_docs_file}...")
    
    with open(merged_path, "r", encoding="utf-8") as fin, open(raw_docs_file, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Generating Documents"):
            record = json.loads(line)
            oid = record["occurrence_id"]
            occ = record["occurrence"]
            vessels = record["vessels"]
            
            seen_texts_for_occurrence = set()
            candidates = []
            
            # Helper to add candidate
            def add_candidate(text, doc_type, source_table, vessel_id=None):
                if not text:
                    return
                norm_text = " ".join(text.strip().split())
                if len(norm_text) < 50:
                    return
                words = norm_text.split()
                if len(words) < 10:
                    return
                candidates.append({
                    "text": norm_text,
                    "type": doc_type,
                    "source": source_table,
                    "vessel_id": vessel_id
                })
            
            # 1. Occurrence Summary (Raw TSB Summary only)
            occ_summary = generate_occurrence_summary(oid, record, v_tpl, i_tpl, e_tpl, dictionary_metadata)
            add_candidate(occ_summary, "occurrence_summary", "MDOTW_VW_OCCURRENCE_PUBLIC")
            
            # 2. Vessels (Fully Integrated Contexts)
            if vessels:
                for v in vessels:
                    vid = v.get("VesselID")
                    vid_val = int(vid) if vid is not None and pd.notna(vid) else None
                    v_int_ctx = build_integrated_context_for_vessel(oid, occ, v, v_tpl, i_tpl, e_tpl, dictionary_metadata)
                    add_candidate(v_int_ctx, "integrated_context", "MULTIPLE_TABLES", vessel_id=vid_val)
            else:
                # Orphan occurrence with no vessels -> export environment details if available
                env_desc = generate_environment(oid, record, dictionary_metadata)
                add_candidate(env_desc, "environment", "MDOTW_VW_OCCURRENCE_PUBLIC")
            
            occurrence_fingerprints = []
            
            for cand in candidates:
                text = cand["text"]
                doc_type = cand["type"]
                source_table = cand["source"]
                vessel_id = cand["vessel_id"]
                
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if text_hash in seen_texts_for_occurrence:
                    continue
                
                concepts = extract_concepts(text)
                
                if doc_type == "occurrence_summary":
                    emit = True
                else:
                    score = calculate_information_density(concepts, occurrence_fingerprints)
                    # Redefined Semantic Density Rule: At least one relationship or two facts (score >= 1)
                    emit = (score >= 1)
                    
                if emit:
                    seen_texts_for_occurrence.add(text_hash)
                    occurrence_fingerprints.append(concepts)
                    
                    output_obj = {
                        "occurrence_id": oid,
                        "vessel_id": vessel_id,
                        "document_type": doc_type,
                        "source_table": source_table,
                        "document": text,
                        "structured": record
                    }
                    fout.write(json.dumps(output_obj) + "\n")
            
    logger.info("Raw documents generation completed successfully!")

if __name__ == "__main__":
    main()
