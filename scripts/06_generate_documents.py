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
    """Generates the primary occurrence-level document summary."""
    occ = record.get("occurrence")
    vessels = record.get("vessels", [])
    
    paragraphs = []
    is_placeholder_occ = record.get("_placeholder_occurrence", False) or (occ is None)
    
    if is_placeholder_occ:
        paragraphs.append("Vessel information is available, although no corresponding occurrence record exists.")
    else:
        raw_summary = occ.get("Summary") if occ else None
        if raw_summary and str(raw_summary).strip() not in ["", "nan", "NaN"]:
            paragraphs.append(str(raw_summary).strip())
            
        if occ:
            env_desc = format_environmental(occ, dictionary_metadata)
            if env_desc:
                paragraphs.append(env_desc)
                
    for v in vessels:
        other_vessels = [oth for oth in vessels if oth != v]
        v_desc = format_vessel(v, v_tpl, i_tpl, e_tpl, dictionary_metadata, occurrence=occ, other_vessels=other_vessels)
        if v_desc:
            paragraphs.append(v_desc)
            
    return "\n\n".join(paragraphs)

def generate_environment(oid: int, record: dict, dictionary_metadata: dict = None) -> str:
    """Generates environment weather and condition document."""
    occ = record.get("occurrence")
    if not occ:
        return ""
    return format_environmental(occ, dictionary_metadata)

def generate_vessel_profile(oid: int, v: dict, v_tpl: dict, occurrence: dict = None, dictionary_metadata: dict = None) -> str:
    """Generates standalone vessel profile describing voyage, cargo, damage, and pollution."""
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
    
    vtype = vtype.lower()
    if vflag: vflag = vflag.title()
    if hull: hull = hull.lower()
    if prop: prop = prop.lower()
    if phase: phase = phase.lower()
    if activity: activity = activity.lower()
        
    vessel_paragraphs = []
    is_placeholder = v.get("_placeholder", False)
    
    if is_placeholder:
        vessel_paragraphs.append("The following details were reported for an unnamed vessel involved in the occurrence:")
    else:
        flag_str = vflag
        hull_str = hull
        
        ton_str = f"{float(tonnage):.0f} GT" if tonnage and pd.notna(tonnage) and float(tonnage) > 0 else None
        built_str = str(int(float(built))) if built and pd.notna(built) and int(float(built)) > 0 else None
        
        profile_parts = []
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
            desc_pieces = [f"'{vname}' is a {vtype}"]
            if flag_str: desc_pieces.append(f"registered in {flag_str}")
            if hull_str: desc_pieces.append(f"constructed of {hull_str}")
            if ton_str: desc_pieces.append(f"with a gross tonnage of {ton_str}")
            if built_str: desc_pieces.append(f"built in {built_str}")
            profile_parts.append(", ".join(desc_pieces) + ".")
            
        vessel_paragraphs.append(" ".join(profile_parts))
        
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
            
        cargo_prod = v.get("CargoProductTypeDisplayEng")
        cargo_qty = v.get("QuantityOnBoard")
        cargo_prod = clean_placeholder(cargo_prod)
        if cargo_prod and cargo_qty is not None and str(cargo_qty).strip() not in ["", "nan", "NaN"]:
            tpl_cargo = random.choice(v_tpl["cargo_info"])
            vessel_paragraphs.append(tpl_cargo.format(
                quantity=f"{cargo_qty}",
                cargo_product=cargo_prod.lower()
            ))
            
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
            
        pollution = v.get("SeaPollutionDegreeDisplayEng")
        pollution = clean_placeholder(pollution)
        if pollution and pollution.upper() not in ["NONE", "NONE APPARENT", "UNKNOWN"]:
            tpl_poll = random.choice(v_tpl["pollution_info"])
            vessel_paragraphs.append(tpl_poll.format(
                pollution_degree=pollution.lower()
            ))
            
    return " ".join(vessel_paragraphs)

def generate_vessel_characteristics(oid: int, v: dict, dictionary_metadata: dict = None) -> list:
    """Generates focused vessel characteristics documents."""
    vname = v.get("VesselName") or "An unnamed vessel"
    vtype = (v.get("VesselTypeDisplayEng") or "vessel").lower()
    vflag = clean_placeholder(v.get("VesselFlagDisplayEng"))
    hull = clean_placeholder(v.get("HullMaterialDisplayEng"))
    prop = clean_placeholder(v.get("PropulsionTypeDisplayEng"))
    tonnage = v.get("GrossTonnage")
    built = v.get("YearBuilt")
    
    if vflag: vflag = vflag.title()
    if hull: hull = hull.lower()
    if prop: prop = prop.lower()
    
    ton_str = f"{float(tonnage):.0f} GT" if tonnage and pd.notna(tonnage) and float(tonnage) > 0 else None
    built_str = str(int(float(built))) if built and pd.notna(built) and int(float(built)) > 0 else None
    
    attrs = [x for x in [vflag, hull, prop, ton_str, built_str] if x is not None]
    if len(attrs) < 2:
        return []
        
    templates = []
    if hull and ton_str:
        templates.append(f"The vessel '{vname}' is constructed of {hull} hull material and has a gross tonnage of {ton_str}.")
    if built_str and vflag and hull:
        templates.append(f"Built in {built_str}, the {vflag}-registered {vtype} '{vname}' features a {hull} hull.")
    if vflag and prop:
        templates.append(f"The {vtype} '{vname}' operates under the {vflag} flag and uses {prop} propulsion.")
    if prop and hull and vflag and ton_str:
        templates.append(f"With {prop} propulsion and a {hull} hull, the {vflag} vessel '{vname}' has a gross tonnage of {ton_str}.")
    if vflag and built_str and hull:
        templates.append(f"The ship '{vname}' is a {vtype} registered in {vflag}, built in {built_str} with a {hull} hull.")
        
    if not templates:
        pieces = [f"'{vname}' is a {vtype}"]
        if vflag: pieces.append(f"registered in {vflag}")
        if hull: pieces.append(f"constructed of {hull}")
        if ton_str: pieces.append(f"with a gross tonnage of {ton_str}")
        if built_str: pieces.append(f"built in {built_str}")
        templates.append(", ".join(pieces) + ".")
        
    return templates

def generate_voyage_activity(oid: int, v: dict, dictionary_metadata: dict = None) -> str:
    """Generates focused voyage activity descriptions."""
    vname = v.get("VesselName") or "An unnamed vessel"
    phase = v.get("VesselPhaseDisplayEng")
    activity = v.get("ActivityTypeDisplayEng")
    
    phase = clean_placeholder(phase)
    activity = clean_placeholder(activity)
    
    phase_valid = phase and phase.upper() not in ["UNKNOWN", "UNSPECIFIED"]
    act_valid = activity and activity.upper() not in ["UNKNOWN", "UNSPECIFIED"]
    
    if not phase_valid and not act_valid:
        return ""
        
    if phase: phase = phase.lower()
    if activity: activity = activity.lower()
    
    templates = []
    if phase_valid and act_valid:
        templates = [
            f"At the time of the occurrence, the vessel '{vname}' was {phase} during a {activity} voyage.",
            f"During the voyage, '{vname}' was operating in the {phase} phase, engaged in {activity}.",
            f"The ship '{vname}' was {phase} while performing {activity} operations.",
            f"While engaged in {activity}, the vessel '{vname}' was operating in the {phase} phase.",
            f"The vessel '{vname}' was {phase} and engaged in {activity} operations at the time of the occurrence."
        ]
    elif phase_valid:
        templates = [
            f"At the time of the occurrence, the vessel '{vname}' was {phase}.",
            f"The vessel '{vname}' was operating in the {phase} phase during the voyage.",
            f"The ship '{vname}' was {phase} when the incident occurred."
        ]
    elif act_valid:
        templates = [
            f"During the voyage, the vessel '{vname}' was engaged in {activity}.",
            f"The ship '{vname}' was performing {activity} operations at the time of the incident.",
            f"'{vname}' was engaged in {activity} when the occurrence took place."
        ]
        
    return random.choice(templates) if templates else ""

def generate_cargo(oid: int, v: dict, dictionary_metadata: dict = None) -> str:
    """Generates cargo-specific document descriptions."""
    vname = v.get("VesselName") or "An unnamed vessel"
    cargo_prod = v.get("CargoProductTypeDisplayEng")
    cargo_qty = v.get("QuantityOnBoard")
    
    cargo_prod = clean_placeholder(cargo_prod)
    if cargo_prod:
        cargo_prod = cargo_prod.lower()
        if cargo_qty is not None and str(cargo_qty).strip() not in ["", "nan", "NaN"]:
            qty_str = str(cargo_qty)
            templates = [
                f"On board the vessel '{vname}' was {qty_str} of {cargo_prod} cargo.",
                f"The vessel '{vname}' was carrying {qty_str} of {cargo_prod} in the cargo holds.",
                f"A cargo of {qty_str} of {cargo_prod} was loaded on board '{vname}'.",
                f"The ship '{vname}' was laden with {qty_str} of {cargo_prod} cargo.",
                f"A shipment consisting of {qty_str} of {cargo_prod} was on board '{vname}'."
            ]
        else:
            templates = [
                f"The vessel '{vname}' was carrying {cargo_prod} cargo during the transit.",
                f"Cargo on board '{vname}' consisted of {cargo_prod}.",
                f"The ship '{vname}' was transporting {cargo_prod} as its primary cargo.",
                f"On board '{vname}' was a shipment of {cargo_prod}."
            ]
        return random.choice(templates)
    return ""

def generate_navigation_equipment(oid: int, v: dict, dictionary_metadata: dict = None) -> list:
    """Generates navigation equipment documents with adaptive granularity levels."""
    nav_list = v.get("navigation_equipment", [])
    if not nav_list:
        return []
        
    vname = v.get("VesselName") or "An unnamed vessel"
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
                    
    if not on_devices and not off_devices:
        return []
        
    docs = []
    
    # Level 1: Summary of all navigation aids on the vessel
    level1_templates = [
        f"The navigation systems active on board '{vname}' included {join_words(on_devices)}." if on_devices else "",
        f"Navigation equipment reported as inactive on board '{vname}' included {join_words(off_devices)}." if off_devices else "",
        f"Active navigation equipment on '{vname}' consisted of {join_words(on_devices)}, while {join_words(off_devices)} was inactive." if on_devices and off_devices else ""
    ]
    level1_text = random.choice([t for t in level1_templates if t])
    if level1_text:
        docs.append(level1_text)
        
    # Level 2: Grouping (e.g. radars vs communication vs positioning)
    radars = [d for d in on_devices if "radar" in d.lower()]
    radios = [d for d in on_devices if any(x in d.lower() for x in ["radio", "vhf", "hf", "inmarsat", "callsign"])]
    positioning = [d for d in on_devices if any(x in d.lower() for x in ["gps", "ecdis", "ais", "gyro", "compass", "navig"])]
    
    level2_parts = []
    if radars:
        level2_parts.append(f"The radar systems operational on '{vname}' included {join_words(radars)}.")
    if radios:
        level2_parts.append(f"For external communication, '{vname}' relied on {join_words(radios)}.")
    if positioning:
        level2_parts.append(f"Positioning and pilotage navigation aids on board '{vname}' featured {join_words(positioning)}.")
        
    level2_text = " ".join(level2_parts)
    if level2_text:
        docs.append(level2_text)
        
    # Level 3: Individual status descriptions
    for dev in on_devices:
        level3_templates = [
            f"The vessel '{vname}' carried an operational {dev} for navigation.",
            f"The {dev} was reported as active on board the vessel '{vname}'.",
            f"For positioning and safety, '{vname}' was equipped with an active {dev}.",
            f"The {dev} equipment on board '{vname}' was functional during the voyage.",
            f"The navigation aids on board '{vname}' included an operational {dev}."
        ]
        docs.append(random.choice(level3_templates))
            
    for dev in off_devices:
        level3_templates_off = [
            f"The {dev} on board '{vname}' was reported to be inactive.",
            f"The vessel '{vname}' was equipped with a {dev}, which was turned off.",
            f"At the time of the occurrence, '{vname}' was not relying on its {dev} as it was inactive.",
            f"The {dev} system on '{vname}' was non-operational during the incident.",
            f"The navigation suite of '{vname}' included a {dev} that was reported as inactive."
        ]
        docs.append(random.choice(level3_templates_off))
            
    return docs

def generate_lsa_equipment(oid: int, v: dict, dictionary_metadata: dict = None) -> list:
    """Generates LSA equipment documents."""
    lsa_list = v.get("lsa_equipment", [])
    if not lsa_list:
        return []
        
    vname = v.get("VesselName") or "An unnamed vessel"
    seen_lsa = set()
    docs = []
    
    all_lsa_details = []
    for lsa in lsa_list:
        lse_type = lsa.get("LsApplianceDisplayEng")
        lse_used = lsa.get("UsedEnumDisplayEng")
        lse_approved = lsa.get("ApprovedEnumDisplayEng")
        if lse_type:
            lse_name = normalize_label(lse_type, dictionary_metadata)
            used_str = "deployed and used" if lse_used == "Yes" else "not used"
            approved_str = "approved" if lse_approved == "Yes" else "not approved"
            
            key = (lse_name.lower(), used_str, approved_str)
            if key not in seen_lsa:
                seen_lsa.add(key)
                all_lsa_details.append((lse_name, used_str, approved_str))
                
    if not all_lsa_details:
        return []
        
    # Level 1 summary
    lsa_names = [x[0] for x in all_lsa_details]
    level1_templates = [
        f"Life-saving appliances carried on board '{vname}' included {join_words(lsa_names)}.",
        f"The safety equipment suite on '{vname}' consisted of {join_words(lsa_names)}.",
        f"For emergency safety, '{vname}' was equipped with {join_words(lsa_names)}."
    ]
    docs.append(random.choice(level1_templates))
        
    # Level 3 details
    for lse_name, used_str, approved_str in all_lsa_details:
        level3_templates = [
            f"The lifesaving equipment '{lse_name}' on board '{vname}' was {used_str}.",
            f"'{vname}' carried '{lse_name}' which was reported as {used_str} and {approved_str}.",
            f"The vessel '{vname}' was equipped with {lse_name}, which was {used_str}.",
            f"Emergency gear on board '{vname}' included {lse_name}, which was {approved_str} and {used_str}.",
            f"The {approved_str} {lse_name} on board '{vname}' was {used_str} during the incident."
        ]
        docs.append(random.choice(level3_templates))
            
    return docs

def generate_recording_equipment(oid: int, v: dict, dictionary_metadata: dict = None) -> list:
    """Generates recording equipment descriptions."""
    rec_list = v.get("rec_equipment", [])
    if not rec_list:
        return []
        
    vname = v.get("VesselName") or "An unnamed vessel"
    seen_rec = set()
    docs = []
    
    for rec in rec_list:
        rec_type = rec.get("RecordingEquipDisplayEng")
        extracted = rec.get("DataExtractedEnumDisplayEng")
        seized = rec.get("EquipSeizedEnumDisplayEng")
        
        if rec_type:
            rec_name = normalize_label(rec_type, dictionary_metadata)
            ext_str = "data was successfully extracted" if extracted == "Yes" else "data could not be extracted"
            if seized == "Yes":
                ext_str += " and the equipment was seized by investigators"
                
            key = (rec_name.lower(), ext_str)
            if key not in seen_rec:
                seen_rec.add(key)
                
                ext_sent = ext_str[0].upper() + ext_str[1:] + "."
                
                templates = [
                    f"The vessel '{vname}' carried a '{rec_name}' recording device. {ext_sent}",
                    f"Investigation logs indicate '{vname}' was fitted with a '{rec_name}'. {ext_sent}",
                    f"The recording system '{rec_name}' on board '{vname}' was analyzed by investigators. {ext_sent}",
                    f"A '{rec_name}' recording system was present on '{vname}', and {ext_str}.",
                    f"Data recovery from the '{rec_name}' on '{vname}' was completed: {ext_sent}"
                ]
                docs.append(random.choice(templates))
                    
    return docs

def generate_injury(oid: int, v: dict, dictionary_metadata: dict = None) -> list:
    """Generates casualty and injury descriptions."""
    injuries = v.get("injuries", [])
    if not injuries:
        return []
        
    vname = v.get("VesselName") or "An unnamed vessel"
    docs = []
    
    for inj in injuries:
        minor = inj.get("VictimMinorInjuries") or 0
        serious = inj.get("VictimSeriousInjuries") or 0
        deaths = inj.get("VictimDeath") or 0
        missing = inj.get("VictimMissing") or 0
        
        if minor > 0 or serious > 0 or deaths > 0 or missing > 0:
            counts = []
            if minor > 0: counts.append(f"{minor} minor injury/injuries")
            if serious > 0: counts.append(f"{serious} serious injury/injuries")
            if deaths > 0: counts.append(f"{deaths} fatality/fatalities")
            if missing > 0: counts.append(f"{missing} person/people missing")
            
            details = "; ".join(counts)
            
            templates = [
                f"The occurrence involving '{vname}' resulted in the following casualties: {details}.",
                f"Casualties reported for the vessel '{vname}' included: {details}.",
                f"An assessment of personnel on '{vname}' following the incident revealed: {details}.",
                f"The incident involving '{vname}' led to the following crew or passenger casualties: {details}.",
                f"Medical reports in connection with '{vname}' confirm the following injury metrics: {details}."
            ]
            docs.append(random.choice(templates))
                
    return docs

def generate_integrated_context(oid: int, record: dict, dictionary_metadata: dict = None) -> list:
    """Generates rich integrated operational contexts combining multiple entity attributes."""
    occ = record.get("occurrence")
    vessels = record.get("vessels", [])
    if not occ or not vessels:
        return []
        
    weather = clean_placeholder(occ.get("WeatherConditionDisplayEng"))
    sea_state = clean_placeholder(occ.get("SeaStateDisplayEng"))
    
    docs = []
    
    for v in vessels:
        vname = v.get("VesselName") or "An unnamed vessel"
        vtype = (v.get("VesselTypeDisplayEng") or "vessel").lower()
        vflag = clean_placeholder(v.get("VesselFlagDisplayEng"))
        cargo = clean_placeholder(v.get("CargoProductTypeDisplayEng"))
        
        nav_list = v.get("navigation_equipment", [])
        on_devices = []
        for nav in nav_list:
            nav_type = nav.get("NavigationAidTypeDisplayEng")
            nav_status = nav.get("OnOffEnumDisplayEng")
            if nav_type and nav_status == "On":
                on_devices.append(normalize_label(nav_type, dictionary_metadata))
                
        if vflag: vflag = vflag.title()
        if cargo: cargo = cargo.lower()
        if weather: weather = weather.lower()
        if sea_state: sea_state = sea_state.lower()
        
        # Weather + Navigation
        if weather and on_devices:
            nav_aids = join_words(on_devices)
            templates_w_n = [
                f"The vessel '{vname}' encountered {weather} weather, requiring the crew to rely heavily on its navigation equipment, including {nav_aids}.",
                f"Under {weather} conditions, navigation on board '{vname}' was maintained using {nav_aids}.",
                f"Weather conditions during the occurrence featured {weather} weather, complicating navigation for '{vname}' as they operated {nav_aids}.",
                f"The crew of '{vname}' navigated through {weather} weather using the vessel's operational {nav_aids}.",
                f"Operational logs for '{vname}' show that the vessel relied on {nav_aids} to navigate through {weather} weather."
            ]
            docs.append(random.choice(templates_w_n))
            
        # Cargo + Weather
        if cargo and weather:
            templates_c_w = [
                f"The {vtype} '{vname}' was carrying a cargo of {cargo} when it encountered {weather} weather.",
                f"While transporting {cargo}, the vessel '{vname}' encountered challenging {weather} weather.",
                f"Under {weather} weather conditions, the cargo vessel '{vname}' was laden with {cargo}.",
                f"The vessel '{vname}' encountered {weather} weather while transporting a shipment of {cargo}.",
                f"The transit of {cargo} on board '{vname}' occurred during reported {weather} weather."
            ]
            docs.append(random.choice(templates_c_w))
            
        # Integrated Operational Context: Vessel Flag/Type + Cargo + Weather + Navigation
        if vflag and cargo and weather and on_devices:
            nav_aids = join_words(on_devices)
            templates_int = [
                f"The {vflag}-registered {vtype} '{vname}' was transporting {cargo} in {weather} weather, relying on {nav_aids} for positioning.",
                f"While navigating through {weather} weather with {cargo} cargo, the {vflag} vessel '{vname}' operated its {nav_aids}.",
                f"The crew of the {vflag} {vtype} '{vname}' utilized {nav_aids} to transport {cargo} during {weather} conditions.",
                f"Under {weather} conditions, the {vflag}-registered cargo ship '{vname}' carried {cargo} using its operational {nav_aids}.",
                f"The transit of '{vname}' ({vflag} flag, laden with {cargo}) encountered {weather} weather while utilizing its {nav_aids} navigation systems."
            ]
            docs.append(random.choice(templates_int))
            
    return docs

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
            
            def write_doc(text, doc_type, source_table, vessel_id=None):
                if not text:
                    return
                norm_text = " ".join(text.strip().split())
                if len(norm_text) < 50:
                    return
                words = norm_text.split()
                if len(words) < 10:
                    return
                    
                text_hash = hashlib.md5(norm_text.encode("utf-8")).hexdigest()
                
                if text_hash not in seen_texts_for_occurrence:
                    seen_texts_for_occurrence.add(text_hash)
                    output_obj = {
                        "occurrence_id": oid,
                        "vessel_id": vessel_id,
                        "document_type": doc_type,
                        "source_table": source_table,
                        "document": norm_text,
                        "structured": record
                    }
                    fout.write(json.dumps(output_obj) + "\n")
            
            # 1. Occurrence Summary
            occ_summary = generate_occurrence_summary(oid, record, v_tpl, i_tpl, e_tpl, dictionary_metadata)
            write_doc(occ_summary, "occurrence_summary", "MDOTW_VW_OCCURRENCE_PUBLIC")
            
            # 2. Environmental description
            env_desc = generate_environment(oid, record, dictionary_metadata)
            write_doc(env_desc, "environment", "MDOTW_VW_OCCURRENCE_PUBLIC")
            
            # 3. Integrated Context
            int_contexts = generate_integrated_context(oid, record, dictionary_metadata)
            for int_text in int_contexts:
                write_doc(int_text, "integrated_context", "MULTIPLE_TABLES")
                
            # 4. Vessel-specific documents
            for v in vessels:
                vid = v.get("VesselID")
                vid_val = int(vid) if vid is not None and pd.notna(vid) else None
                
                # Vessel Profile
                v_prof = generate_vessel_profile(oid, v, v_tpl, occurrence=occ, dictionary_metadata=dictionary_metadata)
                write_doc(v_prof, "vessel_profile", "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC", vessel_id=vid_val)
                
                # Vessel Characteristics
                v_chars = generate_vessel_characteristics(oid, v, dictionary_metadata=dictionary_metadata)
                for v_char in v_chars:
                    write_doc(v_char, "vessel_characteristics", "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC", vessel_id=vid_val)
                    
                # Voyage Activity
                v_voy = generate_voyage_activity(oid, v, dictionary_metadata=dictionary_metadata)
                write_doc(v_voy, "voyage_activity", "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC", vessel_id=vid_val)
                    
                # Cargo
                v_cargo = generate_cargo(oid, v, dictionary_metadata=dictionary_metadata)
                write_doc(v_cargo, "cargo", "MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC", vessel_id=vid_val)
                    
                # Navigation Equipment
                v_navs = generate_navigation_equipment(oid, v, dictionary_metadata=dictionary_metadata)
                for v_nav in v_navs:
                    write_doc(v_nav, "navigation_equipment", "MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC", vessel_id=vid_val)
                    
                # LSA Equipment
                v_lsas = generate_lsa_equipment(oid, v, dictionary_metadata=dictionary_metadata)
                for v_lsa in v_lsas:
                    write_doc(v_lsa, "lsa_equipment", "MDOTW_VW_OCCURRENCE_VESSEL_LSA_EQUIPMENT_PUBLIC", vessel_id=vid_val)
                    
                # Recording Equipment
                v_recs = generate_recording_equipment(oid, v, dictionary_metadata=dictionary_metadata)
                for v_rec in v_recs:
                    write_doc(v_rec, "recording_equipment", "MDOTW_VW_OCCURRENCE_VESSEL_REC_EQUIPMENT_PUBLIC", vessel_id=vid_val)
                    
                # Injury
                v_injs = generate_injury(oid, v, dictionary_metadata=dictionary_metadata)
                for v_inj in v_injs:
                    write_doc(v_inj, "injury", "MDOTW_VW_INJURIES_PUBLIC", vessel_id=vid_val)
            
    logger.info("Raw documents generation completed successfully!")

if __name__ == "__main__":
    main()
