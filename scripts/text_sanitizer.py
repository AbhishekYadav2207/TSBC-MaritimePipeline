import re

ADMINISTRATIVE_NOISE_PATTERNS = [
    r'(?i)note\s*:\s*formerly\s*occno\s*:?\s*[\w\-\.]+',
    r'(?i)formerly\s*occno\s*:?\s*[\w\-\.]+',
    r'(?i)\(data\s+extraction\s+status\s+pending\)',
    r'(?i)data\s+extraction\s+status\s+pending',
    r'(?i)data\s+extraction\s+status\s+unknown',
    r'(?i)extraction\s+status\s+pending',
    r'(?i)record\s+id\s*:?\s*\d+',
    r'(?i)tsb\s+internal\s+ref\s*:?\s*[\w\-]+'
]

def strip_administrative_noise(text: str) -> str:
    """Strips administrative metadata noise from text intended for BERT pretraining."""
    if not text:
        return ""
    
    cleaned = text
    for pat in ADMINISTRATIVE_NOISE_PATTERNS:
        cleaned = re.sub(pat, ' ', cleaned)
        
    # Clean up double spaces, hanging commas or colons
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\s*([,\.])\s*\1+', r'\1', cleaned)
    cleaned = re.sub(r'^\s*[:,\.\-]\s*', '', cleaned)
    cleaned = re.sub(r'^\s*\.\s*', '', cleaned)
    return cleaned.strip()

def join_words_grammatical(words: list, conjunction: str = "and") -> str:
    """Renders a list of words or phrases with proper Oxford commas and natural conjunctions."""
    if not words:
        return ""
    # Filter out empty or whitespace strings
    valid_words = [str(w).strip() for w in words if str(w).strip()]
    if not valid_words:
        return ""
    if len(valid_words) == 1:
        return valid_words[0]
    if len(valid_words) == 2:
        return f"{valid_words[0]} {conjunction} {valid_words[1]}"
    return ", ".join(valid_words[:-1]) + f", {conjunction} {valid_words[-1]}"

def format_cargo_description(cargo_prod: str, cargo_qty=None) -> str:
    """Formats cargo descriptions cleanly without duplicate 'cargo cargo' phrases."""
    if not cargo_prod or str(cargo_prod).upper() in ["NAN", "UNKNOWN", "NONE", "UNSPECIFIED"]:
        return ""
    
    c_lower = str(cargo_prod).strip().lower()
    
    if cargo_qty and str(cargo_qty).strip() and str(cargo_qty).upper() not in ["NAN", "NONE"]:
        return f"carrying {cargo_qty} of {c_lower}"
    
    if c_lower.endswith("cargo"):
        return f"laden with {c_lower}"
    else:
        return f"laden with {c_lower} cargo"

def format_damage_description(degree: str, location: str = None) -> str:
    """Formats vessel damage descriptions without duplicate 'damaged damage' constructions."""
    if not degree or str(degree).upper() in ["NONE", "NONE APPARENT", "UNKNOWN", "NAN", "UNSPECIFIED"]:
        return ""
    
    d_lower = str(degree).strip().lower()
    loc_lower = str(location).strip().lower() if location and str(location).upper() not in ["UNKNOWN", "NAN", "NONE"] else ""
    
    # Normalize degree word
    if d_lower in ["damaged", "damage"]:
        deg_str = "damage"
    elif d_lower.endswith("damage"):
        deg_str = d_lower
    else:
        deg_str = f"{d_lower} damage"
        
    if loc_lower:
        return f"sustaining {deg_str} to the {loc_lower}"
    else:
        return f"sustaining {deg_str}"

def format_casualty_count(count: int, singular_term: str, plural_term: str) -> str:
    """Formats casualty counts with strict singular/plural agreement."""
    if count <= 0:
        return ""
    term = singular_term if count == 1 else plural_term
    return f"{count} {term}"
