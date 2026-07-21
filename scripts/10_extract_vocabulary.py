import os
import json
import re
from collections import Counter
from pathlib import Path
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("10_extract_vocabulary")

MULTIWORD_PATTERNS = [
    r'\b(?:restricted|poor|reduced|good)\s+visibility\b',
    r'\b(?:propulsion|steering|engine|machinery)\s+(?:failure|breakdown|loss)\b',
    r'\b(?:gross|net|deadweight)\s+tonnage\b',
    r'\b(?:life\s+saving|lifesaving)\s+appliances?\b',
    r'\bvoyage\s+data\s+recorder\b',
    r'\bsearch\s+and\s+rescue\b',
    r'\b(?:starboard|port)\s+side\b',
    r'\bengine\s+room\b',
    r'\bcargo\b\s+(?:hold|vessel|tanker)\b',
    r'\bfishing\s+vessel\b',
    r'\bbulk\s+carrier\b',
    r'\bcontainer\s+ship\b',
    r'\bnavigational?\s+aids?\b',
    r'\bvhf\s+radio\b',
    r'\bgps\s+receiver\b',
    r'\bsea\s+pollution\b',
    r'\b(?:hull|deck|keel|bow|stern)\s+damage\b'
]

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents not found at {clean_path}! Run Step 7 first.")
        return
        
    meta_path = output_dir / "dictionary_metadata.json"
    dict_terms = set()
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fm:
            metadata = json.load(fm)
            for table, cols in metadata.get("registry", {}).items():
                for col, info in cols.items():
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', info.get("full_name", "") + " " + info.get("description", ""))
                    for w in words:
                        dict_terms.add(w.lower())
                        
    stop_words = {
        "the", "and", "was", "for", "with", "that", "were", "this", "from", "had", "been", "not", "but", "are", 
        "have", "which", "there", "their", "they", "will", "would", "about", "them", "then", "into",
        "has", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first", "second",
        "reported", "on", "in", "at", "by", "an", "is", "or", "it", "as", "to", "of", "during",
        "resulted", "sustained", "occurred", "took", "place", "time", "date", "year", "month",
        "day", "hour", "minute", "second", "details", "information", "number", "system", "database", "admin",
        "administrative", "purpose", "purposes", "generated", "value", "values", "specified", "unknown",
        "none", "other", "another", "same", "any", "all", "each", "every", "some", "many", "most", "more"
    }
    
    maritime_stems = {
        "vess", "ship", "boat", "craft", "yacht", "tug", "barge", "tanker", "trawler", "carrier", "cruis",
        "port", "starboard", "bow", "stern", "keel", "hull", "deck", "mast", "helm", "rudder", "propel", 
        "anchor", "mooring", "dock", "wharf", "pier", "quay", "berth", "harbo", "harbu", "marit", "marin",
        "seaway", "ocean", "sea", "water", "gulf", "bay", "strait", "channel", "lake", "canal", "river",
        "towing", "towed", "towline", "tow", "bight", "hawser", "line", "cleat", "bollard",
        "navig", "gps", "ais", "vhf", "radar", "sonar", "compass", "gyro", "sounder", "chart", "vdr",
        "lsa", "lifeboat", "lifejack", "liferaft", "buoy", "davit", "flare", "epirb", "sart",
        "crew", "capt", "master", "mate", "pilot", "seaman", "sailor", "engineer", "passenger",
        "collision", "grounding", "stranding", "flooding", "leak", "ingress", "list", "capsiz", "sink", 
        "foundering", "pollution", "spill", "oil", "bilge", "ballast", "tank", "cargo", "machinery"
    }
    
    logger.info("Reading corpus text to extract single-word and multiword maritime vocabulary...")
    single_word_counter = Counter()
    multiword_counter = Counter()
    
    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in fin:
            doc = json.loads(line)["document"].lower()
            
            # Extract multiword phrases first
            for pat in MULTIWORD_PATTERNS:
                matches = re.findall(pat, doc)
                for m in matches:
                    multiword_counter[m] += 1
                    
            # Extract single alphabetic words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc)
            for w in words:
                single_word_counter[w] += 1
                
    # Filter single terms
    maritime_single = set()
    for word, freq in single_word_counter.items():
        if word in stop_words:
            continue
        is_maritime = any(stem in word for stem in maritime_stems)
        if not is_maritime and word in dict_terms and freq >= 5:
            is_maritime = True
        if is_maritime:
            maritime_single.add((word, freq))
            
    sorted_single = sorted(list(maritime_single), key=lambda x: x[1], reverse=True)
    sorted_multi = sorted(list(multiword_counter.items()), key=lambda x: x[1], reverse=True)
    
    vocab_txt_path = output_dir / "maritime_vocabulary.txt"
    logger.info(f"Exporting top single and multiword maritime terms to {vocab_txt_path}...")
    
    # Export top multiword phrases first, then top single terms
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        # Multiword phrases (up to 50)
        for phrase, freq in sorted_multi[:50]:
            f.write(f"{phrase}\n")
        # Single-word terms (up to 300)
        for word, freq in sorted_single[:300]:
            f.write(f"{word}\n")
            
    logger.info(f"Exported {min(50, len(sorted_multi))} multiword phrases and {min(300, len(sorted_single))} single terms to vocabulary file.")

if __name__ == "__main__":
    main()
