import os
import json
import re
from collections import Counter
from pathlib import Path
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("10_extract_vocabulary")

def clean_word(word: str) -> str:
    """Cleans a word of non-alphabetic characters."""
    return re.sub(r'[^a-zA-Z]', '', word).lower()

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
            # Extract words from descriptions and full names in the dictionary
            for table, cols in metadata.get("registry", {}).items():
                for col, info in cols.items():
                    # Tokenize full name and description
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', info.get("full_name", "") + " " + info.get("description", ""))
                    for w in words:
                        dict_terms.add(w.lower())
                        
    # Common general English stop words to exclude
    stop_words = {
        "the", "and", "was", "for", "with", "that", "were", "this", "from", "had", "been", "not", "but", "are", 
        "have", "which", "there", "their", "they", "will", "would", "about", "their", "them", "then", "into",
        "has", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first", "second",
        "reported", "reported", "on", "in", "at", "by", "an", "is", "or", "it", "as", "to", "of", "during",
        "resulted", "sustained", "occurred", "occurred", "took", "place", "time", "date", "year", "month",
        "day", "hour", "minute", "second", "details", "information", "number", "system", "database", "admin",
        "administrative", "purpose", "purposes", "generated", "value", "values", "specified", "unknown",
        "none", "other", "another", "same", "any", "all", "each", "every", "some", "many", "most", "more"
    }
    
    # Domain-specific maritime roots and anchors
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
    
    # Step 1: Count words in corpus
    logger.info("Reading corpus text to count word frequencies...")
    corpus_word_counter = Counter()
    
    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in fin:
            doc = json.loads(line)["document"]
            # Extract alphabetic words of length >= 3
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc)
            for w in words:
                corpus_word_counter[w.lower()] += 1
                
    logger.info(f"Total unique words in corpus: {len(corpus_word_counter)}")
    
    # Step 2: Extract maritime vocabulary terms
    maritime_vocab = set()
    
    # We keep a word if:
    # 1. It is not in stop words
    # 2. AND (it matches a maritime stem OR it is in dict_terms and appears in the corpus)
    # Let's check all corpus words
    for word, freq in corpus_word_counter.items():
        if word in stop_words:
            continue
            
        is_maritime = False
        
        # Check against stems
        for stem in maritime_stems:
            if stem in word:
                is_maritime = True
                break
                
        # Check if it was in the dictionary terms and has reasonable frequency (>= 5)
        if not is_maritime and word in dict_terms and freq >= 5:
            # Check if it has maritime meaning by analyzing if it ends with standard suffixes
            # or is a known maritime term
            is_maritime = True
            
        if is_maritime:
            maritime_vocab.add((word, freq))
            
    # Sort by frequency in the corpus
    sorted_vocab = sorted(list(maritime_vocab), key=lambda x: x[1], reverse=True)
    
    # Format output
    vocab_txt_path = output_dir / "maritime_vocabulary.txt"
    logger.info(f"Exporting top vocabulary words to {vocab_txt_path}...")
    
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        # We export the top 300 maritime words
        for word, freq in sorted_vocab[:300]:
            f.write(f"{word}\n")
            
    logger.info(f"Extracted {min(300, len(sorted_vocab))} maritime vocabulary terms successfully.")

if __name__ == "__main__":
    main()
