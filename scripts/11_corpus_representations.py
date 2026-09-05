import os
import json
import re
import hashlib
from pathlib import Path
from tqdm import tqdm
from pipeline_utils import setup_logging, load_config, get_project_root, get_benchmark_config, load_corpus_documents

logger = setup_logging("11_corpus_representations")

# Standardized deterministic template formats for Extracted Template Representation
TEMPLATES = [
    "EVENT RECORD: Focus: {subject}. Environment/Context: {context}. Narrative: {details}.",
    "Case Documentation | Subject Entity: {subject} | Operational Environment: {context} | Incident Narrative: {details}.",
    "Standardized Report: Under conditions of {context}, observed event involving {subject}. Event Summary: {details}."
]

def extract_text_components(doc_text: str) -> dict:
    """
    Deterministic rule-based extraction of semantic facets directly from document text.
    Does NOT depend on database columns or structured metadata records.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc_text) if s.strip()]
    first_sent = sentences[0] if sentences else doc_text
    other_sents = " ".join(sentences[1:]) if len(sentences) > 1 else first_sent

    # 1. Quoted terms, capitalized titles, or alphanumeric identifiers
    quoted_entities = re.findall(r"['\"]([A-Za-z0-9\s\-]+)['\"]", doc_text)
    capitalized_entities = re.findall(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,})*\b", doc_text)
    entities = list(dict.fromkeys([e.strip() for e in quoted_entities + capitalized_entities if len(e.strip()) > 2]))

    # 2. Measurements and quantitative indicators
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:GT|meters?|knots?|tonnes?|deg(?:rees)?|km|hours?|fatalit(?:y|ies)|injur(?:y|ies)|deaths?)\b', doc_text, flags=re.IGNORECASE)

    # 3. Context / conditions (prepositional / adverbial clauses)
    context_matches = re.findall(r'\b(?:under|during|while|in)\s+([^,.;]{5,40})', doc_text, flags=re.IGNORECASE)
    context_str = ", ".join(context_matches[:3]) if context_matches else "general operational context"

    subject_str = entities[0] if entities else "Primary entity"

    return {
        "primary_statement": first_sent,
        "details": other_sents,
        "subject": subject_str,
        "entities": entities[:6],
        "measurements": measurements[:6],
        "context": context_str
    }

def build_key_value_representation(doc_id: str, doc_text: str, components: dict) -> str:
    """Renders text into a clean Key-Value representation extracted directly from text."""
    parts = [
        f"Document ID: {doc_id}",
        f"Primary Subject: {components['subject']}",
        f"Core Statement: {components['primary_statement']}"
    ]
    if components["entities"]:
        parts.append(f"Identified Entities: {', '.join(components['entities'])}")
    if components["context"] and components["context"] != "general operational context":
        parts.append(f"Operational Context: {components['context']}")
    if components["measurements"]:
        parts.append(f"Key Measurements: {', '.join(components['measurements'])}")
    if components["details"] and components["details"] != components["primary_statement"]:
        parts.append(f"Supplementary Details: {components['details']}")

    return " \n ".join(parts)

def build_template_representation(doc_id: str, doc_text: str, components: dict) -> str:
    """
    Renders text into a semi-structured template representation.
    Uses stable hashlib hashing to ensure deterministic cross-run template selection.
    """
    # Deterministic template selection via stable SHA-256 hash
    hash_val = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16)
    template = TEMPLATES[hash_val % len(TEMPLATES)]

    return template.format(
        subject=components["subject"],
        context=components["context"],
        details=components["details"]
    )

def build_structured_semantic_representation(doc_id: str, doc_text: str, components: dict) -> str:
    """
    Renders text into an Extracted Structured Representation (serialized JSON).
    Derived purely from text processing, representing semantic structure without database schemas.
    """
    structured_payload = {
        "document_id": doc_id,
        "representation_type": "extracted_structured_semantic",
        "primary_subject": components["subject"],
        "statement": components["primary_statement"],
        "context": components["context"],
        "extracted_entities": components["entities"],
        "measurements": components["measurements"],
        "token_count": len(doc_text.split())
    }
    return json.dumps(structured_payload, ensure_ascii=False)

def build_mixed_representation(narrative_doc: str, kv_doc: str) -> str:
    """Combines extracted Key-Value structural header with natural narrative body."""
    return f"[EXTRACTED STRUCTURE]\n{kv_doc}\n[TEXT NARRATIVE]\n{narrative_doc}"

def main():
    root = get_project_root()
    config = load_config()
    bench_cfg = get_benchmark_config()
    output_dir = root / config.get("output_dir", "outputs")

    # Resolve TXT corpus path from config (deterministic, no silent fallback unless explicitly configured)
    corpus_filename = bench_cfg.get("corpus_file", "maritime_corpus.txt")
    corpus_path = output_dir / corpus_filename

    allow_discovery = bench_cfg.get("allow_corpus_auto_discovery", False)
    dedup = bench_cfg.get("deduplicate_corpus", False)

    logger.info(f"Loading documents from plain text corpus: {corpus_path}...")
    documents = load_corpus_documents(corpus_path, deduplicate=dedup, allow_auto_discovery=allow_discovery)
    logger.info(f"Loaded {len(documents)} documents from TXT corpus.")

    reps_dir = output_dir / "corpus_representations"
    reps_dir.mkdir(parents=True, exist_ok=True)

    rep_files = {
        "narrative": open(reps_dir / "narrative.jsonl", "w", encoding="utf-8"),
        "key_value": open(reps_dir / "key_value.jsonl", "w", encoding="utf-8"),
        "template": open(reps_dir / "template.jsonl", "w", encoding="utf-8"),
        "json": open(reps_dir / "json.jsonl", "w", encoding="utf-8"),
        "mixed": open(reps_dir / "mixed.jsonl", "w", encoding="utf-8")
    }

    logger.info("Generating 5 multi-format corpus representations (Narrative, Key-Value, Template, Structured Semantic, Mixed)...")
    doc_count = 0

    try:
        for rec in tqdm(documents, desc="Corpus Representations"):
            doc_id = rec.get("doc_id")
            narrative_doc = rec.get("document", "")

            # Deterministic text-based semantic component extraction
            components = extract_text_components(narrative_doc)

            # 1. Narrative Representation
            rep_files["narrative"].write(json.dumps({
                "doc_id": doc_id,
                "occurrence_id": doc_id,
                "representation": "narrative",
                "document": narrative_doc
            }) + "\n")

            # 2. Key-Value Representation
            kv_doc = build_key_value_representation(doc_id, narrative_doc, components)
            rep_files["key_value"].write(json.dumps({
                "doc_id": doc_id,
                "occurrence_id": doc_id,
                "representation": "key_value",
                "document": kv_doc
            }) + "\n")

            # 3. Template Representation
            tmpl_doc = build_template_representation(doc_id, narrative_doc, components)
            rep_files["template"].write(json.dumps({
                "doc_id": doc_id,
                "occurrence_id": doc_id,
                "representation": "template",
                "document": tmpl_doc
            }) + "\n")

            # 4. Structured Semantic Representation (JSON)
            json_doc = build_structured_semantic_representation(doc_id, narrative_doc, components)
            rep_files["json"].write(json.dumps({
                "doc_id": doc_id,
                "occurrence_id": doc_id,
                "representation": "json",
                "document": json_doc
            }) + "\n")

            # 5. Mixed Representation
            mixed_doc = build_mixed_representation(narrative_doc, kv_doc)
            rep_files["mixed"].write(json.dumps({
                "doc_id": doc_id,
                "occurrence_id": doc_id,
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
