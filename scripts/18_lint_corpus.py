import os
import json
import re
from pathlib import Path
from tqdm import tqdm
from pipeline_utils import (
    setup_logging,
    load_config,
    get_project_root,
    get_benchmark_config,
    load_corpus_documents
)

logger = setup_logging("18_lint_corpus")

# Truly generic linguistic and structural quality rules
GENERIC_LINT_RULES = {
    "repeated_adjacent_words": r'\b([a-zA-Z]{3,})\s+\1\b',
    "malformed_singular_plural": r'\b1\s+(?:persons|items|units|records|events|cases|occurrences)\b',
    "awkward_phrasing": r'(?i)(?:damaged\s+damage|occurred\s+occurrence|failed\s+failure)',
    "duplicated_list_items": r'\b([a-zA-Z\s]+),\s+\1\b'
}

VALID_REPETITIONS = {"that", "had", "was", "york", "long", "far"}

def main():
    root = get_project_root()
    config = load_config()
    bench_cfg = get_benchmark_config()
    output_dir = root / config.get("output_dir", "outputs")

    # Resolve TXT corpus path (deterministic, no silent fallback)
    corpus_filename = bench_cfg.get("corpus_file", "maritime_corpus.txt")
    corpus_path = output_dir / corpus_filename

    allow_discovery = bench_cfg.get("allow_corpus_auto_discovery", False)
    dedup = bench_cfg.get("deduplicate_corpus", False)

    logger.info(f"Loading documents from text corpus for quality linting: {corpus_path}...")
    documents = load_corpus_documents(corpus_path, deduplicate=dedup, allow_auto_discovery=allow_discovery)
    total_docs = len(documents)
    logger.info(f"Loaded {total_docs} documents for quality linting.")

    # Combine generic rules with optional domain-specific patterns from config
    lint_rules = dict(GENERIC_LINT_RULES)
    domain_patterns = bench_cfg.get("domain_lint_patterns", [])
    if domain_patterns:
        # Construct domain artifact rule from configuration
        combined_pat = r'(?i)(?:' + "|".join(domain_patterns) + r')'
        lint_rules["domain_specific_artifacts"] = combined_pat

    issue_counts = {rule: 0 for rule in lint_rules}
    issue_samples = {rule: [] for rule in lint_rules}
    compiled_rules = {rule: re.compile(pat) for rule, pat in lint_rules.items()}

    for doc_item in tqdm(documents, desc="Linting Corpus"):
        doc_text = doc_item.get("document", "")
        doc_id = doc_item.get("doc_id", "")

        for rule_name, pattern in compiled_rules.items():
            matches = pattern.findall(doc_text)
            if matches:
                valid = True
                if rule_name == "repeated_adjacent_words":
                    valid_matches = [m for m in matches if (m[0] if isinstance(m, tuple) else m).lower() not in VALID_REPETITIONS]
                    if not valid_matches:
                        valid = False
                if valid:
                    issue_counts[rule_name] += 1
                    if len(issue_samples[rule_name]) < 5:
                        issue_samples[rule_name].append({
                            "doc_id": doc_id,
                            "occurrence_id": doc_id,  # Compatibility alias
                            "match": matches[0] if isinstance(matches[0], str) else matches[0][0],
                            "snippet": doc_text[:150]
                        })

    total_issues = sum(issue_counts.values())
    issue_rate = (total_issues / total_docs) if total_docs > 0 else 0.0
    status = "PASS" if issue_rate < 0.005 else "WARN"

    report = {
        "status": status,
        "corpus_file": corpus_path.name,
        "total_documents_linted": total_docs,
        "total_violations": total_issues,
        "violation_rate": f"{issue_rate*100:.3f}%",
        "configured_domain_rules_active": bool(domain_patterns),
        "rule_summary": {
            rule: {
                "count": count,
                "percentage": f"{(count/total_docs*100):.3f}%" if total_docs > 0 else "0.000%",
                "samples": issue_samples[rule]
            }
            for rule, count in issue_counts.items()
        }
    }

    out_path = output_dir / "corpus_lint_report.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2)

    logger.info(f"Corpus linting complete. Status: {status}. Report saved to {out_path}")

if __name__ == "__main__":
    main()
