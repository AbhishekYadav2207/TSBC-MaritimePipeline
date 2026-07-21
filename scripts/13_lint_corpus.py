import os
import json
import re
from pathlib import Path
from tqdm import tqdm
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("13_lint_corpus")

LINT_RULES = {
    "repeated_adjacent_words": r'\b([a-zA-Z]{3,})\s+\1\b',
    "malformed_singular_plural": r'\b1\s+(?:persons|injuries|fatalities|deaths|missing persons)\b',
    "administrative_leakage": r'(?i)(?:formerly\s*occno|extraction\s+status\s+pending|record\s+id\s*:?\s*\d+)',
    "awkward_phrasing": r'(?i)(?:carried\s+featured|sustained\s+damaged|damaged\s+damage)',
    "duplicated_list_items": r'\b([a-zA-Z\s]+),\s+\1\b'
}

VALID_REPETITIONS = {"that", "had", "was", "york", "long", "far"}

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")
    
    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents not found at {clean_path}!")
        return
        
    logger.info("Running automated corpus quality linter...")
    
    total_docs = 0
    issue_counts = {rule: 0 for rule in LINT_RULES}
    issue_samples = {rule: [] for rule in LINT_RULES}
    
    compiled_rules = {rule: re.compile(pat) for rule, pat in LINT_RULES.items()}
    
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Linting Corpus"):
            total_docs += 1
            record = json.loads(line)
            doc_text = record.get("document", "")
            oid = record.get("occurrence_id")
            
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
                                "occurrence_id": oid,
                                "match": matches[0] if isinstance(matches[0], str) else matches[0][0],
                                "snippet": doc_text[:150]
                            })
                        
    total_issues = sum(issue_counts.values())
    issue_rate = (total_issues / total_docs) if total_docs > 0 else 0.0
    status = "PASS" if issue_rate < 0.005 else "WARN"
    
    report = {
        "status": status,
        "total_documents_linted": total_docs,
        "total_violations": total_issues,
        "violation_rate": f"{issue_rate*100:.3f}%",
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
    for rule, count in issue_counts.items():
        logger.info(f"  {rule}: {count} violations ({(count/total_docs*100):.3f}%)")

if __name__ == "__main__":
    main()
