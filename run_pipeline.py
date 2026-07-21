import os
import sys
import argparse
import importlib
import time
from pathlib import Path

# Add scripts directory to path to allow dynamic importing of stages
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))

STAGES = {
    "01": ("01_parse_dictionary", "Parse Data Dictionary"),
    "02": ("02_profile_dataset", "Profile Datasets"),
    "03": ("03_discover_relationships", "Discover Schema Relationships"),
    "04": ("04_select_semantic_columns", "Select Semantic Columns"),
    "05": ("05_merge_tables", "Merge Datasets"),
    "05a": ("05a_validate_records", "Validate Records"),
    "06": ("06_generate_documents", "Generate Natural Language Documents"),
    "07": ("07_clean_documents", "Clean and Normalize Documents"),
    "08": ("08_export_corpus", "Export Maritime Corpus & Manifest"),
    "09": ("09_statistics", "Calculate Corpus Statistics & Report"),
    "10": ("10_extract_vocabulary", "Extract Maritime Vocabulary"),
    "11": ("11_tokenizer_analysis", "BERT Tokenizer Split Analysis"),
    "12": ("12_bert_mlm_evaluation", "BERT Masked Language Model Diagnostic Evaluation"),
    "13": ("13_lint_corpus", "Automated Corpus Quality Linting")
}

def run_stage(stage_key: str):
    """Dynamically imports and executes the main() function of a pipeline stage."""
    if stage_key not in STAGES:
        print(f"Error: Unknown stage '{stage_key}'")
        sys.exit(1)
        
    module_name, stage_desc = STAGES[stage_key]
    print(f"\n======================================================================")
    print(f" STARTING STAGE {stage_key}: {stage_desc.upper()}")
    print(f"======================================================================")
    
    t_start = time.time()
    try:
        # Import module dynamically
        module = importlib.import_module(module_name)
        # Run main function
        if hasattr(module, "main"):
            module.main()
        else:
            print(f"Error: Module '{module_name}' has no main() function.")
            sys.exit(1)
            
        elapsed = time.time() - t_start
        print(f"Completed Stage {stage_key} successfully in {elapsed:.2f} seconds.")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] in Stage {stage_key} ({module_name}): {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Maritime Accident Corpus Generation Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        help="Run a specific stage (e.g. 01, 05, 05a, 06, etc.). If omitted, runs the entire pipeline.",
        choices=list(STAGES.keys())
    )
    args = parser.parse_args()
    
    if args.stage:
        run_stage(args.stage)
    else:
        print("Running full Maritime Corpus Generation Pipeline...")
        t_all_start = time.time()
        
        # Execute stages in order
        for stage_key in sorted(STAGES.keys()):
            run_stage(stage_key)
            
        elapsed_all = time.time() - t_all_start
        print(f"\n======================================================================")
        print(f" [SUCCESS] FULL PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_all/60:.2f} MINUTES.")
        print(f"======================================================================")

if __name__ == "__main__":
    main()
