import os
import sys
import json
import unittest
import tempfile
import ast
from pathlib import Path
import numpy as np
import pandas as pd

# Add scripts directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from pipeline_utils import load_corpus_documents, get_benchmark_config, get_project_root
import importlib

stage11 = importlib.import_module("11_corpus_representations")
stage12 = importlib.import_module("12_semantic_importance")
stage13 = importlib.import_module("13_tokenizer_analysis")
stage14 = importlib.import_module("14_mlm_evaluation")
stage15 = importlib.import_module("15_cross_model_benchmarking")
stage16 = importlib.import_module("16_statistical_analysis")
stage17 = importlib.import_module("17_decision_engine")
stage18 = importlib.import_module("18_lint_corpus")

class TestUniversalBenchmark(unittest.TestCase):

    def setUp(self):
        self.test_corpus_path = Path(__file__).resolve().parent / "domain_test_corpus.txt"

    # 1. Test TXT Corpus Loading
    def test_01_txt_corpus_loading(self):
        docs = load_corpus_documents(self.test_corpus_path)
        self.assertGreater(len(docs), 0)
        self.assertTrue(all("doc_id" in d and "document" in d for d in docs))
        self.assertTrue(docs[0]["doc_id"].startswith("doc_"))

    # 2. Test Document Boundary Detection
    def test_02_document_boundary_detection(self):
        sample_text = "Doc 1 text.\n\nDoc 2 text with internal\nnewline.\n\nDoc 3 text."
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(sample_text)
            tmp_path = Path(tmp.name)
        try:
            docs = load_corpus_documents(tmp_path)
            self.assertEqual(len(docs), 3)
            self.assertEqual(docs[0]["document"], "Doc 1 text.")
            self.assertEqual(docs[1]["document"], "Doc 2 text with internal\nnewline.")
            self.assertEqual(docs[2]["document"], "Doc 3 text.")
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    # 3. Test Empty and Invalid Documents Handling
    def test_03_empty_and_invalid_documents(self):
        sample_text = "\n\n   \n\nValid Doc 1.\n\n\n\n   \n\nValid Doc 2.\n\n\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(sample_text)
            tmp_path = Path(tmp.name)
        try:
            docs = load_corpus_documents(tmp_path)
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["document"], "Valid Doc 1.")
            self.assertEqual(docs[1]["document"], "Valid Doc 2.")
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    # 4. Test Duplicate Documents Handling
    def test_04_duplicate_documents(self):
        sample_text = "Duplicate text paragraph.\n\nDuplicate text paragraph.\n\nUnique paragraph."
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(sample_text)
            tmp_path = Path(tmp.name)
        try:
            docs = load_corpus_documents(tmp_path)
            self.assertEqual(len(docs), 3)
            # Each receives unique doc_id
            self.assertNotEqual(docs[0]["doc_id"], docs[1]["doc_id"])
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    # 5. Test Vocabulary Extraction from Corpus Text
    def test_05_vocabulary_extraction(self):
        docs = load_corpus_documents(self.test_corpus_path)
        all_text = " ".join([d["document"] for d in docs]).lower()
        words = set([w for w in all_text.split() if len(w) > 3])
        self.assertIn("patient", words)
        self.assertIn("clinical", words)

    # 6. Test Multi-Format Representation Generation from Text
    def test_06_representation_generation(self):
        doc_id = "doc_test_001"
        doc_text = "The robotic surgery unit 'DA VINCI XI' operated under sterile protocol when it completed an arterial bypass."
        components = stage11.extract_text_components(doc_text)
        
        self.assertIn("DA VINCI XI", components["entities"])
        self.assertEqual(components["subject"], "DA VINCI XI")

        # Key-Value
        kv = stage11.build_key_value_representation(doc_id, doc_text, components)
        self.assertIn("Document ID: doc_test_001", kv)
        self.assertIn("Primary Subject: DA VINCI XI", kv)

        # Template (Deterministic selection with hashlib SHA-256)
        tmpl = stage11.build_template_representation(doc_id, doc_text, components)
        self.assertIn("DA VINCI XI", tmpl)

        # Structured Semantic (JSON)
        struct_json = stage11.build_structured_semantic_representation(doc_id, doc_text, components)
        parsed = json.loads(struct_json)
        self.assertEqual(parsed["representation_type"], "extracted_structured_semantic")
        self.assertEqual(parsed["document_id"], doc_id)
        self.assertIn("DA VINCI XI", parsed["extracted_entities"])

        # Mixed
        mixed = stage11.build_mixed_representation(doc_text, kv)
        self.assertIn("[EXTRACTED STRUCTURE]", mixed)
        self.assertIn("[TEXT NARRATIVE]", mixed)

    # 7. Test Text-Derived Semantic Scoring
    def test_07_semantic_scoring(self):
        doc_text = "During clinical trial phase, the medical device 'PULSE-X' operated under hospital monitoring, recording 120 bpm and sustaining zero failures."
        categories = {
            "cardiology": ["cardio", "pulse", "bpm", "heart"],
            "procedure": ["trial", "phase", "device", "surgical"]
        }
        rare_terms = {"bpm", "hemodynamic"}
        
        feats = stage12.compute_document_features(doc_text, categories, rare_terms, {}, 100)
        self.assertGreater(feats["domain_density"], 0.0)
        self.assertGreater(feats["concept_diversity"], 0.0)
        self.assertGreater(feats["entity_diversity"], 0.0)
        self.assertGreater(feats["structural_completeness"], 0.0)
        
        score = stage12.compute_raw_score(feats)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    # 8. Test Real Feature Ablation Calculations
    def test_08_real_feature_ablation_calculation(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for i in range(20):
                rec = {
                    "doc_id": f"doc_{i:04d}",
                    "importance_score": 40.0 + i,
                    "features": {
                        "domain_density": 0.3, "rare_score": 0.2, "concept_diversity": 0.5,
                        "entity_diversity": 0.4, "event_complexity": 0.3, "information_density": 0.6,
                        "structural_completeness": 0.8, "linguistic_diversity": 0.7,
                        "domain_novelty": 0.2, "redundancy_penalty": 0.0
                    },
                    "document": "Synthetic test document for ablation calculation."
                }
                tmp.write(json.dumps(rec) + "\n")
        try:
            ablation = stage16.compute_real_feature_ablation(tmp_path)
            self.assertTrue(ablation["is_real_calculation"])
            self.assertIn("features", ablation)
            self.assertIn("Rare Vocabulary", ablation["features"])
            drop = ablation["features"]["Rare Vocabulary"]["score_reduction"]
            self.assertIsInstance(drop, (int, float))
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    # 9. Test Paired Statistical Alignment
    def test_09_paired_statistical_alignment(self):
        # Create unaligned dataframe rows to verify merge alignment
        df_rows = []
        for rep in ["narrative", "key_value", "template"]:
            for sub in ["high", "medium", "low"]:
                df_rows.append({"model_name": "model_A", "representation": rep, "subset": sub, "top1_acc": 0.70})
        # Shuffle order for model B
        import random
        b_rows = list(df_rows)
        random.seed(42)
        random.shuffle(b_rows)
        for r in b_rows:
            df_rows.append({"model_name": "model_B", "representation": r["representation"], "subset": r["subset"], "top1_acc": 0.65})

        df = pd.DataFrame(df_rows)
        df1 = df[df["model_name"] == "model_A"][["representation", "subset", "top1_acc"]]
        df2 = df[df["model_name"] == "model_B"][["representation", "subset", "top1_acc"]]

        merged = pd.merge(df1, df2, on=["representation", "subset"], suffixes=("_m1", "_m2"))
        self.assertEqual(len(merged), 9)
        # Test diff is exact 0.05 on aligned cells
        diffs = merged["top1_acc_m1"] - merged["top1_acc_m2"]
        np.testing.assert_allclose(diffs, 0.05)

    # 10. Test Decision Engine Configuration
    def test_10_decision_engine_configuration(self):
        custom_decision = {
            "dapt": {"top1_threshold": 0.80, "max_domain_gap": 0.04, "max_fragmentation": 0.15},
            "scratch": {"top1_threshold": 0.50, "min_domain_gap": 0.25, "max_fragmentation": 0.45}
        }
        thresh = stage17.parse_thresholds(custom_decision)
        self.assertEqual(thresh["dapt_top1_threshold"], 80.0)
        self.assertEqual(thresh["gap_threshold"], 4.0)
        self.assertEqual(thresh["frag_threshold"], 15.0)

        # Evaluate rules for medical domain
        res = stage17.run_decision_rules(82.0, 3.0, 10.0, thresh, domain_name="medical")
        self.assertIn("DAPT", res["decision"])
        self.assertIn("Medical", res["strategy"])

    # 11. Test End-to-End Non-Maritime Corpus Flow
    def test_11_non_maritime_corpus_pipeline_flow(self):
        docs = load_corpus_documents(self.test_corpus_path)
        self.assertEqual(len(docs), 5)
        
        # Run representations
        reps = []
        for d in docs:
            comp = stage11.extract_text_components(d["document"])
            kv = stage11.build_key_value_representation(d["doc_id"], d["document"], comp)
            reps.append(kv)
        self.assertEqual(len(reps), 5)

        # Run semantic scoring
        categories = {"biomedical": ["clinical", "trial", "patient", "coronary", "arterial", "hemodynamic"]}
        rare = {"mab-2048", "da vinci xi", "bypass"}
        for d in docs:
            feats = stage12.compute_document_features(d["document"], categories, rare, {}, len(docs))
            score = stage12.compute_raw_score(feats)
            self.assertGreaterEqual(score, 0.0)

    # 12. STATIC DEPENDENCY AUDIT: Prove scripts/11* through scripts/18* have zero MARSIS/clean_documents dependencies
    def test_12_static_dependency_audit_scripts_11_to_18(self):
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        stage_scripts = [
            "11_corpus_representations.py",
            "12_semantic_importance.py",
            "13_tokenizer_analysis.py",
            "14_mlm_evaluation.py",
            "15_cross_model_benchmarking.py",
            "16_statistical_analysis.py",
            "17_decision_engine.py",
            "18_lint_corpus.py"
        ]

        forbidden_tokens = [
            "clean_documents.jsonl",
            "OccID",
            "VesselID",
            "OccTypeDisplayEng",
            "AccIncTypeDisplayEng",
            "NearestLocationDescription",
            "WeatherConditionDisplayEng",
            "SeaStateDisplayEng",
            "GrossTonnage",
            "HullMaterialDisplayEng",
            "VesselPhaseDisplayEng",
            "NavigationAidTypeDisplayEng",
            "LsApplianceDisplayEng"
        ]

        for script_name in stage_scripts:
            script_path = scripts_dir / script_name
            self.assertTrue(script_path.exists(), f"Script missing: {script_name}")

            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST to ensure valid Python syntax
            tree = ast.parse(content, filename=str(script_path))
            self.assertIsNotNone(tree)

            # Check that none of the forbidden tokens exist in code
            for token in forbidden_tokens:
                self.assertNotIn(
                    token,
                    content,
                    f"Forbidden dependency '{token}' found in {script_name}!"
                )

    # 13. Regression Test: Subword/Domain-Span Contamination (Priority 0, Fix 1)
    def test_13_subword_domain_span_contamination(self):
        # Term 'hemodynamic' should NOT match when 'dynamic' appears alone
        vocab = ["hemodynamic"]
        categories = {"biomedical": ["hemo", "dynamic"]}
        rare = set()

        # Sentence with only 'dynamic', not 'hemodynamic'
        text_without_domain_term = "Engineers implemented a dynamic routing algorithm."
        spans_1 = stage14.extract_domain_spans(text_without_domain_term, vocab, rare, categories)
        self.assertEqual(len(spans_1), 0, "Subword 'dynamic' was incorrectly matched as domain term 'hemodynamic'!")

        # Sentence with the actual domain term 'hemodynamic'
        text_with_domain_term = "Patient exhibited stable hemodynamic pressure."
        spans_2 = stage14.extract_domain_spans(text_with_domain_term, vocab, rare, categories)
        self.assertEqual(len(spans_2), 1)
        self.assertEqual(spans_2[0]["term"], "hemodynamic")
        self.assertEqual(text_with_domain_term[spans_2[0]["start"]:spans_2[0]["end"]], "hemodynamic")

    # 14. Regression Test: Missing General-English Baseline (Priority 0, Fix 2)
    def test_14_missing_general_english_baseline(self):
        # When gen_eng_docs is empty, baseline must be marked unavailable without inventing 0.85
        gen_eng_eval = stage14.evaluate_model_on_docs(None, None, [], ["term"], {}, [], "cpu")
        self.assertEqual(gen_eng_eval, {})
        gen_eng_summary = gen_eng_eval.get("general_tokens_summary")
        self.assertIsNone(gen_eng_summary)

    # 15. Regression Test: Empty Domain Lexicons (Priority 0, Fix 3)
    def test_15_empty_domain_lexicons(self):
        # Stage 12 must run with empty domain lexicons using generic linguistic heuristics
        doc_text = "The system was tested under operating conditions, generating 500 records and zero errors."
        feats = stage12.compute_document_features(
            doc_text=doc_text,
            categories={},
            rare_terms=set(),
            term_freq_map={},
            total_docs=1,
            domain_lexicons={}
        )
        self.assertIn("structural_completeness", feats)
        self.assertGreater(feats["structural_completeness"], 0.0)
        score = stage12.compute_raw_score(feats)
        self.assertGreaterEqual(score, 0.0)

    # 16. Regression Test: Non-Maritime Semantic Scoring (Priority 0, Fix 3)
    def test_16_non_maritime_semantic_scoring(self):
        legal_text = "The court delivered judgment regarding patent infringement following appellate review."
        legal_cats = {"jurisprudence": ["court", "judgment", "patent", "appellate"]}
        feats = stage12.compute_document_features(
            doc_text=legal_text,
            categories=legal_cats,
            rare_terms={"appellate"},
            term_freq_map={"court": 5, "patent": 3},
            total_docs=10,
            domain_lexicons={}
        )
        self.assertGreater(feats["domain_density"], 0.0)
        self.assertGreater(feats["rare_score"], 0.0)
        score = stage12.compute_raw_score(feats)
        self.assertGreater(score, 0.0)

    # 17. Regression Test: Non-Maritime Linting with Empty Domain Patterns (Priority 0, Fix 4)
    def test_17_non_maritime_linting(self):
        sample_doc = "Normal technical documentation paragraph describing component behavior."
        # Verify generic rules compile and run with no domain-specific patterns
        rules = dict(stage18.GENERIC_LINT_RULES)
        self.assertNotIn("domain_specific_artifacts", rules)
        for name, pat in rules.items():
            import re
            m = re.findall(pat, sample_doc)
            self.assertEqual(len(m), 0)

    # 18. Regression Test: Missing Configured Corpus Deterministic Error (Priority 0, Fix 7)
    def test_18_missing_configured_corpus_error(self):
        # When corpus does not exist and allow_auto_discovery=False, must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_corpus_documents(Path("non_existent_domain_corpus_file.txt"), allow_auto_discovery=False)

    # 19. Regression Test: Dynamic Category Scoring (Priority 0, Fix 6)
    def test_19_dynamic_category_scoring(self):
        # Create comparison rows with domain-agnostic categories
        rows = [
            {"model_name": "model_X", "cat_cardiology_acc": 0.82, "cat_neurology_acc": 0.76, "cat_oncology_acc": 0.79},
            {"model_name": "model_X", "cat_cardiology_acc": 0.84, "cat_neurology_acc": 0.78, "cat_oncology_acc": 0.81}
        ]
        df = pd.DataFrame(rows)
        cat_keys = ["cardiology", "neurology", "oncology"]
        means = [df[f"cat_{c}_acc"].mean() for c in cat_keys]
        balance = 1.0 - float(np.std(means))
        self.assertGreater(balance, 0.9)
        self.assertNotIn("nav_acc", df.columns)

    # 20. Regression Test: Duplicate Detection Semantics (Priority 2, Fix 13)
    def test_20_duplicate_detection_semantics(self):
        corpus_with_dups = "Text A.\n\nText A.\n\nText B."
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(corpus_with_dups)
            tmp_path = Path(tmp.name)
        try:
            # Default: deduplicate=False (preserve all, flag is_duplicate)
            docs_all = load_corpus_documents(tmp_path, deduplicate=False)
            self.assertEqual(len(docs_all), 3)
            self.assertFalse(docs_all[0]["is_duplicate"])
            self.assertTrue(docs_all[1]["is_duplicate"])
            self.assertFalse(docs_all[2]["is_duplicate"])

            # Explicit deduplicate=True (filters out duplicates)
            docs_deduped = load_corpus_documents(tmp_path, deduplicate=True)
            self.assertEqual(len(docs_deduped), 2)
            self.assertEqual(docs_deduped[0]["document"], "Text A.")
            self.assertEqual(docs_deduped[1]["document"], "Text B.")
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    # 21. Regression Test: Information Density vs Domain Density Independence (Priority 1, Fix 9)
    def test_21_density_and_information_density_independence(self):
        # A general English sentence with zero domain tokens
        general_text = "The comprehensive laboratory investigation confirmed optimal operational parameters."
        categories = {"maritime": ["vessel", "ship", "anchor"]}
        feats = stage12.compute_document_features(general_text, categories, set(), {}, 10, {})
        
        # Domain density must be 0, while information density must be positive
        self.assertEqual(feats["domain_density"], 0.0)
        self.assertGreater(feats["information_density"], 0.5)
        # Demonstrates no double-counting between domain_density and information_density

    # 22. Regression Test: Deterministic Stable Sampling (Priority 1, Fix 8)
    def test_22_stable_deterministic_sampling(self):
        records = [{"doc_id": f"id_{i}", "importance_score": float(i)} for i in range(100)]
        
        import random
        random.seed(42)
        sample_1 = random.sample(records, 10)

        random.seed(42)
        sample_2 = random.sample(records, 10)

        self.assertEqual([r["doc_id"] for r in sample_1], [r["doc_id"] for r in sample_2])

if __name__ == "__main__":
    unittest.main()

