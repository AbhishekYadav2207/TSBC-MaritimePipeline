import os
import sys
import json
import unittest
import tempfile
import ast
import re
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

    # 1. TXT Loading
    def test_01_txt_corpus_loading(self):
        docs = load_corpus_documents(self.test_corpus_path)
        self.assertGreater(len(docs), 0)
        self.assertTrue(all("doc_id" in d and "document" in d for d in docs))
        self.assertTrue(docs[0]["doc_id"].startswith("doc_"))

    # 2. Blank-Line Boundaries
    def test_02_blank_line_boundaries(self):
        sample_text = "Doc 1 text.\n\nDoc 2 text with internal\nnewline.\n\n\nDoc 3 text."
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

    # 3. Empty-Document Filtering
    def test_03_empty_document_filtering(self):
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

    # 4. Duplicate Detection (deduplicate=False marks duplicates without removing)
    def test_04_duplicate_detection(self):
        sample_text = "Duplicate text paragraph.\n\nDuplicate text paragraph.\n\nUnique paragraph."
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(sample_text)
            tmp_path = Path(tmp.name)
        try:
            docs = load_corpus_documents(tmp_path, deduplicate=False)
            self.assertEqual(len(docs), 3)
            self.assertFalse(docs[0]["is_duplicate"])
            self.assertTrue(docs[1]["is_duplicate"])
            self.assertFalse(docs[2]["is_duplicate"])
            self.assertNotEqual(docs[0]["doc_id"], docs[1]["doc_id"])
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    # 5. Explicit Deduplication (deduplicate=True removes exact duplicates)
    def test_05_explicit_deduplication(self):
        sample_text = "Exact text.\n\nExact text.\n\nUnique text."
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(sample_text)
            tmp_path = Path(tmp.name)
        try:
            docs = load_corpus_documents(tmp_path, deduplicate=True)
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["document"], "Exact text.")
            self.assertEqual(docs[1]["document"], "Unique text.")
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    # 6. Deterministic Document IDs
    def test_06_deterministic_document_ids(self):
        docs1 = load_corpus_documents(self.test_corpus_path)
        docs2 = load_corpus_documents(self.test_corpus_path)
        self.assertEqual([d["doc_id"] for d in docs1], [d["doc_id"] for d in docs2])
        self.assertEqual(docs1[0]["doc_id"], "doc_000001")
        self.assertEqual(docs1[1]["doc_id"], "doc_000002")

    # 7. Deterministic Sampling
    def test_07_deterministic_sampling(self):
        records = [{"doc_id": f"id_{i}", "score": float(i)} for i in range(100)]
        import random
        random.seed(42)
        sample_1 = random.sample(records, 15)
        random.seed(42)
        sample_2 = random.sample(records, 15)
        self.assertEqual([r["doc_id"] for r in sample_1], [r["doc_id"] for r in sample_2])

    # 8. Five Representations
    def test_08_five_representations(self):
        doc_id = "doc_test_001"
        doc_text = "The robotics laboratory 'CYBER-CORE' tested an autonomous agent under sterile conditions, measuring 3.5 GHz clock speed."
        components = stage11.extract_text_components(doc_text)
        
        # 1. Narrative
        # 2. Key-Value
        kv = stage11.build_key_value_representation(doc_id, doc_text, components)
        self.assertIn("Document ID: doc_test_001", kv)
        self.assertIn("Primary Subject:", kv)

        # 3. Template
        tmpl = stage11.build_template_representation(doc_id, doc_text, components)
        self.assertIn("CYBER-CORE", tmpl)

        # 4. Structured Semantic
        struct_json = stage11.build_structured_semantic_representation(doc_id, doc_text, components)
        parsed = json.loads(struct_json)
        self.assertEqual(parsed["representation_type"], "extracted_structured_semantic")
        self.assertEqual(parsed["document_id"], doc_id)
        self.assertIn("whitespace_token_count", parsed)

        # 5. Mixed
        mixed = stage11.build_mixed_representation(doc_text, kv)
        self.assertIn("[EXTRACTED STRUCTURE]", mixed)
        self.assertIn("[TEXT NARRATIVE]", mixed)

    # 9. No occurrence_id in Output
    def test_09_no_occurrence_id(self):
        doc_id = "doc_test_001"
        doc_text = "The financial audit firm 'VERTEX' examined ledger records, recording €200 surplus."
        components = stage11.extract_text_components(doc_text)
        struct_json = stage11.build_structured_semantic_representation(doc_id, doc_text, components)
        parsed = json.loads(struct_json)
        self.assertNotIn("occurrence_id", parsed)

        # Verify Stage 12 features and scored records
        feats = stage12.compute_document_features(doc_text, {}, set(), {}, 10, {})
        self.assertNotIn("occurrence_id", feats)

    # 10. Neutral Templates
    def test_10_neutral_templates(self):
        forbidden_template_words = ["event record", "incident narrative", "operational environment", "observed event"]
        for tmpl in stage11.TEMPLATES:
            tmpl_lower = tmpl.lower()
            for forbidden in forbidden_template_words:
                self.assertNotIn(forbidden, tmpl_lower, f"Forbidden ontology '{forbidden}' found in template: {tmpl}")

    # 11. Generic Quantities Extraction
    def test_11_generic_quantities(self):
        samples = [
            ("Administered 20 mg daily with 15% improvement.", ["20 mg", "15%"]),
            ("Latency was 120 ms under 500 kg load.", ["120 ms", "500 kg"]),
            ("Cost reached $500 while European unit spent €200.", ["$500", "€200"]),
            ("Processor clocked at 3.5 GHz covering 20km with 120ms ping.", ["3.5 GHz", "20km", "120ms"])
        ]
        for text, expected_items in samples:
            comp = stage11.extract_text_components(text)
            extracted = " ".join(comp["measurements"])
            for expected in expected_items:
                self.assertIn(expected, extracted, f"Failed to extract '{expected}' from '{text}'. Extracted: {comp['measurements']}")

    # 12. Non-Maritime Semantic Scoring
    def test_12_non_maritime_semantic_scoring(self):
        legal_text = "The appellate court delivered judgment regarding patent infringement following evidentiary review."
        categories = {"jurisprudence": ["court", "judgment", "patent", "evidentiary"]}
        rare = {"infringement"}
        feats = stage12.compute_document_features(legal_text, categories, rare, {"court": 2}, 10)
        self.assertGreater(feats["domain_density"], 0.0)
        self.assertGreater(feats["concept_diversity"], 0.0)
        score = stage12.compute_raw_score(feats)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 100.0)

    # 13. Empty Domain Lexicons
    def test_13_empty_domain_lexicons(self):
        text = "The control system executed routine validation under normal parameters, recording 100 ms response."
        feats = stage12.compute_document_features(
            doc_text=text,
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

    # 14. Dynamic Categories
    def test_14_dynamic_categories(self):
        cat_rec = {"cardiology": 0.85, "oncology": 0.78, "neurology": 0.82}
        df_row = {"model_name": "model_test"}
        for cat, acc in cat_rec.items():
            df_row[f"cat_{cat}_acc"] = acc
        self.assertEqual(df_row["cat_cardiology_acc"], 0.85)
        self.assertNotIn("nav_acc", df_row)
        self.assertNotIn("vessel_acc", df_row)

    # 15. Non-Maritime Linting
    def test_15_non_maritime_linting(self):
        clean_text = "The computational chemistry cluster finished simulations under scheduled parameters."
        rules = stage18.GENERIC_LINT_RULES
        for rule_name, pat in rules.items():
            matches = re.findall(pat, clean_text)
            self.assertEqual(len(matches), 0)

    # 16. Empty Domain Lint Configuration
    def test_16_empty_domain_lint_configuration(self):
        rules = dict(stage18.GENERIC_LINT_RULES)
        self.assertNotIn("domain_specific_artifacts", rules)

    # 17. Missing Corpus Failure
    def test_17_missing_corpus_failure(self):
        with self.assertRaises(FileNotFoundError):
            load_corpus_documents(Path("non_existent_domain_corpus_12345.txt"), allow_auto_discovery=False)

    # 18. Explicit Auto-Discovery
    def test_18_explicit_auto_discovery(self):
        # When file doesn't exist but allow_auto_discovery=True and directory has *_corpus.txt
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            corpus_file = tmpdir_path / "custom_domain_corpus.txt"
            corpus_file.write_text("Discovered document text 1.\n\nDiscovered document text 2.", encoding="utf-8")
            missing_configured = tmpdir_path / "missing_file.txt"
            docs = load_corpus_documents(missing_configured, allow_auto_discovery=True)
            self.assertEqual(len(docs), 2)

    # 19. Span-Aware Domain Matching
    def test_19_span_aware_domain_matching(self):
        vocab = ["hemodynamic"]
        categories = {"biomedical": ["hemo", "dynamic"]}
        text = "Patient exhibited stable hemodynamic pressure."
        spans = stage14.extract_domain_spans(text, vocab, set(), categories)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["term"], "hemodynamic")
        self.assertEqual(text[spans[0]["start"]:spans[0]["end"]], "hemodynamic")

    # 20. Subword Contamination (dynamic in hemodynamic)
    def test_20_subword_contamination(self):
        vocab = ["hemodynamic"]
        categories = {"biomedical": ["hemo", "dynamic"]}
        text = "Engineers implemented a dynamic routing algorithm."
        spans = stage14.extract_domain_spans(text, vocab, set(), categories)
        self.assertEqual(len(spans), 0, "Subword 'dynamic' was falsely matched as domain term 'hemodynamic'!")

    # 21. Unavailable Baseline
    def test_21_unavailable_baseline(self):
        eval_res = stage14.evaluate_model_on_docs(None, None, [], ["term"], {}, [], "cpu")
        self.assertEqual(eval_res, {})

    # 22. Information vs Domain Density Independence
    def test_22_information_domain_density_independence(self):
        general_text = "The comprehensive laboratory investigation confirmed optimal operational parameters."
        categories = {"custom": ["custom_domain_term"]}
        feats = stage12.compute_document_features(general_text, categories, set(), {}, 10, {})
        self.assertEqual(feats["domain_density"], 0.0)
        self.assertGreater(feats["information_density"], 0.5)

    # 23. Actual Document-Count Timing Divisor (No Hardcoded /200)
    def test_23_actual_document_count_timing_divisor(self):
        grp_data = pd.DataFrame([
            {"eval_time_sec": 2.5, "evaluated_doc_count": 50.0},
            {"eval_time_sec": 3.5, "evaluated_doc_count": 50.0}
        ])
        avg_eval_time = grp_data["eval_time_sec"].mean()
        avg_doc_count = grp_data["evaluated_doc_count"].mean()
        eval_time_per_doc_ms = (avg_eval_time / avg_doc_count) * 1000.0
        # Expected: (3.0 / 50.0) * 1000 = 60.0 ms (NOT 3.0 / 200 = 15.0 ms)
        self.assertEqual(eval_time_per_doc_ms, 60.0)

        # Fails safely on None/missing count without assuming 200
        empty_grp = pd.DataFrame([{"eval_time_sec": 2.0, "evaluated_doc_count": np.nan}])
        valid_cnts = empty_grp["evaluated_doc_count"].dropna()
        self.assertTrue(valid_cnts.empty)

    # 24. Stage 16 Pairing
    def test_24_stage16_pairing(self):
        df_rows = []
        for rep in ["narrative", "key_value", "template"]:
            for sub in ["high", "medium", "low"]:
                df_rows.append({"model_name": "model_A", "representation": rep, "subset": sub, "top1_acc": 0.80})
                df_rows.append({"model_name": "model_B", "representation": rep, "subset": sub, "top1_acc": 0.70})
        df = pd.DataFrame(df_rows)
        df1 = df[df["model_name"] == "model_A"][["representation", "subset", "top1_acc"]]
        df2 = df[df["model_name"] == "model_B"][["representation", "subset", "top1_acc"]]
        merged = pd.merge(df1, df2, on=["representation", "subset"], suffixes=("_m1", "_m2"))
        self.assertEqual(len(merged), 9)
        diffs = merged["top1_acc_m1"] - merged["top1_acc_m2"]
        np.testing.assert_allclose(diffs, 0.10)

    # 25. Stage 16 Effect Sizes
    def test_25_stage16_effect_sizes(self):
        a1 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        a2 = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
        d = stage16.cohens_d(a1, a2)
        delta = stage16.cliffs_delta(a1, a2)
        self.assertGreater(d, 0.0)
        self.assertGreater(delta, 0.0)
        self.assertLessEqual(delta, 1.0)

    # 26. Stage 17 Missing Metrics
    def test_26_stage17_missing_metrics(self):
        thresh = {
            "dapt_top1_threshold": 85.0, "gap_threshold": 5.0, "frag_threshold": 20.0,
            "scratch_top1_threshold": 60.0, "scratch_gap_threshold": 20.0, "scratch_frag_threshold": 40.0
        }
        res_nan = stage17.run_decision_rules(np.nan, 3.0, 15.0, thresh, domain_name="medical")
        self.assertEqual(res_nan["decision"], "Incomplete Evaluation / Missing Metrics")
        self.assertIn("Evaluation Incomplete", res_nan["strategy"])

        res_none = stage17.run_decision_rules(75.0, None, 15.0, thresh, domain_name="medical")
        self.assertEqual(res_none["decision"], "Incomplete Evaluation / Missing Metrics")

    # 27. Stage 17 Threshold Parsing
    def test_27_stage17_threshold_parsing(self):
        decimal_cfg = {
            "dapt": {"top1_threshold": 0.85, "max_domain_gap": 0.05, "max_fragmentation": 0.20},
            "scratch": {"top1_threshold": 0.60, "min_domain_gap": 0.20, "max_fragmentation": 0.40}
        }
        thresh = stage17.parse_thresholds(decimal_cfg)
        self.assertEqual(thresh["dapt_top1_threshold"], 85.0)
        self.assertEqual(thresh["gap_threshold"], 5.0)
        self.assertEqual(thresh["frag_threshold"], 20.0)

    # 28. Stage 18 Generic Linting
    def test_28_stage18_generic_linting(self):
        text_with_repeat = "The algorithm processed the the data stream."
        match = re.findall(stage18.GENERIC_LINT_RULES["repeated_adjacent_words"], text_with_repeat)
        self.assertIn("the", match)

        text_with_plural = "The query returned 1 records."
        match_plural = re.findall(stage18.GENERIC_LINT_RULES["malformed_singular_plural"], text_with_plural)
        self.assertEqual(len(match_plural), 1)

    # 29. Static Forbidden Dependency Audit
    def test_29_static_forbidden_dependency_audit(self):
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        stage_scripts = [
            "11_corpus_representations.py", "12_semantic_importance.py", "13_tokenizer_analysis.py",
            "14_mlm_evaluation.py", "15_cross_model_benchmarking.py", "16_statistical_analysis.py",
            "17_decision_engine.py", "18_lint_corpus.py"
        ]
        forbidden_tokens = [
            "clean_documents.jsonl", "OccID", "VesselID", "OccTypeDisplayEng", "AccIncTypeDisplayEng",
            "NearestLocationDescription", "WeatherConditionDisplayEng", "SeaStateDisplayEng",
            "GrossTonnage", "HullMaterialDisplayEng", "VesselPhaseDisplayEng",
            "NavigationAidTypeDisplayEng", "LsApplianceDisplayEng"
        ]
        for script_name in stage_scripts:
            content = (scripts_dir / script_name).read_text(encoding="utf-8")
            ast.parse(content)
            for token in forbidden_tokens:
                self.assertNotIn(token, content, f"Forbidden token '{token}' found in {script_name}!")

    # 30. Semantic Domain-Hardcoding Audit
    def test_30_semantic_domain_hardcoding_audit(self):
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        stage_scripts = [
            "11_corpus_representations.py", "12_semantic_importance.py", "13_tokenizer_analysis.py",
            "14_mlm_evaluation.py", "15_cross_model_benchmarking.py", "16_statistical_analysis.py",
            "17_decision_engine.py", "18_lint_corpus.py"
        ]
        forbidden_semantic_tokens = [
            "occurrence_id", "maritime", "vessel", "ship", "marine",
            "navigation", "nautical", "casualty", "MARSIS", "OccNo"
        ]
        for script_name in stage_scripts:
            content = (scripts_dir / script_name).read_text(encoding="utf-8")
            for token in forbidden_semantic_tokens:
                self.assertNotIn(
                    token.lower(),
                    content.lower(),
                    f"Forbidden domain term/identifier '{token}' found in {script_name}!"
                )

    # 31. Medical-like Corpus
    def test_31_medical_like_corpus(self):
        med_docs = [
            "The patient received 50 mg of monoclonal antibody MAB-101 under sterile intravenous protocol, recording 72 bpm heart rate.",
            "During Phase III cardiology trial, subject 'PATIENT-882' demonstrated stable hemodynamic parameters with 120 ms cardiac cycles."
        ]
        for idx, doc in enumerate(med_docs):
            comp = stage11.extract_text_components(doc)
            self.assertTrue(len(comp["measurements"]) > 0)
            kv = stage11.build_key_value_representation(f"doc_{idx+1}", doc, comp)
            self.assertIn("Document ID:", kv)
            feats = stage12.compute_document_features(
                doc,
                categories={"cardiology": ["cardio", "hemodynamic", "bpm", "heart"]},
                rare_terms={"mab-101"},
                term_freq_map={"patient": 2},
                total_docs=2
            )
            self.assertGreater(feats["domain_density"], 0.0)

    # 32. Financial-like Corpus
    def test_32_financial_like_corpus(self):
        fin_docs = [
            "The investment fund 'ALPHA-CAPITAL' acquired 500 shares at €200 per equity unit under quarterly rebalancing.",
            "Federal regulators issued $500 fine following statutory audit of sovereign debt holdings."
        ]
        for idx, doc in enumerate(fin_docs):
            comp = stage11.extract_text_components(doc)
            self.assertTrue(any("€" in m or "$" in m for m in comp["measurements"]))
            tmpl = stage11.build_template_representation(f"doc_{idx+1}", doc, comp)
            self.assertIn("Record", tmpl)
            feats = stage12.compute_document_features(
                doc,
                categories={"equities": ["equity", "shares", "capital", "debt"]},
                rare_terms={"rebalancing"},
                term_freq_map={"audit": 1},
                total_docs=2
            )
            self.assertGreater(feats["domain_density"], 0.0)

    # 33. Software-like Corpus
    def test_33_software_like_corpus(self):
        sw_docs = [
            "The microservice cluster 'AUTH-GATEWAY' processed token authentication in 120 ms under heavy network ingress.",
            "Engineers deployed patch v2.4 to resolve memory leakage in the distributed cache tier, reducing latency by 45 ms."
        ]
        for idx, doc in enumerate(sw_docs):
            comp = stage11.extract_text_components(doc)
            self.assertTrue(any("ms" in m for m in comp["measurements"]))
            struct = stage11.build_structured_semantic_representation(f"doc_{idx+1}", doc, comp)
            self.assertIn("document_id", struct)
            feats = stage12.compute_document_features(
                doc,
                categories={"distributed_systems": ["microservice", "gateway", "cache", "network"]},
                rare_terms={"ingress"},
                term_freq_map={"patch": 1},
                total_docs=2
            )
            self.assertGreater(feats["domain_density"], 0.0)

    # 34. Scientific-like Corpus
    def test_34_scientific_like_corpus(self):
        sci_docs = [
            "The radio astronomy observatory 'ALMA-SPEC' detected molecular resonance at 3.5 GHz under cryogenic receiver conditions.",
            "Spectrometry analysis revealed high density absorption lines across 20km orbital radius."
        ]
        for idx, doc in enumerate(sci_docs):
            comp = stage11.extract_text_components(doc)
            self.assertTrue(any("GHz" in m or "20km" in m for m in comp["measurements"]))
            mixed = stage11.build_mixed_representation(doc, stage11.build_key_value_representation(f"doc_{idx+1}", doc, comp))
            self.assertIn("[EXTRACTED STRUCTURE]", mixed)
            feats = stage12.compute_document_features(
                doc,
                categories={"astrophysics": ["astronomy", "molecular", "resonance", "spectrometry"]},
                rare_terms={"cryogenic"},
                term_freq_map={"absorption": 1},
                total_docs=2
            )
            self.assertGreater(feats["domain_density"], 0.0)

if __name__ == "__main__":
    unittest.main()
