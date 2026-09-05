import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline_utils import (
    setup_logging,
    load_config,
    get_project_root,
    get_benchmark_config
)

logger = setup_logging("15_cross_model_benchmarking")

MODEL_PROFILES = {
    "bert-base-uncased": {"params_m": 110, "size_mb": 440},
    "bert-large-uncased": {"params_m": 340, "size_mb": 1340},
    "roberta-base": {"params_m": 125, "size_mb": 500},
    "microsoft/deberta-v3-base": {"params_m": 86, "size_mb": 500},
    "answerdotai/ModernBERT-base": {"params_m": 149, "size_mb": 590},
    "allenai/scibert_scivocab_uncased": {"params_m": 110, "size_mb": 440},
    "dmis-lab/biobert-base-cased-v1.2": {"params_m": 110, "size_mb": 440},
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": {"params_m": 110, "size_mb": 440},
    "emilyalsentzer/Bio_ClinicalBERT": {"params_m": 110, "size_mb": 440},
    "nlpaueb/legal-bert-base-uncased": {"params_m": 110, "size_mb": 440},
    "ProsusAI/finbert": {"params_m": 110, "size_mb": 440},
    "anferico/bert-for-patents": {"params_m": 340, "size_mb": 1340},
    "google/electra-base-discriminator": {"params_m": 110, "size_mb": 440},
    "distilbert-base-uncased": {"params_m": 66, "size_mb": 268}
}

def clean_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")

def apply_legacy_maritime_compatibility(row: dict, cat_rec: dict) -> dict:
    """
    Isolated backwards-compatibility layer.
    Exposes legacy column aliases without polluting universal scoring logic.
    """
    row["nav_acc"] = cat_rec.get("navigation", 0.0)
    row["weather_acc"] = cat_rec.get("weather_environment", 0.0)
    row["safety_acc"] = cat_rec.get("safety_lifesaving", 0.0)
    row["machinery_acc"] = cat_rec.get("machinery_propulsion", 0.0)
    row["vessel_acc"] = cat_rec.get("vessel_terminology", 0.0)
    row["casualty_acc"] = cat_rec.get("casualty_incident", 0.0)
    return row

def main():
    root = get_project_root()
    config = load_config()
    bench_cfg = get_benchmark_config()
    output_dir = root / config.get("output_dir", "outputs")

    domain_name = bench_cfg.get("domain_name", "maritime")
    metric_name = bench_cfg.get("metric_name", "DUI")
    tok_dir = output_dir / "tokenizer_analysis"
    cache_dir = output_dir / "evaluations" / "cache"

    # 1. Load Tokenizer Benchmark Data
    tok_data = {}
    if tok_dir.exists():
        for json_file in tok_dir.glob("*.json"):
            if json_file.name == "tokenizer_comparison.csv":
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                tok_data[data["model_name"]] = data

    # 2. Load MLM Evaluation Cache Data (175 matrix runs)
    mlm_rows = []
    category_keys = set()

    if cache_dir.exists():
        for json_file in cache_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                item = json.load(f)
                model_name = item.get("model_name")
                rep = item.get("representation")
                sub = item.get("subset")
                metrics = item.get("evaluation_metrics", {})

                gen_sum = metrics.get("general_tokens_summary", {})
                dom_sum = metrics.get("domain_tokens_summary") or metrics.get("maritime_tokens_summary", {})
                rare_sum = metrics.get("rare_domain_tokens_summary") or metrics.get("rare_maritime_tokens_summary", {})
                cat_rec = metrics.get("category_recall", {})

                row = {
                    "model_name": model_name,
                    "representation": rep,
                    "subset": sub,
                    "domain_shift_gap": item.get("domain_shift_gap"),
                    "mlm_loss": dom_sum.get("mlm_loss", 0.0),
                    "top1_acc": dom_sum.get("top1_accuracy", 0.0),
                    "top5_acc": dom_sum.get("top5_accuracy", 0.0),
                    "top10_acc": dom_sum.get("top10_accuracy", 0.0),
                    "rare_top1_acc": rare_sum.get("top1_accuracy", 0.0),
                    "performance_gap": metrics.get("performance_gap_top1", 0.0),
                    "eval_time_sec": metrics.get("evaluation_time_sec", 0.0)
                }

                # Purely dynamic category registration
                for cat, acc in cat_rec.items():
                    category_keys.add(cat)
                    row[f"cat_{cat}_acc"] = acc

                # Isolated backward-compatibility layer
                if domain_name == "maritime":
                    row = apply_legacy_maritime_compatibility(row, cat_rec)

                mlm_rows.append(row)

    df_mlm = pd.DataFrame(mlm_rows)
    if df_mlm.empty:
        logger.error("No MLM evaluation cache records found! Run Stage 14 first.")
        return

    # Export full comparison.csv
    df_mlm.to_csv(output_dir / "comparison.csv", index=False)

    # 3. Aggregate Model-Level Metrics & Compute Generic DUI Score
    model_groups = df_mlm.groupby("model_name")
    leaderboard_rows = []

    max_loss = df_mlm["mlm_loss"].max() if not df_mlm["mlm_loss"].empty else 1.0

    for model_name, grp in model_groups:
        t_info = tok_data.get(model_name, {})
        p_info = MODEL_PROFILES.get(model_name, {"params_m": 110, "size_mb": 440})

        avg_loss = grp["mlm_loss"].mean()
        avg_top1 = grp["top1_acc"].mean()
        avg_rare = grp["rare_top1_acc"].mean()
        avg_gap = grp["performance_gap"].mean()

        # Handle potential null domain_shift_gap if general baseline was unavailable
        shift_vals = grp["domain_shift_gap"].dropna()
        avg_shift = shift_vals.mean() if not shift_vals.empty else 0.0

        # Dynamic category balance calculation across configured categories
        if category_keys:
            cat_means = [grp[f"cat_{c}_acc"].mean() for c in category_keys if f"cat_{c}_acc" in grp.columns]
            cat_balance = max(0.0, 1.0 - float(np.std(cat_means))) if len(cat_means) > 1 else 1.0
        else:
            cat_balance = 1.0

        frag_rate = t_info.get("domain_fragmentation_rate", t_info.get("maritime_fragmentation_rate", 0.3))
        oov_rate = t_info.get("oov_rate", 0.0)
        tok_speed = t_info.get("tokenizer_speed_tokens_per_sec", 1000.0)

        # Mathematical Domain Understanding Index (DUI) Formula
        norm_loss = avg_loss / max_loss if max_loss > 0 else 0.5
        score = (
            0.35 * avg_top1 +
            0.20 * avg_rare +
            0.15 * (1.0 - norm_loss) +
            0.15 * (1.0 - frag_rate) +
            0.10 * (1.0 - min(1.0, oov_rate * 10)) +
            0.05 * cat_balance
        ) * 100.0

        # Methodological Hardening: Accurate evaluation timing terminology
        avg_eval_time = grp["eval_time_sec"].mean()
        eval_time_per_doc_ms = (avg_eval_time / 200.0) * 1000.0 if avg_eval_time > 0 else 5.0
        docs_per_sec = 1000.0 / eval_time_per_doc_ms if eval_time_per_doc_ms > 0 else 200.0

        # 95% Confidence Interval Error Bounds (Standard Error * 1.96)
        std_err_top1 = (grp["top1_acc"].std() / np.sqrt(len(grp))) * 1.96 if len(grp) > 1 else 0.01

        leaderboard_rows.append({
            "model_name": model_name,
            "dui_score": round(score, 2),
            "mui_score": round(score, 2),  # Compatibility alias
            "domain_top1_acc": round(avg_top1 * 100, 2),
            "maritime_top1_acc": round(avg_top1 * 100, 2),  # Compatibility alias
            "top1_ci_error": round(std_err_top1 * 100, 2),
            "rare_domain_acc": round(avg_rare * 100, 2),
            "rare_maritime_acc": round(avg_rare * 100, 2),  # Compatibility alias
            "mlm_loss": round(avg_loss, 4),
            "domain_shift_gap_pct": round(avg_shift * 100, 2) if not shift_vals.empty else None,
            "performance_gap_pct": round(avg_gap * 100, 2),
            "single_token_coverage_pct": round(t_info.get("single_token_vocabulary_coverage", 0) * 100, 2),
            "fragmentation_rate_pct": round(frag_rate * 100, 2),
            "oov_rate_pct": round(oov_rate * 100, 4),
            "params_millions": p_info["params_m"],
            "disk_size_mb": p_info["size_mb"],
            "evaluation_time_per_document_ms": round(eval_time_per_doc_ms, 2),
            "inference_latency_ms": round(eval_time_per_doc_ms, 2),  # Compatibility alias
            "throughput_docs_sec": round(docs_per_sec, 2),
            "tokenizer_speed_tok_sec": round(tok_speed, 2)
        })

    df_lb = pd.DataFrame(leaderboard_rows)
    df_lb.sort_values(by="dui_score", ascending=False, inplace=True)
    df_lb.to_csv(output_dir / "leaderboard.csv", index=False)

    # 4. Generate Visualizations with Error Bars & 95% CIs
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Visualization 1: MLM Loss Comparison
    plt.figure(figsize=(12, 6))
    top_df = df_lb.sort_values("mlm_loss")
    plt.barh(top_df["model_name"], top_df["mlm_loss"], color="#2b5c8f", edgecolor="black")
    plt.xlabel("MLM Loss (Lower is Better)")
    plt.title(f"Cross-Model Masked Language Model Loss ({domain_name.capitalize()} Benchmark)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(viz_dir / "mlm_loss_comparison.png", dpi=300)
    plt.close()

    # Visualization 2: Leaderboard Ranks & Scores with 95% CI Error Bars
    plt.figure(figsize=(12, 6))
    plt.bar(df_lb["model_name"], df_lb["domain_top1_acc"], yerr=df_lb["top1_ci_error"], capsize=5, color="#4c9be8", edgecolor="black", alpha=0.85)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"{domain_name.capitalize()} Top-1 Accuracy (%) ± 95% CI")
    plt.title(f"Model Performance Ranking ({metric_name}) with 95% Confidence Intervals")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(viz_dir / "model_leaderboard_ranks.png", dpi=300)
    plt.close()

    # Visualization 3: Category Recall Radar Chart
    categories = sorted(list(category_keys)) if category_keys else ["Subdomain 1", "Subdomain 2", "Subdomain 3"]
    top_3_models = df_lb.head(3)["model_name"].tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for m in top_3_models:
        m_grp = df_mlm[df_mlm["model_name"] == m]
        vals = [
            m_grp[f"cat_{cat}_acc"].mean() * 100 if f"cat_{cat}_acc" in m_grp.columns else 0.0
            for cat in categories
        ]
        vals += vals[:1]
        plt.plot(angles, vals, linewidth=2, label=m)
        plt.fill(angles, vals, alpha=0.15)

    clean_labels = [c.replace("_", " ").capitalize() for c in categories]
    plt.xticks(angles[:-1], clean_labels)
    plt.title(f"Subdomain Category Recall Radar Chart (Top 3 Models - {domain_name.capitalize()})")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(viz_dir / "domain_accuracy_radar.png", dpi=300)
    plt.savefig(viz_dir / "maritime_accuracy_radar.png", dpi=300)
    plt.close()

    # Visualization 4: Tokenizer Fragmentation Heatmap
    plt.figure(figsize=(10, 6))
    heat_df = df_lb[["model_name", "single_token_coverage_pct", "fragmentation_rate_pct", "oov_rate_pct"]].set_index("model_name")
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True)
    plt.title(f"Tokenizer Fragmentation & Vocabulary Coverage Heatmap ({domain_name.capitalize()})")
    plt.tight_layout()
    plt.savefig(viz_dir / "tokenizer_fragmentation_heatmap.png", dpi=300)
    plt.close()

    logger.info(f"Stage 15 completed. Exported comparison.csv, leaderboard.csv, and plots to {viz_dir}")

if __name__ == "__main__":
    main()
