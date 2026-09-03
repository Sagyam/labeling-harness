"""Run local CPU-only pilot benchmark and analysis on labeling harness exports.

This script executes the research analysis pipeline for Nepali-English code-switching ASR:
1. Audits exports and manifests.
2. Computes linguistic and code-mixing demographics.
3. Benchmarks ElevenLabs Scribe vs Gemini 3.8 Flash against human Gold labels.
4. Evaluates annotator anchoring across seed systems.
5. Simulates multi-ASR disagreement gating for pseudo-label selection.
6. Exports publication-ready matplotlib figures and LaTeX/Markdown tables.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Metric & Normalization Functions (CPU Levenshtein Distance)
# -----------------------------------------------------------------------------

DEV_RE = re.compile(r"[\u0900-\u097F]")
TOK_RE = re.compile(r"[A-Za-z']+|[\u0900-\u097F]+|\d+")
PUNCT_CHARS = r""".,!?;:'"|।॥()[]{}-—"""
PUNCT_RE = re.compile(f"[{re.escape(PUNCT_CHARS)}]")


def normalize_text(text: str | None) -> str:
    """Normalize code-switched text for evaluation (NFKC, strip punct, lowercase Latin)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = PUNCT_RE.sub(" ", text)
    tokens = text.split()
    normalized_tokens = []
    for tok in tokens:
        if not DEV_RE.search(tok):
            normalized_tokens.append(tok.lower())
        else:
            normalized_tokens.append(tok)
    return " ".join(normalized_tokens)


def levenshtein_distance(seq1: list[str] | str, seq2: list[str] | str) -> int:
    """Compute standard Levenshtein edit distance using dynamic programming on CPU."""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate: Levenshtein distance on words / reference word count."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate: Levenshtein distance on characters / reference char count."""
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------


def load_exports(export_root: Path = Path("exports")) -> dict[str, Any]:
    """Load all 4 export profiles and their manifest files."""
    datasets = {}
    manifests = {}
    for kind in ["training", "gold", "analytics", "error_mining"]:
        data_file = export_root / kind / f"{kind}.jsonl"
        manifest_file = export_root / kind / "manifest.json"

        records = []
        if data_file.is_file():
            with data_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        datasets[kind] = records

        manifest = {}
        if manifest_file.is_file():
            with manifest_file.open(encoding="utf-8") as f:
                manifest = json.load(f)
        manifests[kind] = manifest

    return {"datasets": datasets, "manifests": manifests}


# -----------------------------------------------------------------------------
# Analysis Pipeline
# -----------------------------------------------------------------------------


def run_pilot():
    print("=" * 70)
    print("NEPANGLISH CODE-SWITCHING ASR: PILOT BENCHMARK & DATASET STUDY")
    print("=" * 70)

    data = load_exports()
    datasets = data["datasets"]
    manifests = data["manifests"]

    # 1. Dataset Auditing
    print("\n[1] Export Profiles Audit:")
    for kind in ["training", "gold", "analytics", "error_mining"]:
        recs = datasets[kind]
        m = manifests.get(kind, {})
        splits_str = str(m.get("filters", {}).get("splits", []))
        disp_str = str(m.get("filters", {}).get("dispositions", []))
        print(
            f"  * {kind:<13}: {len(recs):>3} rows | splits={splits_str} | dispositions={disp_str}"
        )

    analytics = datasets["analytics"]
    df_analytics = pd.DataFrame(analytics)
    print(f"\nTotal segments in analytics: {len(df_analytics)}")
    dispositions = df_analytics["disposition"].value_counts().to_dict()
    splits = df_analytics["split"].value_counts().to_dict()
    print(f"Dispositions: {dispositions}")
    print(f"Splits:       {splits}")

    durations = df_analytics["duration_seconds"].dropna()
    cmi_values = df_analytics["code_switch_density"].dropna() * 100.0

    print(
        f"Audio Duration: total={durations.sum() / 3600:.3f} h, "
        f"mean={durations.mean():.2f} s (std={durations.std():.2f} s)"
    )
    print(
        f"Code-Switching Index (CMI %): mean={cmi_values.mean():.2f}%, "
        f"median={cmi_values.median():.2f}%, max={cmi_values.max():.2f}%"
    )

    # 2. Token-level language demographics
    print("\n[2] Script & Language Token Demographics:")
    total_dev = 0
    total_lat = 0
    for text in df_analytics["text"]:
        tokens = TOK_RE.findall(text or "")
        for t in tokens:
            if not t.isdigit():
                if DEV_RE.search(unicodedata.normalize("NFKC", t)):
                    total_dev += 1
                else:
                    total_lat += 1
    total_tok = total_dev + total_lat
    print(f"  Devanagari tokens: {total_dev} ({total_dev / total_tok * 100:.1f}%)")
    print(f"  Latin/English tokens: {total_lat} ({total_lat / total_tok * 100:.1f}%)")
    print(f"  Total non-numeric tokens: {total_tok}")

    # 3. Gold Split Multi-ASR Benchmark
    print("\n[3] Gold Benchmark Evaluation (Test Split):")
    gold_records = [r for r in analytics if r.get("split") == "test"]
    print(f"  Evaluating {len(gold_records)} gold test segments against upstream models...")

    rows_eval = []
    for rec in gold_records:
        ref_raw = rec["text"]
        ref_norm = normalize_text(ref_raw)
        cmi = rec.get("code_switch_density", 0.0) or 0.0
        seed_sys = rec.get("seed_system_id")
        disposition = rec.get("disposition")

        hyps = {h["system_id"]: h["text"] for h in rec.get("hypotheses", [])}

        row = {
            "segment_id": rec["segment_id"],
            "cmi": cmi,
            "seed_system_id": seed_sys,
            "disposition": disposition,
            "ref_raw": ref_raw,
            "ref_norm": ref_norm,
        }

        for sys_name in ["elevenlabs-scribe-v2", "gemini-3.8-flash"]:
            hyp_raw = hyps.get(sys_name, "")
            hyp_norm = normalize_text(hyp_raw)
            row[f"{sys_name}_wer_raw"] = compute_wer(ref_raw, hyp_raw)
            row[f"{sys_name}_wer_norm"] = compute_wer(ref_norm, hyp_norm)
            row[f"{sys_name}_cer_raw"] = compute_cer(ref_raw, hyp_raw)
            row[f"{sys_name}_cer_norm"] = compute_cer(ref_norm, hyp_norm)

        rows_eval.append(row)

    df_eval = pd.DataFrame(rows_eval)

    # Benchmark summary table
    bench_results = []
    for sys_name in ["elevenlabs-scribe-v2", "gemini-3.8-flash"]:
        bench_results.append(
            {
                "System": sys_name,
                "Raw WER (%)": df_eval[f"{sys_name}_wer_raw"].mean() * 100,
                "Norm WER (%)": df_eval[f"{sys_name}_wer_norm"].mean() * 100,
                "Raw CER (%)": df_eval[f"{sys_name}_cer_raw"].mean() * 100,
                "Norm CER (%)": df_eval[f"{sys_name}_cer_norm"].mean() * 100,
            }
        )
    df_bench = pd.DataFrame(bench_results)
    print("\n--- Model Benchmark on Gold Test Split ---")
    print(df_bench.to_string(index=False))

    # Error breakdown by CMI tier
    print("\n--- Model Performance by Code-Switching Density Tier ---")
    df_eval["cmi_tier"] = pd.cut(
        df_eval["cmi"],
        bins=[-0.01, 0.15, 0.35, 1.0],
        labels=["Low (<15%)", "Moderate (15-35%)", "High (>35%)"],
    )
    tier_summary = (
        df_eval.groupby("cmi_tier", observed=False)
        .agg(
            segments=("segment_id", "count"),
            scribe_norm_wer=("elevenlabs-scribe-v2_wer_norm", lambda s: s.mean() * 100),
            gemini_norm_wer=("gemini-3.8-flash_wer_norm", lambda s: s.mean() * 100),
            scribe_norm_cer=("elevenlabs-scribe-v2_cer_norm", lambda s: s.mean() * 100),
            gemini_norm_cer=("gemini-3.8-flash_cer_norm", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    print(tier_summary.to_string(index=False))

    # 4. Annotator Anchoring Analysis (Paper Idea 2)
    print("\n[4] Annotator Anchoring & Seed System Bias Analysis:")
    seed_analysis = (
        df_eval.groupby("seed_system_id")
        .agg(
            segments=("segment_id", "count"),
            edit_rate=("disposition", lambda d: (d == "edited").mean() * 100),
            mean_wer_scribe=("elevenlabs-scribe-v2_wer_norm", lambda s: s.mean() * 100),
            mean_wer_gemini=("gemini-3.8-flash_wer_norm", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    print(seed_analysis.to_string(index=False))

    # 5. Multi-ASR Disagreement Gating Simulation (Paper Idea 1)
    print("\n[5] Pseudo-Label Disagreement Gating Simulation:")
    scores_list = []
    for r in analytics:
        s = r.get("scores") or {}
        scores_list.append(
            {
                "segment_id": r["segment_id"],
                "split": r["split"],
                "disposition": r["disposition"],
                "is_edited": 1 if r["disposition"] == "edited" else 0,
                "word_disagreement": s.get("word_disagreement_rate", 0.0) or 0.0,
                "cer_disagreement": s.get("cer_between_hypotheses", 0.0) or 0.0,
                "cmi": r.get("code_switch_density", 0.0) or 0.0,
            }
        )
    df_scores = pd.DataFrame(scores_list)

    # Disagreement vs Edit correlation
    corr_word = df_scores["word_disagreement"].corr(df_scores["is_edited"])
    corr_cer = df_scores["cer_disagreement"].corr(df_scores["is_edited"])
    corr_cmi = df_scores["cmi"].corr(df_scores["is_edited"])
    print("  Correlation with Human Edits:")
    print(f"    * Word Disagreement Rate: r = {corr_word:.3f}")
    print(f"    * Hypothesis CER:         r = {corr_cer:.3f}")
    print(f"    * Code-Switch Density:    r = {corr_cmi:.3f}")

    # Simulated Gating: Thresholding pseudo-labels by disagreement
    thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]
    gating_results = []
    for th in thresholds:
        retained = df_scores[df_scores["word_disagreement"] <= th]
        n_ret = len(retained)
        edit_pct = retained["is_edited"].mean() * 100 if n_ret > 0 else 0
        gating_results.append(
            {
                "Max Disagreement": th,
                "Retained Segments": n_ret,
                "Retention Rate (%)": n_ret / len(df_scores) * 100,
                "Human Edit / Error Rate in Pool (%)": edit_pct,
            }
        )
    df_gating = pd.DataFrame(gating_results)
    print("\n--- Pseudo-Label Quality at Various Disagreement Gating Thresholds ---")
    print(df_gating.to_string(index=False))

    # 6. Generate Publication Figures
    fig_dir = Path("notebooks/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Figure 1: CMI Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cmi_values, bins=15, color="#2563eb", edgecolor="black", alpha=0.75)
    ax.set_title("Distribution of Code-Mixing Index (CMI)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Code-Mixing Index (%)")
    ax.set_ylabel("Segment Count")
    ax.axvline(
        cmi_values.mean(), color="#dc2626", linestyle="--", label=f"Mean ({cmi_values.mean():.1f}%)"
    )
    ax.axvline(
        cmi_values.median(),
        color="#16a34a",
        linestyle=":",
        label=f"Median ({cmi_values.median():.1f}%)",
    )
    ax.legend()
    fig1_path = fig_dir / "fig1_cmi_distribution.png"
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)

    # Figure 2: WER across CMI Tiers
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(tier_summary))
    width = 0.35
    ax.bar(
        x - width / 2,
        tier_summary["scribe_norm_wer"],
        width,
        label="ElevenLabs Scribe v2",
        color="#3b82f6",
    )
    ax.bar(
        x + width / 2,
        tier_summary["gemini_norm_wer"],
        width,
        label="Gemini 3.8 Flash",
        color="#10b981",
    )
    ax.set_title("ASR Word Error Rate (WER) by Code-Switching Tier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Code-Switching Tier (CMI)")
    ax.set_ylabel("Normalized WER (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_summary["cmi_tier"])
    ax.legend()
    fig2_path = fig_dir / "fig2_wer_by_cmi_tier.png"
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)

    # Figure 3: Disagreement Gating Tradeoff
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    color_ret = "#2563eb"
    color_err = "#dc2626"

    ax1.set_xlabel("Disagreement Threshold (Filter)")
    ax1.set_ylabel("Retained Segments (%)", color=color_ret)
    ax1.plot(
        df_gating["Max Disagreement"],
        df_gating["Retention Rate (%)"],
        marker="o",
        color=color_ret,
        label="Retention %",
    )
    ax1.tick_params(axis="y", labelcolor=color_ret)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Human Edit Rate / Noise in Pool (%)", color=color_err)
    ax2.plot(
        df_gating["Max Disagreement"],
        df_gating["Human Edit / Error Rate in Pool (%)"],
        marker="s",
        color=color_err,
        linestyle="--",
        label="Noise %",
    )
    ax2.tick_params(axis="y", labelcolor=color_err)

    ax1.set_title("Pseudo-Label Disagreement Gating Trade-off", fontsize=13, fontweight="bold")
    fig3_path = fig_dir / "fig3_disagreement_gating.png"
    fig.savefig(fig3_path, dpi=200)
    plt.close(fig)

    print(f"\n[6] Figures saved to: {fig_dir}/")
    print(f"  * {fig1_path.name}")
    print(f"  * {fig2_path.name}")
    print(f"  * {fig3_path.name}")

    # 7. Print Publication LaTeX Table
    print("\n[7] Publication-Ready LaTeX Benchmark Table:")
    latex_table = df_bench.to_latex(index=False, float_format="%.2f")
    print(latex_table)

    return {
        "df_analytics": df_analytics,
        "df_eval": df_eval,
        "df_bench": df_bench,
        "df_gating": df_gating,
        "tier_summary": tier_summary,
    }


if __name__ == "__main__":
    run_pilot()
