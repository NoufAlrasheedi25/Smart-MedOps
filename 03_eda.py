# -*- coding: utf-8 -*-
"""
03 — EDA: 10 figures (English labels only), light styling, high DPI.
Captions for documentation: figure_captions_en.json (English).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _paths import CLEAN_CSV, FIG_DIR


def apply_plot_style():
    """Clean, light figures — no Arabic in plot text."""
    sns.set_theme(style="ticks", context="notebook", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#c8c8c8",
            "axes.linewidth": 0.85,
            "axes.labelcolor": "#333333",
            "axes.titlecolor": "#1a1a1a",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.85,
            "lines.linewidth": 1.35,
            "lines.antialiased": True,
            "patch.linewidth": 0.6,
        }
    )


def savefig(name: str):
    p = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Figure:", p)


def main():
    apply_plot_style()
    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1
    plt.figure(figsize=(6.2, 4.0))
    df["readmitted"].value_counts().reindex(["NO", ">30", "<30"]).plot(
        kind="bar", color=["#66bb6a", "#ffb74d", "#e57373"], edgecolor="white", linewidth=0.5
    )
    plt.title("Raw readmitted label distribution")
    plt.ylabel("Count")
    sns.despine()
    savefig("01_readmitted_raw_distribution.png")

    # 2
    plt.figure(figsize=(5.2, 4.0))
    df["readmit_lt30"].value_counts().sort_index().plot(
        kind="bar", color=["#42a5f5", "#ef5350"], edgecolor="white", linewidth=0.5
    )
    plt.title("Binary target readmit_lt30 (1 = readmit within 30 days)")
    plt.xticks(rotation=0)
    sns.despine()
    savefig("02_binary_target_distribution.png")

    # 3
    plt.figure(figsize=(6.2, 4.0))
    df["time_in_hospital"].hist(bins=14, color="#7986cb", edgecolor="white", linewidth=0.6, alpha=0.92)
    plt.title("Length of stay (days)")
    plt.xlabel("Days")
    plt.ylabel("Count")
    sns.despine()
    savefig("03_time_in_hospital_distribution.png")

    # 4
    plt.figure(figsize=(8.0, 4.2))
    df["age"].value_counts().sort_index().plot(kind="bar", color="#26a69a", edgecolor="white", linewidth=0.4)
    plt.title("Encounters by age bracket")
    plt.xticks(rotation=45, ha="right")
    sns.despine()
    savefig("04_age_bracket_counts.png")

    # 5
    plt.figure(figsize=(6.2, 4.0))
    df["race"].fillna("Missing").value_counts().head(8).plot(
        kind="barh", color="#ab47bc", edgecolor="white", linewidth=0.4
    )
    plt.title("Top race categories (Missing shown)")
    sns.despine()
    savefig("05_race_distribution.png")

    # 6
    num_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    plt.figure(figsize=(8.5, 6.8))
    corr = df[num_cols + ["readmit_lt30"]].corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"shrink": 0.72, "label": "r"},
        vmin=-1,
        vmax=1,
    )
    plt.title("Numeric feature correlation (incl. target)")
    savefig("06_numeric_correlation_heatmap.png")

    # 7
    plt.figure(figsize=(6.0, 4.0))
    g = df.groupby("diabetesMed", dropna=False)["readmit_lt30"].mean()
    g.plot(kind="bar", color=["#90a4ae", "#66bb6a"], edgecolor="white", linewidth=0.5)
    plt.ylabel("P(readmit < 30d)")
    plt.title("Readmit rate by diabetesMed")
    plt.xticks(rotation=0)
    sns.despine()
    savefig("07_readmit_rate_by_diabetesMed.png")

    # 8
    plt.figure(figsize=(7.2, 4.2))
    top_spec = df["medical_specialty"].fillna("Missing").value_counts().head(10)
    top_spec.plot(kind="barh", color="#ffa726", edgecolor="white", linewidth=0.4)
    plt.title("Top 10 medical_specialty (Missing = NA)")
    sns.despine()
    savefig("08_top_medical_specialty.png")

    # 9
    plt.figure(figsize=(6.0, 4.0))
    df.groupby("gender", dropna=False)["readmit_lt30"].mean().plot(
        kind="bar", color=["#42a5f5", "#ec407a"], edgecolor="white", linewidth=0.5
    )
    plt.ylabel("P(readmit < 30d)")
    plt.title("Readmit_lt30 rate by gender")
    plt.xticks(rotation=0)
    sns.despine()
    savefig("09_readmit_by_gender.png")

    # 10
    nb = 20
    plt.figure(figsize=(7.2, 4.0))
    dtmp = df.copy()
    dtmp["nmed_bin"] = pd.cut(dtmp["num_medications"], bins=nb, duplicates="drop")
    dtmp.groupby("nmed_bin", observed=True)["readmit_lt30"].mean().plot(color="#7e57c2", linewidth=1.35)
    plt.xlabel("num_medications (binned)")
    plt.ylabel("P(readmit < 30d)")
    plt.title("Readmit rate vs number of medications")
    sns.despine()
    savefig("10_medications_vs_readmit_rate.png")

    captions_en = {
        "01_readmitted_raw_distribution.png": "Class imbalance across the three original readmitted labels.",
        "02_binary_target_distribution.png": "Binary training target; positive class is relatively rare.",
        "03_time_in_hospital_distribution.png": "Most stays are short; a right tail may indicate higher risk.",
        "04_age_bracket_counts.png": "Older age brackets contribute more encounters in this extract.",
        "05_race_distribution.png": "Race mix; missing values are imputed later in the ML pipeline.",
        "06_numeric_correlation_heatmap.png": "Guides numeric feature selection and multicollinearity checks.",
        "07_readmit_rate_by_diabetesMed.png": "Small rate differences may still help non-linear models.",
        "08_top_medical_specialty.png": "Specialty reflects care pathway and may associate with readmission.",
        "09_readmit_by_gender.png": "Descriptive readmit rate comparison by gender at encounter level.",
        "10_medications_vs_readmit_rate.png": "Non-linear pattern motivates tree/boosting models.",
    }
    cap_path = FIG_DIR / "figure_captions_en.json"
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(captions_en, f, ensure_ascii=False, indent=2)
    # Legacy path: same English text (no Arabic in figure-related assets)
    (FIG_DIR / "figure_captions_ar.json").write_text(
        json.dumps(captions_en, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote:", cap_path)
    print("Wrote:", FIG_DIR / "figure_captions_ar.json")


if __name__ == "__main__":
    main()
