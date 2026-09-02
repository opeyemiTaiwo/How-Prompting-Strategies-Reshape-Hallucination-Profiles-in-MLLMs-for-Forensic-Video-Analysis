#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-generation persistence analysis (Table III + Spearman persistence).

Compares each model family's hallucination *profile shape* between the
previous-generation 19-video pilot and the current-generation 807-video study,
to test whether the architecture-level signature persists across a model
generation. Reproduces Table III (per-axis rates, previous vs. current) and the
rank-correlation persistence statistics.

Aggregation: paper rule -> per report, a type is positive iff all judges agree
(logical AND); ties -> no hallucination. Rates are pooled per model over all
techniques/videos (the axis rates are invariant to the pilot's finer technique
labels). Absolute rates are NOT comparable across generations (different
generator and judge models); only the profile *shape* is compared.

Inputs (panel raw judge labels; columns include model, technique, video, H1..H6):
    PREV : previous-generation pilot  (default: data/pilot_19video/panel_raw_judge_labels.csv)
    CURR : current-generation study   (default: data/main_study_807/panel_raw_judge_labels_full.csv)

Run:  pip install pandas numpy scipy
      python cross_generation_analysis.py [PREV_LABELS] [CURR_LABELS]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREV = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "PREV_LABELS", "data/pilot_19video/panel_raw_judge_labels.csv")
CURR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "CURR_LABELS", "data/main_study_807/panel_raw_judge_labels_full.csv")

H = ["H1", "H2", "H3", "H4", "H5", "H6"]
AXES = {"Fabrication": ["H1", "H5", "H6"], "Omission": ["H3"], "Distortion": ["H2", "H4"]}
MODELS = ["Claude", "GPT", "Gemini"]


def per_type_rates(path):
    """Per-model per-type prevalence (%) under the two-judge AND rule."""
    df = pd.read_csv(path)
    for h in H:
        df[h] = pd.to_numeric(df[h], errors="coerce")
    df = df.dropna(subset=H)
    rep = (df.groupby(["model", "technique", "video"])[H]
             .agg(lambda s: int(s.sum() == len(s))).reset_index())
    return {m: {h: g[h].mean() * 100 for h in H} for m, g in rep.groupby("model")}


prev = per_type_rates(PREV)
curr = per_type_rates(CURR)

# ---- Table III: per-axis rates, previous vs. current ----
print("=" * 58)
print("Table III  --  cross-generation persistence of signatures")
print("axis rate = mean per-type prevalence (%) within the axis")
print("=" * 58)
print(f'{"Model":8}{"Generation":12}{"Fabric.":>9}{"Omiss.":>9}{"Distort.":>10}')
for m in MODELS:
    for gen, R in [("previous", prev), ("current", curr)]:
        ax = {a: np.mean([R[m][h] for h in cols]) for a, cols in AXES.items()}
        dom = max(ax, key=ax.get)
        print(f'{m:8}{gen:12}{ax["Fabrication"]:9.1f}{ax["Omission"]:9.1f}'
              f'{ax["Distortion"]:10.1f}   (dominant: {dom})')

# ---- Spearman rho: profile persistence ----
print("\n" + "=" * 58)
print("Spearman rho  --  per-type profile (H1..H6), previous vs. current")
print("=" * 58)
print("within family:")
for m in MODELS:
    rho, p = spearmanr([prev[m][h] for h in H], [curr[m][h] for h in H])
    print(f"  {m:8} rho={rho:+.2f}  p={p:.2f}")
print("cross-model (previous GPT vs. current ...):")
for m in ["GPT", "Claude"]:
    rho, _ = spearmanr([prev["GPT"][h] for h in H], [curr[m][h] for h in H])
    print(f"  previous GPT vs current {m:8} rho={rho:+.2f}")
print("\n(The dominant axis is preserved per family across the generation gap. "
      "Within-family n=6 types, so the GPT/Gemini rho are positive but "
      "underpowered; the robust claim is the qualitative axis persistence.)")
