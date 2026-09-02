#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the paper's eight figures (fig1-fig8) from the analysis outputs.

Inputs (in METRICS_DIR; default = data/main_study_807/metrics):
    metrics_full.csv                model x technique aggregates (rates, claim_density, n_triplets)
    ablation_results_full.json      detectability / feature ablation / training-size
    kappa_confidence_intervals.json inter-judge kappa with bootstrap CIs
    crime_type_analysis.csv         per-crime rates
Outputs: PDF + PNG for each figure, written to FIG_DIR (default = ./figures).

Run:  pip install pandas numpy scipy matplotlib
      python make_figures.py [METRICS_DIR] [FIG_DIR]
"""

# ============================================================
#  Hallucination study — figure generation (fig1-fig8)
#  Reads the real analysis files; no synthetic data.
# ============================================================

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

# ---- paths (override via env vars or CLI args) -------------
# METRICS_DIR must contain: metrics_full.csv, ablation_results_full.json,
# kappa_confidence_intervals.json, crime_type_analysis.csv
METRICS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("METRICS_DIR", "data/main_study_807/metrics")
FIG_DIR     = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FIG_DIR", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- style (IEEE-ish) --------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
})
COL_W, FULL_W = 3.5, 7.16            # IEEE single / double column (inches)
MODELS = ["Claude", "GPT", "Gemini"]
MCOL   = {"Claude": "#2c7fb8", "GPT": "#d95f02", "Gemini": "#1b9e77"}
HS     = ["H1", "H2", "H3", "H4", "H5", "H6"]
HNAME  = {"H1": "Scene\nFab.", "H2": "Crime\nMisclass.", "H3": "Crime\nMissed",
          "H4": "Severity\nMin.", "H5": "Entity\nFab.", "H6": "Phantom\nActors"}
AXIS   = {"H1": "Fabrication", "H5": "Fabrication", "H6": "Fabrication",
          "H3": "Omission", "H2": "Distortion", "H4": "Distortion"}
ACOL   = {"Fabrication": "#3182bd", "Omission": "#e6550d", "Distortion": "#756bb1"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG_DIR}/{name}.{ext}")
    print("saved", name)
    plt.close(fig)

# ---- load --------------------------------------------------
m = pd.read_csv(f"{METRICS_DIR}/metrics_full.csv", encoding="utf-8-sig")
abl = json.load(open(f"{METRICS_DIR}/ablation_results_full.json"))
kci = json.load(open(f"{METRICS_DIR}/kappa_confidence_intervals.json"))
crime = pd.read_csv(f"{METRICS_DIR}/crime_type_analysis.csv", encoding="utf-8-sig")

# model-level weighted (by n_triplets) per-H rates
def model_rates():
    rows = {}
    for mod in MODELS:
        d = m[m.model == mod]; w = d.n_triplets.values
        rows[mod] = {h: np.average(d[f"{h}_rate"].values, weights=w) * 100 for h in HS}
        rows[mod]["ANY"] = np.average(d.any_rate.values, weights=w) * 100
    return rows

# ============================================================
# FIG 1 — model signatures (radar)            [Table II, RQ1]
# ============================================================
def fig_radar():
    R = model_rates()
    ang = np.linspace(0, 2*np.pi, len(HS), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(COL_W, COL_W))
    ax = fig.add_subplot(111, polar=True)
    for mod in MODELS:
        v = [R[mod][h] for h in HS]; v += v[:1]
        ax.plot(ang, v, color=MCOL[mod], lw=1.8, label=mod)
        ax.fill(ang, v, color=MCOL[mod], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([HNAME[h].replace("\n", " ") for h in HS])
    ax.set_ylim(0, 100); ax.set_yticks([25, 50, 75, 100])
    ax.set_title("Hallucination signatures by model (% of reports)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    save(fig, "fig1_signatures_radar")

# ============================================================
# FIG 2 — 24-cell heatmap                     [Table III, RQ2]
# ============================================================
def fig_heatmap():
    order = (m.assign(o=m.model.map({k: i for i, k in enumerate(MODELS)}))
               .sort_values(["o", "technique"]))
    mat = order[[f"{h}_rate" for h in HS]].values * 100
    ylab = [f"{r.model[:3]}·{r.technique}" for _, r in order.iterrows()]
    fig, ax = plt.subplots(figsize=(FULL_W*0.6, FULL_W))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(HS))); ax.set_xticklabels([HNAME[h] for h in HS])
    ax.set_yticks(range(len(ylab))); ax.set_yticklabels(ylab, fontsize=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=5, color="black" if mat[i, j] < 60 else "white")
    for s in (8, 16): ax.axhline(s-0.5, color="k", lw=1)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="% flagged")
    ax.set_title("Per-cell hallucination rates")
    save(fig, "fig2_master_heatmap")

# ============================================================
# FIG 3 — training-size learning curves          [ablation3]
# ============================================================
def fig_trainsize():
    a3 = abl["ablation3_training_size"]
    fracs = sorted(a3, key=lambda k: a3[k]["fraction"])
    x = [a3[f]["fraction"]*100 for f in fracs]
    fig, ax = plt.subplots(figsize=(COL_W, COL_W*0.8))
    for h in HS + ["ANY"]:
        y = [a3[f][h]["auc"] for f in fracs]
        ax.plot(x, y, marker="o", ms=3, lw=1.3,
                label=h, ls="--" if h == "ANY" else "-")
    ax.set_xlabel("Training data (%)"); ax.set_ylabel("Held-out AUC")
    ax.set_title("Detector learning curves"); ax.grid(alpha=.3)
    ax.legend(ncol=2, fontsize=7)
    save(fig, "fig3_training_size")

# ============================================================
# FIG 4 — inter-judge kappa forest plot           [Table VIII]
# ============================================================
def fig_kappa_forest():
    pairs = ["GPT_vs_Gemini", "GPT_vs_Claude", "Gemini_vs_Claude"]
    pcol = {"GPT_vs_Gemini": "#1b9e77", "GPT_vs_Claude": "#d95f02",
            "Gemini_vs_Claude": "#7570b3"}
    offs = {pairs[0]: 0.22, pairs[1]: 0.0, pairs[2]: -0.22}
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    for p in pairs:
        for i, h in enumerate(HS):
            d = kci[p][h]; y = (len(HS)-i) + offs[p]
            ax.errorbar(d["kappa"], y,
                        xerr=[[d["kappa"]-d["ci_low_95"]], [d["ci_high_95"]-d["kappa"]]],
                        fmt="o", ms=4, color=pcol[p], capsize=2, lw=1,
                        label=p.replace("_vs_", "–") if i == 0 else None)
    ax.axvline(0.6, ls=":", color="grey", lw=.8)
    ax.set_yticks([len(HS)-i for i in range(len(HS))]); ax.set_yticklabels(HS)
    ax.set_xlabel("Cohen's $\\kappa$ (95% CI)"); ax.set_xlim(0, 1)
    ax.set_title("Inter-judge agreement by type"); ax.legend(fontsize=7)
    save(fig, "fig4_kappa_forest")

# ============================================================
# FIG 5 — thoroughness vs hallucination scatter   [Table IV, RQ3]
# ============================================================
def fig_scatter():
    fig, ax = plt.subplots(figsize=(COL_W, COL_W*0.85))
    for mod in MODELS:
        d = m[m.model == mod]
        ax.scatter(d.claim_density, d.any_rate*100, color=MCOL[mod],
                   s=28, alpha=.8, label=mod, edgecolor="k", lw=.3)
    z = np.polyfit(m.claim_density, m.any_rate*100, 1)
    xs = np.linspace(m.claim_density.min(), m.claim_density.max(), 50)
    ax.plot(xs, np.polyval(z, xs), "k--", lw=1)
    rho, p = spearmanr(m.claim_density, m.any_rate)
    ax.text(.05, .05, f"$\\rho={rho:.2f},\\ p={p:.3f}$", transform=ax.transAxes,
            fontsize=8, bbox=dict(fc="white", ec="grey", alpha=.8))
    ax.set_xlabel("Claim density"); ax.set_ylabel("Any-hallucination rate (%)")
    ax.set_title("Thoroughness vs. hallucination"); ax.legend(fontsize=7)
    save(fig, "fig5_thoroughness_scatter")

# ============================================================
# FIG 6 — per-type detectability bars              [Table V]
# ============================================================
def fig_detect():
    full = abl["ablation2_features"]["full"]
    order = sorted(HS, key=lambda h: full[h]["auc"], reverse=True)
    vals = [full[h]["auc"] for h in order]; cols = [ACOL[AXIS[h]] for h in order]
    fig, ax = plt.subplots(figsize=(COL_W, COL_W*0.7))
    ax.bar(range(len(order)), vals, color=cols)
    ax.axhline(0.5, ls=":", color="grey", lw=.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([HNAME[h] for h in order], fontsize=7)
    ax.set_ylim(0.5, 0.9); ax.set_ylabel("5-fold CV AUC")
    ax.set_title("Per-type detectability (surface features)")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in ACOL.items()], fontsize=7)
    save(fig, "fig6_detectability")

# ============================================================
# FIG 7 — feature ablation bars                    [Table VI]
# ============================================================
def fig_feature_ablation():
    feats = abl["ablation2_features"]
    labels = ["full", "minus_verbosity", "minus_hedging", "minus_multiturn",
              "minus_model_id", "only_model_id", "only_verbosity",
              "only_hedging", "only_multiturn"]
    vals = [feats[k]["ANY"]["auc"] for k in labels]
    fig, ax = plt.subplots(figsize=(COL_W, COL_W*0.9))
    ax.barh(range(len(labels)), vals, color="#4575b4")
    ax.axvline(feats["full"]["ANY"]["auc"], ls="--", color="k", lw=.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0.5, 0.8)
    ax.set_xlabel("AUC (ANY)"); ax.set_title("Feature ablation")
    save(fig, "fig7_feature_ablation")

# ============================================================
# FIG 8 — crime-type heatmap                       [Table IX]
# ============================================================
def fig_crime_heatmap():
    c = crime.sort_values("ANY_rate", ascending=False)
    mat = c[[f"{h}_rate" for h in HS]].values * 100
    fig, ax = plt.subplots(figsize=(COL_W*1.1, COL_W*1.1))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=80)
    ax.set_xticks(range(len(HS))); ax.set_xticklabels([HNAME[h] for h in HS], fontsize=7)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.crime_type, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=6, color="black" if mat[i, j] < 50 else "white")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="% flagged")
    ax.set_title("Hallucination by crime type")
    save(fig, "fig8_crime_heatmap")

# ---- run all -----------------------------------------------
for f in (fig_radar, fig_heatmap, fig_trainsize, fig_kappa_forest,
          fig_scatter, fig_detect, fig_feature_ablation, fig_crime_heatmap):
    f()
print("\nAll figures written to:", FIG_DIR)
