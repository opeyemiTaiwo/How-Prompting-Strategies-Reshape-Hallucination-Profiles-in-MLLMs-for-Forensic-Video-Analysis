#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_study_statistical_analyses.py -- inter-judge kappa CIs, per-crime
chi-square tests, and two-way ANOVA on the panel-labeled (main-study) dataset.

No API calls. Reads the panel labels produced by the panel-judging script and
writes the JSON/CSV/TXT summaries consumed by the kappa table, Fig. 4 (kappa
forest), Fig. 8 (crime heatmap), and the variance-partition discussion.

Aggregation: per report a type is positive iff both judges agree (the mode of a
2-judge split resolves to 0), i.e. the same unanimity/AND rule (ties -> 0) used
study-wide.

Inputs (in DATA_DIR; default data/main_study_807):
    panel_raw_judge_labels_full.csv   columns: row_idx, model, technique, video,
                                      crime_type, judge, H1..H6
Outputs (in METRICS_DIR; default data/main_study_807/metrics):
    kappa_confidence_intervals.json, crime_type_analysis.csv/json,
    anova_results.json, statistical_summary.txt

Run:  pip install numpy pandas scipy scikit-learn statsmodels
      python full_study_statistical_analyses.py [DATA_DIR] [METRICS_DIR]
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

DATA_DIR    = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATA_DIR", "data/main_study_807")
METRICS_DIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("METRICS_DIR", "data/main_study_807/metrics")
PANEL_RAW   = os.path.join(DATA_DIR, "panel_raw_judge_labels_full.csv")
os.makedirs(METRICS_DIR, exist_ok=True)

# sanity: confirm the labels file resolves before loading
print(('OK   ' if os.path.exists(PANEL_RAW) else 'MISSING  ') + PANEL_RAW)

KAPPA_CI_OUT  = os.path.join(METRICS_DIR, 'kappa_confidence_intervals.json')
CRIME_CSV     = os.path.join(METRICS_DIR, 'crime_type_analysis.csv')
CRIME_JSON    = os.path.join(METRICS_DIR, 'crime_type_analysis.json')
ANOVA_OUT     = os.path.join(METRICS_DIR, 'anova_results.json')
SUMMARY_OUT   = os.path.join(METRICS_DIR, 'statistical_summary.txt')

HCOLS = ['H1','H2','H3','H4','H5','H6']
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Load panel labels
print('Loading panel labels...')
labels = pd.read_csv(PANEL_RAW)
for h in HCOLS:
    labels[h] = pd.to_numeric(labels[h], errors='coerce')
clean = labels[(labels[HCOLS] >= 0).all(axis=1)].copy()
print(f'  Panel rows: {len(clean):,}')

# Per-triplet verdict (mode of the 2-judge panel; split -> 0, i.e. AND rule)
def majority(col):
    return col.mode().iloc[0] if not col.mode().empty else 0
mv = clean.groupby(['row_idx','model','technique','video','crime_type'])[HCOLS].agg(majority).reset_index()
mv['ANY'] = (mv[HCOLS].sum(axis=1) > 0).astype(int)
mv['hall_count'] = mv[HCOLS].sum(axis=1)
print(f'  Triplets (panel verdicts): {len(mv):,}')

# --- coverage check: confirm all 8 prompting techniques are present ---
EXPECTED_TECHNIQUES = ['Chain-of-Thought','Least-to-Most','Meta-Prompting','ReAct',
                       'Self-Consistency','Sequential','True-Iterative','Zero-Shot']
techs = sorted(mv['technique'].unique())
print(f"\nTechniques present ({len(techs)}): {techs}")
_missing = [t for t in EXPECTED_TECHNIQUES if t not in techs]
if _missing:
    print(f'  WARNING: missing techniques -> {_missing} (analyses will NOT cover the full study)')
else:
    print('  OK: all 8 techniques present -- analyses cover the full study.')
print('Rows per technique:')
print(mv.groupby('technique').size().to_string())


def kappa_bootstrap_ci(y1, y2, n_boot=1000, alpha=0.05, seed=42):
    """Compute bootstrap 95% CI for Cohen's kappa."""
    rng_local = np.random.default_rng(seed)
    n = len(y1)
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    boot_kappas = []
    for _ in range(n_boot):
        idx = rng_local.integers(0, n, size=n)
        try:
            k = cohen_kappa_score(y1[idx], y2[idx])
            if not np.isnan(k):
                boot_kappas.append(k)
        except Exception:
            pass
    boot_kappas = np.array(boot_kappas)
    if len(boot_kappas) < 100:
        return None, None, None
    lo = float(np.percentile(boot_kappas, 100 * alpha / 2))
    hi = float(np.percentile(boot_kappas, 100 * (1 - alpha / 2)))
    return lo, hi, float(boot_kappas.std())

# Build pivot: row_idx -> {judge: H1...H6}
pivot = clean.pivot_table(index='row_idx', columns='judge', values=HCOLS)

PAIRS = [('GPT', 'Gemini'), ('GPT', 'Claude'), ('Gemini', 'Claude')]

ci_results = {}
print('Computing bootstrap CIs (1000 iterations per pair x per H)...')
for j1, j2 in PAIRS:
    pair_key = f'{j1}_vs_{j2}'
    print(f'\n  {pair_key}:')
    per_dim = {}

    # Per-H
    for h in HCOLS:
        if (h, j1) not in pivot.columns or (h, j2) not in pivot.columns:
            continue
        sub = pivot[[(h, j1), (h, j2)]].dropna()
        if len(sub) < 50:
            per_dim[h] = {'note': 'too few rows'}
            continue
        y1 = sub[(h, j1)].astype(int).values
        y2 = sub[(h, j2)].astype(int).values
        try:
            point = float(cohen_kappa_score(y1, y2))
        except Exception:
            point = None
        if point is None:
            per_dim[h] = {'note': 'kappa undefined'}
            continue
        lo, hi, se = kappa_bootstrap_ci(y1, y2, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
        per_dim[h] = {
            'kappa': round(point, 4),
            'ci_low_95': round(lo, 4) if lo is not None else None,
            'ci_high_95': round(hi, 4) if hi is not None else None,
            'bootstrap_se': round(se, 4) if se is not None else None,
            'n': int(len(sub)),
        }
        print(f'    {h}: kappa = {point:.3f}  [{lo:.3f}, {hi:.3f}]  n={len(sub):,}')

    # Overall (concatenated across all 6 dimensions)
    all_y1, all_y2 = [], []
    for h in HCOLS:
        if (h, j1) not in pivot.columns or (h, j2) not in pivot.columns:
            continue
        sub = pivot[[(h, j1), (h, j2)]].dropna()
        all_y1.extend(sub[(h, j1)].astype(int).values)
        all_y2.extend(sub[(h, j2)].astype(int).values)
    if len(all_y1) > 100:
        try:
            point = float(cohen_kappa_score(all_y1, all_y2))
        except Exception:
            point = None
        if point is not None:
            lo, hi, se = kappa_bootstrap_ci(all_y1, all_y2, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
            per_dim['overall'] = {
                'kappa': round(point, 4),
                'ci_low_95': round(lo, 4) if lo is not None else None,
                'ci_high_95': round(hi, 4) if hi is not None else None,
                'bootstrap_se': round(se, 4) if se is not None else None,
                'n': int(len(all_y1)),
            }
            print(f'    overall: kappa = {point:.3f}  [{lo:.3f}, {hi:.3f}]  n={len(all_y1):,}')

    ci_results[pair_key] = per_dim

with open(KAPPA_CI_OUT, 'w') as f:
    json.dump(ci_results, f, indent=2)
print(f'\nSaved: {KAPPA_CI_OUT}')


# Per-crime per-H positive rates
crime_rows = []
for crime in mv['crime_type'].unique():
    sub = mv[mv['crime_type'] == crime]
    row = {'crime_type': crime, 'n_triplets': len(sub)}
    for h in HCOLS:
        row[f'{h}_rate'] = round(sub[h].mean(), 4)
        row[f'{h}_count'] = int(sub[h].sum())
    row['ANY_rate'] = round(sub['ANY'].mean(), 4)
    row['hall_per_triplet'] = round(sub['hall_count'].mean(), 4)
    crime_rows.append(row)

crime_df = pd.DataFrame(crime_rows).sort_values('hall_per_triplet', ascending=False)
crime_df.to_csv(CRIME_CSV, index=False)
print(f'Saved: {CRIME_CSV}\n')

print('=== HALLUCINATION RATE BY CRIME TYPE ===')
print(f'{"Crime":<18}{"n":>6}{"H1":>7}{"H2":>7}{"H3":>7}{"H4":>7}{"H5":>7}{"H6":>7}'
      f'{"ANY":>7}{"halls/T":>10}')
print('-' * 80)
for _, r in crime_df.iterrows():
    print(f'{r["crime_type"]:<18}{r["n_triplets"]:>6,}', end='')
    for h in HCOLS:
        print(f'{r[f"{h}_rate"]*100:>6.1f}%', end='')
    print(f'{r["ANY_rate"]*100:>6.1f}%{r["hall_per_triplet"]:>10.3f}')

# Chi-square: is ANY hallucination independent of crime_type?
print('\n=== CHI-SQUARE: ANY-hallucination by crime_type ===')
ct = pd.crosstab(mv['crime_type'], mv['ANY'])
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f'chi2 = {chi2:.3f}  p = {p:.4g}  dof = {dof}')
if p < 0.001:
    print('  -> Highly significant: hallucination rate VARIES by crime type.')
elif p < 0.05:
    print('  -> Significant: crime type and hallucination rate are not independent.')
else:
    print('  -> Not significant: hallucination rate is independent of crime type.')

# Also: per-H chi-square
print('\nPer-H chi-square (crime_type independence):')
crime_chi = {}
for h in HCOLS:
    ct_h = pd.crosstab(mv['crime_type'], mv[h])
    if ct_h.shape[1] < 2:
        crime_chi[h] = {'note': 'no variance'}
        continue
    chi2_h, p_h, dof_h, _ = stats.chi2_contingency(ct_h)
    crime_chi[h] = {'chi2': float(chi2_h), 'p': float(p_h), 'dof': int(dof_h)}
    sig = '***' if p_h < 0.001 else ('**' if p_h < 0.01 else ('*' if p_h < 0.05 else ''))
    print(f'  {h}: chi2={chi2_h:.2f}  p={p_h:.4g}  dof={dof_h}  {sig}')

crime_results = {
    'per_crime_rates': {row['crime_type']: row for row in crime_rows},
    'overall_chi2': {'chi2': float(chi2), 'p': float(p), 'dof': int(dof)},
    'per_H_chi2': crime_chi,
}
with open(CRIME_JSON, 'w') as f:
    json.dump(crime_results, f, indent=2, default=str)
print(f'\nSaved: {CRIME_JSON}')


try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAVE_STATSMODELS = True
except ImportError:
    print('statsmodels not installed. Falling back to scipy alternatives.')
    print('  For full ANOVA, run: pip install statsmodels')
    HAVE_STATSMODELS = False

# Use hall_count as the dependent variable
df_anova = mv[['model','technique','hall_count','ANY']].copy()

print(f'ANOVA inputs: {len(df_anova):,} rows')
print('Model x technique cell counts:')
print(df_anova.groupby(['model','technique']).size().unstack(fill_value=0))

anova_results = {}

if HAVE_STATSMODELS:
    # Two-way ANOVA on hall_count with interaction term
    print('\n=== Two-way ANOVA: hall_count ~ model * technique ===')
    model_lm = ols('hall_count ~ C(model) + C(technique) + C(model):C(technique)',
                    data=df_anova).fit()
    table = sm.stats.anova_lm(model_lm, typ=2)
    print(table.to_string())

    # Convert to JSON-serializable
    anova_count = {}
    for idx in table.index:
        row = table.loc[idx]
        anova_count[idx] = {
            'sum_sq': float(row['sum_sq']),
            'df': float(row['df']),
            'F': float(row['F']) if not pd.isna(row['F']) else None,
            'p': float(row['PR(>F)']) if not pd.isna(row['PR(>F)']) else None,
        }

    # Compute eta-squared (effect size = SS_factor / SS_total)
    ss_total = table['sum_sq'].sum()
    eta_squared = {}
    for idx in table.index:
        if idx == 'Residual':
            continue
        eta_squared[idx] = float(table.loc[idx, 'sum_sq'] / ss_total)

    print('\nEffect sizes (eta-squared = proportion of total variance explained):')
    for k, v in eta_squared.items():
        print(f'  {k}: eta^2 = {v:.4f}  ({v*100:.2f}% of total variance)')

    anova_results['hall_count_two_way'] = {
        'anova_table': anova_count,
        'eta_squared': eta_squared,
        'interpretation': 'Larger eta-squared indicates the factor explains more variance.',
    }

    # Same for ANY (binary outcome -- OLS used as approximation here)
    print('\n=== Two-way ANOVA: ANY ~ model * technique ===')
    model_any = ols('ANY ~ C(model) + C(technique) + C(model):C(technique)',
                     data=df_anova).fit()
    table_any = sm.stats.anova_lm(model_any, typ=2)
    print(table_any.to_string())

    anova_any = {}
    for idx in table_any.index:
        row = table_any.loc[idx]
        anova_any[idx] = {
            'sum_sq': float(row['sum_sq']),
            'df': float(row['df']),
            'F': float(row['F']) if not pd.isna(row['F']) else None,
            'p': float(row['PR(>F)']) if not pd.isna(row['PR(>F)']) else None,
        }
    ss_total_any = table_any['sum_sq'].sum()
    eta_any = {idx: float(table_any.loc[idx, 'sum_sq'] / ss_total_any)
               for idx in table_any.index if idx != 'Residual'}
    print('\nEffect sizes:')
    for k, v in eta_any.items():
        print(f'  {k}: eta^2 = {v:.4f}  ({v*100:.2f}%)')

    anova_results['ANY_two_way'] = {
        'anova_table': anova_any,
        'eta_squared': eta_any,
    }

else:
    # Fallback: scipy one-way ANOVAs
    print('\n=== One-way ANOVA: hall_count by model ===')
    groups = [df_anova[df_anova['model']==m]['hall_count'].values
              for m in df_anova['model'].unique()]
    f_m, p_m = stats.f_oneway(*groups)
    print(f'F = {f_m:.3f}  p = {p_m:.4g}')

    print('\n=== One-way ANOVA: hall_count by technique ===')
    groups = [df_anova[df_anova['technique']==t]['hall_count'].values
              for t in df_anova['technique'].unique()]
    f_t, p_t = stats.f_oneway(*groups)
    print(f'F = {f_t:.3f}  p = {p_t:.4g}')

    anova_results['fallback'] = {
        'one_way_model': {'F': float(f_m), 'p': float(p_m)},
        'one_way_technique': {'F': float(f_t), 'p': float(p_t)},
        'note': 'statsmodels not installed; full two-way ANOVA skipped',
    }

with open(ANOVA_OUT, 'w') as f:
    json.dump(anova_results, f, indent=2)
print(f'\nSaved: {ANOVA_OUT}')


lines = []
lines.append('=' * 76)
lines.append('FULL-STUDY STATISTICAL ANALYSES -- SUMMARY')
lines.append('=' * 76)
lines.append('')

# Kappa CIs
lines.append('-- INTER-JUDGE KAPPA WITH 95% BOOTSTRAP CIs (n=1,000 iterations) --')
lines.append(f'{"Pair":<22}{"H":<6}{"kappa":>8}{"95% CI":>20}{"n":>10}')
lines.append('-' * 66)
for pair, dims in ci_results.items():
    for h, v in dims.items():
        if 'kappa' in v and v.get('ci_low_95') is not None:
            lines.append(f'  {pair:<20}{h:<6}{v["kappa"]:>8.3f}'
                         f'  [{v["ci_low_95"]:.3f}, {v["ci_high_95"]:.3f}]'
                         f'{v["n"]:>10,}')
lines.append('')

# Crime
lines.append('-- HALLUCINATION RATE BY CRIME TYPE --')
lines.append(f'  Overall chi-square: chi2={chi2:.2f}, p={p:.4g}, dof={dof}')
lines.append('  -> ' + ('crime type AFFECTS hallucination rate' if p < 0.05
             else 'crime type does NOT significantly affect hallucination rate'))
lines.append('')
lines.append('  Top 5 highest-hallucination crime types (per triplet):')
top5 = crime_df.head(5)
for _, r in top5.iterrows():
    lines.append(f'    {r["crime_type"]:<18}  ANY={r["ANY_rate"]*100:>5.1f}%  '
                 f'halls/triplet={r["hall_per_triplet"]:.2f}  n={r["n_triplets"]:,}')
lines.append('')
lines.append('  Bottom 3 lowest-hallucination crime types:')
bot3 = crime_df.tail(3)
for _, r in bot3.iterrows():
    lines.append(f'    {r["crime_type"]:<18}  ANY={r["ANY_rate"]*100:>5.1f}%  '
                 f'halls/triplet={r["hall_per_triplet"]:.2f}  n={r["n_triplets"]:,}')

# ANOVA
lines.append('')
lines.append('-- VARIANCE PARTITION (two-way ANOVA on hall_count) --')
if 'hall_count_two_way' in anova_results:
    eta = anova_results['hall_count_two_way']['eta_squared']
    for factor, e in eta.items():
        lines.append(f'  {factor:<32} eta^2 = {e:.4f}  ({e*100:.2f}% of variance)')
    # Identify dominant factor
    dom = max(eta.items(), key=lambda kv: kv[1])
    lines.append(f'  -> Dominant factor: {dom[0]} ({dom[1]*100:.2f}% of variance)')

summary = '\n'.join(lines)
with open(SUMMARY_OUT, 'w', encoding='utf-8') as f:
    f.write(summary)
print(summary)
print(f'\nSaved: {SUMMARY_OUT}')
