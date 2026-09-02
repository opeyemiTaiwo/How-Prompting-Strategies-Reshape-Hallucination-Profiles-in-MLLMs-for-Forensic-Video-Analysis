#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human validation spot-check: builds the stratified rater package (Word docs +
blank scoring sheets) and computes inter-rater agreement, each rater vs. the
LLM panel, and the majority-human-vote vs. panel, as Cohen's kappa.

Raters are anonymized as Rater A / Rater B.

Inputs (place in EVAL_DIR, default = current directory):
    full_labeled_dataset_subset.csv      panel-labeled current-gen subset
    RaterA-scores.xlsx, RaterB-scores.xlsx   completed rater sheets (in RATERS_DIR)
Outputs: human_panel_agreement.json (in EVAL_DIR).

Run:  pip install pandas numpy scikit-learn openpyxl    # python-docx, tqdm optional (sheet generation)
      python human_spotcheck_analysis.py [EVAL_DIR] [RATERS_DIR]
"""

# A1: imports + paths
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Word doc generation (only needed if regenerating sheets)
try:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    print('Note: python-docx not installed. Cells A3 / A4 will not run, but analysis cells (B1-B5) will.')

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EVAL_DIR", ".")
RATERS_DIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("RATERS_DIR", "human-raters")
os.makedirs(RATERS_DIR, exist_ok=True)

# Per-rater folders (only used if regenerating Word docs / blank sheets)
RATER_A_DIR = os.path.join(RATERS_DIR, 'Rater A')
RATER_B_DIR = os.path.join(RATERS_DIR, 'Rater B')
os.makedirs(RATER_A_DIR, exist_ok=True)
os.makedirs(RATER_B_DIR, exist_ok=True)

FULL_LABELED_PATH  = os.path.join(OUTPUT_DIR, 'full_labeled_dataset_subset.csv')
AGREEMENT_PATH     = os.path.join(OUTPUT_DIR, 'human_panel_agreement.json')

# Blank scoring sheet output paths (only written if you regenerate them)
RATER_A_SCORES     = os.path.join(RATER_A_DIR, 'Rater A-scores.xlsx')
RATER_B_SCORES     = os.path.join(RATER_B_DIR, 'Rater B-scores.xlsx')

# === FILLED rater files — these are the files you uploaded ===
# Both raters' completed score sheets live directly in HUMAN-RATERS/.
RATER_A_FILLED     = os.path.join(RATERS_DIR, 'Rater A-scores.xlsx')
RATER_B_FILLED     = os.path.join(RATERS_DIR, 'Rater B-scores.xlsx')

HCOLS       = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
N_SAMPLE    = 100
RANDOM_SEED = 42

print(f'Output dir       : {OUTPUT_DIR}')
print(f'Raters dir       : {RATERS_DIR}')
print(f'Rater A filled   : {RATER_A_FILLED}')
print(f'Rater B filled   : {RATER_B_FILLED}')
print(f'  Rater A exists : {os.path.exists(RATER_A_FILLED)}')
print(f'  Rater B exists : {os.path.exists(RATER_B_FILLED)}')
print(f'Sample size      : {N_SAMPLE}')



# A2: load full labeled dataset and stratified-sample N_SAMPLE rows
df_full = pd.read_csv(FULL_LABELED_PATH)
print(f'Loaded full labeled dataset: {len(df_full):,} rows')
print(f'\nPer (model, technique):')
print(df_full.groupby(['model', 'technique']).size().unstack(fill_value=0))

# Stratified across model × technique (12 cells)
groups = df_full.groupby(['model', 'technique'])
n_groups = len(groups)
base_per_group = N_SAMPLE // n_groups
remainder = N_SAMPLE - base_per_group * n_groups

print(f'\nSampling: {base_per_group} per (model, technique) cell + {remainder} extras')

rng = np.random.RandomState(RANDOM_SEED)
sample_dfs = []
extras_pool = []

for (model, tech), g in groups:
    take = min(base_per_group, len(g))
    picked = g.sample(n=take, random_state=RANDOM_SEED)
    sample_dfs.append(picked)
    leftover = g.drop(picked.index)
    extras_pool.append(leftover)

df_sample = pd.concat(sample_dfs, ignore_index=True)

# Top up to exactly N_SAMPLE
if remainder > 0 and len(df_sample) == base_per_group * n_groups:
    extras_df = pd.concat(extras_pool, ignore_index=True)
    if len(extras_df) >= remainder:
        topup = extras_df.sample(n=remainder, random_state=RANDOM_SEED + 1)
        df_sample = pd.concat([df_sample, topup], ignore_index=True)
elif len(df_sample) < N_SAMPLE:
    extras_df = pd.concat(extras_pool, ignore_index=True)
    need = N_SAMPLE - len(df_sample)
    if len(extras_df) >= need:
        topup = extras_df.sample(n=need, random_state=RANDOM_SEED)
        df_sample = pd.concat([df_sample, topup], ignore_index=True)

df_sample = df_sample.reset_index(drop=True)
df_sample['row_id'] = [f'{i+1:03d}' for i in range(len(df_sample))]

print(f'\nFinal sample: {len(df_sample)} rows')
print(f'\nSample distribution per (model, technique):')
print(df_sample.groupby(['model', 'technique']).size().unstack(fill_value=0))
print(f'\nSample distribution per crime_type:')
print(df_sample.groupby('crime_type').size())


# A3: helper — generate one Word doc per row
def safe_filename(s):
    """Make string safe for Windows filenames."""
    return re.sub(r'[\\/:*?"<>|]', '_', str(s))

def make_rater_docx(out_path, row):
    """Create one Word doc with full ground truth + full model output + scoring checklist."""
    doc = Document()

    # Default font sizing
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # ===== Header / metadata =====
    h = doc.add_heading(f'Spotcheck Row {row["row_id"]}', level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT

    p = doc.add_paragraph()
    p.add_run('Video: ').bold = True
    p.add_run(str(row['video']))

    p = doc.add_paragraph()
    p.add_run('Crime type: ').bold = True
    p.add_run(str(row['crime_type']))

    p = doc.add_paragraph()
    p.add_run('Model / Technique: ').bold = True
    p.add_run(f'{row["model"]} / {row["technique"]}')

    doc.add_paragraph()   # spacer

    # ===== Ground truth =====
    doc.add_heading('Ground Truth', level=2)
    doc.add_paragraph(str(row['ground_truth']))

    # ===== Model output =====
    doc.add_heading('Model Output', level=2)
    # Word handles unlimited length — split paragraphs on \n\n for readability
    text = str(row['model_output'])
    for chunk in text.split('\n\n'):
        if chunk.strip():
            doc.add_paragraph(chunk.strip())

    # ===== Scoring instructions =====
    doc.add_page_break()
    doc.add_heading('Hallucination Scoring Instructions', level=2)

    instructions = (
        'Compare the MODEL OUTPUT against the GROUND TRUTH above. For each '
        'hallucination type below, decide whether it is PRESENT (1) or ABSENT (0) '
        'in the model output. Then enter your scores in the file '
        '"<Your-Name>-scores.xlsx" in the row matching this Row ID.'
    )
    doc.add_paragraph(instructions)

    descriptions = [
        ('H1  SCENE_FABRICATION',       'Invents setting details not in ground truth (wrong location, objects, environment).'),
        ('H2  CRIME_MISCLASSIFICATION', 'Identifies the wrong crime type (e.g. says Robbery when it is Assault).'),
        ('H3  CRIME_MISSED',            'Fails to mention the primary crime that is clearly described in ground truth.'),
        ('H4  SEVERITY_MINIMIZATION',   'Describes the crime as less serious than the ground truth indicates.'),
        ('H5  ENTITY_FABRICATION',      'Invents people, vehicles, or objects not present in ground truth.'),
        ('H6  PHANTOM_ACTORS',          'Adds perpetrators or victims not mentioned in ground truth.'),
    ]

    for label, desc in descriptions:
        p = doc.add_paragraph()
        p.add_run(label).bold = True
        p.add_run(' — ' + desc)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run('Reminder: ').bold = True
    p.add_run(
        'Score independently. Do NOT discuss with the other rater until both '
        'spreadsheets are submitted.'
    )

    doc.save(out_path)

print('docx generator ready.')


# A4: generate all 100 Word docs into BOTH rater folders
from tqdm.auto import tqdm

print(f'Generating {len(df_sample)} Word docs per rater...')

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc='Word docs'):
    fname_base = (
        f'{row["row_id"]}_'
        f'{safe_filename(row["video"])}_'
        f'{safe_filename(row["model"])}_'
        f'{safe_filename(row["technique"])}.docx'
    )
    op_path = os.path.join(RATER_A_DIR, fname_base)
    dk_path = os.path.join(RATER_B_DIR, fname_base)
    make_rater_docx(op_path, row)
    make_rater_docx(dk_path, row)

print(f'\nWrote {len(df_sample)} docs into each of:')
print(f'  {RATER_A_DIR}')
print(f'  {RATER_B_DIR}')


# A5: build the scoring spreadsheets — one per rater
# These are SMALL — just the row identifiers and the H1-H6 columns to fill in.
# Panel labels are kept on Rater B's sheet for direct comparison in Stage B.

scores_cols = ['row_id', 'video', 'crime_type', 'model', 'technique',
               'human_H1', 'human_H2', 'human_H3', 'human_H4', 'human_H5', 'human_H6',
               'human_notes']

df_scores = df_sample.copy()
for h in HCOLS:
    df_scores[f'human_{h}'] = ''   # blank for the rater to fill in
df_scores['human_notes'] = ''

# Rater A's scoring file — panel labels included for direct comparison in Stage B
df_op_out = df_scores[scores_cols].copy()
for h in HCOLS:
    df_op_out[f'panel_{h}'] = df_sample[h].astype(int).values
df_op_out.to_excel(RATER_A_SCORES, index=False)
print(f'Saved: {RATER_A_SCORES}')

# Rater B's scoring file — same panel labels for symmetric comparison
df_dk_out = df_scores[scores_cols].copy()
for h in HCOLS:
    df_dk_out[f'panel_{h}'] = df_sample[h].astype(int).values
df_dk_out.to_excel(RATER_B_SCORES, index=False)
print(f'Saved: {RATER_B_SCORES}')

print('\n=== Rater package summary ===')
print(f'Each rater receives:')
print(f'  - 100 Word docs ({len(df_sample)} files)')
print(f'  - 1 scoring spreadsheet (Rater A-scores.xlsx / Rater B-scores.xlsx)')
print()
print('Workflow for each rater:')
print('  1. Open each Word doc to read the ground truth and model output')
print('  2. Open the scoring spreadsheet')
print('  3. For each row_id, enter 0 or 1 for human_H1 through human_H6')
print('  4. Optionally add a note in human_notes')
print('  5. Save the spreadsheet as <Name>-scores_FILLED.xlsx and return it')


# B1: load both filled scoring sheets, validate, and compute ANY columns
assert os.path.exists(RATER_A_FILLED), f'Missing: {RATER_A_FILLED}'
assert os.path.exists(RATER_B_FILLED), f'Missing: {RATER_B_FILLED}'

df_op = pd.read_excel(RATER_A_FILLED)
df_dk = pd.read_excel(RATER_B_FILLED)

# Sort both by row_id to ensure alignment
df_op = df_op.sort_values('row_id').reset_index(drop=True)
df_dk = df_dk.sort_values('row_id').reset_index(drop=True)

# Verify alignment
assert (df_op['row_id'] == df_dk['row_id']).all(), 'row_id mismatch between rater files!'
assert (df_op['video']  == df_dk['video']).all(),  'video mismatch between rater files!'
n_rows = len(df_op)
print(f'Loaded {n_rows} rows from each rater (aligned by row_id).')

# Coerce blanks to 0 / ints
for h in HCOLS:
    df_op[f'human_{h}'] = pd.to_numeric(df_op[f'human_{h}'], errors='coerce').fillna(0).astype(int)
    df_dk[f'human_{h}'] = pd.to_numeric(df_dk[f'human_{h}'], errors='coerce').fillna(0).astype(int)
    df_dk[f'panel_{h}'] = pd.to_numeric(df_dk[f'panel_{h}'], errors='coerce').fillna(0).astype(int)

# ANY columns
any_op = (df_op[[f'human_{h}' for h in HCOLS]].sum(axis=1) > 0).astype(int)
any_dk = (df_dk[[f'human_{h}' for h in HCOLS]].sum(axis=1) > 0).astype(int)
any_pa = (df_dk[[f'panel_{h}'  for h in HCOLS]].sum(axis=1) > 0).astype(int)

print('\n=== Rating Summary (count of 1s) ===')
print(f'{"H-type":<8} {"n+_Rater A":>12} {"n+_Rater B":>12} {"n+_Panel":>10}')
print('-' * 47)
for h in HCOLS:
    print(f'{h:<8} {df_op[f"human_{h}"].sum():>12} '
          f'{df_dk[f"human_{h}"].sum():>12} '
          f'{df_dk[f"panel_{h}"].sum():>10}')
print(f'{"ANY":<8} {any_op.sum():>12} {any_dk.sum():>12} {any_pa.sum():>10}')


# B2: Inter-rater agreement (Rater A vs Rater B)
print('=== Inter-Rater Agreement: Rater A vs Rater B ===')
print(f'{"H-type":<8} {"Agree%":>8} {"Kappa":>8}')
print('-' * 28)

kappas_irr = []
irr_results = {}

for h in HCOLS:
    y_op = df_op[f'human_{h}'].values
    y_dk = df_dk[f'human_{h}'].values
    agree = (y_op == y_dk).mean() * 100
    try:
        k = cohen_kappa_score(y_op, y_dk)
    except Exception:
        k = float('nan')
    if not np.isnan(k):
        kappas_irr.append(k)
    irr_results[h] = {'agreement_pct': float(agree), 'kappa': float(k)}
    print(f'{h:<8} {agree:>7.1f}% {k:>8.3f}')

k_any_irr     = cohen_kappa_score(any_op.values, any_dk.values)
agree_any_irr = (any_op.values == any_dk.values).mean() * 100
irr_results['ANY']   = {'agreement_pct': float(agree_any_irr), 'kappa': float(k_any_irr)}
irr_results['macro'] = float(np.nanmean(kappas_irr))

print(f'{"ANY":<8} {agree_any_irr:>7.1f}% {k_any_irr:>8.3f}')
print(f'\nMacro kappa (H1-H6): {irr_results["macro"]:.3f}')
print(f'ANY kappa          : {k_any_irr:.3f}')


# B3: Each human rater vs LLM panel
rater_panel_results = {}

for rater_name, df_rater, any_rater in [
    ('Rater A', df_op, any_op),
    ('Rater B', df_dk, any_dk)
]:
    print(f'=== {rater_name} vs Panel ===')
    print(f'{"H-type":<8} {"Agree%":>8} {"Kappa":>8}')
    print('-' * 28)

    kappas_r = []
    r_results = {}

    for h in HCOLS:
        y_rater = df_rater[f'human_{h}'].values
        y_panel = df_dk[f'panel_{h}'].values
        agree   = (y_rater == y_panel).mean() * 100
        try:
            k = cohen_kappa_score(y_rater, y_panel)
        except Exception:
            k = float('nan')
        if not np.isnan(k):
            kappas_r.append(k)
        r_results[h] = {'agreement_pct': float(agree), 'kappa': float(k)}
        print(f'{h:<8} {agree:>7.1f}% {k:>8.3f}')

    k_any_r     = cohen_kappa_score(any_rater.values, any_pa.values)
    agree_any_r = (any_rater.values == any_pa.values).mean() * 100
    r_results['ANY']   = {'agreement_pct': float(agree_any_r), 'kappa': float(k_any_r)}
    r_results['macro'] = float(np.nanmean(kappas_r))

    print(f'{"ANY":<8} {agree_any_r:>7.1f}% {k_any_r:>8.3f}')
    print(f'Macro kappa (H1-H6): {r_results["macro"]:.3f}\n')
    rater_panel_results[rater_name] = r_results


# B4: Majority human vote vs panel (rows where both raters agree)
print('=== Majority Human Vote vs Panel ===')
print('(only rows where Rater A and Rater B agree)')
print(f'{"H-type":<8} {"n_agreed":>9} {"Agree%":>8} {"Kappa":>8}')
print('-' * 38)

kappas_maj = []
maj_results = {}

for h in HCOLS:
    y_op    = df_op[f'human_{h}'].values
    y_dk    = df_dk[f'human_{h}'].values
    y_panel = df_dk[f'panel_{h}'].values
    mask    = y_op == y_dk
    n_agree = mask.sum()
    if n_agree > 0:
        agree = (y_op[mask] == y_panel[mask]).mean() * 100
        try:
            k = cohen_kappa_score(y_op[mask], y_panel[mask])
        except Exception:
            k = float('nan')
    else:
        agree, k = 0.0, float('nan')
    if not np.isnan(k):
        kappas_maj.append(k)
    maj_results[h] = {
        'n_rows_agreed': int(n_agree),
        'agreement_pct': float(agree),
        'kappa': float(k)
    }
    print(f'{h:<8} {n_agree:>9} {agree:>7.1f}% {k:>8.3f}')

mask_any = any_op.values == any_dk.values
if mask_any.sum() > 0:
    k_any_maj     = cohen_kappa_score(any_op.values[mask_any], any_pa.values[mask_any])
    agree_any_maj = (any_op.values[mask_any] == any_pa.values[mask_any]).mean() * 100
else:
    k_any_maj, agree_any_maj = float('nan'), 0.0

maj_results['ANY']   = {
    'n_rows_agreed': int(mask_any.sum()),
    'agreement_pct': float(agree_any_maj),
    'kappa': float(k_any_maj)
}
maj_results['macro'] = float(np.nanmean(kappas_maj))

print(f'{"ANY":<8} {mask_any.sum():>9} {agree_any_maj:>7.1f}% {k_any_maj:>8.3f}')
print(f'\nMacro kappa (H1-H6): {maj_results["macro"]:.3f}')
print(f'ANY kappa          : {k_any_maj:.3f}')


# B5: summary + save JSON
INTER_JUDGE_PATH = os.path.join(OUTPUT_DIR, 'inter_judge_agreement_subset.json')
panel_macro_range = '(see inter_judge_agreement_subset.json)'
if os.path.exists(INTER_JUDGE_PATH):
    with open(INTER_JUDGE_PATH) as f:
        ij = json.load(f)
    overall_kappas = [v['overall'] for k, v in ij.items() if 'overall' in v]
    if overall_kappas:
        panel_macro_range = f'{min(overall_kappas):.3f}-{max(overall_kappas):.3f}'

print('=' * 62)
print('SUMMARY: Human Validation Results')
print('=' * 62)
print(f'{"Comparison":<38} {"Macro kappa":>11} {"ANY kappa":>9}')
print('-' * 62)
print(f'{"Rater A vs Rater B (inter-rater)":<38} '
      f'{irr_results["macro"]:>11.3f} '
      f'{irr_results["ANY"]["kappa"]:>9.3f}')
print(f'{"Rater A vs Panel":<38} '
      f'{rater_panel_results["Rater A"]["macro"]:>11.3f} '
      f'{rater_panel_results["Rater A"]["ANY"]["kappa"]:>9.3f}')
print(f'{"Rater B vs Panel":<38} '
      f'{rater_panel_results["Rater B"]["macro"]:>11.3f} '
      f'{rater_panel_results["Rater B"]["ANY"]["kappa"]:>9.3f}')
print(f'{"Majority human vote vs Panel":<38} '
      f'{maj_results["macro"]:>11.3f} '
      f'{maj_results["ANY"]["kappa"]:>9.3f}')
print(f'{"LLM panel inter-judge overall kappa":<38} {panel_macro_range:>11}')
print(f'\nn spotcheck rows: {n_rows}')

agreement_results = {
    'n_rows': int(n_rows),
    'raters': ['Rater A', 'Rater B'],
    'inter_rater_agreement': irr_results,
    'rater_a_vs_panel': rater_panel_results['Rater A'],
    'rater_b_vs_panel': rater_panel_results['Rater B'],
    'majority_human_vs_panel': maj_results,
}

with open(AGREEMENT_PATH, 'w') as f:
    json.dump(agreement_results, f, indent=2)
print(f'\nSaved: {AGREEMENT_PATH}')
