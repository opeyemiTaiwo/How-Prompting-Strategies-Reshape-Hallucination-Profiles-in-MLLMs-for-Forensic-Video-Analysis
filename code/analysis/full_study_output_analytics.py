#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-study output analytics: corpus statistics (words/sentences/claims),
refusal rates, claim density, and Spearman correlations between output
characteristics and panel-labeled hallucination rates.

Inputs (place in EVAL_DIR, default = current directory):
    all_triplets_cache.csv          final-answer reports
    panel_raw_judge_labels_full.csv panel H1-H6 labels
    generation_corpus_full.json     (optional) precomputed generation corpus
Outputs are written to OUT_DIR (default = ./output_analytics).

Run:  pip install pandas numpy scipy
      python full_study_output_analytics.py [EVAL_DIR] [OUT_DIR]
"""

import os, sys, json, re
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EVAL_DIR", ".")
TRIPLETS_PATH = os.path.join(OUTPUT_DIR, 'all_triplets_cache.csv')
PANEL_RAW     = os.path.join(OUTPUT_DIR, 'panel_raw_judge_labels_full.csv')

WC_DIR     = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUT_DIR", "output_analytics")
os.makedirs(WC_DIR, exist_ok=True)

# Inputs (precomputed elsewhere)
GENERATION_CORPUS_JSON = os.path.join(WC_DIR, 'generation_corpus_full.json')

# Outputs from this notebook
WORD_COUNT_OUT     = os.path.join(WC_DIR, 'word_count_final_answers.json')
REFUSAL_OUT        = os.path.join(WC_DIR, 'refusal_rate_full.csv')
CLAIM_DENSITY_OUT  = os.path.join(WC_DIR, 'claim_density_full.csv')
CORRELATIONS_OUT   = os.path.join(WC_DIR, 'correlations_full.json')
SUMMARY_OUT        = os.path.join(WC_DIR, 'summary_full.txt')
METRICS_OUT        = os.path.join(WC_DIR, 'metrics_full.csv')

HCOLS = ['H1','H2','H3','H4','H5','H6']

for _p in [TRIPLETS_PATH, PANEL_RAW]:
    print(('OK   ' if os.path.exists(_p) else 'MISSING  ') + _p)

# Load triplets (final answers = the text the models generated from each video)
df = pd.read_csv(TRIPLETS_PATH)
df['model_output'] = df['model_output'].fillna('').astype(str)

# Load generation corpus (precomputed)
if not os.path.exists(GENERATION_CORPUS_JSON):
    print(f'WARNING: {GENERATION_CORPUS_JSON} not found.')
    print('  Run the generation_corpus_counter notebook first to produce it.')
    print('  Sections that need generation-corpus data will fail until you do.')
    gen_corpus = None
else:
    with open(GENERATION_CORPUS_JSON, 'r') as f:
        gen_corpus = json.load(f)
    print(f'Generation corpus loaded: {gen_corpus["grand_totals"]["words"]:,} words across '
          f'{gen_corpus["grand_totals"]["experiments"]:,} experiments')

print(f'\nTriplets (final answers): {len(df):,} rows')
print(f'  Models:     {sorted(df["model"].unique().tolist())}')
print(f'  Techniques: {sorted(df["technique"].unique().tolist())}')
print(f'\nPer (model, technique) counts:')
print(df.groupby(["model","technique"]).size().unstack(fill_value=0))



SENT_RE  = re.compile(r'[.!?]+(?:\s|$)')
CLAIM_RE = re.compile(r'(?:[.!?;]|--|—|\n\s*[-*•])')

def count_words(text): return len(text.split()) if text else 0
def count_sentences(text):
    return sum(1 for p in SENT_RE.split(text) if p.strip()) if text else 0
def count_claims(text):
    if not text: return 0
    parts = [p.strip() for p in CLAIM_RE.split(text) if p.strip()]
    return sum(1 for p in parts if len(p.split()) >= 3)

print('Counting words / sentences / claims for all final-answer outputs...')
df['word_count']  = df['model_output'].apply(count_words)
df['sent_count']  = df['model_output'].apply(count_sentences)
df['claim_count'] = df['model_output'].apply(count_claims)

# Per-model totals
model_totals = df.groupby('model').agg(
    experiments=('model','size'),
    words=('word_count','sum'),
    sentences=('sent_count','sum'),
    claims=('claim_count','sum'),
).to_dict('index')

# Per (model, technique)
tech_details = {}
for model in df['model'].unique():
    tech_details[model] = {}
    for tech in df[df['model']==model]['technique'].unique():
        sub = df[(df['model']==model) & (df['technique']==tech)]
        tech_details[model][tech] = {
            'experiments': int(len(sub)),
            'words': int(sub['word_count'].sum()),
            'sentences': int(sub['sent_count'].sum()),
            'claims': int(sub['claim_count'].sum()),
            'mean_words': round(float(sub['word_count'].mean()), 1),
        }

grand = {
    'experiments': int(len(df)),
    'words': int(df['word_count'].sum()),
    'sentences': int(df['sent_count'].sum()),
    'claims': int(df['claim_count'].sum()),
}

final_answer_data = {
    'source': TRIPLETS_PATH,
    'description': 'Final-answer corpus: words in the synthesized final reports judged by the LLM panel.',
    'n_rows': int(len(df)),
    'model_results': {m: {k: int(v) for k, v in d.items()} for m, d in model_totals.items()},
    'grand_totals': grand,
    'technique_details': tech_details,
}

with open(WORD_COUNT_OUT, 'w') as f:
    json.dump(final_answer_data, f, indent=2)
print(f'\nSaved: {WORD_COUNT_OUT}')

print('\n=== FINAL-ANSWER CORPUS SUMMARY ===')
print(f'Grand total: {grand["words"]:,} words across {grand["experiments"]:,} reports')
print('\nPer-model:')
for m, d in model_totals.items():
    print(f'  {m:<8} reports={d["experiments"]:>5,}  words={d["words"]:>12,}  '
          f'claims={d["claims"]:>10,}')
print('\nPer (model, technique):')
print(f'{"Model":<10}{"Technique":<16}{"n":>5}{"Words":>12}{"MeanW":>8}{"Claims":>10}')
print('-'*60)
for model, techs in tech_details.items():
    for tech, d in techs.items():
        print(f'{model:<10}{tech:<16}{d["experiments"]:>5,}{d["words"]:>12,}'
              f'{d["mean_words"]:>8.1f}{d["claims"]:>10,}')



df['char_count'] = df['model_output'].str.len()

n_videos  = df['video'].nunique()
n_outputs = len(df)
TW = int(df['word_count'].sum()); TS = int(df['sent_count'].sum())
TC = int(df['claim_count'].sum()); TCH = int(df['char_count'].sum())

print('=' * 60)
print('TOTAL CORPUS GENERATED FROM THE VIDEOS')
print('=' * 60)
print(f'  Videos:             {n_videos:,}')
print(f'  Generated outputs:  {n_outputs:,}  '
      f'({df["model"].nunique()} models x {df["technique"].nunique()} techniques x {n_videos} videos)')
print(f'  TOTAL WORDS:        {TW:,}')
print(f'  Total sentences:    {TS:,}')
print(f'  Total claims:       {TC:,}')
print(f'  Total characters:   {TCH:,}')
print(f'  Mean words/output:  {TW / max(n_outputs,1):,.1f}')
print(f'  Mean words/video:   {TW / max(n_videos,1):,.1f}')

# Per-video: words generated about each video, summed across all model x technique outputs
per_video = (df.groupby(['video','crime_type'])
               .agg(outputs=('word_count','size'), words=('word_count','sum'),
                    sentences=('sent_count','sum'), claims=('claim_count','sum'),
                    mean_words=('word_count','mean'))
               .reset_index().sort_values('words', ascending=False))
per_video['mean_words'] = per_video['mean_words'].round(1)
PER_VIDEO_OUT = os.path.join(WC_DIR, 'corpus_by_video.csv')
per_video.to_csv(PER_VIDEO_OUT, index=False)
print(f'\n  Saved per-video corpus -> {PER_VIDEO_OUT}')

print('\nWords by crime type:')
print(df.groupby('crime_type')['word_count'].agg(outputs='size', words='sum', mean_words='mean')
        .round(1).sort_values('words', ascending=False).to_string())

corpus_total = {
    'n_videos': int(n_videos), 'n_outputs': int(n_outputs),
    'total_words': TW, 'total_sentences': TS, 'total_claims': TC, 'total_characters': TCH,
    'mean_words_per_output': round(TW / max(n_outputs,1), 1),
    'mean_words_per_video':  round(TW / max(n_videos,1), 1),
    'by_model':     {k:int(v) for k,v in df.groupby('model')['word_count'].sum().items()},
    'by_technique': {k:int(v) for k,v in df.groupby('technique')['word_count'].sum().items()},
    'by_crime_type':{k:int(v) for k,v in df.groupby('crime_type')['word_count'].sum().items()},
}
CORPUS_TOTAL_OUT = os.path.join(WC_DIR, 'corpus_total.json')
with open(CORPUS_TOTAL_OUT, 'w') as f:
    json.dump(corpus_total, f, indent=2)
print(f'  Saved corpus totals    -> {CORPUS_TOTAL_OUT}')



refusal_phrases = [
    'i cannot', "i can't", "i'm unable", 'i am unable', "i'm not able",
    'cannot analyze', 'cannot process', 'not appropriate', 'i must decline',
    'i apologize', 'unable to provide', 'cannot provide', 'i refuse',
    "i'm sorry, but", 'against my guidelines', 'potentially harmful',
    'sensitive content', 'cannot assist', 'not able to assist',
    'i cannot fulfill', 'inappropriate', "i won't", 'i will not',
]

def is_refusal(text):
    if not text: return False
    t = text.lower()
    return any(p in t for p in refusal_phrases)

df['is_refusal'] = df['model_output'].apply(is_refusal)

print('=== Refusal rate by (model, technique) ===')
print(f'{"Model":<10}{"Technique":<16}{"n":>5}{"refusals":>10}{"rate":>8}{"mean_words":>12}')
print('-'*61)
ref_rows = []
for (m, t), grp in df.groupby(['model','technique']):
    n = len(grp)
    n_ref = int(grp['is_refusal'].sum())
    rate = n_ref / n * 100 if n else 0
    mean_w = grp['word_count'].mean()
    flag = ' <<<' if rate > 30 else ''
    print(f'{m:<10}{t:<16}{n:>5,}{n_ref:>10}{rate:>7.1f}%{mean_w:>11.0f}{flag}')
    ref_rows.append({
        'model': m, 'technique': t,
        'n': n, 'refusals': n_ref,
        'refusal_rate_pct': round(rate, 2),
        'mean_words_final': round(float(mean_w), 1),
    })

ref_df = pd.DataFrame(ref_rows)
ref_df.to_csv(REFUSAL_OUT, index=False)
print(f'\nSaved: {REFUSAL_OUT}')

print('\n=== Overall refusal rate per model (final answers) ===')
for m in df['model'].unique():
    sub = df[df['model']==m]
    rate = sub['is_refusal'].mean() * 100
    print(f'  {m:<10} {sub["is_refusal"].sum()}/{len(sub)} ({rate:.1f}%)')

# Sample one refusal per model
print('\n=== Sample refusals (first 250 chars) ===')
for m in df['model'].unique():
    sub = df[(df['model']==m) & (df['is_refusal'])]
    if len(sub) > 0:
        s = sub.iloc[0]
        print(f'\n  {m} / {s["technique"]} / {s["video"]}:')
        print(f'    "{s["model_output"][:250]}..."')



# Load panel labels and aggregate hallucination rates per (model, technique)
labels = pd.read_csv(PANEL_RAW)
for h in HCOLS:
    labels[h] = pd.to_numeric(labels[h], errors='coerce')
clean = labels[(labels[HCOLS] >= 0).all(axis=1)].copy()
print(f'Clean panel rows (no -1): {len(clean):,}')

def majority(col):
    return col.mode().iloc[0] if not col.mode().empty else 0

mv = clean.groupby(['row_idx','model','technique'])[HCOLS].agg(majority).reset_index()
print(f'Triplets with majority labels: {len(mv):,}')

agg_rows = []
for (m, t), grp in mv.groupby(['model','technique']):
    row = {'model': m, 'technique': t, 'n_triplets': len(grp)}
    for h in HCOLS:
        row[f'{h}_rate'] = round(grp[h].mean(), 4)
    row['hall_total']   = int(grp[HCOLS].sum().sum())
    row['hall_per_row'] = round(grp[HCOLS].sum().sum() / len(grp), 4)
    row['any_rate']     = round((grp[HCOLS].sum(axis=1) > 0).mean(), 4)
    agg_rows.append(row)
hall_df = pd.DataFrame(agg_rows)

# Corpus basis for claim density: precomputed generation corpus if present, else final answers.
if gen_corpus is not None:
    gen_rows = []
    for model, techs in gen_corpus['technique_details'].items():
        for tech, d in techs.items():
            gen_rows.append({'model': model, 'technique': tech, 'gen_words': d['words'],
                             'gen_claims': d['claims'], 'gen_sentences': d['sentences'],
                             'gen_mean_words': d['mean_words'], 'gen_experiments': d['experiments']})
    gen_df = pd.DataFrame(gen_rows)
    CORPUS_BASIS = 'generation corpus (all stages)'
else:
    print('NOTE: generation_corpus_full.json missing -> claim density uses the FINAL-ANSWER corpus.')
    gen_df = (df.groupby(['model','technique'])
                .agg(gen_words=('word_count','sum'), gen_claims=('claim_count','sum'),
                     gen_sentences=('sent_count','sum'), gen_mean_words=('word_count','mean'),
                     gen_experiments=('word_count','size')).reset_index())
    gen_df['gen_mean_words'] = gen_df['gen_mean_words'].round(1)
    CORPUS_BASIS = 'final-answer corpus (generation corpus not available)'

gen_df['claim_density'] = gen_df['gen_claims'] / gen_df['gen_words']

cd_df = gen_df.merge(hall_df, on=['model','technique'], how='outer').sort_values(['model','technique']).reset_index(drop=True)
cd_df.to_csv(CLAIM_DENSITY_OUT, index=False)
print(f'\nClaim-density basis: {CORPUS_BASIS}')
print(f'Saved: {CLAIM_DENSITY_OUT}')

print('\n=== CLAIM DENSITY vs HALLUCINATION RATE ===')
print(f'{"Model":<10}{"Technique":<16}{"GenWords":>12}{"Claims":>10}{"Density":>9}{"Hall/row":>10}{"AnyRate":>9}')
print('-'*76)
for _, r in cd_df.iterrows():
    print(f'{r["model"]:<10}{r["technique"]:<16}{int(r["gen_words"]):>12,}'
          f'{int(r["gen_claims"]):>10,}{r["claim_density"]:>9.4f}'
          f'{r["hall_per_row"]:>10.3f}{r["any_rate"]:>9.3f}')



# Combine claim density, hallucination, refusal, mean-words into one frame
metrics = cd_df.merge(ref_df[['model','technique','refusal_rate_pct','mean_words_final']],
                       on=['model','technique'], how='left')
metrics['refusal_rate'] = metrics['refusal_rate_pct'] / 100.0

correlations = {}
print('=== Spearman correlations across (model, technique) cells (n=12) ===\n')

pairs = [
    ('claim_density',     'hall_per_row',  'Claim density vs Halls per row'),
    ('claim_density',     'any_rate',      'Claim density vs ANY-hall rate'),
    ('gen_mean_words',    'hall_per_row',  'Mean words per generation vs Halls per row'),
    ('gen_mean_words',    'any_rate',      'Mean words per generation vs ANY-hall rate'),
    ('mean_words_final',  'hall_per_row',  'Mean words (final) vs Halls per row'),
    ('mean_words_final',  'any_rate',      'Mean words (final) vs ANY-hall rate'),
    ('refusal_rate',      'hall_per_row',  'Refusal rate vs Halls per row'),
    ('refusal_rate',      'any_rate',      'Refusal rate vs ANY-hall rate'),
    ('gen_words',         'gen_claims',    'Total gen words vs Total gen claims'),
]
for x, y, label in pairs:
    valid = metrics[[x, y]].dropna()
    if len(valid) < 3:
        print(f'  {label}: too few rows')
        continue
    rho, p = stats.spearmanr(valid[x], valid[y])
    correlations[label] = {'rho': round(float(rho), 4), 'p': round(float(p), 4),
                           'n': int(len(valid))}
    print(f'  {label:<46} rho = {rho:+.3f}  p = {p:.3f}  (n={len(valid)})')

# Per-H correlations with claim density
print('\n=== Claim density vs each H-type rate ===')
for h in HCOLS:
    col = f'{h}_rate'
    valid = metrics[['claim_density', col]].dropna()
    if len(valid) < 3:
        continue
    rho, p = stats.spearmanr(valid['claim_density'], valid[col])
    correlations[f'Claim density vs {h}'] = {'rho': round(float(rho), 4),
                                               'p': round(float(p), 4),
                                               'n': int(len(valid))}
    print(f'  {h}: rho = {rho:+.3f}  p = {p:.3f}')

with open(CORRELATIONS_OUT, 'w') as f:
    json.dump(correlations, f, indent=2)
print(f'\nSaved: {CORRELATIONS_OUT}')

metrics.to_csv(METRICS_OUT, index=False)
print(f'Saved: {METRICS_OUT}')



lines = []
lines.append('=' * 72)
lines.append('FULL-STUDY OUTPUT ANALYTICS — SUMMARY')
lines.append('=' * 72)
lines.append('')
lines.append(f'Final-answer corpus (reports judged by panel):')
lines.append(f'  Reports:    {grand["experiments"]:>15,}')
lines.append(f'  Words:      {grand["words"]:>15,}')
lines.append(f'  Claims:     {grand["claims"]:>15,}')
lines.append('')
if gen_corpus is not None:
    gen_g = gen_corpus["grand_totals"]
    lines.append('Generation corpus (all stages summed):')
    lines.append(f'  Experiments:{gen_g["experiments"]:>15,}')
    lines.append(f'  Words:      {gen_g["words"]:>15,}')
    lines.append(f'  Claims:     {gen_g["claims"]:>15,}')
else:
    lines.append('Generation corpus: not available (claim density used the final-answer corpus).')
lines.append('')
lines.append('-- REFUSAL RATES (final answers) --')
for m in sorted(df['model'].unique()):
    sub = df[df['model']==m]
    rate = sub['is_refusal'].mean() * 100
    lines.append(f'  {m:<10} {sub["is_refusal"].sum():>4}/{len(sub):>4} ({rate:>5.1f}%)')
lines.append('')
lines.append('-- CLAIM DENSITY (gen claims / gen words) --')
for _, r in cd_df.sort_values(['model','technique']).iterrows():
    lines.append(f'  {r["model"]:<10}{r["technique"]:<16} {r["claim_density"]:.4f}')
lines.append('')
lines.append('-- KEY CORRELATIONS --')
for label, vals in correlations.items():
    lines.append(f'  {label:<48} rho={vals["rho"]:+.3f}  p={vals["p"]:.3f}')

summary = '\n'.join(lines)
with open(SUMMARY_OUT, 'w') as f:
    f.write(summary)
print(summary)
print(f'\nSaved: {SUMMARY_OUT}')

