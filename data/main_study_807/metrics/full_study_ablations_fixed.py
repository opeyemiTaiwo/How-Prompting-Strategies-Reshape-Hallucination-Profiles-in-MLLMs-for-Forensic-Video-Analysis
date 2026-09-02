"""
full_study_ablations_fixed_FINAL.py  -- ablation studies on the panel-labeled
dataset: single-judge vs panel, leave-one-out, feature ablation, training-size
sweep, and taxonomy-granularity comparison.

No API calls. Reads the panel labels, triplets, and (optionally) embeddings
produced by the panel + embeddings scripts; writes JSON/TXT summaries.
"""


import os, json, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score

# ========================= EDIT THESE =========================
EVAL_DIR = 'EVALUATION'              # labels + triplets
HALL_DIR = 'HALLUCINATION-METRICS'   # embeddings
OUT_DIR  = HALL_DIR
EMB_KIND = 'custom'                   # filename suffix only: must match embeddings_<EMB_KIND>.npy on disk
# ==============================================================

TRIPLETS_PATH   = os.path.join(EVAL_DIR, 'all_triplets_cache.csv')
PANEL_RAW       = os.path.join(EVAL_DIR, 'panel_raw_judge_labels_full.csv')
EMB_PATH        = os.path.join(HALL_DIR, f'embeddings_{EMB_KIND}.npy')
EMB_INDEX_PATH  = os.path.join(HALL_DIR, 'embedding_index.csv')
ABLATION_OUT    = os.path.join(OUT_DIR, 'ablation_results_full.json')
SUMMARY_OUT     = os.path.join(OUT_DIR, 'ablation_summary.txt')

HCOLS = ['H1','H2','H3','H4','H5','H6']
RANDOM_SEED = 42

# --- sanity: confirm every required file resolves before loading ---
for _p in [TRIPLETS_PATH, PANEL_RAW, EMB_PATH, EMB_INDEX_PATH]:
    print(('OK   ' if os.path.exists(_p) else 'MISSING  ') + _p)

labels = pd.read_csv(PANEL_RAW)
for h in HCOLS:
    labels[h] = pd.to_numeric(labels[h], errors='coerce')
clean_labels = labels[(labels[HCOLS] >= 0).all(axis=1)].copy()

def majority(col):
    return col.mode().iloc[0] if not col.mode().empty else 0

mv = clean_labels.groupby(['row_idx','model','technique','video','crime_type'])[HCOLS].agg(majority).reset_index()
mv['ANY'] = (mv[HCOLS].sum(axis=1) > 0).astype(int)

# Each item is scored by 2 of the 3 judges (rotating pairs) -> mv is a 2-judge majority.
print('Judges per item:', clean_labels.groupby('row_idx')['judge'].nunique().value_counts().to_dict())

max_idx = int(clean_labels['row_idx'].max())
df_trip = pd.read_csv(TRIPLETS_PATH, nrows=max_idx + 1)
df_trip['model_output'] = df_trip['model_output'].fillna('').astype(str)
df_trip = df_trip.reset_index().rename(columns={'index': 'row_idx'})

# FULL labeled set (ablations 1, 1b, 2 use this)
data = df_trip.merge(mv[['row_idx'] + HCOLS + ['ANY']], on='row_idx', how='inner')
print(f'Merged labeled rows (full): {len(data):,}  techniques: {sorted(data["technique"].unique())}')

# ---- Embeddings: OPTIONAL. Ablations 3 & 4 run on whatever subset they cover. ----
emb = None; data_emb = None; HAS_EMB = False
if os.path.exists(EMB_PATH) and os.path.exists(EMB_INDEX_PATH):
    emb = np.load(EMB_PATH)
    ei = pd.read_csv(EMB_INDEX_PATH)            # one row per embedding, in emb order
    KEY = ['model','technique','video','crime_type']   # stable join key
    # NOTE: embedding_index 'row_idx' does NOT match the triplets ordering, so we join
    # embeddings to labels on content keys (unique in both files), not on row_idx.
    data_emb = ei.merge(mv[KEY + HCOLS + ['ANY']], on=KEY, how='left')
    if len(data_emb) == len(emb) and not data_emb[HCOLS].isna().any().any():
        for h in HCOLS:
            data_emb[h] = data_emb[h].astype(int)
        data_emb['ANY'] = data_emb['ANY'].astype(int)
        HAS_EMB = True
        cov = sorted(ei['technique'].unique())
        print(f'Embeddings ({EMB_KIND}): {emb.shape} aligned to {len(data_emb):,} labeled rows.')
        print(f'  Techniques covered: {cov}')
        if len(data_emb) < len(data):
            print(f'  NOTE: embeddings cover {len(data_emb):,} of {len(data):,} labeled rows '
                  f'-> ablations 3 & 4 run on this SUBSET only.')
    else:
        print('WARNING: embeddings could not be aligned to labels; skipping 3 & 4.')
else:
    print('NOTE: embeddings not found -> ablations 1 & 2 run; 3 & 4 skipped.')
    print(f'      Looked in: {HALL_DIR}')


ablation1 = {}
for judge in ['Claude','GPT','Gemini']:
    judge_rows = clean_labels[clean_labels['judge'] == judge].copy()

    # Pivot: one row per row_idx with this judge's labels
    judge_lookup = judge_rows.set_index('row_idx')[HCOLS]

    per_h = {}
    for h in HCOLS:
        # Find row_idx values where both this judge AND the majority verdict exist
        common = judge_lookup.index.intersection(mv['row_idx'])
        if len(common) == 0:
            per_h[h] = {'note': 'no common rows'}
            continue
        # Align
        mv_idx = mv.set_index('row_idx').loc[common, h].values
        ju_idx = judge_lookup.loc[common, h].values
        # Cast to int
        mv_int = mv_idx.astype(int)
        ju_int = ju_idx.astype(int)
        agree = (mv_int == ju_int).mean() * 100
        try:
            k = cohen_kappa_score(mv_int, ju_int)
        except Exception:
            k = None
        per_h[h] = {
            'n': int(len(common)),
            'agree_pct': float(agree),
            'kappa': float(k) if k is not None else None,
        }

    # Also ANY
    any_lookup = (judge_lookup[HCOLS].sum(axis=1) > 0).astype(int)
    common = any_lookup.index.intersection(mv['row_idx'])
    if len(common):
        mv_any = mv.set_index('row_idx').loc[common, 'ANY'].values.astype(int)
        ju_any = any_lookup.loc[common].values
        agree = (mv_any == ju_any).mean() * 100
        try:
            k = cohen_kappa_score(mv_any, ju_any)
        except Exception:
            k = None
        per_h['ANY'] = {
            'n': int(len(common)),
            'agree_pct': float(agree),
            'kappa': float(k) if k is not None else None,
        }

    ablation1[judge] = per_h
    print(f'\n{judge} vs Panel (majority):')
    for h, v in per_h.items():
        if 'kappa' in v and v['kappa'] is not None:
            print(f'  {h}: n={v["n"]:>5,}  agree={v["agree_pct"]:>5.1f}%  kappa={v["kappa"]:.3f}')
        else:
            print(f'  {h}: {v}')

# Summary: would a single judge suffice?
print('\n=== Macro kappa (single judge vs panel majority) ===')
for judge in ['Claude','GPT','Gemini']:
    ks = [ablation1[judge][h]['kappa'] for h in HCOLS
          if 'kappa' in ablation1[judge][h] and ablation1[judge][h]['kappa'] is not None]
    if ks:
        print(f'  {judge}: macro kappa = {np.mean(ks):.3f}')


COLS = HCOLS + ['ANY']
cl = clean_labels.copy()
cl['ANY'] = (cl[HCOLS].sum(axis=1) > 0).astype(int)
def _maj(s): return s.mode().iloc[0] if not s.mode().empty else 0

ablation1_loo = {}
print('Leave-one-out: judge vs majority of the REMAINING judges')
print('=' * 60)
for J in ['Claude','GPT','Gemini']:
    jrows = cl[cl.judge==J].drop_duplicates('row_idx').set_index('row_idx')
    rest_mv = cl[cl.judge!=J].groupby('row_idx')[COLS].agg(_maj)
    common = jrows.index.intersection(rest_mv.index)
    per = {}
    for c in COLS:
        a = jrows.loc[common,c].astype(int).values; b = rest_mv.loc[common,c].astype(int).values
        try: k = float(cohen_kappa_score(a,b))
        except Exception: k = None
        per[c] = {'n':int(len(common)),'agree_pct':float((a==b).mean()*100),'kappa':k}
    ablation1_loo[J] = per
    ks = [per[h]['kappa'] for h in HCOLS if per[h]['kappa'] is not None]
    print(f'\n{J} vs rest-of-panel (n={len(common):,}): macro kappa = {np.mean(ks):.3f}')
    for c in COLS:
        if per[c]['kappa'] is not None:
            print(f'  {c}: agree={per[c]["agree_pct"]:>5.1f}%  kappa={per[c]["kappa"]:.3f}')

print('\n=== self-included vs leave-one-out (macro kappa, H1-H6) ===')
mv_h = mv.set_index('row_idx')
for J in ['Claude','GPT','Gemini']:
    jl = cl[cl.judge==J].drop_duplicates('row_idx').set_index('row_idx')
    com = jl.index.intersection(mv_h.index)
    inc = np.mean([cohen_kappa_score(mv_h.loc[com,h].astype(int), jl.loc[com,h].astype(int)) for h in HCOLS])
    loo = np.mean([ablation1_loo[J][h]['kappa'] for h in HCOLS])
    print(f'  {J:<7}: self-included={inc:.3f}  leave-one-out={loo:.3f}  inflation={inc-loo:+.3f}')


# Re-compute hand-crafted features (same definitions as detectability notebook)
HEDGE_WORDS = [
    ' may ',' might ',' possibly ',' perhaps ',' likely ',' probably ',
    ' appears ',' seems ',' suggests ',' suggesting ',' could ',
    ' presumably ',' apparently ',' uncertain',' approximately ',
    ' roughly ',' somewhat ',' arguably ',
]
SENT_RE  = re.compile(r'[.!?]+(?:\s|$)')
CLAIM_RE = re.compile(r'(?:[.!?;]|--|—|\n\s*[-*•])')

def hand_features(text, model_name, technique):
    if not isinstance(text, str): text = ''
    t = text.lower()
    nw = len(text.split())
    n_sent  = sum(1 for p in SENT_RE.split(text) if p.strip())
    n_claim = sum(1 for p in CLAIM_RE.split(text) if len(p.split()) >= 3)
    n_hedge = sum(t.count(h) for h in HEDGE_WORDS)
    return {
        'verbosity_words': nw,
        'verbosity_log':   np.log1p(nw),
        'hedging_density': n_hedge / max(nw, 1),
        'claim_density':   n_claim / max(nw, 1),
        'sent_density':    n_sent / max(nw, 1),
        'is_claude':       int(model_name == 'Claude'),
        'is_gpt':          int(model_name == 'GPT'),
        'is_gemini':       int(model_name == 'Gemini'),
        'is_zero_shot':    int(technique == 'Zero-Shot'),
        'is_sequential':   int(technique == 'Sequential'),
        'is_least_to_most':int(technique == 'Least-to-Most'),
        'is_react':        int(technique == 'ReAct'),
        'is_multiturn':    int(technique in ('Sequential','Least-to-Most','ReAct')),
    }

print('Computing hand features...')
feats = [hand_features(r['model_output'], r['model'], r['technique']) for _, r in data.iterrows()]
hand_df = pd.DataFrame(feats)
print(f'Hand features: {hand_df.shape}')

# Define feature ablation groups
verbosity_cols  = [c for c in hand_df.columns if 'verbosity' in c or c == 'sent_density'
                   or c == 'claim_density']
hedging_cols    = [c for c in hand_df.columns if 'hedging' in c]
model_id_cols   = ['is_claude','is_gpt','is_gemini']
multiturn_cols  = ['is_zero_shot','is_sequential','is_least_to_most','is_react','is_multiturn']

CONFIGS = {
    'full':              list(hand_df.columns),
    'minus_verbosity':   [c for c in hand_df.columns if c not in verbosity_cols],
    'minus_hedging':     [c for c in hand_df.columns if c not in hedging_cols],
    'minus_model_id':    [c for c in hand_df.columns if c not in model_id_cols],
    'minus_multiturn':   [c for c in hand_df.columns if c not in multiturn_cols],
    'only_verbosity':    verbosity_cols,
    'only_hedging':      hedging_cols,
    'only_model_id':     model_id_cols,
    'only_multiturn':    multiturn_cols,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def cv_auc_f1(X, y):
    if y.sum() < 10 or y.sum() > len(y) - 10:
        return None, None
    Xs = StandardScaler().fit_transform(X)
    try:
        scores = cross_validate(
            LogisticRegression(max_iter=2000, class_weight='balanced',
                               random_state=RANDOM_SEED),
            Xs, y, cv=skf,
            scoring=('f1','roc_auc'), n_jobs=-1, error_score='raise',
        )
        return float(np.mean(scores['test_f1'])), float(np.mean(scores['test_roc_auc']))
    except Exception:
        return None, None

ablation2 = {}
for name, cols in CONFIGS.items():
    if not cols:
        continue
    X = hand_df[cols].values.astype(np.float32)
    out = {'config': cols}
    print(f'\n  {name} ({len(cols)} cols):')
    for h in HCOLS + ['ANY']:
        y = data[h].astype(int).values
        f1, auc = cv_auc_f1(X, y)
        out[h] = {'f1': f1, 'auc': auc, 'n_pos': int(y.sum())}
        if auc is not None:
            print(f'    {h}: AUC={auc:.3f}  F1={f1:.3f}')
        else:
            print(f'    {h}: imbalanced')
    ablation2[name] = out


if not HAS_EMB:
    print('Ablation 3 skipped: embeddings not available.')
    ablation3 = {}
else:
    # Stratified 80/20 split for the held-out evaluation; vary the training fraction within
    # the 80% training portion. Use the precomputed embeddings as the feature set (best from the
    # detectability run).
    
    from sklearn.metrics import roc_auc_score, f1_score
    
    # Pre-split: stratified on ANY so all training fractions test on the same held-out set
    y_any = data_emb['ANY'].astype(int).values
    X_train_full, X_test, idx_train, idx_test = train_test_split(
        emb, np.arange(len(data_emb)), test_size=0.2, random_state=RANDOM_SEED, stratify=y_any,
    )
    
    ablation3 = {}
    for frac in (0.25, 0.50, 0.75, 1.00):
        # Subsample within training set
        n_take = int(len(X_train_full) * frac)
        rng = np.random.default_rng(RANDOM_SEED)
        sub = rng.choice(len(X_train_full), size=n_take, replace=False)
        X_tr_sub = X_train_full[sub]
        idx_tr_sub = idx_train[sub]
    
        out = {'fraction': frac, 'n_train': int(n_take)}
        print(f'\n  fraction={frac}  n_train={n_take:,}')
        for h in HCOLS + ['ANY']:
            y_full = data_emb[h].astype(int).values
            y_tr = y_full[idx_tr_sub]
            y_te = y_full[idx_test]
            if y_tr.sum() < 10 or y_te.sum() < 5:
                out[h] = {'note': f'imbalanced (train_pos={int(y_tr.sum())}, test_pos={int(y_te.sum())})'}
                continue
            scaler = StandardScaler().fit(X_tr_sub)
            Xtr_s = scaler.transform(X_tr_sub)
            Xte_s = scaler.transform(X_test)
            clf = LogisticRegression(max_iter=2000, class_weight='balanced',
                                      random_state=RANDOM_SEED)
            clf.fit(Xtr_s, y_tr)
            ypred = clf.predict(Xte_s)
            try:
                yprob = clf.predict_proba(Xte_s)[:, 1]
                auc = float(roc_auc_score(y_te, yprob))
            except Exception:
                auc = None
            f1 = float(f1_score(y_te, ypred, zero_division=0))
            out[h] = {'f1': f1, 'auc': auc, 'n_pos': int(y_tr.sum())}
            if auc is not None:
                print(f'    {h}: AUC={auc:.3f}  F1={f1:.3f}  n_pos_train={int(y_tr.sum())}')
    
        ablation3[f'frac_{frac:.2f}'] = out


if not HAS_EMB:
    print('Ablation 4 skipped: embeddings not available.')
    ablation4 = {}
else:
    GRANULARITIES = {
        '6-type (full H1-H6)': {
            'H1': ['H1'], 'H2': ['H2'], 'H3': ['H3'],
            'H4': ['H4'], 'H5': ['H5'], 'H6': ['H6'],
        },
        '3-type (fabrication / omission / distortion)': {
            'FABRICATION': ['H1','H5','H6'],
            'OMISSION':    ['H3'],
            'DISTORTION':  ['H2','H4'],
        },
        '2-type (fabrication / non-fabrication)': {
            'FABRICATION':     ['H1','H5','H6'],
            'NON_FABRICATION': ['H2','H3','H4'],
        },
        'Binary (any hallucination)': {
            'ANY': ['H1','H2','H3','H4','H5','H6'],
        },
    }
    
    # Re-use the same train/test split from ablation 3 for consistency
    ablation4 = {}
    for gran_name, targets in GRANULARITIES.items():
        print(f'\n  {gran_name}:')
        out = {}
        f1s, aucs = [], []
        for tname, components in targets.items():
            # OR across components
            y_full = (data_emb[components].sum(axis=1) > 0).astype(int).values
            y_tr = y_full[idx_train]
            y_te = y_full[idx_test]
            if y_tr.sum() < 10 or y_te.sum() < 5 or y_tr.sum() > len(y_tr) - 10:
                out[tname] = {'components': components, 'note': 'imbalanced'}
                continue
            scaler = StandardScaler().fit(X_train_full)
            Xtr_s = scaler.transform(X_train_full)
            Xte_s = scaler.transform(X_test)
            clf = LogisticRegression(max_iter=2000, class_weight='balanced',
                                      random_state=RANDOM_SEED)
            clf.fit(Xtr_s, y_tr)
            ypred = clf.predict(Xte_s)
            try:
                yprob = clf.predict_proba(Xte_s)[:, 1]
                auc = float(roc_auc_score(y_te, yprob))
            except Exception:
                auc = None
            f1 = float(f1_score(y_te, ypred, zero_division=0))
            out[tname] = {
                'components': components,
                'n_pos': int(y_full.sum()),
                'n_neg': int(len(y_full) - y_full.sum()),
                'f1': f1, 'auc': auc,
            }
            if auc is not None:
                aucs.append(auc); f1s.append(f1)
                print(f'    {tname}: AUC={auc:.3f}  F1={f1:.3f}  '
                      f'n_pos={int(y_full.sum())}/{len(y_full)}')
        out['_summary'] = {
            'n_targets': len(targets),
            'macro_f1':  float(np.mean(f1s)) if f1s else None,
            'macro_auc': float(np.mean(aucs)) if aucs else None,
        }
        if out['_summary']['macro_auc']:
            print(f'    --> macro AUC = {out["_summary"]["macro_auc"]:.3f}')
        ablation4[gran_name] = out


ablation_results = {
    'ablation1_panel_vs_single':    ablation1,
    'ablation1b_leave_one_out':      ablation1_loo,
    'ablation2_features':            ablation2,
    'ablation3_training_size':       ablation3,
    'ablation4_taxonomy_granularity': ablation4,
}
with open(ABLATION_OUT, 'w') as f:
    json.dump(ablation_results, f, indent=2)
print(f'Saved: {ABLATION_OUT}')

# Build summary
lines = []
lines.append('=' * 76)
lines.append('FULL-STUDY ABLATION RESULTS — SUMMARY')
lines.append('=' * 76)

# Ablation 1 summary
lines.append('')
lines.append('-- ABLATION 1: Single judge vs panel-of-3 (macro kappa) --')
for judge in ['Claude','GPT','Gemini']:
    ks = [ablation1[judge][h]['kappa'] for h in HCOLS
          if ablation1[judge][h].get('kappa') is not None]
    if ks:
        lines.append(f'  {judge:<8}  macro kappa = {np.mean(ks):.3f}  (per-H range: '
                     f'{min(ks):.3f}-{max(ks):.3f})')
lines.append('  Interpretation: high kappa (>0.85) means single judge could substitute for panel.')

lines.append('')
lines.append('-- ABLATION 1b: Leave-one-out (judge vs rest-of-panel, macro kappa) --')
mv_h = mv.set_index('row_idx')
for judge in ['Claude','GPT','Gemini']:
    ks = [ablation1_loo[judge][h]['kappa'] for h in HCOLS if ablation1_loo[judge][h].get('kappa') is not None]
    jl = clean_labels[clean_labels['judge']==judge].drop_duplicates('row_idx').set_index('row_idx')
    com = jl.index.intersection(mv_h.index)
    inc = float(np.mean([cohen_kappa_score(mv_h.loc[com,h].astype(int), jl.loc[com,h].astype(int)) for h in HCOLS]))
    if ks:
        lines.append(f'  {judge:<8}  leave-one-out kappa = {np.mean(ks):.3f}  (self-included {inc:.3f}, inflation {inc-np.mean(ks):+.3f})')
lines.append('  Note: each item scored by 2 of 3 judges (rotating pairs); leave-one-out = pairwise.')

# Ablation 2 summary
lines.append('')
lines.append('-- ABLATION 2: Feature ablation (5-fold CV AUC for ANY) --')
for cfg, res in ablation2.items():
    auc_any = res.get('ANY', {}).get('auc')
    if auc_any is not None:
        lines.append(f'  {cfg:<22}  AUC(ANY) = {auc_any:.3f}')

# Ablation 3
lines.append('')
lines.append('-- ABLATION 3: Training size sweep (held-out AUC for ANY) --')
if HAS_EMB and len(data_emb) < len(data):
    lines.append(f'  (on embedded subset: {len(data_emb):,} rows, techniques {sorted(data_emb["technique"].unique())})')
for k in ['frac_0.25','frac_0.50','frac_0.75','frac_1.00']:
    if ablation3 and k in ablation3:
        v = ablation3[k]
        any_auc = v.get('ANY', {}).get('auc')
        n = v.get('n_train', 0)
        if any_auc is not None:
            lines.append(f'  {k}: n={n:>5,}  AUC(ANY) = {any_auc:.3f}')

# Ablation 4
lines.append('')
lines.append('-- ABLATION 4: Taxonomy granularity (held-out macro AUC) --')
for gran, res in (ablation4 or {}).items():
    summary = res.get('_summary', {})
    macro_auc = summary.get('macro_auc')
    macro_f1  = summary.get('macro_f1')
    if macro_auc is not None:
        lines.append(f'  {gran:<48} macro AUC = {macro_auc:.3f}  macro F1 = {macro_f1:.3f}')

summary = '\n'.join(lines)
with open(SUMMARY_OUT, 'w') as f:
    f.write(summary)
print('\n' + summary)
print(f'\nSaved: {SUMMARY_OUT}')
