"""
complete_embeddings.py  -- resumable embedding completion (token-budget batching)

Finds labeled rows not yet embedded, embeds only those, appends them.
Safe to re-run: recomputes the missing set each time, so it resumes after interruption.

API REMOVED: the hosted-embedding API client and key loading have been
stripped out. Implement the single stub `embed_texts(texts)` with whatever
backend you like, OR set EMB_KIND='minilm' to use the local
sentence-transformers model (no API needed).

MiniLM mode :  pip install sentence-transformers pandas numpy   (then EMB_KIND='minilm')
Run:           python complete_embeddings.py
"""
import os, sys, re, time
import numpy as np
import pandas as pd

# ========================= EDIT THESE =========================
EVAL_DIR     = 'EVALUATION'              # labels + triplets
HALL_DIR     = 'HALLUCINATION-METRICS'   # embeddings
EMB_KIND     = 'custom'                  # 'custom' (your embed_texts stub) or 'minilm'
MAX_CHARS        = 24000                 # per-input truncation (~6k tokens)
TOKEN_BUDGET     = 200000                # max estimated tokens per request
MAX_BATCH        = 200                   # max inputs per request
CHECKPOINT_EVERY = 500                   # save progress every N new rows
MAX_RETRIES      = 6                     # backoff retries on transient errors
# ==============================================================

HCOLS = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
KEY   = ['model', 'technique', 'video', 'crime_type']
EMB_PATH = os.path.join(HALL_DIR, f'embeddings_{EMB_KIND}.npy')
IDX_PATH = os.path.join(HALL_DIR, 'embedding_index.csv')
TRIPLETS = os.path.join(EVAL_DIR, 'all_triplets_cache.csv')
PANEL    = os.path.join(EVAL_DIR, 'panel_raw_judge_labels_full.csv')

def log(*a): print(*a, flush=True)

# ---- 1. existing embeddings + index ----
if os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH):
    emb = np.load(EMB_PATH).astype(np.float32)
    idx = pd.read_csv(IDX_PATH)
    assert len(emb) == len(idx), f'existing emb/index length mismatch: {len(emb)} vs {len(idx)}'
    DIM = emb.shape[1]
    log(f'Existing: {len(idx):,} rows, dim={DIM}')
else:
    emb, idx, DIM = None, pd.DataFrame(columns=KEY), None
    log('No existing embeddings found; starting fresh.')

# ---- 2. full labeled set + missing rows (content-key set difference) ----
lab = pd.read_csv(PANEL)
for h in HCOLS:
    lab[h] = pd.to_numeric(lab[h], errors='coerce')
clean = lab[(lab[HCOLS] >= 0).all(axis=1)]
max_idx = int(clean['row_idx'].max())
trip = pd.read_csv(TRIPLETS, nrows=max_idx + 1)
trip['model_output'] = trip['model_output'].fillna('').astype(str)

labeled = (clean[KEY].drop_duplicates()
           .merge(trip[KEY + ['model_output']].drop_duplicates(KEY), on=KEY, how='left'))
done = idx[KEY].drop_duplicates() if len(idx) else pd.DataFrame(columns=KEY)
missing = labeled.merge(done.assign(_d=1), on=KEY, how='left')
missing = missing[missing['_d'].isna()].drop(columns='_d').reset_index(drop=True)

log(f'Labeled: {len(labeled):,} | already embedded: {len(done):,} | MISSING: {len(missing):,}')
if len(missing):
    log('Missing by technique:\n' + missing['technique'].value_counts().to_string())
if len(missing) == 0:
    log('Nothing to do -- embeddings already cover the full labeled set.'); sys.exit(0)

# ---- 3. embedder ----
# IMPLEMENT THIS (API removed on purpose), or use EMB_KIND='minilm' below.
def embed_texts(texts):
    """
    Embed a list of strings; return an (len(texts), DIM) float32 numpy array.
    Plug in your own embedding backend here.
    """
    raise NotImplementedError(
        "Implement embed_texts() with your own embedding backend, or set "
        "EMB_KIND='minilm' to use the local sentence-transformers model."
    )

if EMB_KIND == 'minilm':
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    def embed_raw(texts):
        return model.encode(texts, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=False).astype(np.float32)
else:
    def embed_raw(texts):
        return np.asarray(embed_texts(texts), dtype=np.float32)

def embed_with_retry(texts):
    """Retry with backoff; if a request is too large, split it and recurse."""
    for attempt in range(MAX_RETRIES):
        try:
            return embed_raw(texts)
        except NotImplementedError:
            raise
        except Exception as e:
            msg = str(e).lower()
            too_big = ('maximum context length' in msg or 'max_tokens_per_request' in msg
                       or 'reduce' in msg or 'too large' in msg or '400' in msg)
            if too_big and len(texts) > 1:
                mid = len(texts) // 2
                return np.vstack([embed_with_retry(texts[:mid]), embed_with_retry(texts[mid:])])
            wait = min(60, 2 ** attempt)
            log(f'    retry {attempt+1}/{MAX_RETRIES}: {str(e)[:110]} (sleep {wait}s)')
            time.sleep(wait)
    raise RuntimeError('embedding failed after retries')

def save(all_emb, all_idx):
    np.save(EMB_PATH + '.tmp.npy', all_emb)
    all_idx.to_csv(IDX_PATH + '.tmp', index=False)
    os.replace(EMB_PATH + '.tmp.npy', EMB_PATH)
    os.replace(IDX_PATH + '.tmp', IDX_PATH)

# ---- 4. token-budgeted, resumable batching ----
cur_emb, cur_idx = emb, idx.copy()
pend_e, pend_i, since_ckpt, processed = [], [], 0, 0
t0 = time.time()

buf_t, buf_k, buf_tok = [], [], 0
def flush_batch():
    global cur_emb, cur_idx, pend_e, pend_i, since_ckpt, processed, buf_t, buf_k, buf_tok, DIM
    if not buf_t: return
    vecs = embed_with_retry(buf_t)
    if DIM is None:
        DIM = vecs.shape[1]; cur_emb = np.zeros((0, DIM), np.float32)
    assert vecs.shape[1] == DIM, f'dim mismatch {vecs.shape[1]} vs {DIM} -- wrong embedder?'
    pend_e.append(vecs); pend_i.append(pd.DataFrame(buf_k, columns=KEY))
    processed += len(buf_t); since_ckpt += len(buf_t)
    log(f'  embedded {processed:,}/{len(missing):,}   ({time.time()-t0:.0f}s)')
    buf_t, buf_k, buf_tok = [], [], 0
    if since_ckpt >= CHECKPOINT_EVERY:
        cur_emb = np.vstack([cur_emb] + pend_e)
        cur_idx = pd.concat([cur_idx, pd.concat(pend_i)], ignore_index=True)
        save(cur_emb, cur_idx); pend_e, pend_i, since_ckpt = [], [], 0
        log(f'    ...checkpoint saved ({len(cur_idx):,} total)')

for r in missing.itertuples(index=False):
    t = (r.model_output or '')[:MAX_CHARS] or ' '
    est = max(1, len(t) // 4)                       # ~4 chars/token
    if buf_t and (buf_tok + est > TOKEN_BUDGET or len(buf_t) >= MAX_BATCH):
        flush_batch()
    buf_t.append(t); buf_k.append((r.model, r.technique, r.video, r.crime_type)); buf_tok += est
flush_batch()

if pend_e:                                          # final checkpoint
    cur_emb = np.vstack([cur_emb] + pend_e)
    cur_idx = pd.concat([cur_idx, pd.concat(pend_i)], ignore_index=True)
    save(cur_emb, cur_idx)

# ---- 5. verify ----
final = pd.read_csv(IDX_PATH)
cov = labeled.merge(final[KEY].assign(_e=1), on=KEY, how='left')['_e'].notna().mean() * 100
log(f'\nDONE. embeddings now: {np.load(EMB_PATH).shape} | labeled coverage: {cov:.1f}%')
log('Re-run the ablation notebook -- 3 & 4 will use the full 8-technique set automatically.')
