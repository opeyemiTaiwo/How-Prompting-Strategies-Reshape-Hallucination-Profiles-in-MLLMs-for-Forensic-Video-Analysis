"""
panel_judges_FULL_v7_final.py  -- panel-of-judges hallucination labeling

Builds (ground_truth, model_output) triplets from saved analysis results,
then has a panel of judges label 6 hallucination types per triplet, votes
by majority, and reports inter-judge agreement.

API REMOVED: the judge model SDK clients and API keys have been stripped
out. Implement the single stub `call_judge_model(judge_name, system_prompt,
user_prompt)` with whatever backend you like; everything else is unchanged.
"""

import os
import json
import re
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm.auto import tqdm

sys.stdout.flush()

# =====================================================================
# PATHS  — update if your folder layout differs
# =====================================================================
RESULTS_BASE = "RESULTS"        # root with CLAUDE/ GPT/ GEMINI/ subfolders
GT_PATH      = "ANNOTATION"     # ground-truth JSON files
OUTPUT_DIR   = "EVALUATION"     # all outputs land here

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_PATHS = {
    'Claude': os.path.join(RESULTS_BASE, 'CLAUDE'),
    'GPT':    os.path.join(RESULTS_BASE, 'GPT'),
    'Gemini': os.path.join(RESULTS_BASE, 'GEMINI'),
}

GT_CACHE_PATH        = os.path.join(OUTPUT_DIR, 'gt_data_cache.json')
TRIPLETS_CACHE_PATH  = os.path.join(OUTPUT_DIR, 'all_triplets_cache.csv')
PANEL_CHECKPOINT     = os.path.join(OUTPUT_DIR, 'panel_checkpoint_full.json')
PANEL_RAW_PATH       = os.path.join(OUTPUT_DIR, 'panel_raw_judge_labels_full.csv')
JUDGE_LABELS_PATH    = os.path.join(OUTPUT_DIR, 'llm_judge_labels_full.csv')
JUDGE_AGREEMENT_PATH = os.path.join(OUTPUT_DIR, 'inter_judge_agreement_full.json')
FULL_LABELED_PATH    = os.path.join(OUTPUT_DIR, 'full_labeled_dataset_full.csv')

# =====================================================================
# TECHNIQUES  (folder name -> display label)  — 8 total
# =====================================================================
TECHNIQUES = {
    'CHAIN-OF-THOUGHT': 'Chain-of-Thought',
    'LEAST-TO-MOST':    'Least-to-Most',
    'META-PROMPTING':   'Meta-Prompting',
    'REACT':            'ReAct',
    'SELF-CONSISTENCY': 'Self-Consistency',
    'SEQUENTIAL':       'Sequential',
    'TRUE-ITERATIVE':   'True-Iterative',
    'ZERO':             'Zero-Shot',
}

ALL_JUDGES   = ['Claude', 'GPT', 'Gemini']
JUDGE_FILTER = None   # None = all 3 judges (cross-judging, 2 per row)

def judges_for(model_name):
    cross = [j for j in ALL_JUDGES if j != model_name]
    if JUDGE_FILTER is not None:
        cross = [j for j in cross if j in JUDGE_FILTER]
    return cross

HTYPE_MAP = {
    'SCENE_FABRICATION':       'H1',
    'CRIME_MISCLASSIFICATION': 'H2',
    'CRIME_MISSED':            'H3',
    'SEVERITY_MINIMIZATION':   'H4',
    'ENTITY_FABRICATION':      'H5',
    'PHANTOM_ACTORS':          'H6',
}

# =====================================================================
# SANITY CHECKS
# =====================================================================
assert os.path.isdir(RESULTS_BASE), f'Missing: {RESULTS_BASE}'
assert os.path.isdir(GT_PATH),      f'Missing: {GT_PATH}'

print('=' * 70)
print('FULL PANEL RUN v7 — Configuration')
print('=' * 70)
print(f'Results base:  {RESULTS_BASE}')
print(f'Output dir:    {OUTPUT_DIR}')
print(f'Panel:         {ALL_JUDGES}')
print(f'Techniques ({len(TECHNIQUES)}):', list(TECHNIQUES.keys()))
print()
print('Output files (will be created):')
for p in [TRIPLETS_CACHE_PATH, PANEL_CHECKPOINT, PANEL_RAW_PATH,
          JUDGE_LABELS_PATH, JUDGE_AGREEMENT_PATH, FULL_LABELED_PATH]:
    print(f'  {p}')


# =====================================================================
# CHECKPOINT AUTO-CLEANUP
# Permanent policy blocks are preserved. Other -1 entries are cleared
# so they get retried.
# =====================================================================
if os.path.exists(PANEL_CHECKPOINT):
    with open(PANEL_CHECKPOINT) as f:
        _ckpt = json.load(f)
    _initial = len(_ckpt)
    _bad_keys = []
    _kept_permanent = 0
    for k, v in _ckpt.items():
        if not isinstance(v, dict):
            continue
        has_neg1 = any(v.get(h, 0) == -1 for h in ['H1','H2','H3','H4','H5','H6'])
        if not has_neg1:
            continue
        if 'PROHIBITED_CONTENT' in str(v.get('reasoning', '')):
            _kept_permanent += 1
            continue
        _bad_keys.append(k)
    for k in _bad_keys:
        del _ckpt[k]
    if _bad_keys:
        with open(PANEL_CHECKPOINT, 'w') as f:
            json.dump(_ckpt, f)
        print(f'Auto-cleanup: removed {len(_bad_keys)} retryable errors')
        print(f'  Checkpoint: {_initial} -> {len(_ckpt)} entries')
    else:
        print(f'Auto-cleanup: nothing retryable to clean ({_initial} entries)')
    if _kept_permanent:
        print(f'  Preserved {_kept_permanent} permanent policy-block entries')
else:
    print('No existing checkpoint — starting fresh.')


# =====================================================================
# GROUND TRUTH
# =====================================================================
GT_FILES = ['UCFCrime_Train.json', 'UCFCrime_Val.json', 'UCFCrime_Test.json']

def derive_crime_type(video_id):
    name = video_id.replace('_x264', '')
    m = re.match(r'^([A-Za-z]+?)\d', name)
    return m.group(1) if m else 'Unknown'

def is_anomalous(video_id):
    return not video_id.startswith('Normal_')

def parse_ground_truth(gt_base_path):
    gt_data = {}
    for fname in GT_FILES:
        fpath = os.path.join(gt_base_path, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for video_id, info in data.items():
            if not is_anomalous(video_id):
                continue
            gt_data[video_id] = {
                'crime_type': derive_crime_type(video_id),
                'sentences':  info.get('sentences', []),
                'timestamps': info.get('timestamps', []),
                'duration':   info.get('duration'),
            }
    return gt_data

if os.path.exists(GT_CACHE_PATH):
    with open(GT_CACHE_PATH) as f:
        gt_data = json.load(f)
    print(f'Loaded GT from cache: {len(gt_data):,} videos')
else:
    gt_data = parse_ground_truth(GT_PATH)
    with open(GT_CACHE_PATH, 'w') as f:
        json.dump(gt_data, f)
    print(f'Cached {len(gt_data):,} GT videos')

VID_RE = re.compile(r'^([A-Za-z]+_[A-Za-z]+\d+(?:_x264)?)')

def video_id_from_filename(filename):
    m = VID_RE.match(filename)
    return m.group(1) if m else None

def resolve_gt_key(video_id, gt_data):
    candidates = [video_id, video_id.replace('_x264', '')]
    parts = video_id.split('_')
    if len(parts) >= 2:
        wp = '_'.join(parts[1:])
        candidates += [wp, wp.replace('_x264', '')]
    for c in candidates:
        if c in gt_data:
            return c
    return None

print('Ground truth helpers ready.')


# =====================================================================
# LOADERS  (v9 - technique-aware field paths, unified Claude+GPT)
# =====================================================================
def _latest_summary_file(tech_path):
    """Return latest *_summary_*.json file (excluding checkpoints), or None."""
    if not os.path.isdir(tech_path):
        return None
    files = sorted([
        f for f in os.listdir(tech_path)
        if f.endswith('.json') and 'summary' in f and 'checkpoint' not in f
    ])
    return files[-1] if files else None


def _try_path(d, path):
    """Walk a nested-dict path; return stripped string or ''."""
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return ''
        cur = cur.get(k)
        if cur is None:
            return ''
    return cur.strip() if isinstance(cur, str) else ''


def extract_final_text(info, technique):
    """Get the 'final output' text for one video record (model-agnostic).

    Strategy: try simple top-level fields first (Claude's flat schema),
    then technique-specific nested paths (GPT's nested schema).
    For META/SC, concatenate per-batch analysis text.
    """
    if not isinstance(info, dict):
        return ''

    # 1) Top-level fallbacks (work for Claude on most techniques)
    for k in ('final_analysis', 'final_answer', 'final_report', 'analysis'):
        v = info.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 2) Technique-specific extraction
    if technique == 'CHAIN-OF-THOUGHT':
        return _try_path(info, ['cot_steps', 'final_answer'])

    if technique == 'LEAST-TO-MOST':
        return (_try_path(info, ['stages', 'level5_final_report'])
                or _try_path(info, ['stages', 'level4_classification']))

    if technique == 'REACT':
        return _try_path(info, ['react_log', 'final_answer'])

    if technique == 'SEQUENTIAL':
        return (_try_path(info, ['stages', 'stage4_final_summary'])
                or _try_path(info, ['stages', 'stage3_crime_classification']))

    if technique == 'TRUE-ITERATIVE':
        return (_try_path(info, ['iteration_log', 'final_report'])
                or _try_path(info, ['convergence_summary', 'final_report']))

    if technique == 'META-PROMPTING':
        # Concatenate per-batch responses; prepend final_label header
        br = info.get('batch_results', [])
        if isinstance(br, list):
            parts = []
            for b in br:
                if isinstance(b, dict):
                    r = b.get('response') or b.get('analysis') or b.get('text', '')
                    if isinstance(r, str) and r.strip():
                        parts.append(r.strip())
            if parts:
                lbl = info.get('final_label', '')
                conf = info.get('final_avg_confidence', '')
                header = f'CLASSIFICATION: {lbl}  (avg confidence {conf})\n\n' if lbl else ''
                return header + '\n\n---\n\n'.join(parts)

    if technique == 'SELF-CONSISTENCY':
        # For each batch, take one sample whose label matches winner_label
        br = info.get('batch_results', [])
        if isinstance(br, list):
            parts = []
            for b in br:
                if not isinstance(b, dict):
                    continue
                winner = b.get('winner_label')
                samples = b.get('samples', [])
                picked = None
                if isinstance(samples, list):
                    for s in samples:
                        if isinstance(s, dict) and s.get('label') == winner:
                            picked = s
                            break
                    if picked is None and samples and isinstance(samples[0], dict):
                        picked = samples[0]
                if picked:
                    r = picked.get('reasoning') or picked.get('response', '')
                    if isinstance(r, str) and r.strip():
                        parts.append(r.strip())
            if parts:
                lbl = info.get('final_label', '')
                conf = info.get('final_avg_confidence', '')
                header = f'CLASSIFICATION: {lbl}  (avg confidence {conf})\n\n' if lbl else ''
                return header + '\n\n---\n\n'.join(parts)

    return ''


def _load_checkpoint_results(tech_path):
    """If a *_checkpoint.json exists with {'results': {video: {...}}} structure,
    return that results dict; else {}."""
    if not os.path.isdir(tech_path):
        return {}
    ckpts = [f for f in os.listdir(tech_path) if f.endswith('checkpoint.json')]
    if not ckpts:
        return {}
    fpath = os.path.join(tech_path, ckpts[0])
    try:
        with open(fpath, encoding='utf-8') as f:
            obj = json.load(f)
    except Exception:
        return {}
    if isinstance(obj, dict):
        r = obj.get('results')
        if isinstance(r, dict):
            return r
    return {}


def _load_summary_based(model_name, tech_folder, tech_label):
    """Generic loader for Claude & GPT - merges *_summary_*.json with
    *_checkpoint.json's `results` dict (checkpoint wins on conflict)."""
    tech_path = os.path.join(BASE_PATHS[model_name], tech_folder)

    summary_data = {}
    fname = _latest_summary_file(tech_path)
    if fname:
        try:
            with open(os.path.join(tech_path, fname), encoding='utf-8') as f:
                summary_data = json.load(f)
        except Exception as e:
            print(f'  [{model_name}/{tech_label}] ERROR reading {fname}: {e}')
            summary_data = {}

    ckpt_results = _load_checkpoint_results(tech_path)

    merged = {}
    if isinstance(summary_data, dict):
        merged.update(summary_data)
    if isinstance(ckpt_results, dict):
        merged.update(ckpt_results)

    if not merged:
        return {}

    results = {}
    for video_key, info in merged.items():
        text = extract_final_text(info, tech_folder)
        if text:
            results[video_key] = text
    return results


def load_claude_technique(tech_folder, tech_label):
    return _load_summary_based('Claude', tech_folder, tech_label)


def load_gpt_technique(tech_folder, tech_label):
    return _load_summary_based('GPT', tech_folder, tech_label)


# --- Gemini config (per-technique file filter, per-video JSON files) -----
GEMINI_FOLDER_REMAP = {'TRUE-ITERATIVE': 'ITERATIVE'}

GEMINI_FILE_FILTER = {
    'CHAIN-OF-THOUGHT': lambda f: '_cot_synthesis_' in f or '_cot_complete_' in f,
    'LEAST-TO-MOST':    lambda f: '_ltm_synthesis_' in f or '_ltm_complete_' in f,
    'META-PROMPTING':   lambda f: '_meta_comprehensive_synthesis_' in f,
    'REACT':            lambda f: '_react_synthesis_' in f or '_react_complete_' in f,
    'SELF-CONSISTENCY': lambda f: '_self_consistency_' in f and 'sample' not in f,
    'SEQUENTIAL':       lambda f: '_sequential_synthesis_' in f or '_sequential_complete_' in f,
    'TRUE-ITERATIVE':   lambda f: ('_true_iterative_analysis_' in f
                                   or '_iterative_synthesis_' in f
                                   or '_iterative_complete_' in f
                                   or '_iterative_final_' in f),
    'ZERO':             lambda f: '_zero_shot_analysis_' in f,
}


def _gemini_file_rank(fname):
    if '_synthesis_' in fname or '_comprehensive_synthesis_' in fname: return 0
    if '_complete_'  in fname: return 1
    if '_final_'     in fname: return 2
    return 3


def _gemini_extract_text(obj, min_len=150):
    """Recursive deep search for the 'final analysis' string in a Gemini JSON."""
    # ---- Special case: TRUE-ITERATIVE nested structure ----
    if isinstance(obj, dict):
        tia = obj.get('True_Iterative_Analysis')
        if isinstance(tia, dict):
            iters = tia.get('all_iterations')
            if isinstance(iters, dict) and iters:
                def _it_num(k):
                    try:    return int(str(k).rsplit('_', 1)[-1])
                    except: return -1
                last_key = max(iters.keys(), key=_it_num)
                last = iters[last_key]
                if isinstance(last, dict):
                    resp = last.get('response')
                    if isinstance(resp, str) and resp.strip():
                        return resp.strip()
    TIER1 = ('synthesis', 'final_analysis', 'final_answer', 'final_report',
             'final_summary', 'final_label', 'comprehensive', 'final_')
    TIER2 = ('analysis', 'report', 'summary', 'reasoning', 'response')

    best = {'t1': '', 't2': '', 'any': ''}

    def walk(x, parent_key=''):
        if isinstance(x, str):
            if len(x) < min_len:
                return
            pk = (parent_key or '').lower()
            if len(x) > len(best['any']):
                best['any'] = x
            if any(p in pk for p in TIER1) and len(x) > len(best['t1']):
                best['t1'] = x
            elif any(p in pk for p in TIER2) and len(x) > len(best['t2']):
                best['t2'] = x
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, parent_key)

    walk(obj)
    return best['t1'] or best['t2'] or best['any']


def load_gemini_technique(tech_folder, tech_label):
    """Gemini: per-video JSON files; pick the best 'final' file per video,
    extract the longest synthesis-like string recursively."""
    actual_folder = GEMINI_FOLDER_REMAP.get(tech_folder, tech_folder)
    tech_path = os.path.join(BASE_PATHS['Gemini'], actual_folder)
    if not os.path.isdir(tech_path):
        return {}
    filt = GEMINI_FILE_FILTER.get(tech_folder, lambda f: False)
    files = [f for f in os.listdir(tech_path) if f.endswith('.json') and filt(f)]

    per_video_files = defaultdict(list)
    for fname in files:
        vid = video_id_from_filename(fname)
        if vid:
            per_video_files[vid].append(fname)

    results = {}
    for vid, fnames in per_video_files.items():
        fnames.sort(key=lambda f: (_gemini_file_rank(f), f), reverse=False)
        best_rank = _gemini_file_rank(fnames[0])
        same_rank = [f for f in fnames if _gemini_file_rank(f) == best_rank]
        chosen = sorted(same_rank)[-1]
        fpath = os.path.join(tech_path, chosen)
        try:
            with open(fpath, encoding='utf-8') as f:
                obj = json.load(f)
        except Exception:
            continue
        text = _gemini_extract_text(obj)
        if text and text.strip():
            results[vid] = text.strip()
    return results


LOADER_MAP = {
    'Claude': load_claude_technique,
    'GPT':    load_gpt_technique,
    'Gemini': load_gemini_technique,
}

# =====================================================================
# BUILD TRIPLETS
# =====================================================================
rows = []
print(f'Building triplets for {len(LOADER_MAP)} models x {len(TECHNIQUES)} techniques...')
print()

for model, loader_fn in LOADER_MAP.items():
    for tech_folder, tech_label in TECHNIQUES.items():
        outputs    = loader_fn(tech_folder, tech_label)
        added      = 0
        skipped_gt = 0
        for raw_key, output_text in outputs.items():
            gt_key = resolve_gt_key(raw_key, gt_data)
            if gt_key is None:
                skipped_gt += 1
                continue
            gt_info = gt_data[gt_key]
            rows.append({
                'model':                 model,
                'technique':             tech_label,
                'video':                 gt_key,
                'crime_type':            gt_info['crime_type'],
                'ground_truth':          ' '.join(gt_info['sentences']),
                'model_output':          output_text,
                'model_output_full_len': len(output_text),
            })
            added += 1
        status = 'OK' if added > 0 else 'EMPTY'
        print(f'  [{model:6s} / {tech_label:<18s}] {status:5s}  loaded={len(outputs):4d}  matched_gt={added:4d}  no_gt={skipped_gt}')

df_triplets = pd.DataFrame(rows)

before = len(df_triplets)
df_triplets = df_triplets.drop_duplicates(
    subset=['model','technique','video'], keep='last'
).reset_index(drop=True)
if before != len(df_triplets):
    print(f'\nDropped {before - len(df_triplets)} duplicate rows.')

df_triplets.to_csv(TRIPLETS_CACHE_PATH, index=False)

print()
print('=' * 70)
print(f'Triplets cache: {TRIPLETS_CACHE_PATH}')
print(f'Total rows:     {len(df_triplets):,}')
print(f'Judge calls:    {len(df_triplets) * 2:,}  (2 cross-judges per row)')
print()
print('Rows per (model x technique):')
print(df_triplets.groupby(['model','technique']).size().unstack(fill_value=0).to_string())


# =====================================================================
# JUDGE MODEL CALL  -  IMPLEMENT THIS (API removed on purpose)
# =====================================================================
def call_judge_model(judge_name: str, system_prompt: str, user_prompt: str) -> str:
    """
    Single entry point for every judge. Plug in your own backend here.

    `judge_name` is one of ALL_JUDGES ('Claude', 'GPT', 'Gemini') in case you
    want to route each judge to a different model. Must return the judge's raw
    text response (expected to contain the JSON object described in
    JUDGE_SYSTEM) as a string.
    """
    raise NotImplementedError(
        "Implement call_judge_model() with your own backend. It receives the "
        "judge name, a system prompt, and a user prompt, and must return the "
        "model's raw text response as a string."
    )


JUDGE_SYSTEM = """You are an expert forensic video analysis evaluator.
Your task is to detect hallucinations in an AI model's analysis of a crime video.
You will be given:
  1. GROUND TRUTH: the human-annotated description of what actually happened.
  2. MODEL OUTPUT: the AI system's analysis of the same video.

Evaluate whether the MODEL OUTPUT contains any of the following hallucination types.
For each type respond ONLY with 0 (not present) or 1 (present).

Hallucination types:
  H1 SCENE_FABRICATION    - invents setting details not in ground truth
  H2 CRIME_MISCLASSIFICATION - identifies the wrong crime type
  H3 CRIME_MISSED         - fails to mention the primary crime in ground truth
  H4 SEVERITY_MINIMIZATION - describes crime as less serious than ground truth
  H5 ENTITY_FABRICATION   - invents people, vehicles, or objects not in ground truth
  H6 PHANTOM_ACTORS       - adds perpetrators or victims not in ground truth

Respond ONLY with a JSON object in this exact format, no extra text:
{
  "H1": 0,
  "H2": 0,
  "H3": 0,
  "H4": 0,
  "H5": 0,
  "H6": 0,
  "reasoning": "brief one-sentence explanation"
}
"""

def make_judge_prompt(ground_truth: str, model_output: str) -> str:
    return (
        f'GROUND TRUTH:\n{ground_truth[:1500]}\n\n'
        f'MODEL OUTPUT:\n{model_output[:3000]}\n\n'
        'Now evaluate for hallucinations and return the JSON.'
    )

def _retry_call(fn, max_retries=7, base_delay=4.0, max_delay=300.0):
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            err_low = str(e).lower()
            is_rate    = any(x in err_low for x in ['429','rate limit','quota','resource exhausted','resource_exhausted'])
            is_server  = any(x in err_low for x in ['500','502','503','504','server error','unavailable','overloaded'])
            is_timeout = any(x in err_low for x in ['timeout','timed out','deadline'])
            is_trunc   = 'truncated response' in err_low
            if (is_rate or is_server or is_timeout or is_trunc) and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                tqdm.write(f'    [{type(e).__name__} attempt {attempt+1}/{max_retries}] sleeping {delay:.0f}s | {str(e)[:200]}')
                time.sleep(delay)
            else:
                raise
    if last_err:
        raise last_err
    return None

def parse_judge_response(text: str) -> dict:
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass
    if obj is not None:
        labels = {}
        for h in ['H1','H2','H3','H4','H5','H6']:
            val = obj.get(h, -1)
            labels[h] = int(bool(val)) if val in (0, 1, True, False) else -1
        labels['reasoning'] = str(obj.get('reasoning', ''))
        n_miss = sum(1 for h in ['H1','H2','H3','H4','H5','H6'] if labels[h] == -1)
        if n_miss > 0:
            raise ValueError(f'Truncated response: {n_miss} of 6 fields missing')
        return labels
    # Regex fallback
    labels = {}
    for h in ['H1','H2','H3','H4','H5','H6']:
        m = re.search(r'["\']?' + h + r'["\']?\s*:\s*([01])', text)
        labels[h] = int(m.group(1)) if m else -1
    rm = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']+)["\']', text)
    labels['reasoning'] = rm.group(1) if rm else 'parsed via regex fallback'
    n_miss = sum(1 for h in ['H1','H2','H3','H4','H5','H6'] if labels[h] == -1)
    if n_miss > 0:
        raise ValueError(f'Truncated response: {n_miss} of 6 fields missing')
    return labels

print('Prompt + retry helper + parse_judge_response ready.')


# =====================================================================
# JUDGE WRAPPERS  (all route through call_judge_model)
# =====================================================================
def _make_judge_caller(judge_name):
    def _judge(ground_truth: str, model_output: str) -> dict:
        prompt = make_judge_prompt(ground_truth, model_output)
        return parse_judge_response(
            _retry_call(lambda: call_judge_model(judge_name, JUDGE_SYSTEM, prompt))
        )
    return _judge

call_claude_judge = _make_judge_caller('Claude')
call_gpt_judge    = _make_judge_caller('GPT')
call_gemini_judge = _make_judge_caller('Gemini')

JUDGE_CALL_MAP = {'Claude': call_claude_judge, 'GPT': call_gpt_judge, 'Gemini': call_gemini_judge}
print('Judge functions ready.')


# =====================================================================
# OPTIONAL QUICK TEST  (requires call_judge_model to be implemented)
# =====================================================================
def quick_test():
    test_gt  = 'A person enters a store and takes items without paying.'
    test_out = 'The video shows shoplifting at a convenience store.'
    print('Claude:', call_claude_judge(test_gt, test_out))
    print('GPT:   ', call_gpt_judge(test_gt, test_out))
    print('Gemini:', call_gemini_judge(test_gt, test_out))


# =====================================================================
# RUN THE PANEL
# =====================================================================
def load_checkpoint():
    if os.path.exists(PANEL_CHECKPOINT):
        with open(PANEL_CHECKPOINT) as f:
            return json.load(f)
    return {}

def save_checkpoint(ckpt):
    with open(PANEL_CHECKPOINT, 'w') as f:
        json.dump(ckpt, f)

def run_judge_panel(df, max_workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    ckpt = load_checkpoint()
    ckpt_lock  = threading.Lock()
    rows_lock  = threading.Lock()
    rows_out   = []
    tasks      = []
    cached_rows = []
    h_keys = ['H1','H2','H3','H4','H5','H6']

    for idx, row in df.iterrows():
        model      = row['model']
        technique  = row['technique']
        video      = row['video']
        crime_type = row['crime_type']
        gt         = str(row['ground_truth'])
        output     = str(row['model_output'])
        for judge in judges_for(model):
            ck_key   = f'{idx}_{judge}'
            base_row = {
                'row_idx': idx, 'model': model, 'technique': technique,
                'video': video, 'crime_type': crime_type, 'judge': judge,
            }
            if ck_key in ckpt:
                labels = ckpt[ck_key]
                cached_rows.append({
                    **base_row,
                    **{h: labels.get(h, -1) for h in h_keys},
                    'reasoning': labels.get('reasoning', ''),
                })
            else:
                tasks.append((idx, base_row, judge, gt, output, ck_key))

    n_judges = len(JUDGE_FILTER) if JUDGE_FILTER else 2
    print(f'\nPanel: {len(df):,} triplets x {n_judges} judges = {len(tasks)+len(cached_rows):,} calls')
    print(f'Already cached: {len(cached_rows):,}')
    print(f'Remaining:      {len(tasks):,}')
    print(f'Workers:        {max_workers}')

    rows_out.extend(cached_rows)
    if not tasks:
        print('Nothing to do.')
        return pd.DataFrame(rows_out)

    pbar = tqdm(total=len(tasks), desc='Judge calls', unit='call', dynamic_ncols=True)

    def _process_task(task):
        idx, base_row, judge, gt, output, ck_key = task
        try:
            labels = JUDGE_CALL_MAP[judge](gt, output)
        except Exception as e:
            tqdm.write(f'  ERROR [{judge}] row {idx}: {str(e)[:200]}')
            labels = {h: -1 for h in h_keys}
            labels['reasoning'] = f'ERROR: {str(e)[:300]}'
        row_out = {
            **base_row,
            **{h: labels.get(h, -1) for h in h_keys},
            'reasoning': labels.get('reasoning', ''),
        }
        with ckpt_lock:
            ckpt[ck_key] = labels
            if len(ckpt) % 25 == 0:
                save_checkpoint(ckpt)
        with rows_lock:
            rows_out.append(row_out)
        return row_out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_task, t) for t in tasks]
        for _ in as_completed(futures):
            pbar.update(1)
    pbar.close()
    with ckpt_lock:
        save_checkpoint(ckpt)
    return pd.DataFrame(rows_out)

# ── EXECUTE ──────────────────────────────────────────────────────────
if os.path.exists(PANEL_RAW_PATH):
    df_panel_raw = pd.read_csv(PANEL_RAW_PATH)
    print(f'Loaded existing raw labels: {len(df_panel_raw):,} rows')
else:
    df_panel_raw = run_judge_panel(df_triplets, max_workers=8)
    df_panel_raw.to_csv(PANEL_RAW_PATH, index=False)
    print(f'\nSaved raw labels: {len(df_panel_raw):,} rows')

df_panel_raw.to_csv(JUDGE_LABELS_PATH, index=False)


# =====================================================================
# INTER-JUDGE AGREEMENT
# =====================================================================
def cohens_kappa(y1, y2):
    valid = [(a, b) for a, b in zip(y1, y2) if a in (0, 1) and b in (0, 1)]
    if len(valid) < 2:
        return float('nan')
    a = [v[0] for v in valid]
    b = [v[1] for v in valid]
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_a = (sum(a)/n)*(sum(b)/n) + ((n-sum(a))/n)*((n-sum(b))/n)
    return (p_o - p_a) / (1 - p_a) if p_a < 1 else 1.0

def compute_inter_judge_agreement(df_raw):
    h_cols  = ['H1','H2','H3','H4','H5','H6']
    judges  = df_raw['judge'].unique().tolist()
    pairs   = [(judges[i], judges[j])
               for i in range(len(judges)) for j in range(i+1, len(judges))]
    results = {}
    for j1, j2 in pairs:
        pk = f'{j1}_vs_{j2}'
        results[pk] = {}
        d1 = df_raw[df_raw['judge']==j1].set_index('row_idx')
        d2 = df_raw[df_raw['judge']==j2].set_index('row_idx')
        ci = d1.index.intersection(d2.index)
        for h in h_cols:
            results[pk][h] = round(cohens_kappa(d1.loc[ci,h].tolist(), d2.loc[ci,h].tolist()), 4)
        y1 = [v for h in h_cols for v in d1.loc[ci,h].tolist()]
        y2 = [v for h in h_cols for v in d2.loc[ci,h].tolist()]
        results[pk]['overall'] = round(cohens_kappa(y1, y2), 4)
    return results

print('Computing inter-judge agreement...')
agreement = compute_inter_judge_agreement(df_panel_raw)
with open(JUDGE_AGREEMENT_PATH, 'w') as f:
    json.dump(agreement, f, indent=2)

print("\nInter-judge agreement (Cohen's Kappa):")
for pair, kappas in agreement.items():
    print(f'  {pair}:')
    for h, k in kappas.items():
        print(f'    {h}: {k:.4f}')


# =====================================================================
# AGGREGATE PANEL VOTES
# =====================================================================
def majority_vote(vals):
    valid = [v for v in vals if v in (0, 1)]
    if not valid:
        return -1
    return 1 if sum(valid) >= len(valid) / 2 else 0

def aggregate_panel(df_raw, df_triplets):
    h_cols  = ['H1','H2','H3','H4','H5','H6']
    agg = (df_raw
           .groupby(['row_idx','model','technique','video','crime_type'])[h_cols]
           .agg(majority_vote)
           .reset_index())
    inv_map = {v: k for k, v in HTYPE_MAP.items()}
    for h in h_cols:
        agg[inv_map[h]] = agg[h]
    df_t = df_triplets.reset_index().rename(columns={'index': 'row_idx'})
    agg  = agg.merge(
        df_t[['row_idx','ground_truth','model_output','model_output_full_len']],
        on='row_idx', how='left'
    )
    agg['hallucination_count'] = agg[h_cols].apply(
        lambda r: sum(v for v in r if v == 1), axis=1)
    agg['any_hallucination'] = (agg['hallucination_count'] > 0).astype(int)
    return agg

print('Aggregating votes...')
df_full = aggregate_panel(df_panel_raw, df_triplets)
df_full.to_csv(FULL_LABELED_PATH, index=False)
print(f'Saved: {FULL_LABELED_PATH}  ({len(df_full):,} rows)')


# =====================================================================
# SUMMARY
# =====================================================================
h_cols  = ['H1','H2','H3','H4','H5','H6']
h_names = {
    'H1': 'Scene Fabrication',
    'H2': 'Crime Misclassification',
    'H3': 'Crime Missed',
    'H4': 'Severity Minimization',
    'H5': 'Entity Fabrication',
    'H6': 'Phantom Actors',
}

valid_df = df_full[df_full[h_cols].apply(lambda r: all(v >= 0 for v in r), axis=1)]

print('=' * 70)
print('FULL DATASET — Hallucination Summary (v7, 8 techniques)')
print('=' * 70)
print(f'\nValid rows: {len(valid_df):,} of {len(df_full):,}')

print('\nHallucination rate by type:')
for h in h_cols:
    rate = valid_df[h].mean() * 100
    print(f'  {h} {h_names[h]:<26} {rate:5.1f}%')

print('\nOverall hallucination rate by model:')
print(valid_df.groupby('model')['any_hallucination'].mean().mul(100).round(1).to_string())

print('\nOverall hallucination rate by technique:')
print(valid_df.groupby('technique')['any_hallucination'].mean().mul(100).round(1).to_string())

print('\nHallucination rate by (model x technique):')
pivot = (valid_df.groupby(['model','technique'])['any_hallucination']
         .mean().mul(100).round(1).unstack(fill_value=0))
print(pivot.to_string())

print('\nRow count per (model x technique):')
print(valid_df.groupby(['model','technique']).size().unstack(fill_value=0).to_string())

print('\n' + '=' * 70)
print('Done. All output files saved to:', OUTPUT_DIR)
print('=' * 70)
