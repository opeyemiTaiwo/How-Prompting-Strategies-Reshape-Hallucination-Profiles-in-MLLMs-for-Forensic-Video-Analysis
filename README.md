# Beyond Accuracy: A Type-Aware Taxonomy and Evaluation Framework for Prompted Hallucination in Forensic Multimodal LLMs

Replication package for the ICDM 2026 submission. **Anonymous — under triple-blind review.**

This repository contains the code, panel-derived labels, and evaluation data needed to
reproduce the tables and figures in the paper: a six-type hallucination taxonomy
(H1–H6) evaluated over 807 anomalous UCF-Crime videos × 8 prompting techniques × 3
frontier multimodal LLMs, producing 19,361 forensic reports scored by a three-judge
LLM panel, plus a previous-generation 19-video pilot for the cross-generation analysis.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── code/
│   ├── generation/                 # report generation, 8 techniques (call_model() stub)
│   ├── judging/                    # panel-of-judges labeling (2 judges/report, no self-eval)
│   ├── analysis/
│   │   ├── full_study_output_analytics.py   # corpus stats, refusal, RQ3 correlations
│   │   ├── limitation_v_vi.py               # RQ1 logistic + RQ5 multi-turn vs single-turn
│   │   ├── human_spotcheck_analysis.py      # human-vs-panel agreement (Rater A / Rater B)
│   │   └── cross_generation_analysis.py     # Table III + Spearman persistence
│   └── figures/                    # plotting scripts for fig1–fig8
├── data/
│   ├── main_study_807/             # current-generation main study
│   │   ├── panel_raw_judge_labels_full.csv  # per-judge H1–H6 (source of truth)
│   │   ├── all_triplets_cache.csv(.gz)      # report corpus (see "Large files")
│   │   ├── full_labeled_dataset_full.csv(.gz)
│   │   ├── inter_judge_agreement_full.json
│   │   ├── metrics/                # kappa_confidence_intervals.json, anova_results.json,
│   │   │                           # crime_type_analysis.*, ablation_results_full.json,
│   │   │                           # embeddings (see "Large files"), embedding_index.csv
│   │   └── human_validation/       # R2 (130-report) human labels
│   │       ├── human_labels.csv    # slim Rater A / Rater B H1–H6 labels
│   │       └── human_panel_agreement.json
│   └── pilot_19video/              # previous-generation cross-generation pilot
│       ├── panel_raw_judge_labels.csv
│       ├── all_triplets_cache.csv
│       └── human_validation/       # R1 (51-report) human labels
│           └── human_spotcheck_50_labels.csv
├── docs/
│   ├── rater_instructions.md       # human-validation rubric given to raters
│   └── judge_prompt.md             # panel judge prompt + forensic H1–H6 rubric
└── figures/                        # fig1–fig8, meth1 (pre-rendered)
```

---

## Datasets (not redistributed here)

The source video data are third-party datasets with their own licenses and are **not**
re-hosted in this repository. Obtain them from the original sources:

- **UCF-Crime** (surveillance videos): https://www.crcv.ucf.edu/projects/real-world/
- **UCA — UCF-Crime Annotation** (expert text descriptions used as ground truth):
  https://github.com/Xuange923/Surveillance-Video-Understanding

We release only our *derived* artifacts: the generated reports, the panel labels, and
the analysis code.

## Model versions

- **Main study (current generation):** Claude Opus 4.7, GPT-5.5, Gemini 3.1 Pro.
- **Pilot (previous generation, cross-generation analysis):** Claude Sonnet 4, GPT-4o,
  Gemini 2.0 Flash. In `data/pilot_19video/`, the `model` column is labeled by family
  (Claude / GPT / Gemini) but refers to these previous-generation versions.

Raters in the human-validation files are anonymized as **Rater A** and **Rater B**.

---

## Environment

```bash
pip install -r requirements.txt
```

Python 3.12. Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`,
`scikit-learn`, `sentence-transformers`, `openpyxl` (and `python-docx`, `tqdm`,
`matplotlib` for sheet/figure generation). API keys for the model providers are read
from environment variables and are **not** included; the generation/judging scripts use
a `call_model()` stub you can wire to your own credentials.

---

## Large files

Two corpus files and the embedding matrix exceed GitHub's per-file limits, so they are
shipped gzipped and/or split. All scripts read them transparently.

| File | Form provided | How to read |
|---|---|---|
| `all_triplets_cache.csv` (162 MB) | `*.csv.gz` (42 MB) or three per-model `*.csv.gz` | `pd.read_csv("…csv.gz")`, or concat the parts (below) |
| `full_labeled_dataset_full.csv` (162 MB) | same | same — *note: this is the corpus **plus** labels; it duplicates `all_triplets_cache.csv`, so you typically only need one of the two* |
| `embeddings_openai.npy` (119 MB, float32) | `embeddings_openai_fp16.npy` (60 MB) or three `*_fp16_part*.npy` | `np.load(...)`, or concat the parts (below) |

Reassemble per-model CSV parts:

```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("all_triplets_cache_*.csv.gz"))],
               ignore_index=True)
```

Reassemble embedding parts (kept in row order, aligned with `embedding_index.csv`):

```python
import numpy as np, glob
emb = np.concatenate([np.load(f) for f in sorted(glob.glob("embeddings_openai_fp16_part*.npy"))]).astype(np.float32)
```

> The embeddings are a lossy float16 downcast (cosine similarity matches float32 to five
> decimals — clustering/detector results are unaffected). For bit-exact float32, the
> embeddings are also available at the archival mirror, or can be regenerated from the
> corpus with `text-embedding-3-small`.

---

## Reproducing the paper

All hallucination rates use the paper's aggregation rule: two judges per report, **both**
must agree (logical AND), ties resolved to "no hallucination." Running
`limitation_v_vi.py` first prints a pooled ANY rate / mean count that should match
Table II (`91.1%`, `2.58`).

| Result in paper | Script | Inputs |
|---|---|---|
| Corpus composition (Table II), refusal rates, RQ3 Spearman correlations | `analysis/full_study_output_analytics.py` | `all_triplets_cache.csv`, `panel_raw_judge_labels_full.csv` |
| RQ1 logistic model×technique (vi) and RQ5 multi-turn vs single-turn (v) | `analysis/limitation_v_vi.py` | `panel_raw_judge_labels_full.csv` |
| Per-cell rates (master table), RQ1/RQ2 ANOVA | `analysis/full_study_output_analytics.py` / metrics JSONs | `panel_raw_judge_labels_full.csv`, `metrics/anova_results.json` |
| Detectability, feature ablation, training-size, granularity | metrics in `data/main_study_807/metrics/` (`ablation_results_full.json`, etc.) + embeddings | `embeddings_openai*.npy`, `embedding_index.csv` |
| Inter-judge κ + CIs (Table, forest plot) | `metrics/kappa_confidence_intervals.json` | `panel_raw_judge_labels_full.csv` |
| Crime-type variation (Table, heatmap) | `metrics/crime_type_analysis.*` | `panel_raw_judge_labels_full.csv` |
| Human validation (Table, R1/R2) | `analysis/human_spotcheck_analysis.py` | `data/*/human_validation/` (Rater A / Rater B labels) + panel labels |
| Cross-generation persistence (Table III, ρ) | `analysis/cross_generation_analysis.py` | `data/pilot_19video/`, `data/main_study_807/` |
| Methodology diagram (Fig. 1) | provided pre-rendered as `figures/meth1.png` | — |

Example commands (each script takes its data path as the first argument or an env var):

```bash
# RQ1 (vi) + RQ5 (v); prints the Table II check first
python code/analysis/limitation_v_vi.py data/main_study_807/panel_raw_judge_labels_full.csv

# corpus stats + RQ3 correlations
EVAL_DIR=data/main_study_807 OUT_DIR=out python code/analysis/full_study_output_analytics.py

# human validation (Rater A / Rater B)
EVAL_DIR=data/main_study_807 RATERS_DIR=data/main_study_807/human_validation \
  python code/analysis/human_spotcheck_analysis.py
```

(Scripts originally accepted a zipped archive path; point them at the corresponding CSV
or directory in this layout, or set the documented environment variables.)

---

## Notes

- **Anonymity:** this package is for blind review; it contains no author names,
  affiliations, or credentials. Raters are Rater A / Rater B.
- **Technique labels:** the pilot data uses finer technique labels (e.g., a per-model
  `REACT-*` split and separate `ITERATIVE`/`TRUE-ITERATIVE`) that the cross-generation
  analysis collapses to the paper's eight techniques.
- **No model weights:** the evaluated models are accessed via vendor APIs and are not
  redistributable; the lightweight detectors are trained in-script.

## License

Code: MIT (or your choice). Derived labels/data: CC BY 4.0 (or your choice). The
underlying UCF-Crime and UCA datasets remain under their original licenses.
