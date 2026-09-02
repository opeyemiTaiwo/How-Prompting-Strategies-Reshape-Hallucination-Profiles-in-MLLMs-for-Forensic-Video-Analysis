# Beyond Accuracy: A Type-Aware Taxonomy and Evaluation Framework for Prompted Hallucination in Forensic Multimodal LLMs

Replication package for the IEEE ICDM 2026 paper, by Opeyemi Adeniran, Jamell
Dacon, Derrick Cook, Temitope Ajibola, Kelechi Nwachukwu, Peter Taiwo, and Kofi
Nyarko (Center for Equitable AI and Machine Learning Systems, Morgan State
University).

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
├── APPENDIX.md                          # supplementary material (judge prompt, rubric, extra tables)
├── code/
│   ├── generation/
│   │   └── prompting_techniques.py      # report generation, 8 techniques (call_model() stub)
│   ├── judging/
│   │   └── panel_judes.py               # panel-of-judges labeling (2 judges/report, no self-eval)
│   ├── analysis/
│   │   ├── full_study_output_analytics.py     # corpus stats, refusal, RQ3 correlations
│   │   ├── full_study_statistical_analyses.py # inter-judge κ CIs, per-crime χ², two-way ANOVA
│   │   ├── limitation_v_vi.py                  # RQ1 logistic + RQ5 multi-turn vs single-turn
│   │   ├── human_spotcheck_analysis.py         # human-vs-panel agreement (Rater A / Rater B)
│   │   └── cross_generation_analysis.py        # Table III + Spearman persistence
│   └── figures/
│       └── make_figures.py              # renders fig1–fig8 from the analysis outputs
├── data/
│   ├── main_study_807/                  # current-generation main study
│   │   ├── panel_raw_judge_labels_full.csv   # per-judge H1–H6 (source of truth)
│   │   ├── all_triplets_cache_{Claude,GPT,Gemini}.csv.gz  # report corpus, per model (see "Large files")
│   │   ├── metrics/
│   │   │   ├── anova_results.json             # two-way ANOVA (pre-computed)
│   │   │   ├── ablation_results_full.json     # detectability / feature ablation / training-size
│   │   │   ├── crime_type_analysis.csv        # per-crime rates
│   │   │   ├── embedding_index.csv            # row key (model, technique, video, crime_type)
│   │   │   ├── embeddings_openai_fp16_part{0,1,2}.npy  # report embeddings (see "Large files")
│   │   │   ├── complete_embeddings.py         # regenerate embeddings from the corpus
│   │   │   └── full_study_ablations_fixed.py  # detectability / ablation / training-size driver
│   │   └── human_validation/            # R2 (130-report) human labels
│   │       ├── Rater-A-scores-807.xlsx - Sheet1.csv
│   │       ├── Rater-B-scores-807.xlsx - Sheet1.csv
│   │       ├── Panel-A-scores-807.xlsx - Sheet1.csv   # panel labels aligned to Rater A's rows
│   │       └── Panel-B-Scores-807.xlsx - Sheet1.csv   # panel labels aligned to Rater B's rows
│   └── pilot_19video/                   # previous-generation cross-generation pilot
│       ├── panel_raw_judge_labels.csv
│       ├── all_triplets_cache.csv
│       └── human_validation/            # R1 (51-report) human labels
│           └── human_spotcheck_50.xlsx
├── docs/
│   ├── rater_instructions.md            # human-validation rubric given to raters
│   └── judge_prompt.md                  # panel judge prompt + forensic H1–H6 rubric
└── figures/                             # fig1–fig8, meth1 (pre-rendered)
```

---

## Datasets (not redistributed here)

The source video data are third-party datasets with their own licenses and are **not**
re-hosted in this repository. Obtain them from the original sources:

- **UCF-Crime** (surveillance videos): https://www.crcv.ucf.edu/projects/real-world/
- **UCA — UCF-Crime Annotation** (expert text descriptions used as ground truth):
  https://github.com/Xuange923/Surveillance-Video-Understanding
- **UCF-Crime Frames via UCA Crime Annotations** (the fixed 19-video subset used for the
  cross-generation pilot):
  https://www.kaggle.com/datasets/ahdabdulrahaman/ucf-crime-frames-via-uca-crime-annotations

We release only our *derived* artifacts: the generated reports, the panel labels, and
the analysis code.

## Model versions

- **Main study (current generation):** Claude Opus 4.7, GPT-5.5, Gemini 3.1 Pro.
- **Pilot (previous generation, cross-generation analysis):** Claude Sonnet 4, GPT-4o,
  Gemini 2.0 Flash. In `data/pilot_19video/`, the `model` column is labeled by family
  (Claude / GPT / Gemini) but refers to these previous-generation versions.

Human raters are released under the neutral labels **Rater A** and **Rater B** to
preserve rater privacy.

---

## Environment

```bash
pip install -r requirements.txt
```

Python 3.12+. Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`,
`scikit-learn`, `sentence-transformers`, `openpyxl` (and `python-docx`, `tqdm`,
`matplotlib` for sheet/figure generation). API keys for the model providers are read
from environment variables and are **not** included; the generation/judging scripts use
a `call_model()` stub you can wire to your own credentials.

---

## Large files

The report corpus and the embedding matrix exceed GitHub's per-file limits, so they are
shipped gzipped and split. All scripts read them transparently.

| Artifact | Form provided | How to read |
|---|---|---|
| Report corpus (`all_triplets_cache.csv`) | three per-model `all_triplets_cache_{Claude,GPT,Gemini}.csv.gz` | `pd.read_csv("…csv.gz")`, or concat the parts (below) |
| Report embeddings (`embeddings_openai.npy`, float32) | three `embeddings_openai_fp16_part*.npy` | `np.load(...)`, or concat the parts (below) |

Reassemble the per-model corpus:

```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("all_triplets_cache_*.csv.gz"))],
               ignore_index=True)
```

Reassemble the embedding parts (row order aligned with `embedding_index.csv`):

```python
import numpy as np, glob
emb = np.concatenate([np.load(f) for f in sorted(glob.glob("embeddings_openai_fp16_part*.npy"))]).astype(np.float32)
```

> The embeddings are a lossy float16 downcast (cosine similarity matches float32 to five
> decimals — clustering/detector results are unaffected). For bit-exact float32, regenerate
> them from the corpus with `text-embedding-3-small` via `metrics/complete_embeddings.py`.

Each report joins to its embedding and labels on the **composite content key**
`(model, technique, video, crime_type)` — the embedding `row_idx` is a positional index
and does **not** correspond to triplet-file ordering.

---

## Reproducing the paper

All hallucination rates use the paper's aggregation rule: two judges per report, **both**
must agree (unanimity / logical AND; a 2-judge split resolves to "no hallucination").
Running `limitation_v_vi.py` first prints a pooled ANY rate / mean count that should match
Table II (`91.1%`, `2.58`).

| Result in paper | Script | Key inputs |
|---|---|---|
| Corpus composition (Table I), refusal rates, RQ3 Spearman correlations | `analysis/full_study_output_analytics.py` | corpus `.csv.gz`, `panel_raw_judge_labels_full.csv` |
| RQ1 logistic model×technique and RQ5 multi-turn vs single-turn | `analysis/limitation_v_vi.py` | `panel_raw_judge_labels_full.csv` |
| Inter-judge κ + CIs, per-crime χ², two-way ANOVA (Tables + Fig. 4, Fig. 8) | `analysis/full_study_statistical_analyses.py` | `panel_raw_judge_labels_full.csv` → writes `metrics/kappa_confidence_intervals.json`, `metrics/crime_type_analysis.*`, `metrics/anova_results.json` |
| Detectability, feature ablation, training-size, granularity | `metrics/full_study_ablations_fixed.py` | `embeddings_openai_fp16_part*.npy`, `embedding_index.csv`, panel labels → `metrics/ablation_results_full.json` |
| Human validation (R1/R2) | `analysis/human_spotcheck_analysis.py` | `data/*/human_validation/` (Rater A / Rater B + panel labels) |
| Cross-generation persistence (Table III, ρ) | `analysis/cross_generation_analysis.py` | `data/pilot_19video/`, `data/main_study_807/` |
| All figures (fig1–fig8) | `figures/make_figures.py` | `metrics/` JSON/CSV outputs from the scripts above |
| Methodology diagram (Fig. 1 / meth1) | pre-rendered in `figures/` | — |

Typical order: (1) `full_study_statistical_analyses.py` and `full_study_ablations_fixed.py`
regenerate the `metrics/` JSON/CSV summaries; (2) `make_figures.py` renders the figures from
them. `anova_results.json`, `ablation_results_full.json`, and `crime_type_analysis.csv` are
also shipped pre-computed so the figures can be rebuilt without rerunning every analysis.

Example commands (each script takes its data path as the first argument or an env var):

```bash
# RQ1 + RQ5; prints the Table II check first
python code/analysis/limitation_v_vi.py data/main_study_807/panel_raw_judge_labels_full.csv

# inter-judge kappa CIs + crime chi-square + ANOVA (writes to metrics/)
python code/analysis/full_study_statistical_analyses.py data/main_study_807 data/main_study_807/metrics

# corpus stats + RQ3 correlations
EVAL_DIR=data/main_study_807 OUT_DIR=out python code/analysis/full_study_output_analytics.py

# human validation (Rater A / Rater B)
EVAL_DIR=data/main_study_807 RATERS_DIR=data/main_study_807/human_validation \
  python code/analysis/human_spotcheck_analysis.py

# figures
python code/figures/make_figures.py data/main_study_807/metrics figures
```

> Note: `full_study_output_analytics.py` expects a single `all_triplets_cache.csv`; concat
> the per-model `.csv.gz` parts (see "Large files") into that path first, or point the
> script at the concatenated frame. The human-validation scripts read the exported
> `Rater-*`/`Panel-*` sheets in `data/*/human_validation/`; set the documented environment
> variables to match this layout.

---

## Notes

- **Technique labels:** the pilot data uses finer technique labels (e.g., a per-model
  `REACT-*` split and separate `ITERATIVE`/`TRUE-ITERATIVE`) that the cross-generation
  analysis collapses to the paper's eight techniques.
- **No model weights:** the evaluated models are accessed via vendor APIs and are not
  redistributable; the lightweight detectors are trained in-script.

---

## Citation

```bibtex
@inproceedings{adeniran2026beyond,
  title     = {Beyond Accuracy: A Type-Aware Taxonomy and Evaluation Framework
               for Prompted Hallucination in Forensic Multimodal LLMs},
  author    = {Adeniran, Opeyemi and Dacon, Jamell and Cook, Derrick and
               Ajibola, Temitope and Nwachukwu, Kelechi and Taiwo, Peter and
               Nyarko, Kofi},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

## License

Code: MIT (or your choice). Derived labels/data: CC BY 4.0 (or your choice). The
underlying UCF-Crime and UCA datasets remain under their original licenses.
