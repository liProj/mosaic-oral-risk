# MOSAIC: occupational oral risk prediction

Research code for **Occupational Oral Risk Prediction through Interval Supervision
and Adaptive Auxiliary Learning**. MOSAIC combines a window-probability neural
head, primary-task-driven auxiliary health supervision, and neural–tree fusion.
The targets are detected caries, gingivitis, and their combined outcome within
each worker's recorded observation window.

The repository includes the original model and evaluation implementations with
portable paths, English figure generation, and saved **aggregate** results.
Raw health records, worker-level predictions, and model checkpoints are excluded.

## Installation

Use Python 3.11 or 3.12 in a virtual environment, then install:

```bash
git clone https://github.com/liProj/mosaic-oral-risk.git
cd mosaic-oral-risk
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
```

The requirements describe compatible dependency ranges, not an exact lockfile
from the historical training environment. Numerical results may vary with library
versions and hardware. GPU support is optional; install an appropriate PyTorch
build for your environment if using a GPU.

## Data and preparation

Obtain [Database Ocupacional](https://doi.org/10.6084/m9.figshare.27098296)
and place `Base Ocupacional.dta` in `data/`. The scripts verify 79,660 source rows,
12,049 eligible workers, and the outcome-specific counts before proceeding.

```bash
python run_pipeline.py prepare
```

Alternatively, set `MOSAIC_DATA_PATH` to the dataset file. Set `MOSAIC_ROOT` to
an alternative working directory if outputs should be stored outside the clone.
These environment variables are read when each process starts. Delete the
generated feature caches if changing the dataset or feature definitions.

The feature sets contain 18 (`paper`), 52 (`base`), and 114 (`traj`) columns.
Main experiments use `base`. Follow-up trajectory features are a retrospective
sensitivity analysis and are unavailable for prediction at baseline.

## Training and evaluation

```bash
python run_pipeline.py main
python run_pipeline.py ablation
python run_pipeline.py era
python run_pipeline.py extras
python run_pipeline.py stats
```

Training is computationally intensive. Run only the stages you need. `extras`
also accepts `--extra weights`, `hazard`, `conformal`, `importance`, `curve`, or
`featureset`. The `main` and `ablation` stages accept `--features base|paper|traj`.
Each main run evaluates repeated five-fold CV (three repeats), leave-one-baseline-
year-out (LOYO), and a chronological split (training through 2017; testing from
2018). LOYO can train on years later than the held-out year; it is distinct from
chronological validation. Inspect run logs for failed model/fold fits before
using generated summaries.

Results and worker-level predictions are saved under `results/` and are ignored
by Git. Original model labels such as `cox_multivar`, `mosaic`, and `lightgbm`
are retained for compatibility with the saved experiment files.

## Inspect the saved results without training

```bash
python run_pipeline.py reference-stats
python -m unittest discover -s tests -v
```

`reference-stats` recalculates paired corrected tests and Holm adjustment from
the shipped fold-level summaries, writing to `reference_analysis/`. It does not
require the health dataset or training. Its dependencies are NumPy, pandas,
SciPy, and scikit-learn; the unit tests use the same minimal set.

The following values are mean follow-up-stratified AUCs across CV folds:

| Outcome | Multivariable Cox | LightGBM | MOSAIC |
|---|---:|---:|---:|
| Caries | 0.6753 | 0.8299 | 0.8339 |
| Gingivitis | 0.6035 | 0.7746 | 0.7772 |
| Combined | 0.6699 | 0.8304 | 0.8288 |

MOSAIC improves on the specified linear reference under random validation.
Differences from LightGBM are small and not significant after adjustment.
Chronological caries sAUC is 0.5698 for MOSAIC and 0.6267 for LightGBM. Component
ablations support window supervision, budget-matched auxiliary weighting, and
representation transfer in selected settings. All protocols and ablations are
included in `reference_results/`; these files contain aggregate statistics,
not individual records.

## Paper figures

`code/plot_paper_figures.py` produces the 15 English paper figures as PDF and PNG.
Run `main`, `ablation`, `era`, and `extras` first. Copy the aggregate source-audit
file `reference_results/data_check.csv` to `results/data_check.csv`, then run:

```bash
python run_pipeline.py figures
```

Pooled ROC, precision–recall, calibration, and decision curves require
`results/experiments/oof_predictions.csv` from training. It is intentionally
excluded from this public repository because it contains worker-level outcomes
and predictions. Fold-mean AUCs and pooled AUCs summarize different quantities.

## Implementation map

| File | Purpose |
|---|---|
| `code/data_prep.py`, `code/features.py` | Cohort validation and feature construction |
| `code/mosaic.py` | Neural window head, interactions, auxiliary learning |
| `code/mosaic_stack.py`, `code/icboost.py` | Tree branches, fusion, window-likelihood objective |
| `code/baselines.py` | Comparator models |
| `code/experiment.py`, `code/extras.py` | Validation protocols, ablations, additional analyses |
| `code/metrics.py`, `code/stats_tests.py` | Risk metrics and corrected comparisons |
| `code/era_analysis.py` | Year-specific rates and association sensitivity analysis |
| `code/plot_paper_figures.py` | English publication figures |

See [method notes](docs/method_notes.md) for interpretation and validation details.
This is research software; the experiments do not establish a clinical referral
threshold or independent clinical deployment performance. Author and publication
metadata will be added when finalized.
