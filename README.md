# DiaFuse-PRC++

Research code for **DiaFuse-PRC** (variety-aware multi-encoder fusion with hierarchical class–variety prototypes, cell-rebalanced contrastive learning, linguistic consistency and variety-wise calibration) and for **DiaFuse-PRC++**, a pre-registered attempt to improve it. The task is binary sentiment classification under synthetic-to-human script transfer for Tunisian Arabic. Models are trained and selected on Arabic-script TSAC data and synthetic Latin-script transliterations of it. They are then evaluated along a fixed chain of increasingly shifted test sets, S0–S6:

| Stage | Test set | Description |
|---|---|---|
| S0 | `ar` | TSAC, Arabic script |
| S1 | `gs` | seen transliteration generator |
| S2 | `gu` | unseen transliteration generator |
| S3 | `tl` | human TSAC-Latin |
| S4 | `tz` | TUNIZI |
| S5 | `na` | NArabizi, Algerian |
| S6 | `my` | MYC, Moroccan |

Script-controlled counterparts are also evaluated: S5AR (`naar`), S6AR (`myar`) and S6SYN (`gsmy`).

## Which code is which model

| Model (manuscript label) | Code that trains it | Configuration | Model card | Results folder of a run |
|---|---|---|---|---|
| **B11** — Robust + calibrated (Shao et al. style) | `original/code/finetune.py --system B11` (unmodified) | hard-coded defaults of `finetune.py` → [`configs/B11/`](configs/B11/README.md) | [`models/B11_robust_calibrated`](models/B11_robust_calibrated/README.md) | `results/B11/` |
| **B12** — DiaFuse-PRC (Setting A) | `original/code/diafuse_run.py` (unmodified; features from `extract.py`, synthetic Arabizi from `translit.py`) | `HP` in `diafuse_run.py`, mirrored in `configs/B12_setting_A/defaults.json` | [`models/B12_diafuse_setting_A`](models/B12_diafuse_setting_A/README.md) | `results/B12_setting_A/` |
| **B12′** — DiaFuse-PRC (Setting B, LoRA) | **no code: never executed** in the manuscript (row "TBO") | — | [`models/B12_prime_diafuse_setting_B_lora`](models/B12_prime_diafuse_setting_B_lora/README.md) | — |
| B12++ — DiaFuse-PRC++ (Setting A) | `src/diafuse_pp_setting_A.py` | `configs/B12_setting_A/` | [`models/B12pp_diafuse_pp_setting_A`](models/B12pp_diafuse_pp_setting_A/README.md) | `results/B12pp_setting_A/` |
| B12′++ — DiaFuse-PRC++ (Setting B, LoRA) | `src/diafuse_pp_setting_B_lora.py` | `configs/B12_setting_B_lora/` | [`models/B12_prime_diafuse_setting_B_lora`](models/B12_prime_diafuse_setting_B_lora/README.md) | `results/B12_setting_B_lora/` |

The manuscript's B12′ was specified but **not executed**, and no implementation of it exists in the original code. `src/diafuse_pp_setting_B_lora.py` is a **new, partial** LoRA implementation written for DiaFuse-PRC++. It has not been executed on a GPU yet, and it does not implement every component of the B12′ specification; its model card lists what is missing. Its results must never be reported as the manuscript's B12′.

## Status of each system

Statuses: **EXECUTED** (run, outputs archived), **VERIFIED** (checked in this repository), **NOT EXECUTED**, **PARTIAL**, **MISSING**.

| System | Implementation | Execution | Evidence |
|---|---|---|---|
| B11 Robust + calibrated | original, unmodified (VERIFIED: MD5) | EXECUTED in the archived reduced run (3 seeds); NOT EXECUTED with this repository | `original/archived/preds/B11`, `finetune_dev_log.json` |
| B12 DiaFuse-PRC (Setting A) | original, unmodified (VERIFIED: MD5) | EXECUTED in the archived reduced run (3 seeds, quick mode, no tuning); NOT EXECUTED with this repository | `original/archived/preds/B12`, `dev_scores.json` |
| B12′ DiaFuse-PRC (Setting B, LoRA) | MISSING (no original implementation exists) | NOT EXECUTED (manuscript and original notebook) | `models/B12_prime_diafuse_setting_B_lora/README.md` |
| B12++ DiaFuse-PRC++ (Setting A) | new research code; static and dry-run tests only | NOT EXECUTED | — |
| B12′++ DiaFuse-PRC++ (Setting B, LoRA) | experimental, PARTIAL with respect to the B12′ specification | NOT EXECUTED | missing components listed in its model card |

The archived run is the reduced run described in the manuscript as provisional: 3 seeds instead of 10, no tuning, splits reconstructed from the public releases. **No real training has been executed with this repository.** Only syntax checks, static audits and dry runs on synthetic data with stand-in experiments were performed (see `REPRODUCIBILITY.md`).

## Repository structure

```text
original/      reference implementation (read-only) and the archived 3-seed run
  code/        original pipeline: data, transliteration, features, B0–B12 trainer, B1/B11 fine-tuning, evaluation
  notebook/    the original Colab notebook that produced the archived run (provenance only)
  archived/    archived predictions and summaries of the reduced run, used to verify splits and reproduction
models/        one model card per system: exact code, configuration, entry point, status
src/           DiaFuse-PRC++ implementations (Setting A, Setting B / LoRA)
configs/       protocol (seeds, tolerances, selection and verdict rules), per-setting defaults and screening grids
scripts/       one-click orchestrator, model selection, gate diagnostics
evaluation/    per-seed metrics (original evaluator functions), statistics, tables, figures, report
notebooks/     DiaFuse_PRCPP_ONE_CLICK.ipynb, the single entry point on Google Colab
docs/          collaborator guide, diagnosis report, provenance of the repository layout
tests/         static audit and dry-run tests (synthetic data only)
data/, results/  empty in git; data are rebuilt from public releases, results go to Google Drive
```

## Installation

The pipeline is designed for **Google Colab with an A100 GPU**. Nothing needs to be installed by hand: the notebook installs the pinned packages of `configs/requirements_colab.txt` (`transformers==5.18.0`, plus missing packages only). It uses Colab's preinstalled PyTorch with CUDA.

On another machine you need a CUDA GPU with PyTorch, the packages of `configs/requirements_colab.txt`, `git`, and Internet access to GitHub and the Hugging Face Hub. The data repositories and the three encoders are downloaded automatically by `original/code/download_models.py`. Python versions actually used: 3.13.15 for the archived run and 3.13.16 for the repository tests; other versions have not been tested.

## Dataset reconstruction

No data are distributed. At stage 02, `original/code/download_models.py` fetches the public releases of TSAC, TUNIZI, NArabizi and MYC from GitHub. `original/code/prep_data.py` (unmodified) rebuilds every split: deduplication, script split, and synthetic transliterated views generated with `original/code/translit.py`. Stage 03 then verifies the rebuilt splits against the archived run, and any mismatch stops the run. See `data/README.md`.

## Quick start (Colab)

1. Open `notebooks/DiaFuse_PRCPP_ONE_CLICK.ipynb` in Google Colab (*File → Open notebook → GitHub*).
2. Select *Runtime → Change runtime type → A100 GPU*, then *Runtime → Run all*, and allow Google Drive access.
3. Wait for **RUN COMPLETED**. If the session stops, run all cells again: the run resumes.

Step-by-step instructions for a collaborator are in [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md).

Outside Colab, the same pipeline is run with:

```bash
python scripts/run_one_click.py --run_dir <output dir> --work_dir <local scratch dir> --mode FULL
```

## Pipeline stages (`scripts/run_one_click.py`)

| Stage | What it does |
|---|---|
| 00–01 | Environment record; MD5 check that `original/code` is unchanged; code version (git commit) recorded. |
| 02–03 | Data rebuilt with the original `prep_data.py`, then checked against the archived run: counts, id/label pairing, exact re-scoring of the archived predictions and leakage. Any mismatch stops the run (FATAL). |
| 04–05 | Encoder download with a checksum manifest; cached layer features (original `extract.py`). |
| 06 | **B12 (Setting A)** reproduction with the original trainer against the archived dev scores. Failure is FATAL. |
| 07 | **B11** reproduction with the original `finetune.py`. Failure is FATAL. |
| 08–09 | Screening of DiaFuse-PRC++ Setting A and Setting B on the development sets only, then selection by the pre-registered rule in `configs/protocol.json`. |
| 10–11 | Final runs with 10 seeds (B11, B12, selected models), then ablations with 5 seeds. |
| 12–16 | Evaluation (original `diafuse_eval.py`, plus per-seed metrics), protocol statistics, tables (CSV/TeX), figures (PNG/PDF), `FINAL_REPORT.md` and archives. |

Test sets are used only once, after selection.

## Reproducibility

- **Seeds** (`configs/protocol.json`): search 0–2, final 0–9, ablations 0–4, reproduction 0–2.
- **Configurations**:
  - `configs/B12_setting_A/defaults.json` holds the original B12 hyper-parameters.
  - `configs/B12_setting_B_lora/defaults.json` holds the LoRA settings.
  - `*/screening_configs.json` hold the pre-declared variants.
  - The run directory stores every resolved configuration (`experiment_registry.csv`, `SELECTED_CONFIG.json`).
- **Evaluation**: metric definitions are those of the unmodified `original/code/diafuse_eval.py`.
- **Checkpoints**: written to `checkpoints/<system>/seed<k>.pt` in the run directory on Google Drive, and zipped separately at the end. They are not stored in git.
- **Persistence**: every output is written to `My Drive/DiaFuse_PRCPP_RUN/`. `run_state.json` and the per-experiment state files make the run resumable.
- **Trace**: each model's results folder contains `trace.json` and `README.md`, linking the git commit, code files (SHA-256), configuration, split manifest, seeds, checkpoints, predictions and metrics.

Details and known limitations are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Not in git (intentionally)

| Excluded | Reason | Where it lives |
|---|---|---|
| raw corpora and rebuilt splits | redistribution terms not checked | rebuilt by the pipeline in `My Drive/DiaFuse_PRCPP_RUN/` |
| `original/archived/audit_anchors/` | contains corpus texts; redistribution rights not established | held by the project owner; optional input of the run |
| encoder weights, Hugging Face caches, cached features | large, re-downloadable | `My Drive/DiaFuse_PRCPP_RUN/cache/`, Colab local disk |
| checkpoints | large generated artifacts | `My Drive/DiaFuse_PRCPP_RUN/checkpoints/` |
| run outputs: predictions, metrics, logs, tables, figures, archives | generated | `My Drive/DiaFuse_PRCPP_RUN/` |
| the executed copy of the original notebook | identical cell sources; only its outputs differ | not committed |

## Limitations

- No GPU training, feature extraction or download has been executed with this repository.
- B12′ has no implementation. B12′++ is a partial experimental substitute and must not be reported as B12′.
- The archived reference run is reduced and provisional.
- Without the private text anchors, split identity relies on counts, id/label pairing, leakage checks and exact re-scoring of the archived predictions.
- B11 (`finetune.py`) does not save model weights. This is original behaviour.

## Reference implementations

`original/` contains the reference pipeline exactly as received; [`original/README.md`](original/README.md) explains its provenance. DiaFuse-PRC++ imports it but never modifies it, and each run verifies its MD5 checksums.

## Results

This repository contains no new experimental results. `original/archived/` holds the predictions and summaries of the reduced 3-seed run described in the manuscript as provisional. Results of a one-click run are written to Google Drive and are not committed here.

## License

**LICENSE: NOT SPECIFIED.** The repository owner has not chosen a license. Without one, default copyright applies, and reuse requires the authors' permission.

## Citation

Citation metadata (authors and their order, title, venue, date, DOI) have not been provided, so there is no `CITATION.cff`. The repository and the associated manuscript will need separate citation entries once this information is available.
