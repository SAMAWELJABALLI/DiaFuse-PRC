# Reproducibility manifest

Each entry below was checked against a file of this repository or against the environment in which the repository was built. Anything not checked is marked as such.

## Code version

- The commit of a checkout is given by `git rev-parse HEAD`. A file cannot contain the hash of the commit that adds it, so the hash of the initial commit is not written here.
- Every one-click run records the commit it used in `run_state.json` (`code_version`) and in each `results/<model>/trace.json`.
- With a GitHub URL, the notebook pins the commit of the first session in `<run_dir>/state/code_commit.txt`, and later sessions check out that same commit.
- `original/code/` is verified against `original/code/MD5SUMS_unmodified.txt` at the start of every session.

## Software

| | Archived run (`original/archived/environment.json`) | One-click run on Colab | Repository build and tests (this environment) |
|---|---|---|---|
| Python | 3.13.15 | Colab's | 3.13.16 |
| PyTorch | 2.11.0+cu130 | Colab's preinstalled build (not reinstalled; version recorded in `logs/environment.txt`) | **not installed**: no training code was executed |
| transformers | 5.18.0 | pinned `==5.18.0` (`configs/requirements_colab.txt`) | not installed |
| numpy / scipy / scikit-learn / pandas / matplotlib | not recorded | installed only if missing | 2.5.3 / 1.18.1 / 1.9.1 / 3.0.5 / 3.11.2 |
| deepsig | — | optional (almost-stochastic-order test) | not installed |

## Hardware

- The archived run used an NVIDIA A100-SXM4-40GB.
- The one-click run expects an A100 on Colab. Two parallel Setting A workers are used only if the GPU has ≥ 39 GB and RAM is ≥ 50 GB; otherwise the workers run sequentially.
- The repository build ran without a GPU.

## Data

- Public releases are fetched by `original/code/download_models.py`: TSAC, TUNIZI, NArabizi and MYC from GitHub.
- Splits are rebuilt by `original/code/prep_data.py`; no data are stored in git (`data/README.md`).
- Split identity with the archived run is verified at stage 03:
  - counts against `original/archived/prep_report.json`;
  - id/label pairing of the parallel views;
  - leakage;
  - exact re-scoring of the archived predictions against `original/archived/summary.json`;
  - text anchors, only when they are provided (see `original/README.md`).

  Any mismatch stops the run.

## Models

The encoders come from the Hugging Face Hub, through `original/code/download_models.py`:
- `FacebookAI/xlm-roberta-large`;
- `intfloat/multilingual-e5-large`;
- `sentence-transformers/LaBSE`.

They are cached on Google Drive with a checksum and revision manifest.

## Entry points

| Purpose | Entry point |
|---|---|
| Whole protocol (Colab) | `notebooks/DiaFuse_PRCPP_ONE_CLICK.ipynb` |
| Whole protocol (command line) | `python scripts/run_one_click.py --run_dir <dir> --work_dir <dir> [--mode FULL\|PRIORITY]` |
| B11 | `original/code/finetune.py --system B11` (`models/B11_robust_calibrated/`) |
| B12 DiaFuse-PRC (Setting A) | `original/code/diafuse_run.py`; inside the one-click run, configuration `V0` (`models/B12_diafuse_setting_A/`) |
| B12++ DiaFuse-PRC++ (Setting A) | `src/diafuse_pp_setting_A.py {search,final}` with `configs/B12_setting_A/` |
| B12′++ DiaFuse-PRC++ (Setting B, LoRA) | `src/diafuse_pp_setting_B_lora.py {search,final}` with `configs/B12_setting_B_lora/` |
| Selection | `scripts/select_config.py` |
| Evaluation | `original/code/diafuse_eval.py` and `evaluation/evaluate.py` |

## Configuration

`configs/protocol.json` defines:
- the seeds, the reproduction tolerances and the selection and verdict rules;
- the stage mapping S0–S6 and the explicit Table 5 mapping;
- the FULL and PRIORITY lists.

Per-setting defaults and screening grids are in `configs/B12_setting_A/` and `configs/B12_setting_B_lora/`. A run stores a copy of all of them, together with the resolved configuration of every experiment.

## Random seeds

From `configs/protocol.json`:
- search: 0, 1, 2;
- final: 0–9;
- ablations: 0–4;
- reproduction: 0, 1, 2.

## Large artifacts (not in git)

| Artifact | Location |
|---|---|
| Encoder weights | `My Drive/DiaFuse_PRCPP_RUN/cache/models/` |
| Raw corpora and rebuilt splits | `My Drive/DiaFuse_PRCPP_RUN/cache/raw/`, `.../data/` |
| Cached features | Colab local disk (`/content/pp_work`); recomputed after a disconnection |
| Checkpoints | `My Drive/DiaFuse_PRCPP_RUN/checkpoints/<system>/seed<k>.pt` and `final_archive/DiaFuse_PRCPP_CHECKPOINTS_<date>.zip` |
| Predictions, metrics, tables, figures, logs | `My Drive/DiaFuse_PRCPP_RUN/` and `final_archive/DiaFuse_PRCPP_FINAL_RESULTS_<date>.zip` |

No storage outside Google Drive has been decided.

## Tests performed when the repository was built

All tests were run without a GPU, on synthetic data, with stand-in experiments:
- `tests/static_audit.py`:
  - every `.py` compiles and all imports resolve;
  - no absolute user paths and no `input()`;
  - configs are valid JSON;
  - the notebook is valid and the paths it uses exist;
  - the code, configs and model cards of every model are present;
  - `original/code` matches its MD5 file.
- `tests/run_dry_test.sh`: three sessions (interruption → failures → resume), with assertions on the final state.
- A single session with Setting B selected and without text anchors.
- `tests/test_notebook.py`: the notebook cells executed with Colab stubbed.

## Known limitations

- **No training, feature extraction or download has been executed with this repository.** PyTorch was unavailable when it was built.
- The manuscript's **B12′ (Setting B, LoRA) was never executed and has no original implementation**. `src/diafuse_pp_setting_B_lora.py` is partial with respect to its specification; see `models/B12_prime_diafuse_setting_B_lora/README.md`.
- The archived run is a reduced run (3 seeds, quick mode, no tuning), described as provisional in the manuscript.
- `finetune.py` (B1/B11) does not save weights. This is original behaviour, kept unchanged.
- The text anchors are not distributed, so split identity relies on the other checks unless the anchors are supplied privately.
- No license has been selected and no citation metadata provided.
