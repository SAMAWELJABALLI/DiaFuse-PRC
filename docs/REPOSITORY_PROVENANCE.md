# Provenance of the repository layout

The repository was built from the distributed one-click package, `DiaFuse-PRC/` with `original/` and `DiaFuse-PRC++/`. That package is the version whose dry runs are described in `tests/`. No file was deleted. The table lists every move and every content change.

## Moves

| Package path | Repository path | Content |
|---|---|---|
| `original/code/*` | `original/code/*` | unchanged (MD5 verified) |
| `original/archived/*` except `audit_anchors/` | `original/archived/*` | unchanged |
| `original/archived/audit_anchors/*.tsv` | **not in git** (dataset texts, see `original/README.md`) | — |
| `DiaFuse-PRC++/src/diafuse_pp.py` | `src/diafuse_pp_setting_A.py` | paths only (see below) |
| `DiaFuse-PRC++/src/diafuse_pp_ft.py` | `src/diafuse_pp_setting_B_lora.py` | paths only |
| `DiaFuse-PRC++/configs/defaults_settingA.json` | `configs/B12_setting_A/defaults.json` | unchanged |
| `DiaFuse-PRC++/configs/phase3_settingA.json` | `configs/B12_setting_A/screening_configs.json` | unchanged |
| `DiaFuse-PRC++/configs/defaults_settingB.json` | `configs/B12_setting_B_lora/defaults.json` | unchanged |
| `DiaFuse-PRC++/configs/phase3_settingB.json` | `configs/B12_setting_B_lora/screening_configs.json` | unchanged |
| `DiaFuse-PRC++/configs/{protocol.json,requirements_colab.txt}` | `configs/` | unchanged |
| `DiaFuse-PRC++/scripts/*.py` | `scripts/` | `run_one_click.py`: see below; the others: usage strings only |
| `DiaFuse-PRC++/evaluation/*.py` | `evaluation/` | `evaluate.py`: path; `tables.py`, `figures.py`, `report.py`: labels and folders, see below |
| `DiaFuse-PRC++/tests/*` | `tests/` | paths; new assertions |
| `DiaFuse-PRC++/docs/PP_DIAGNOSIS_REPORT.md` | `docs/` | unchanged |
| `DiaFuse_PRCPP_ONE_CLICK.ipynb` | `notebooks/` | repository detection and clone, see below |
| `RUN_INSTRUCTIONS.md` | `docs/` | rewritten for the GitHub workflow |
| original notebook `DiaFuse-PRC_Colab.ipynb` (provided by the project owner, 2026-10-08) | `original/notebook/` | unchanged; embedded code verified identical to `original/code/` |

## Changes relative to the distributed package

### Scientific changes

**None.** No model, loss, hyper-parameter, dataset, split, seed, metric, statistical test, selection rule or verdict rule was changed. `original/code/` is byte-identical to the received code (MD5-checked).

### Engineering changes

1. **Paths.** Configuration and source paths follow the repository layout. The original-code directory is resolved two levels up instead of three. Files were renamed: `diafuse_pp.py` → `src/diafuse_pp_setting_A.py` and `diafuse_pp_ft.py` → `src/diafuse_pp_setting_B_lora.py`.
2. **Run results.** Each model gets its own folder: `results/B11`, `results/B12_setting_A`, `results/B12pp_setting_A` and `results/B12_setting_B_lora`. Previously B12++ models shared `results/selected_model/`.
3. **Labels.** Table and figure labels now use the manuscript labels: "B11 Robust + calibrated (Shao et al. style)" and "B12 DiaFuse-PRC (Setting A)". DiaFuse-PRC++ systems are labelled "B12++ … (Setting A)" and "B12'++ … (Setting B, LoRA)"; the `++` marks that they are not the manuscript's systems.
4. **Text anchors.** Their absence is now a recorded warning instead of a crash, because the texts are not distributed. The other split checks are unchanged and still FATAL.
5. **Notebook.**
   - It clones the repository from `REPO_URL` (`https://github.com/SAMAWELJABALLI/diafuse-prc-pp.git`), or finds a copy in My Drive when `REPO_URL` is empty.
   - It copies the full checkout, so that git status reflects real modifications only.
6. **Labelling of Setting B.** The docstring of `src/diafuse_pp_setting_B_lora.py` now states "B12'++ (experimental implementation), NOT B12' from the paper" and lists the missing components. The comment change does not affect the code.
7. **Tests.** Test paths were updated. New tests: assertions on the final dry-run state, a test knob for Setting B selection, and a clone-mode notebook test.

### Reproducibility changes

1. **Code version.** `run_one_click.py` records the git commit, and whether there are uncommitted changes, in `run_state.json`. In clone mode the notebook pins the commit of the first session (`<run_dir>/state/code_commit.txt`), and later sessions check out that commit.
2. **Trace cards.** Each results folder gets `trace.json` and `README.md`, linking the git commit, code files (SHA-256), configuration, split manifest, seeds, checkpoints, predictions and metrics.
3. **B11 metadata.** The registry notes that `batch: 8` means `finetune.py --batch 16 // 2` anchors. The B11 command is unchanged.
4. **Line endings.** `.gitattributes` disables end-of-line conversion under `original/`, so a checkout on any OS keeps the MD5-checked bytes.
5. **Bug fixes in the orchestrator, made before the first commit and found by the dry runs.** Parallel workers wrote the registry through a shared temporary file (race condition). A selection failure was frozen as a negative result. Both are fixed: unique temporary files with a lock, and a failed selection is never frozen.
