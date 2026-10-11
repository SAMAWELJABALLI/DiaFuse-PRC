# Reference implementation (read-only)

The files in `code/` are the original DiaFuse-PRC pipeline, exactly as received. Every DiaFuse-PRC++ experiment uses them as a reference and **they must not be modified**. `scripts/run_one_click.py` checks them against `code/MD5SUMS_unmodified.txt` at the start of every session and stops (FATAL) if one has changed. If something in them looks wrong, document it in an issue; do not change the file.

## `code/` — original pipeline

| File | Role | Used for |
|---|---|---|
| `prep_data.py` | rebuilds every split from the public releases: TSAC, TUNIZI, NArabizi, MYC; deduplication; script split; transliterated views | all systems |
| `translit.py` | rule-based Arabic → Latin (Arabizi) transliteration generators (seen / unseen) and the perturbation families | synthetic Arabizi, all systems |
| `download_models.py` | downloads the 3 encoders (Hugging Face) and the 4 corpora (GitHub) | all systems |
| `extract.py` | caches per-layer pooled sentence vectors of the frozen encoders | Setting A systems (B0, B2–B10, B12) |
| `diafuse_run.py` | trainer of B0, B2–B10 and **B12 DiaFuse-PRC (Setting A)**: LayerMix, convex gate with surface descriptor, hierarchical class × variety prototypes, cell-rebalanced SupCon, hard mixup, consistency, MMCE, logit adjustment, variety-wise temperature (`HP` dict) | **B12** |
| `finetune.py` | full fine-tuning of XLM-R-large: **B1** (cross-entropy) and **B11** (attention pooling, ℓ2 clean–perturbed consistency, temperature scaling) | **B11** |
| `diafuse_eval.py` | metrics: macro-F1, accuracy, minority F1, ECE (raw / global T / variety-wise T), Brier, NLL, AURC, exact match on paired sets, WGA; bootstrap and Holm | evaluation |
| `export_audit.py` | exports the native-speaker audit sheets | audits (not run by the one-click pipeline) |
| `gen_llm_perturb.py` | optional LLM perturbations | not executed (as in the manuscript) |
| `run_all.sh` | original order of the steps on a lab server; also gives the exact B11 invocation | documentation of the original run |

DiaFuse-PRC++ loads `diafuse_run.py` (`import diafuse_run as base`) and `diafuse_eval.py` as libraries. Copies or modified versions of these files exist nowhere in this repository.

## `notebook/` — the original Colab notebook

`notebook/DiaFuse-PRC_Colab.ipynb` (MD5 `291f8a65bbfde4b5a94c0b51928b2d5f`) is the notebook that produced the archived results, provided by the project owner without cell outputs. It writes the ten scripts of `code/` with `%%writefile` and runs them. The following was verified when building the repository:

- the code embedded in each `%%writefile` cell is identical to the corresponding file of `code/`;
- the separately provided `DiaFuse-PRC_code` archive and `download_models.py` are byte-identical to `code/`;
- the executed copy of the notebook (with outputs, not committed) has exactly the same cell sources.

The notebook states in cells 0 and 32 that B12′ (Setting B, LoRA) was **not executed**. It is kept for provenance only. Do not run it as part of DiaFuse-PRC++; the one-click notebook is `notebooks/DiaFuse_PRCPP_ONE_CLICK.ipynb`.

## `archived/` — the reduced run described in the manuscript

These files come from the executed original Colab notebook, `DiaFuse_PRC_Colab.ipynb`: the results bundle and `runs/` of the run of 2026-10-07/08 (quick mode, 3 seeds, A100). The manuscript describes this run as **provisional**. The one-click pipeline uses the files only to verify that a new run rebuilds the same splits and reproduces B12 and B11.

| File | Content |
|---|---|
| `preds/{B0,B6,B10,B11,B12}/<seed>/<set>.tsv` | archived test predictions: id, predicted label and probabilities (no texts) |
| `summary.json`, `ci.csv` | archived aggregated metrics |
| `dev_scores.json` | archived development scores (B12 reproduction targets) |
| `finetune_dev_log.json` | B1/B11 development macro-F1 per epoch. **Derived file**: parsed from the stdout of `finetune.py` in cell 25 of the executed notebook, because that run did not write these values to a file. |
| `prep_report.json`, `manual_counts.json`, `manual.json` | split counts and cost/manual entries of the archived run |
| `environment.json` | software and hardware of the archived run |

Four audit sheets (`audit_anchors/*.tsv`) are **not in git**: they contain dataset texts, whose redistribution rights have not been established. The one-click run uses them when they are present in `original/archived/audit_anchors/` or in `<run_dir>/private_inputs/audit_anchors/`. Without them, split identity is verified by counts, id/label pairing and exact re-scoring of the archived predictions.

## Changes made when building the repository

Contents unchanged; MD5 verified. Previously this folder sat next to `DiaFuse-PRC++/` in the distributed package. See `docs/REPOSITORY_PROVENANCE.md`.
