# B12 — DiaFuse-PRC (Setting A)

**Role**: the manuscript's DiaFuse-PRC system. **Status**: executed in the archived reduced run (3 seeds, default hyper-parameters, no tuning). It is reproduced, then extended to 10 seeds, by the one-click pipeline.

In Setting A the three encoders are frozen and their per-layer pooled states are cached once. All trainable parameters are in light heads.

| Item | Value |
|---|---|
| Training code | `original/code/diafuse_run.py` (unmodified, MD5 `9288806e6d5a8d64b6bf5342987b2aec`), class `DiaFuseTrainer` |
| Components (all in `diafuse_run.py`) | `LayerMix` (layer mixtures per encoder); input-dependent convex gate whose input includes the surface descriptor `Store.surface`; hierarchical class × variety prototypes with κ shrinkage (`proto_cv`); cell-rebalanced SupCon (`cl_loss`); hard mixup (`hard_sets`, `make_mixes`); consistency (cosine + JS); MMCE; logit adjustment (`ce_la`); prior correction and variety-wise temperature (`calibrate`, `fit_temperature`) |
| Synthetic Arabizi | `original/code/translit.py` (seen generator `gs`, unseen generator `gu`) and the perturbation families TR, INF, NOISE, CS |
| Preprocessing | `original/code/prep_data.py`; features: `original/code/extract.py` (`--layer_stride 2`) |
| Encoders | XLM-R-large, multilingual-E5-large, LaBSE (`original/code/download_models.py`) |
| Hyper-parameters | the `HP` dict at the top of `diafuse_run.py`, mirrored in `configs/B12_setting_A/defaults.json` |
| Entry point in the one-click run | configuration `V0` of `configs/B12_setting_A/screening_configs.json` (`impl: orig`). `src/diafuse_pp_setting_A.py` then calls `base.DiaFuseTrainer` from the unmodified original file, with no DiaFuse-PRC++ code in the training path. Stage 06 reproduces seeds 0–2; stage 10 trains seeds 0–9 as system `B12`. |
| Original entry point | `python original/code/diafuse_run.py --data data --feats feats --out runs --mode quick --encoders xlmr,me5,labse` (`run_all.sh`) |
| Reproduction target | archived dev scores `original/archived/dev_scores.json` (`B12/<seed>`); tolerance in `configs/protocol.json` → `reproduction.B12` |
| Evaluation | `original/code/diafuse_eval.py` |
| Outputs of a run | `results/B12_setting_A/` (predictions, `seed_metrics.csv`, `reproduction.txt`, `trace.json`); checkpoints `checkpoints/B12/seed<k>.pt` |
