# Results (not stored in git)

A one-click run writes its outputs to Google Drive, in `My Drive/DiaFuse_PRCPP_RUN/`, never to this folder:

```text
results/B11/                 B11 Robust + calibrated (Shao et al. style)
results/B12_setting_A/       B12 DiaFuse-PRC (Setting A), original trainer
results/B12pp_setting_A/     B12++ DiaFuse-PRC++ (Setting A)
results/B12_setting_B_lora/  B12'++ DiaFuse-PRC++ (Setting B, LoRA), not the manuscript's B12'
results/ablations/           one folder per ablation
```

Each model folder contains `predictions/<seed>/<set>.tsv`, `seed_metrics.csv`, `trace.json` and `README.md`. The trace files link model → code version → configuration → split → seeds → checkpoints → results.

Checkpoints are written to `My Drive/DiaFuse_PRCPP_RUN/checkpoints/<system>/`. They are zipped separately in `final_archive/DiaFuse_PRCPP_CHECKPOINTS_<date>.zip`, and no other storage location has been decided.
