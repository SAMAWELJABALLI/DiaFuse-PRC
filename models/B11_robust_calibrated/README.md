# B11 — Robust + calibrated (Shao et al. style)

**Role**: reference baseline. **Status**: executed in the archived reduced run (3 seeds). It is reproduced, then extended to 10 seeds, by the one-click pipeline.

| Item | Value |
|---|---|
| Code | `original/code/finetune.py` (unmodified, MD5 `16f6f4d47296ca756bd8ab09dfc064e5`) |
| Entry point | `python original/code/finetune.py --system B11 --model <models>/xlmr --data <data> --out <runs> --seeds <k>`. This is the invocation of `original/code/run_all.sh`; the one-click pipeline calls it once per seed (stages 07 and 10). |
| Encoder | XLM-R-large (`FacebookAI/xlm-roberta-large`), fully fine-tuned |
| Method (manuscript) | attention pooling, ℓ2 clean–perturbed consistency on the same transformation families, post-hoc temperature scaling |
| Hyper-parameters | `finetune.py` defaults: `--epochs 3 --lr 1e-5 --batch 16 --max_len 64 --lam 0.5`. B11 uses `batch // 2 = 8` anchors per step, each with a perturbed partner. See `configs/B11/README.md`. |
| Data | splits rebuilt by `original/code/prep_data.py`; trained on `train_ar` and its synthetic views; selected on `val_ar` + `val_gs` |
| Seeds | reproduction 0–2; final 0–9 (`configs/protocol.json`) |
| Reproduction target | archived best dev macro-F1 per seed (`original/archived/finetune_dev_log.json`); tolerance in `configs/protocol.json` → `reproduction.B11` |
| Outputs of a run | `results/B11/` (predictions, `seed_metrics.csv`, `reproduction.txt`, `trace.json`); predictions also in `runs/preds/B11/` |
| Checkpoints | `finetune.py` does not save model weights. This is the original behaviour, kept unchanged. |
