# B12++ — DiaFuse-PRC++ (Setting A)

**Role**: pre-registered attempt to improve B12 with frozen encoders. **Status**: implemented; only static and dry-run tests so far. **No GPU execution yet.** The selected configuration is decided by the run itself, from development data only.

| Item | Value |
|---|---|
| Code | `src/diafuse_pp_setting_A.py`, which imports the unmodified `original/code/diafuse_run.py` as `base` (data store, layer mixtures, losses) and adds configurable fusion, gate, head and loss variants |
| Defaults | `configs/B12_setting_A/defaults.json`: the original B12 values (`HP` of `diafuse_run.py`) |
| Screening grid | `configs/B12_setting_A/screening_configs.json`: `V0` (original trainer, reference), `P0` (implementation check: new code with default values, must match V0 on dev), families V1 gate, V2 fusion, V3 head, V4 prototypes, V5 losses, V6 inputs. Each variant states its motivation in `why`. |
| Selection | `scripts/select_config.py` with the rule of `configs/protocol.json` → `selection_rule`: mean dev macro-F1 over `val_ar` and `val_gs`, at least 3 seeds, gain larger than the seed noise. Family winners are then combined; ties within 0.2 points go to fewer parameters. If no variant qualifies, the selected model is the reference (a negative result). |
| Implementation check | stage 08: P0 must match V0 (`configs/protocol.json` → `reproduction.P0_vs_V0`). If it does not, Setting A variants are disabled. |
| Final / ablations | 10 seeds as system `PPA-FINAL`; pre-declared ablations with 5 seeds as `PPA-ABL-*` |
| Outputs of a run | `results/B12pp_setting_A/`, `results/ablations/settingA_*/`, checkpoints `checkpoints/PPA-FINAL/seed<k>.pt` |
| Diagnosis behind the variants | `docs/PP_DIAGNOSIS_REPORT.md` |
