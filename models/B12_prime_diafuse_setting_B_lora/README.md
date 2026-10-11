# B12′ — DiaFuse-PRC (Setting B, LoRA)

## Status

```text
SETTING B / LoRA CODE STATUS
  Manuscript B12' (Setting B, LoRA): MISSING. Specified, never executed, no original implementation exists.
  DiaFuse-PRC++ Setting B (this repository): PARTIAL with respect to the B12' specification; never executed on a GPU.
```

Evidence that B12′ was never executed:
- The manuscript lists "B12′ (Setting B)" among the experiments that were not executed, and its row in the main table reads TBO.
- The executed original notebook says "Non exécuté : B12′ (Setting B, LoRA)".
- `original/code/` contains no LoRA or Setting B code.
- The original code and notebook provided by the project owner as "the code that produced the results" are byte-identical to `original/code/` and `original/notebook/`, and contain no LoRA code either.

**Results of `src/diafuse_pp_setting_B_lora.py` must not be reported as the manuscript's B12′.** Tables label them `B12'++ DiaFuse-PRC++ (Setting B, LoRA)`.

## What the manuscript specifies for B12′

From sections 4 and 9 and the hyper-parameter appendix:
- LoRA adapters on the encoders, r = 8, lr 2·10⁻⁴;
- attention pooling instead of mean pooling;
- a cell-indexed FIFO queue for the contrastive targets;
- otherwise the DiaFuse-PRC objective.

## What `src/diafuse_pp_setting_B_lora.py` implements

| Item | Value | Source |
|---|---|---|
| LoRA | manual `LoRALinear` (no PEFT dependency) wrapping `nn.Linear` modules named in `lora_targets` | code |
| LoRA rank / α / dropout | r = 8, α = 16, dropout 0.1 | `configs/B12_setting_B_lora/defaults.json` |
| Target modules | `query`, `key`, `value` (self-attention projections of every layer) | same file |
| Learning rates | LoRA 2·10⁻⁴, head 10⁻³, full fine-tuning (FT0/FT1 only) 10⁻⁵; weight decay 0.01; warm-up 10 % | same file |
| Epochs / batch / max length | 3 / 8 anchors (+ perturbed partners) / 64 | same file |
| Encoders | XLM-R-large, mE5-large, LaBSE, each with its own LoRA adapters; base weights frozen | same file |
| Pooling | attention pooling (`pool: attn`) | same file |
| Fusion / head | input-dependent convex gate (zero-init, τ = 2, load balancing 0.01) / MLP head | same file |
| Losses | cross-entropy on clean and perturbed partner, ℓ2 consistency (0.5), optional JS consistency, optional class-prototype loss | same file + code |
| Calibration | global and variety-wise temperature fitted on the development sets | code |
| Trainable / frozen parameters | counted at run time and written to `trainable_params` in `experiment_registry.csv`. **Not measured yet.** | code |
| Checkpoints | trainable parameters only, `checkpoints/PPB-FINAL/seed<k>.pt`; per-epoch resume checkpoints are deleted after success | code |

Screening variants (`configs/B12_setting_B_lora/screening_configs.json`):

| Name | Kind | Description |
|---|---|---|
| FT0 | reference, full fine-tuning | the B11 recipe re-implemented; must match the original B11 on dev, otherwise Setting B is disabled |
| FT1 | control | FT0 on mE5 |
| FT2 | LoRA | 3 encoders, gate, MLP head, ℓ2 + JS |
| FT3 | LoRA | static fusion |
| FT4 | LoRA | stronger anti-collapse |
| FT5 | LoRA | class prototypes |

## Missing with respect to the B12′ specification

- `LayerMix` (layer mixtures per encoder): only the attention-pooled last layer is used;
- surface descriptor s(x) as gate input;
- hierarchical class × variety prototypes with κ shrinkage (only an optional class-prototype loss exists);
- cell-rebalanced supervised contrastive loss with a cell-indexed FIFO queue;
- hard mixup;
- MMCE;
- logit adjustment.

Completing these components would be new scientific code. It has not been written, pending the project owner's decision.

## Outputs of a run

`results/B12_setting_B_lora/` (predictions, `seed_metrics.csv`, `trace.json`) and `results/ablations/settingB_*/`. If the selection rule keeps the reference FT0, no B12′++ final model is trained, and the run records this as a negative result.
