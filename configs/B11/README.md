# B11 configuration

B11 has no configuration file: its hyper-parameters are the command-line defaults of the unmodified `original/code/finetune.py`, which must not be edited.

```text
--system B11 --model <models>/xlmr --epochs 3 --lr 1e-5 --batch 16 --max_len 64 --lam 0.5
```

The invocation is the one of `original/code/run_all.sh`:

```bash
python finetune.py --system B11 --model models/xlmr --data data --out runs --seeds <seeds>
```

`--batch 16` gives `batch // 2 = 8` clean anchors per step for B11, each paired with a perturbed partner. Seeds come from `configs/protocol.json`.
