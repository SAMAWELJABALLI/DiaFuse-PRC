#!/usr/bin/env bash
# DiaFuse-PRC on a lab GPU server (same steps as the Colab notebook).
# 1) on a machine with Internet:   python download_models.py --models_dir models --raw_dir raw
#    then copy models/ and raw/ next to these scripts on the GPU server.
# 2) on the GPU server:            MODE=quick bash run_all.sh      (or MODE=full)
set -euo pipefail
MODE=${MODE:-quick}
ENCODERS=${ENCODERS:-xlmr,me5,labse}
FINETUNE_SEEDS=${FINETUNE_SEEDS:-$([ "$MODE" = full ] && echo 0,1,2,3,4,5,6,7,8,9 || echo 0,1,2)}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results audit
[ -f data/prep_report.json ] || python prep_data.py            # OWN_SPLITS=/path and MADAR_DIR=/path are honoured
python extract.py --models_dir models --data data --out feats --encoders "$ENCODERS" --layer_stride 2 --batch 128
python diafuse_run.py --data data --feats feats --out runs --mode "$MODE" --encoders "$ENCODERS" ${SKIP:+--skip $SKIP}
for S in B1 B11; do python finetune.py --system $S --model models/xlmr --data data --out runs --seeds "$FINETUNE_SEEDS"; done
python export_audit.py --data data --runs runs --out audit_todo
python - <<'PY'
import json, os
m = {}
for p in ("data/manual_counts.json", "runs/manual.json"):
    if os.path.exists(p): m.update(json.load(open(p)))
json.dump(m, open("results/manual_all.json", "w"), indent=1)
PY
AUDIT=$(ls audit/*.tsv >/dev/null 2>&1 && echo "--audit audit" || true)
python diafuse_eval.py --data data --preds runs/preds --diag runs/diag --emb runs/emb --manual results/manual_all.json --out results $AUDIT
zip -qr results_bundle.zip results data/prep_report.json data/manual_counts.json runs/dev_scores.json runs/manual.json \
    $( [ -f runs/b12_hparams.json ] && echo runs/b12_hparams.json )
echo "send back: results_bundle.zip"
