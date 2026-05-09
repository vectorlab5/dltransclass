#!/usr/bin/env bash
# Reproduce Table 3 (main results) across the six benchmark datasets.

set -euo pipefail

DATASETS=(
    udc_oldbooks
    lcsh_theses
    jucs_meta
    compscholar_meta
    arxiv_cs_metadata
    library_sci_meta
)

for ds in "${DATASETS[@]}"; do
    echo "=== Training DLTransClass on ${ds} ==="
    python -m dltransclass.scripts.train --config "configs/experiments/${ds}.yaml"

    echo "=== Evaluating on test split ==="
    python -m dltransclass.scripts.evaluate \
        --config "configs/experiments/${ds}.yaml" \
        --checkpoint "checkpoints/${ds}/model.pt" \
        --split test
done
