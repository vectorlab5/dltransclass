#!/usr/bin/env bash
# Reproduce Table 9 / Figure 8 — realism-calibrated missingness curves.

set -euo pipefail

for r in 0.0 0.2 0.4 0.6; do
    echo "=== Realism-calibrated removal ratio: ${r} ==="
    python -m dltransclass.scripts.evaluate \
        --config configs/experiments/stability.yaml \
        --checkpoint checkpoints/library_sci_meta/model.pt \
        --split test \
        --missingness "${r}" \
        --mode realistic

    echo "=== Uniform-random control: ${r} ==="
    python -m dltransclass.scripts.evaluate \
        --config configs/experiments/stability.yaml \
        --checkpoint checkpoints/library_sci_meta/model.pt \
        --split test \
        --missingness "${r}" \
        --mode uniform
done
