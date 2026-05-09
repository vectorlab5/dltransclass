#!/usr/bin/env bash
# Reproduce Table 4 (ablation) — runs the full model and the six ablation variants
# on each dataset, then averages the results.

set -euo pipefail

VARIANTS=(
    full
    title_abstract_only
    shared_encoder_only
    barycenter_no_wasserstein
    no_wasserstein
    uniform_barycenter
    fixed_radius
)

for v in "${VARIANTS[@]}"; do
    echo "=== Variant: ${v} ==="
    python -m dltransclass.scripts.train \
        --config "configs/experiments/ablation_${v}.yaml"
done
