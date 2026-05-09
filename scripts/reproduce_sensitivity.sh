#!/usr/bin/env bash
# Reproduce sensitivity sweeps in Section 4.4 of the paper:
#   - Tables 5, 6 and the four panels of Figure 4
#   - lambda penalty weight (4.4.1)
#   - abstract truncation (4.4.2)
#   - rho0, beta_m, beta_v, num fields

set -euo pipefail

python -m dltransclass.scripts.train --config configs/experiments/sensitivity_lambda.yaml
python -m dltransclass.scripts.train --config configs/experiments/sensitivity_truncation.yaml
python -m dltransclass.scripts.train --config configs/experiments/sensitivity_rho0.yaml
python -m dltransclass.scripts.train --config configs/experiments/sensitivity_beta_m.yaml
python -m dltransclass.scripts.train --config configs/experiments/sensitivity_beta_v.yaml
