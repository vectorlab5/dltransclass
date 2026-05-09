#!/usr/bin/env bash
# =============================================================================
# Push DLTransClass to git@github.com:vectorlab5/dltransclass.git
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE="git@github.com:vectorlab5/dltransclass.git"

echo "==> Working in: ${REPO_DIR}"
cd "${REPO_DIR}"

# 1. Clean up any half-initialized .git folder left behind by sandbox attempts.
if [ -d .git ]; then
    echo "==> Removing any pre-existing .git/"
    rm -rf .git
fi

# 2. Initialize a fresh repository on main.
git init -b main
git config user.email "Cy_Carmen@163.com"
git config user.name  "Xiaorui Ma"

# 3. Stage everything, respecting .gitignore.
git add -A
git status --short

# 4. Initial commit.
git commit -m "Initial commit: DLTransClass v1.0.0

Reference implementation accompanying:
  Ma, X. and Zeng, Z. (2026). Metadata-Aware Transformer Classification
  for Digital Libraries with Wasserstein Field Fusion.
  Alexandria Engineering Journal.

Includes:
  - Shared transformer field encoder with field-specific token budgets
  - Wasserstein metadata barycenter (closed-form Dirac case)
  - Missingness-aware Wasserstein surrogate loss with explicit lambda_var
  - Realism-calibrated missingness simulator
  - BERT-Concat baseline for the field-naive comparison
  - Reference configs, reproduction scripts, unit tests, and CI"

# 5. Wire up the GitHub remote (idempotent).
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "${REMOTE}"
else
    git remote add origin "${REMOTE}"
fi
git remote -v

# 6. Push.
echo "==> Pushing to ${REMOTE}"
git push -u origin main

echo
echo "==> Done. Visit https://github.com/vectorlab5/dltransclass"
