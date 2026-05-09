# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] – 2026 (planned, on paper acceptance)

### Added
- Initial public release of DLTransClass.
- Shared transformer field encoder with field-specific token budgets.
- Wasserstein metadata barycenter (closed-form Dirac case).
- Missingness-aware Wasserstein surrogate loss with explicit `lambda_var` coefficient.
- Realism-calibrated missingness simulator with per-dataset observation rates.
- BERT-Concat baseline for the field-naive comparison.
- Reference configurations for the six benchmark datasets.
- End-to-end training/evaluation scripts and ablation/sensitivity reproduction shells.
- Field-attribution utilities for librarian-facing inspection.
- Pre-trained checkpoints and metric logs (released alongside the paper on Zenodo).
