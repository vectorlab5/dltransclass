# DLTransClass Architecture

This document mirrors Section 3 of the paper and serves as a reading guide for the source code.

## Overall pipeline

```
Title ──┐
Abstract┤    Shared    ┌── h_title
Keywords┼── Transformer ┼── h_abstract ─┐                ┌── logits
Subject ┘   (BERT-base) └── h_keywords  ┼─ Barycenter b ─┤
                            h_subject   ┘   (Eq. 8)      └── rho_i (Eq. 10)
                                                                   │
                                                                   ▼
                                              missingness-aware Wasserstein surrogate
                                                              (Eq. 12)
```

| Section | Module | Eq. | Code |
|---------|--------|-----|------|
| 3.1 Problem formulation, field budgets | `models/field_encoder.py` | 1, 2 | `SharedFieldEncoder.encode_field` |
| 3.3 Optimal-transport barycenter        | `models/barycenter.py`    | 5–9 | `MetadataBarycenter.forward`     |
| 3.4 Missingness-aware radius            | `models/dltransclass.py`  | 10  | `DLTransClass.forward` (rho)     |
| 3.4 Variation surrogate                 | `training/surrogate.py`   | 11–12 | `wasserstein_surrogate_loss`   |
| 3.5 Optimization & complexity           | `training/trainer.py`     | 13  | `Trainer.train`                  |

## Field-specific token budgets

We use `FieldBudget(title=64, abstract=256, keywords=48, subject=48)` to match the empirical length distribution of bibliographic fields. Section 4.4.2 of the paper analyses the effect of the abstract budget on Macro-F1 and per-record latency; the 256-token setting is the empirical Pareto knee.

## Realism-calibrated missingness

The mask process used in Section 4.7 is **not** uniform random over fields. The simulator in `data/missingness.py` replicates the empirical missingness pattern measured on each catalog, then scales each field's drop probability proportionally so that the global removal ratio reaches the target value (20%, 40%, or 60%). A `mode="uniform"` flag is provided as a sanity-check baseline.

## Reproducibility

- All experiments fix Python, NumPy, and PyTorch seeds via `utils/seed.py`.
- Set `DLT_DETERMINISTIC=1` to enable cuDNN deterministic mode.
- Hardware, framework versions, batch size, warm-up, and timing methodology are documented in `README.md` and Section 4.1 of the paper.
