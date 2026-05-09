# DLTransClass

**Metadata-Aware Transformer Classification for Digital Libraries with Wasserstein Field Fusion**

This repository contains the official implementation of *DLTransClass*, a transformer-based framework for automatic text classification in digital libraries. The model encodes title, abstract, keyword, and subject-related fields with a shared transformer, fuses them through a Wasserstein metadata barycenter, and minimizes a missingness-aware Wasserstein objective whose first-order surrogate yields a local variation penalty on the fused representation.

> **Paper:** Ma, X. and Zeng, Z. *Metadata-Aware Transformer Classification for Digital Libraries with Wasserstein Field Fusion.* Alexandria Engineering Journal, 2026. *(to appear)*

---

## Highlights

- **Field-aware barycentric fusion.** Each bibliographic field is encoded by a shared transformer and treated as a Dirac mass in the embedding space; the fused record representation is the weighted Wasserstein barycenter.
- **Missingness-aware Wasserstein objective.** A record-specific radius driven by missingness and cross-field disagreement scales a first-order local-variation penalty on the cross-entropy loss, providing principled robustness to incomplete metadata.
- **State-of-the-art performance.** Average Macro-F1 of 82.2% across six benchmarks, including library and scholarly subject-assignment corpora.
- **Production-ready latency.** 9.8 ms per record on a single NVIDIA A100 (40 GB) under fp16 inference with batch size 32.
- **Robust under sparsity.** Retains an 8.1-point Macro-F1 advantage over competitive fusion baselines under realism-calibrated 60% field removal.

## Repository Layout

```
dltransclass/
├── src/dltransclass/        # Core package
│   ├── models/              # DLTransClass model and barycenter fusion
│   ├── data/                # Dataset loaders, field tokenizers, missingness simulator
│   ├── training/            # Training loops, Wasserstein surrogate, schedulers
│   ├── baselines/           # 13 baselines (BERT-Concat, Mamba-Doc, LongLoRA, ...)
│   ├── visualization/       # t-SNE, field-attribution, latent-space plotting
│   └── utils/               # Logging, config, reproducibility utilities
├── configs/                 # YAML configs for datasets, baselines, experiments
├── scripts/                 # Train, evaluate, ablation, sensitivity, error-analysis
├── tests/                   # Unit tests
├── docs/                    # Documentation, architecture notes
├── checkpoints/             # Pretrained weights (released on Zenodo)
├── results/                 # Logged metrics and figures
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/vectorlab5/dltransclass.git
cd dltransclass
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The package targets **Python 3.10+**, **PyTorch 2.2+**, and **CUDA 12.1**. CPU-only inference is supported but not recommended for full benchmark reproduction.

### Quick start

```bash
# Reproduce the headline number on UDC-OldBooks
python -m dltransclass.scripts.train --config configs/experiments/udc_oldbooks.yaml

# Evaluate a released checkpoint
python -m dltransclass.scripts.evaluate \
    --checkpoint checkpoints/dltransclass_udc_oldbooks.pt \
    --config configs/experiments/udc_oldbooks.yaml
```

## Datasets

We evaluate DLTransClass on six benchmark datasets:

| Dataset            | Classes | Train  | Val   | Test  | Source |
|--------------------|---------|--------|-------|-------|--------|
| UDC-OldBooks       | 18      | 8,400  | 1,200 | 2,400 | Kragelj & Kljajić Borštnar (2021) |
| LCSH-Theses        | 26      | 11,200 | 1,600 | 3,200 | Usta (2025) |
| JUCS-Meta          | 12      | 6,800  | 900   | 1,900 | Mitra et al. (2025) |
| CompScholar-Meta   | 15      | 9,500  | 1,300 | 2,700 | Curated |
| ArXiv-CS-Metadata  | 40      | 15,000 | 2,000 | 4,000 | arXiv (CS prefix) |
| Library-Sci-Meta   | 32      | 9,200  | 1,200 | 2,500 | Curated |

Pre-processing scripts and field-mask utilities live in `src/dltransclass/data/`. Data files themselves are not redistributed; see `docs/datasets.md` for download and preparation instructions.

## Reproducing Paper Results

```bash
# Main results across six benchmarks (Table 3)
bash scripts/reproduce_main_results.sh

# Ablation study (Table 4)
bash scripts/reproduce_ablation.sh

# Sensitivity sweeps (Tables 5–6, Figure 4)
bash scripts/reproduce_sensitivity.sh

# Stability under realism-calibrated missingness (Table 9, Figure 8)
bash scripts/reproduce_stability.sh

# Error analysis (Section 4.8)
python -m dltransclass.scripts.error_analysis --dataset library_sci_meta
```

All scripts log JSON metrics to `results/` and generate publication-ready PDF figures.

## Hardware and Reproducibility

All efficiency numbers reported in the paper were measured on:

- **GPU:** NVIDIA A100 40 GB (PCIe)
- **CPU:** AMD EPYC 7763
- **Memory:** 256 GB DDR4
- **Software:** CUDA 12.1, PyTorch 2.2.1, HuggingFace Transformers 4.41
- **Precision:** fp16 with NVIDIA Apex-O2 autocast
- **Batch size (inference):** 32
- **Warm-up:** 50 batches; latency measured as median over 1,000 timed batches

For determinism, all experiments fix `PYTHONHASHSEED`, NumPy, PyTorch, and CUDA seeds in `src/dltransclass/utils/seed.py`. Set `DLT_DETERMINISTIC=1` to enable cuDNN deterministic mode at the cost of ~10% throughput.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Xiaorui Ma** — Department of Library, Jiangsu Vocational College of Finance and Economics — `Cy_Carmen@163.com`
- **Zhengyu Zeng** — Department of State-owned Assets Management, Jiangsu Vocational College of Finance and Economics — `51797998@qq.com`

For bug reports and feature requests please use the [GitHub issue tracker](https://github.com/vectorlab5/dltransclass/issues).
