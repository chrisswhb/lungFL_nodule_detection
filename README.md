# Federated Learning for Lung Nodule Detection on LIDC-IDRI

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-orange.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/flwr-1.5.0-green.svg)](https://flower.dev/)

## Overview

End-to-end federated learning pipeline for lung nodule detection on raw LIDC-IDRI CT data. The pipeline spans DICOM ingestion, 1mm isotropic resampling, lung segmentation, radiologist consensus extraction (≥3/4 annotators), and 3D patch generation with spatial jitter augmentation — integrated with federated training across 5 simulated clients without any patient data leaving its source institution, as required by HIPAA and GDPR.

The detection model is a 3D U-Net trained on Gaussian heatmap targets with Online Hard Example Mining (OHEM), evaluated using the Competition Performance Metric (CPM) and Free-Response ROC (FROC) analysis. Three federated algorithms are studied — FedAvg, FedProx, and DP-FedAvg — across two clinically motivated non-IID data regimes.

---

## Key Results

| Method | Non-IID | Privacy | Sens | Prec | FP/scan | CPM |
|--------|---------|---------|------|------|---------|-----|
| Client 1 local | — | None | 0.923 | 0.358 | 3.58 | 0.690 |
| FedAvg | Moderate | None | 0.914 | 0.259 | 3.44 | **0.741** |
| FedProx μ=0.01 | Moderate | None | 0.880 | 0.289 | 2.84 | 0.703 |
| FedAvg | Severe | None | 0.743 | 0.274 | 2.67 | 0.583 |
| FedProx μ=0.01 | Severe | None | 0.782 | 0.333 | 2.12 | 0.636 |
| FedProx μ=0.10 | Severe | None | 0.402 | 0.576 | 0.40 | 0.358 |
| DP-FedAvg | Moderate | ε≈5 | 0.897 | 0.266 | 3.26 | 0.724 |

**Key findings:**
- FL outperforms single-site baseline by **+7.4% CPM** (0.690 → 0.741)
- Heterogeneity cost: **−21.3% CPM** under severe vs moderate non-IID
- **FedProx crossover**: hurts under moderate (−5.1%) but helps under severe (+9.1%)
- Wrong proximal coefficient is catastrophic: μ=0.10 vs μ=0.01 → **−43.7% CPM**
- DP cost at ε≈5: only **−2.3% CPM** — privacy-preserving FL is feasible
- Full pipeline runs on a single **NVIDIA A10G (24GB VRAM, peak 17GB)**

---

## Architecture

3D U-Net with symmetric encoder-decoder and skip connections.

| Property | Value |
|----------|-------|
| Parameters | 22,456,321 |
| Input / Output | [B, 1, 96, 96, 96] |
| Base filters | 32 → 64 → 128 → 256 → 512 |
| Normalization | BatchNorm3D |
| Loss | OHEM BCE (neg:pos = 3:1) |
| Optimizer | Adam lr=1e-4 |
| Scheduler | CosineAnnealingLR |
| VRAM | ~7.3 GB model / ~17 GB total with Ray |

```
INPUT [B,1,96³]
  ↓
DoubleConv3D(1→32)    [B,32,96³]  ──────────────────────────────  skip e1
  ↓ MaxPool3D(2)
DoubleConv3D(32→64)   [B,64,48³]  ─────────────────────────────  skip e2
  ↓ MaxPool3D(2)
DoubleConv3D(64→128)  [B,128,24³] ────────────────────────────── skip e3
  ↓ MaxPool3D(2)
DoubleConv3D(128→256) [B,256,12³] ────────────────────────────── skip e4
  ↓ MaxPool3D(2)
DoubleConv3D(256→512) [B,512,6³]  ← BOTTLENECK
  ↓ ConvTranspose3D(×2)
Concat(e4) → DoubleConv3D(512→256) [B,256,12³]
  ↓ ConvTranspose3D(×2)
Concat(e3) → DoubleConv3D(256→128) [B,128,24³]
  ↓ ConvTranspose3D(×2)
Concat(e2) → DoubleConv3D(128→64)  [B,64,48³]
  ↓ ConvTranspose3D(×2)
Concat(e1) → DoubleConv3D(64→32)   [B,32,96³]
  ↓ Conv3D(1×1×1) + Sigmoid
HEATMAP OUTPUT [B,1,96³]  P(nodule) ∈ [0,1]
```

---

## Non-IID Partitioning

### Moderate Non-IID
Random stratified split across 5 clients. Differs only in quantity.
Patient IDs stored in `federation_splits.json`.

### Severe Non-IID (clinically motivated)
Joint stratification across 3 dimensions: slice thickness × nodule size × clean patient %.

| Client | Scanner | Nodule size | Clean% | N | Clinical analog |
|--------|---------|-------------|--------|---|-----------------|
| C1 | Thin ≤1.25mm | Small <8mm | 55.9% | 245 | Modern lung screening program |
| C2 | Thin ≤1.25mm | Large ≥8mm | 10.0% | 229 | Tertiary oncology referral center |
| C3 | Medium 1.25–2mm | Mixed | 21.6% | 125 | Regional general hospital |
| C4 | Thick >2mm | Small <8mm | 66.1% | 127 | Resource-limited rural clinic |
| C5 | Thick >2mm | Large ≥8mm | 10.0% | 170 | Older hospital, symptomatic population |

| Metric | Moderate CV | Severe CV | Ratio |
|--------|------------|-----------|-------|
| Slice thickness | 13.0% | 28.4% | 2.2× |
| Nodule diameter | 12.2% | 34.5% | 2.8× |
| Clean patient % | 24.4% | 72.4% | 3.0× |
| Nodules/patient | 19.3% | 33.1% | 1.7× |

Split parameters: `experiments/severe_noniid/description.json`

---


## Setup

### 1. Clone

```bash
git clone https://github.com/chrisswhb/lungFL_nodule_detection.git
cd LungFL_nodule_detection
```

### 2. Environment

```bash
conda env create -f environment.yml
conda activate lungfl
```

### 3. LIDC-IDRI Dataset

Download from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/).
Configure pylidc to point to your local copy:

```ini
# ~/.pylidcrc  (Linux/Mac) or %APPDATA%\pylidc\pylidc.conf (Windows)
[dicom]
path = /path/to/LIDC-IDRI
```

---

## Reproducing Experiments

| Step | Notebook | Output |
|------|----------|--------|
| Preprocessing + local baseline | client1_unet.ipynb |
| Client data prep | client_{2-5}.ipynb | client{2-5}/ |
| Severe split generation | fl_end.ipynb (split cell) | experiments/severe_noniid/ |
| FedAvg moderate | fl_end.ipynb | weights/global_fl_final.pt |
| FedProx moderate μ=0.01 | fl_end.ipynb | experiments/fedprox_moderate/ |
| FedAvg severe | fl_end.ipynb | experiments/severe_noniid/fl_results/ |
| FedProx severe μ=0.01 | fl_end.ipynb | experiments/fedprox_severe_mu001/ |
| FedProx severe μ=0.10 | fl_end.ipynb | experiments/fedprox_severe/ |
| DP-FedAvg moderate ε≈5 | fl_end.ipynb | experiments/dp_fedavg_moderate_e5/ |
| Full analysis | analysis.ipynb | experiments/full_analysis/ |

---

## Hardware

| Component | Used | Minimum |
|-----------|------|---------|
| GPU | NVIDIA A10G 24GB | Any CUDA GPU ≥12GB |
| Peak VRAM | 17 GB (model 7.3GB + Ray 6GB) | 12 GB (model only) |
| Training time | ~15h per FL experiment | — |
| Inference time | ~30s per patient | — |
| OS | Windows Server (EC2 g5.2xlarge) | — |

---

## Known Limitations

- **No FP reduction stage** — precision (0.26–0.58) is below two-stage SOTA systems (0.70–0.90+)
- **BatchNorm in FL** — running statistics mismatch under non-IID; GroupNorm is recommended
- **Single DP budget** — only ε≈5 tested; full privacy-utility curve not provided
- **Sequential Ray simulation** — 5× slower than real parallel FL deployment
- **Single dataset** — no external validation beyond LIDC-IDRI

---

## License

MIT

---

## Dataset

This project uses the publicly available [LIDC-IDRI dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/).
No patient CT data is included in this repository.
All preprocessing is performed locally from the original DICOM files.

---

## Authors

Chris Wehbe, Nassif El Hachem, Kevin El Khoury, Guy-Anthony Zakhia, Ali Al Karrar Zeaiter