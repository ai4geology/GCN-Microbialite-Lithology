<!-- 
  GCN-Based Microbialite Lithology Identification
  Petroleum Science 2025
-->

<div align="center">

## Spectral Graph Convolution Networks for Microbialite Lithology Identification Based on Conventional Well Logs

</div>

Keran Li<sup>a,1</sup>, Jinmin Song<sup>a,*</sup>, Han Wang<sup>a</sup>, Haijun Yan<sup>b</sup>, Shugen Liu<sup>a</sup>, Yang Lan<sup>c,2</sup>, Xin Jin<sup>a</sup>, Jiaxin Ren<sup>a</sup>, Lizhou Tian<sup>a</sup>, Haoshuang Deng<sup>a</sup>, Wei Chen<sup>a</sup>

<sup>a</sup>State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Chengdu University of 
Technology, Chengdu 610059, China

<sup>b</sup>Research Institute of Petroleum Exploration and Development, Beijing, 100083, China

<sup>c</sup>University College London, Gower Street, London, WC1E 6BT, UK

<sup>1</sup>Present address: State Key Laboratory of Critical Earth Material Cycling and Mineral Deposits, Frontiers Science Center for Critical Earth Material Cycling, School of Earth Sciences and Engineering, Nanjing University, Nanjing, 210023, China

<sup>2</sup>Present address: School of Economics and Management, Beihang University, Beijing, 100191, China

<sup>*</sup>Corresponding authors

---

[![Petroleum Science](https://img.shields.io/badge/Petroleum%20Science-2025-blue.svg?style=flat-square)](https://www.sciencedirect.com/journal/petroleum-science)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND-green.svg?style=flat-square)](LICENSE)

[📄 Paper](https://doi.org/10.1016/j.petsci.2025.02.008) • 
[🌐 Project Page](https://github.com/KeranLi/GCN-Microbialite-Lithology) • 
[📊 Dataset](#dataset) • 
[🚀 Quick Start](#quick-start)

---

## 🎯 Overview

This repository provides the official implementation of **Spectral Graph Convolutional Networks (GCN)** for automated microbialite lithology identification from conventional well logs. Unlike traditional methods that shuffle time-series data (destroying sedimentary sequence information), this approach treats well logs as **graph-structured spectral data**, preserving both vertical temporal dependencies and inter-log correlations.

### 🔬 Key Innovations

- **🕸️ Graph Representation**: Transforms well logs into latent graphs (spectra + adjacency matrix) using GRU and self-attention
- **🎵 Spectral Processing**: Utilizes Graph Fourier Transform (GFT) and Discrete Fourier Transform (DFT) to capture frequency-domain features
- **⏱️ Sequence Preservation**: Maintains depth-series (time-series) order without shuffling, modeling actual sedimentary deposition sequences
- **⚖️ Data Balance**: Implements SMOTE to handle class imbalance in microbialite distribution
- **🔄 Transfer Learning**: Demonstrates fine-tuning strategies for adapting to new formations (Dengying-2, Leikoupo-4³) with limited samples

### 📊 Performance Highlights

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|:-----:|:--------:|:---------:|:------:|:--------:|:---:|
| **GCN (Ours)** | **0.90** | **0.93** | **0.94** | **0.90** | **0.95** |
| LSTM | 0.80 | 0.79 | 0.78 | 0.80 | 0.78 |
| RNN | 0.61 | 0.60 | 0.65 | 0.61 | 0.72 |
| TCN | 0.70 | 0.69 | 0.72 | 0.70 | 0.78 |
| ANN | 0.61 | 0.50 | 0.56 | 0.61 | 0.58 |

*Results on Dengying Formation (Z2dn4), Moxi Gas Field, Sichuan Basin*

---

## 🏗️ Architecture

**Workflow**: Raw Logs → GRU Encoder → Self-Attention (Adjacency) → GFT/DFT → GLU → Graph Conv → Classification

### Core Components

1. **📈 GRU Block**: Processes depth-series sequences to generate latent graph representations
2. **🔗 Self-Attention**: Dynamically constructs adjacency matrices from hidden states (Q, K, V mechanism)
3. **🌊 Spectral Transform**: 
   - **GFT**: Graph Fourier Transform using Laplacian eigen-decomposition
   - **DFT**: Discrete Fourier Transform for frequency-domain convolution
4. **🎛️ GLU (Gated Linear Unit)**: Controls information flow in spectral domain
5. **🕸️ Graph Convolution**: Spectral graph convolution with symmetric normalized Laplacian

---

## 📁 Dataset

### Geological Setting
- **Location**: Moxi Gas Field, Central Sichuan Basin, China
- **Formation**: 4th Member of Ediacaran Dengying Formation (Z2dn4)
- **Wells**: 44 wells (42 for training, 2 preserved for generalization testing)
- **Samples**: 10,367 valid data points (after SMOTE augmentation: 12,570)

### Well Log Features (8 Curves)

| Curve | Description | Unit | Geological Significance |
|:-----:|:------------|:----:|:------------------------|
| **AC** | Acoustic (Sonic) | μs/m | Velocity, porosity indicator |
| **CAL** | Caliper | inch | Borehole diameter, caving |
| **CNL** | Compensated Neutron | % | Porosity, hydrogen index |
| **DEN** | Density | g/cm³ | Bulk density, lithology |
| **GR** | Gamma Ray | API | Shale content, clay volume |
| **PE** | Photoelectric Factor | b/e | Mineral composition, lithology |
| **RLLD** | Deep Resistivity | Ω·m | True formation resistivity |
| **RLLS** | Shallow Resistivity | Ω·m | Invasion zone resistivity |

### Lithological Classes (5 Microbialite Types)

| Code | Full Name | Description | Characteristics |
|:----:|:----------|:------------|:----------------|
| **MICR** | Dolomicrite | Micritic dolomite | Dark, fine-grained, rare structures |
| **SSTR** | Stratiform Stromatolite | Layered microbial mats | Parallel laminations, intermittent dark lines |
| **WSTR** | Wavy Stromatolite | Undulating microbial structures | Large curvature, semi-circular, porous |
| **THRO** | Thrombolite | Clotted microbial structure | Dark clots, diffusing fabric, dissolving pores |
| **SILIS** | Siliceous Stromatolite | Silica-rich microbialite | Curved stripes, interlayer quartz, brittle |

---

## 📦 Installation

### Prerequisites
- Python ≥ 3.8
- CUDA ≥ 11.3 (optional, for GPU acceleration)
- 8GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/KeranLi/GCN-Microbialite-Lithology.git
cd GCN-Microbialite-Lithology

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

**Key Dependencies**:
```text
torch>=1.11.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
imbalanced-learn>=0.8.0  # For SMOTE
```

---

## 🚀 Quick Start

### 1. Data Preparation

Prepare CSV files with the following columns:
```csv
Depth,AC,CAL,CNL,DEN,GR,PE,RLLD,RLLS,Lithology
5000.0,55.2,8.5,2.1,2.45,15.0,3.2,100.0,95.0,SSTR
5000.125,56.1,8.6,2.2,2.46,16.2,3.1,105.0,98.0,SSTR
...
```

### 2. Training

#### Train GCN Model (Full Pipeline)
```bash
python scripts/train.py --config configs/config.yaml --model gcn
```

#### Train Baseline Models (for Comparison)
```bash
# LSTM (Time-series sequential)
python scripts/train.py --model lstm

# RNN (Basic recurrent)
python scripts/train.py --model rnn

# TCN (Temporal Convolutional Network)
python scripts/train.py --model tcn

# ANN (Standard feed-forward)
python scripts/train.py --model ann
```

### 3. Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_gcn.pth --test_data data/test.csv
```

Expected output:
```text
Test Metrics:
- Accuracy: 0.90
- Precision: 0.93
- Recall: 0.94
- F1-Score: 0.90
- AUC: 0.95

Confusion Matrix:
       SSTR  THRO  WSTR  SILIS  MICR
SSTR   0.97  0.01  0.02   0.00  0.00
THRO   0.00  0.99  0.00   0.00  0.00
WSTR   0.03  0.01  0.95   0.00  0.01
SILIS  0.00  0.00  0.00   0.99  0.01
MICR   0.00  0.04  0.00   0.00  0.95
```

### 4. Inference on New Wells

```python
import torch
from models.gcn import MicrobialiteGCN
from utils.data_loader import WellLogDataset

# Load model
model = MicrobialiteGCN(input_dim=8, num_classes=5)
checkpoint = torch.load('checkpoints/best_gcn.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Prepare data (8 log curves)
logs = [[55.2, 8.5, 2.1, 2.45, 15.0, 3.2, 100.0, 95.0], ...]  # [seq_len, 8]
input_tensor = torch.FloatTensor(logs).unsqueeze(0)  # [1, seq_len, 8]

# Predict
logits, adjacency_matrix = model(input_tensor)
predicted_class = torch.argmax(logits, dim=1)
# Returns: SSTR (Stratiform Stromatolite)
```

---

## 🔄 Transfer Learning (Fine-tuning)

The model supports rapid adaptation to new formations with limited data, as demonstrated in the paper for:
- **Dengying-2 Member** (Taihe Gas Field) - 6 lithologies
- **Leikoupo-4³ Submember** (Pengzhou Gas Field) - 3 lithologies

### Strategy 1: Standard Fine-tuning (Medium Dataset: >1000 samples)
```bash
python scripts/transfer_learning.py     --source_model checkpoints/best_gcn.pth     --target_data data/dengying2.csv     --num_classes 6     --strategy fine_tune     --epochs 50
```

### Strategy 2: GCN + SVM (Small Dataset: <500 samples)
When target data is extremely limited, freeze GCN layers and use SVM classifier:
```python
from scripts.transfer_learning import TransferLearningGCN

# Initialize transfer learning
tl = TransferLearningGCN(
    pretrained_path='checkpoints/best_gcn.pth',
    num_new_classes=3,  # Leikoupo-43 has 3 classes
    freeze_layers=True
)

# Use GCN as feature extractor + SVM
tl.fine_tune_with_svm(X_train, y_train, X_test, y_test)
```

**Results with Fine-tuning**:
| Formation | Samples | Strategy | Accuracy | Notes |
|:----------|:-------:|:---------|:--------:|:------|
| Dengying-4 (Source) | 8,000 | Baseline | 0.90 | Original training |
| Dengying-2 | 500 | GCN+SVM | 0.86 | Limited data |
| Dengying-2 | 8,000 | Fine-tune | 0.91 | Full adaptation |
| Leikoupo-4³ | 2,000 | Fine-tune | 0.84 | Cross-formation |

---

## 📊 Experimental Results

### Main Results (Test Set, Moxi Gas Field)

<div align="center">

| Model | Architecture | Acc | Pre | Rec | F1 | AUC | Params |
|:-----:|:-------------|:---:|:---:|:---:|:---:|:---:|:------:|
| **GCN** | GRU+GFT+GLU+GCN | **0.90** | **0.93** | **0.94** | **0.90** | **0.95** | 2.1M |
| LSTM | 5-layer LSTM | 0.80 | 0.79 | 0.78 | 0.80 | 0.78 | 1.8M |
| RNN | 5-layer RNN | 0.61 | 0.60 | 0.65 | 0.61 | 0.72 | 1.2M |
| TCN | Dilated Conv | 0.70 | 0.69 | 0.72 | 0.70 | 0.78 | 1.5M |
| FC-ANN | Fully Connected | 0.61 | 0.50 | 0.56 | 0.61 | 0.58 | 2.8M |
| Dropout-ANN | ANN + Dropout | 0.67 | 0.70 | 0.63 | 0.67 | 0.62 | 2.8M |

</div>

### Key Findings

1. **GCN Superiority**: GCN achieves **90% accuracy**, outperforming LSTM by 10% and RNN by 29%
2. **Overfitting in ANNs**: Standard ANNs show severe overfitting (train acc 0.82 → test acc 0.61), mitigated by dropout but still inferior to sequential models
3. **Temporal Information**: Shuffling time-series destroys sedimentary patterns, reducing all models to ~20% accuracy
4. **Class-wise Performance**:
   - **THRO** (Thrombolite): Best identified (Accuracy >0.95)
   - **SSTR vs WSTR**: Main confusion pair (3% misclassification due to similar lamination patterns)
   - **SILIS**: Easily distinguished by quartz signature in PE logs

### Ablation Study

Components removed and performance impact:

| Configuration | Accuracy Drop | Analysis |
|:-------------|:-------------:|:---------|
| **Full Model** | **0.90** | Baseline |
| w/o GRU | -0.18 | Graph construction is critical |
| w/o Self-Attention | -0.06 | Attention provides moderate gain |
| w/o DFT | -0.08 | Frequency domain important |
| w/o GFT | -0.10 | Graph Fourier Transform essential |
| w/o Convolution | -0.28 | Most vital component |
| Single GCN Layer | -0.08 | Two layers optimal |

---

## 🔍 Geological Insights

### Stratigraphic Sequence Analysis
The model captures **Walther's Law** in the vertical direction:
- **Window=2 (0.25m)**: Dominated by same-lithology transitions (self-transitions)
- **Window=3 (0.375m)**: Optimal for detecting lithology changes
- **Window>4**: Self-transitions dominate again

This 0.375m scale matches the GCN's receptive field, validating that the model learns actual depositional cyclicity.

### Log Correlation Analysis
Despite high correlation between **RLLD** and **RLLS** (Pearson 0.81), removing either reduces accuracy by ~5%, indicating they provide complementary latent information through spectral graph convolution.

---

## 📂 Project Structure

```text
gcn-lithology-identification/
├── configs/
│   └── config.yaml              # Configuration parameters
├── models/
│   ├── gcn.py                   # Main GCN architecture (GRU + Spectral GCN)
│   ├── layers.py                # Custom layers (GLU, GFT, GraphConv)
│   └── baselines.py             # LSTM, RNN, TCN, ANN comparators
├── utils/
│   ├── data_loader.py           # Data loading with SMOTE augmentation
│   ├── graph_utils.py           # Adjacency matrix construction
│   ├── metrics.py               # Evaluation metrics (Acc, F1, AUC)
│   └── visualizer.py            # Confusion matrix & log visualization
├── scripts/
│   ├── train.py                 # Main training loop
│   ├── evaluate.py              # Model evaluation
│   ├── predict.py               # Inference script
│   └── transfer_learning.py     # Fine-tuning for new formations
├── notebooks/
│   └── tutorial.ipynb           # Step-by-step tutorial
├── requirements.txt
└── README.md
```

---

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{li2025spectral,
  title={Spectral graph convolution networks for microbialite lithology identification based on conventional well logs},
  author={Li, Ke-Ran and Song, Jin-Min and Wang, Han and Yan, Hai-Jun and Liu, Shu-Gen and Lan, Yang and Jin, Xin and Ren, Jia-Xin and Zhao, Ling-Li and Tian, Li-Zhou and Deng, Hao-Shuang and Chen, Wei},
  journal={Petroleum Science},
  volume={22},
  pages={1513--1533},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.petsci.2025.02.008}
}
```

---

## ⚠️ Usage Notes

1. **Data Quality**: Ensure logs are environmentally corrected and depth-aligned. Missing values should be interpolated before training.

2. **Class Imbalance**: Microbialite distributions are naturally imbalanced (SSTR: 33%, THRO: 27%, WSTR: 12%, etc.). Always use **SMOTE** or class-weighted loss to avoid bias toward majority classes.

3. **Sequence Preservation**: **Do not shuffle** the training data along the depth axis. The model relies on temporal dependencies in sedimentary sequences. Shuffling reduces accuracy to ~20%.

4. **Transfer Learning**: When applying to new formations:
   - If >1000 samples available: Use standard fine-tuning
   - If 500-1000 samples: Use GCN+SVM strategy
   - If <500 samples: Consider domain adaptation or data augmentation

5. **Hyperparameters**: The paper uses 1200 epochs with early stopping (patience=50). Learning rate 5e-4 works best for Adam optimizer.

---

## 📝 License

This project is licensed under MIT License.