# TaskAwareCompressor

### Task-Aware Image Reconstruction via Normalized Loss Balancing

**A Deep Learning framework for generating compact, machine-optimized image representations that prioritize classification accuracy over human visual fidelity.**

-----

## 📖 Overview

As machine vision systems migrate to edge devices, traditional compression methods (like JPEG) become a bottleneck. They optimize for human perception, often discarding high-frequency features essential for neural networks while preserving visually pleasing but semantically irrelevant details.

**TaskAwareCompressor** introduces a **Normalized Loss Balancing** framework. By repurposing a pre-trained ResNet-18 encoder and training custom decoders, we generate abstract image reconstructions that:

1.  **Maintain or Improve Accuracy:** Achieves **78.40%** accuracy on CIFAR-10 (vs. 77.71% baseline).
2.  **Reduce Storage:** Achieves **15-23%** natural compaction compared to raw inputs.
3.  **Enable Privacy:** Produces visually obfuscated images that are unrecognizable to humans but interpretable by machines.

This project solves the instability of multi-objective optimization (Accuracy vs. Reconstruction) using **Exponential Moving Averages (EMA)** to normalize loss scales, creating a stable, interpretable control parameter $\alpha \in [0, 1]$.

-----

## 🚀 Key Features

  * **Normalized Loss Balancing:** A novel loss formulation that stabilizes the scale disparity between Cross-Entropy (CE) and Mean Squared Error (MSE).
    $$L_{total} = CE_{norm} + \alpha \cdot MSE_{norm}$$
  * **Multi-Stage Evaluation:** \* **Binary Sweep:** Exhaustive hyperparameter tuning on a binary subset (Cat vs. Dog).
      * **Multiclass Expansion:** Scalable validation on the full CIFAR-10 dataset.
  * **Modular Decoder Architectures:**
      * **Simple Decoder:** Optimized for maximum compaction.
      * **Mirrored ResNet (MR) Decoder:** Optimized for feature fidelity.
  * **Privacy by Design:** High extractor specificity means images are only useful to the specific model trained on them, preventing unauthorized visual inspection.
  * **Interactive Dashboard:** Includes a Streamlit app to visualize reconstruction trade-offs and metrics.

-----

## 🛠️ System Requirements & Installation

This project relies on **Python 3.10**. We recommend using Anaconda to manage dependencies.

### 1\. Environment Setup

```bash
# Create a new environment with Python 3.10
conda create -n miniproject python=3.10 -y

# Activate the environment
conda activate miniproject
```

### 2\. Install Dependencies

You can install the core scientific stack via Conda, and the specific library versions via pip to ensure reproducibility.

**Option A: Quick Install**

```bash
# Core Utilities
conda install numpy pandas matplotlib seaborn scikit-learn jupyter -c conda-forge -y
pip install streamlit==1.28.1 plotly==5.17.0 pillow==10.0.0 tqdm==4.66.1
```

**Option B: PyTorch Installation**

Select the command matching your hardware:

  * **For CPU (Universal):**

    ```bash
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch -y
    ```

  * **For GPU (NVIDIA with CUDA 11.8/12.1):**

    ```bash
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ```

### 3\. Jupyter Setup

To run the notebooks:

```bash
conda install ipykernel -c conda-forge -y
python -m ipykernel install --user --name=miniproject --display-name "Python (miniproject)"
jupyter notebook
```

-----

## 📂 Project Structure

```text
.
├── Cifar10_binaryClassifier_Train.ipynb  # Initial Binary classifier training
├── Cifar10_FullClassifer_Train.ipynb     # Full Multiclass classifier training
├── config.py                             # Global configuration variables
├── streamlit_dashboard.py                # Interactive results dashboard
├── model.pth                             # Saved model states
│
├── Decoders/                             # Saved binary decoder models
│   ├── Accuracy/                         # Models optimized for Accuracy
│   ├── CE/                               # Models optimized for Cross-Entropy
│   ├── Compression/                      # Models optimized for Size
│   └── Task-Aware/                       # Balanced models
│
├── HyperparameterSweep/                  # BINARY EXPERIMENTS (Phase 1)
│   ├── ...HyperparamTuning.ipynb         # Notebooks for Simple/MR Decoders
│   ├── decoder_summary_results_full.csv  # Aggregated results
│   └── PlotsAndObservations.ipynb        # Analysis of binary sweep
│
├── MulticlassExpansion/                  # MULTICLASS EXPERIMENTS (Phase 2)
│   ├── DeepAE.ipynb                      # Deep Autoencoder Baseline comparison
│   ├── MultiClassTargetedEvaluation.ipynb# Strategic Alpha evaluation
│   ├── NewClassifierTests/               # Extractor specificity tests
│   └── ...TargetAlphaTraining.ipynb      # Training scripts for optimal Alphas
│
├── activation_plots/                     # Visualizations of Tanh vs Sigmoid
├── architectural_plots/                  # Visualizations of Simple vs MR
└── custom_dataset/                       # Sample images for testing
```

-----

## 📊 Usage Guide

### 1\. Training & Experimentation

  * **Binary Phase:** Start with `HyperparameterSweep/SimpleTanhDecoderHyperparamTuning.ipynb` to see how $\alpha$ impacts the trade-off between accuracy and compression on the binary subset.
  * **Multiclass Phase:** Navigate to `MulticlassExpansion/` to explore how these findings scale to the full CIFAR-10 dataset using `MultiClassTargetedEvaluation.ipynb`.

### 2\. Interactive Dashboard

We provide a Streamlit dashboard to easily visualize the results and compare original vs. reconstructed images.

```bash
streamlit run streamlit_dashboard.py
```

-----

## 📈 Key Results

Our proposed method outperforms traditional baselines (JPEG) and standard Deep Autoencoders.

| Method | Accuracy | Cross-Entropy | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline (Raw Images)** | **77.71%** | 1.0591 | Reference |
| **Our Method (Simple Tanh)** | **78.40%** | 1.2725 | **+0.69% Accuracy Gain** |
| Deep Autoencoder | 69.37% | 1.1514 | Optimized only for MSE |
| Lossy Processing (JPEG) | \~45.00% | 2.2292 | Catastrophic failure |

  * **Specificity:** Classifiers trained on our reconstructions perform poorly on original images (11.23% accuracy), proving the method effectively obfuscates visual data while retaining machine utility.

-----

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. You are free to use, modify, and distribute this code, provided that derivative works remain open-source. See [LICENSE](https://www.google.com/search?q=./LICENSE) for details.

-----

## 👥 Acknowledgments

  * **Authors:** Rishabh Ranjan, Mumtaz Parveen, Mithun Raj, Mallanagowda Hosamani.
  * **Guide:** Mr. Muhammad Owais (Asst. Professor, Dept of CSE, ACSCE).
  * **Institution:** ACS College of Engineering, Bangalore (Affiliated to VTU).
  * **Resources:** Original ResNet-18 implementation via PyTorch Torchvision; CIFAR-10 Dataset.
