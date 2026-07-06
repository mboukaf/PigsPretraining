# PigsBF: Aortic Pressure to Flow Transformer

This repository contains the codebase for a Transformer-based deep learning architecture designed to predict continuous Aortic Flow from Aortic Pressure measurements in swine models. 

## 📂 Project Structure

- `data/`: Contains raw physiological datasets (MATLAB `.mat` files) and experiment tracking logs (`experiments.csv`).
- `models/`: Stores the PyTorch `.pth` model weights (both pre-trained Masked Autoencoder weights and fine-tuned predictive weights).
- `src/`: Contains the executable Python scripts.
  - `main.py`: The core classes (`WindowedBPDataset`, `BPTransformer`) and the original Time-Split training loop.
  - `main_pigs_split.py`: The rigorous Leave-One-Out (Pig-Split) cross-validation training loop utilizing ResNet stems and physiological data augmentation.
- `notebooks/`: Jupyter Notebooks used for extracting evaluation metrics, visualizing Self-Attention heatmaps, and generating Bland-Altman plots.
- `images/`: Stores generated figures and scientific visualizations.

## 🧠 Architecture Highlights
- **V1 Architecture**: Standard 1D Convolutional Stem followed by a vanilla Transformer Encoder.
- **V2 Architecture**: Advanced 1D ResNet block for high-frequency local feature extraction, paired with aggressive physiological data augmentation (Gaussian noise, amplitude scaling) and a strict 35% masking ratio during pre-training to enforce robust generalization across unseen pigs.

## 🛠 Setup

1. Create a virtual environment: 
```bash
python3 -m venv .venv
```
2. Activate the environment: 
```bash
source .venv/bin/activate
```
3. Install the required dependencies: 
```bash
pip install -r requirements.txt
```

## 🚀 Usage

To train the V2 architecture using the Leave-One-Out validation split (Trains on Pigs 4 & 7, Tests on Pig 8):
```bash
python3 src/main_pigs_split.py
```
