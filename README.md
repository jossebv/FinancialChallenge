# FinanceChallenge: Market Regime & Return Prediction

This repository implements specific Deep Learning solutions for financial time-series analysis (S&P 500). It covers tasks including Supervised Classification, Regression, Hyperparameter Tuning, Transfer Learning, and Unsupervised Clustering.

## 📂 Project Structure

```
├── data/                   # Processed data
├── dataloader/             # Data ingestion and PyTorch Dataset
│   └── dataloader.py       # Custom SP500Dataset and DataLoader logic
├── model/                  # Deep Learning Architectures
│   ├── model.py            # FiT (Financial Transformer)
│   └── autoencoder.py      # FiTAE (Autoencoder wrapper)
├── tests/                  # Verification scripts for modules
├── checkpoints/            # Saved model weights, not in the shared repository
├── config.yaml             # Main configuration (Task type, Params)
├── config_ae.yaml          # Autoencoder configuration
├── train.py                # Main training script (Supervised)
├── tune.py                 # Hyperparameter tuning (Optuna)
├── finetune.py             # Transfer Learning script
├── train_ae.py             # Autoencoder training
└── cluster_analysis.py     # Unsupervised Regime Discovery
```

## 🚀 Tasks & Usage

### 1. Market Regime Classification / Regression
Train the **FiT (Financial Transformer)** model to predict market direction (Classification) or exact returns (Regression).
1.  Edit `config.yaml`: Set `task_type: "classification"` (or `"regression"`).
2.  Run training:
    ```bash
    python train.py
    ```

### 2. Hyperparameter Tuning
Optimize model parameters (Layers, LR, Kernel Size) using Optuna.
1.  Run the tuning loop:
    ```bash
    python tune.py
    ```
2.  The script optimizes `kernel_size`, `window_size`, `dropout`, etc. and saves the best parameters.

### 3. Transfer Learning
Fine-tune a pre-trained Classification backbone for the Regression task.
1.  Ensure you have a classification checkpoint (e.g., `checkpoints/best_model_classification.pth`).
2.  Run fine-tuning:
    ```bash
    python finetune.py --checkpoint checkpoints/best_model_classification.pth
    ```

### 4. Deep Clustering (Unsupervised)
Discover hidden market regimes using a Transformer Autoencoder.
1.  Train the Autoencoder:
    ```bash
    python train_ae.py
    ```
2.  Analyze clusters (K-Means on Latent Vectors):
    ```bash
    python cluster_analysis.py
    ```

## 🛠 Prerequisites
- Python 3.x
- PyTorch
- Neptune (Logging)
- Optuna (Tuning)
- Scikit-learn, Pandas, NumPy

## 📊 Models
- **FiT**: Hybrid architecture combining Conv1d (Tokenizer), Transformer Encoder, and MLP Heads.
- **FiTAE**: Autoencoder variant that learns to compress and reconstruct market windows.
