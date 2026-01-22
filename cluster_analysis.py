import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

from dataloader.dataloader import DataLoader
from model.autoencoder import FiTAE

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    future_returns = []
    
    with torch.no_grad():
        for x_past, x_curr, y_reg, _ in tqdm(loader, desc="Extracting Embeddings"):
            x_past, x_curr = x_past.to(device), x_curr.to(device)
            
            _, z = model(x_past, x_curr) # z is (Batch, d_model * 2)
            
            embeddings.append(z.cpu().numpy())
            future_returns.append(y_reg.numpy())
            
    return np.vstack(embeddings), np.hstack(future_returns)

def analyze_clusters(embeddings, returns, n_clusters=4):
    print(f"\nRunning K-Means with K={n_clusters}...")
    
    # Normalize embeddings before clustering (Euclidean distance sensitivity)
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(z_scaled)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'Cluster': labels,
        'Future_Return_Norm': returns
    })
    
    # Analyze
    stats = df.groupby('Cluster')['Future_Return_Norm'].agg(['count', 'mean', 'std', 'min', 'max'])
    stats['freq_pct'] = stats['count'] / len(df) * 100
    
    print("\nCluster Statistics (Future Return Normalized):")
    print(stats)
    
    return stats, df

def main():
    parser = argparse.ArgumentParser(description='Cluster Analysis')
    parser.add_argument('--config', type=str, default='config_ae.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load Config
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data (Use Test set for analysis to see generalization, or ALL data?)
    # Let's use Test set first to be safe, or Train+Test? 
    # Usually clustering analysis is done on the whole dataset to find all regimes.
    # But let's stick to Test for cleanliness, or just handle Train/Test separately.
    # Let's do TEST set analysis.
    loader_wrapper = DataLoader(config['data_path'])
    train_df, test_df = loader_wrapper.split_data()
    _, test_df = loader_wrapper.create_targets(train_df, test_df)
    
    features_to_scale = config['past_features'] + config['current_features']
    _, test_df = loader_wrapper.scale_data(
        train_df, test_df, 
        features=features_to_scale,
        scale_target=False
    )
    
    test_loader = loader_wrapper.get_torch_dataloader(
        test_df, 
        window_size=config['window_size'], 
        batch_size=config['batch_size'],
        shuffle=False,
        past_features=config['past_features'],
        current_features=config['current_features']
    )
    
    # Load Model
    model = FiTAE(
        n_past_features=len(config['past_features']),
        n_current_features=len(config['current_features']),
        window_size=config['window_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        kernel_size=config.get('kernel_size', 1)
    ).to(device)
    
    checkpoint_path = os.path.join(config['output_dir'], "best_model_ae.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded AE from {checkpoint_path}")
    
    # Extract
    embeddings, returns = extract_embeddings(model, test_loader, device)
    
    # Cluster
    analyze_clusters(embeddings, returns, n_clusters=4)

if __name__ == "__main__":
    main()
