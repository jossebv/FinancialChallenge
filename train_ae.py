import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import neptune
import numpy as np

from dataloader.dataloader import DataLoader
from model.autoencoder import FiTAE

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training AE", leave=False)
    for x_past, x_curr, _, _ in pbar:
        x_past, x_curr = x_past.to(device), x_curr.to(device)
        
        optimizer.zero_grad()
        
        recon, _ = model(x_past, x_curr)
        
        # Loss: MSE(Reconstructed Past, Original Past)
        loss = criterion(recon, x_past)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for x_past, x_curr, _, _ in loader:
            x_past, x_curr = x_past.to(device), x_curr.to(device)
            
            recon, _ = model(x_past, x_curr)
            
            loss = criterion(recon, x_past)
            running_loss += loss.item()
    
    return running_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description='Train FiT Autoencoder')
    parser.add_argument('--config', type=str, default='config_ae.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load Config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Initialize Neptune
    run = neptune.init_run(
        project="FinancialChallenge",
        source_files=["*.py", "*.yaml"],
        tags=["autoencoder", "unsupervised"]
    )
    run["parameters"] = config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Loading
    print("Loading Data...")
    loader_wrapper = DataLoader(config['data_path'])
    train_df, test_df = loader_wrapper.split_data() # Strict split Still useful to validate generalization of reconstruction
    train_df, test_df = loader_wrapper.create_targets(train_df, test_df)
    
    # Scale Data
    features_to_scale = config['past_features'] + config['current_features']
    print("Normalizing Data...")
    train_df, test_df = loader_wrapper.scale_data(
        train_df, test_df, 
        features=features_to_scale,
        scale_target=False # No target scaling needed for AE (we reconstruct features)
    )
    
    # Create Torch Loaders
    train_loader = loader_wrapper.get_torch_dataloader(
        train_df, 
        window_size=config['window_size'], 
        batch_size=config['batch_size'],
        shuffle=True,
        past_features=config['past_features'],
        current_features=config['current_features']
    )
    test_loader = loader_wrapper.get_torch_dataloader(
        test_df, 
        window_size=config['window_size'], 
        batch_size=config['batch_size'],
        shuffle=False,
        past_features=config['past_features'],
        current_features=config['current_features']
    )
    
    # 2. Model Setup
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
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    criterion = nn.MSELoss()

    # 4. Training Loop
    os.makedirs(config['output_dir'], exist_ok=True)
    best_val_loss = float('inf')
    
    print("\nStarting Autoencoder Training...")
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        # Logging
        log_msg = f"Epoch {epoch+1}/{config['epochs']} | Train Recon Loss: {train_loss:.6f} | Val Recon Loss: {val_loss:.6f}"
        
        run["train/recon_loss"].append(train_loss)
        run["val/recon_loss"].append(val_loss)
        
        print(log_msg)
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['output_dir'], f"best_model_ae.pth")
            torch.save(model.state_dict(), save_path)
            
    print(f"\nAE Training Complete. Best Val Recon Loss: {best_val_loss:.6f}")
    run.stop()

if __name__ == "__main__":
    main()
