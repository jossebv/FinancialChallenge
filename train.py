import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy as np
import neptune
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataloader.dataloader import DataLoader
from model.model import FiT

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, criterion, optimizer, device, task_type):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for x_past, x_curr, y_reg, y_cls in pbar:
        x_past, x_curr = x_past.to(device), x_curr.to(device)
        y_reg, y_cls = y_reg.to(device), y_cls.to(device)
        
        optimizer.zero_grad()
        
        output = model(x_past, x_curr)
        
        if task_type == 'regression':
            loss = criterion(output.squeeze(), y_reg)
        else: # classification
            loss = criterion(output, y_cls)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device, task_type):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_past, x_curr, y_reg, y_cls in loader:
            x_past, x_curr = x_past.to(device), x_curr.to(device)
            y_reg, y_cls = y_reg.to(device), y_cls.to(device)
            
            output = model(x_past, x_curr)
            
            if task_type == 'regression':
                loss = criterion(output.squeeze(), y_reg)
                running_loss += loss.item()
            else: # classification
                loss = criterion(output, y_cls)
                running_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += y_cls.size(0)
                correct += (predicted == y_cls).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

def log_confusion_matrix(model, loader, device, run):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_past, x_curr, _, y_cls in loader:
            x_past, x_curr = x_past.to(device), x_curr.to(device)
            y_cls = y_cls.to(device)
            
            output = model(x_past, x_curr)
            _, predicted = torch.max(output.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_cls.cpu().numpy())
            
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bearish', 'Neutral', 'Bullish'],
                yticklabels=['Bearish', 'Neutral', 'Bullish'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Log to Neptune
    run["confusion_matrix"].upload(plt.gcf())
    plt.close()

def train_model(config, run_neptune=True):
    # Initialize Neptune
    run = None
    if run_neptune:
        run = neptune.init_run(
            project="FinancialChallenge",
            source_files=["*.py", "*.yaml"],
            tags=[config['task_type']]
        )
        run["parameters"] = config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Loading
    print("Loading Data...")
    loader_wrapper = DataLoader(config['data_path'])
    train_df, test_df = loader_wrapper.split_data() # Uses default 0.2 split
    train_df, test_df = loader_wrapper.create_targets(train_df, test_df)
    
    # Scale Data
    features_to_scale = config['past_features'] + config['current_features']
    
    print("Normalizing Data...")
    train_df, test_df = loader_wrapper.scale_data(
        train_df, test_df, 
        features=features_to_scale,
        scale_target=(config['task_type'] == 'regression')
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
    model = FiT(
        n_past_features=len(config['past_features']),
        n_current_features=len(config['current_features']),
        window_size=config['window_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        task_type=config['task_type'],
        kernel_size=config.get('kernel_size', 1)
    ).to(device)
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    if config['task_type'] == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    os.makedirs(config['output_dir'], exist_ok=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config['task_type'])
        val_loss, val_acc = validate(model, test_loader, criterion, device, config['task_type'])
        
        scheduler.step()
        
        # Logging
        log_msg = f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        
        if run:
            run["train/loss"].append(train_loss)
            run["val/loss"].append(val_loss)
        
        if config['task_type'] == 'classification':
             log_msg += f" | Val Acc: {val_acc:.2f}%"
             if run:
                 run["val/acc"].append(val_acc)
             
             if val_acc > best_val_acc:
                 best_val_acc = val_acc
        
        print(log_msg)
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['output_dir'], f"best_model_{config['task_type']}.pth")
            torch.save(model.state_dict(), save_path)
    
    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.6f}")
    
    # 5. Confusion Matrix (Classification only)
    if config['task_type'] == 'classification' and run:
        print("Generating Confusion Matrix...")
        best_model_path = os.path.join(config['output_dir'], f"best_model_{config['task_type']}.pth")
        model.load_state_dict(torch.load(best_model_path))
        log_confusion_matrix(model, test_loader, device, run)
        
    if run:
        run.stop()
        
    return best_val_loss, best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Train FiT Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load Config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Task: {config['task_type']}")
    
    train_model(config)

if __name__ == "__main__":
    main()
