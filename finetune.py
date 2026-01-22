import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import neptune
import copy

from dataloader.dataloader import DataLoader
from model.model import FiT

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_backbone(model, checkpoint_path):
    """
    Load weights from checkpoint, excluding the head.
    """
    print(f"Loading backbone from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Filter out head weights
    # The head layer in FiT is usually self.head.net if using the Sequential definition
    # Let's inspect model.py structure implicitly. 
    # Usually: 'head.net.0.weight', etc. depending on definition.
    # We will exclude any key starting with 'head'.
    
    backbone_dict = {k: v for k, v in checkpoint.items() if not k.startswith('head')}
    
    # Load with strict=False because we are missing head weights
    missing, unexpected = model.load_state_dict(backbone_dict, strict=False)
    print(f"Missing keys (Expected, as we are replacing head): {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, task_type):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Finetuning", leave=False)
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
    
    return running_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description='Finetune FiT Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pre-trained classification checkpoint')
    args = parser.parse_args()
    
    # Load Config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # OVERRIDE TASK TYPE FOR FINETUNING
    config['task_type'] = 'regression'
    print(f"Task set to: {config['task_type']}")
    
    # Initialize Neptune
    run = neptune.init_run(
        project="FinancialChallenge",
        source_files=["*.py", "*.yaml"],
        tags=["finetuning", "regression"]
    )
    run["parameters"] = config
    run["parameters/pretrained_checkpoint"] = args.checkpoint
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Loading
    print("Loading Data...")
    loader_wrapper = DataLoader(config['data_path'])
    train_df, test_df = loader_wrapper.split_data()
    train_df, test_df = loader_wrapper.create_targets(train_df, test_df)
    
    # Scale Data
    features_to_scale = config['past_features'] + config['current_features']
    print("Normalizing Data...")
    train_df, test_df = loader_wrapper.scale_data(
        train_df, test_df, 
        features=features_to_scale,
        scale_target=True # Always scale target for regression
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
    # Initialize with Config (Classification Architecture) but Regression Task Type
    model = FiT(
        n_past_features=len(config['past_features']),
        n_current_features=len(config['current_features']),
        window_size=config['window_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        task_type='regression', # Force Regression Head initialization
        kernel_size=config.get('kernel_size', 1)
    ).to(device)
    
    # Load Backbone
    model = load_backbone(model, args.checkpoint)
    
    # 3. Optimization
    # Use config LR? Maybe lower? Let's use config LR first.
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    criterion = nn.MSELoss()

    # 4. Training Loop
    os.makedirs(config['output_dir'], exist_ok=True)
    best_val_loss = float('inf')
    
    print("\nStarting Fine-tuning...")
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, 'regression')
        val_loss = validate(model, test_loader, criterion, device, 'regression')
        
        scheduler.step()
        
        # Logging
        log_msg = f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        
        run["train/loss"].append(train_loss)
        run["val/loss"].append(val_loss)
        
        print(log_msg)
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['output_dir'], f"best_model_finetuned_regression.pth")
            torch.save(model.state_dict(), save_path)

    print(f"\nFine-tuning Complete. Best Val Loss: {best_val_loss:.6f}")
    run.stop()

if __name__ == "__main__":
    main()
