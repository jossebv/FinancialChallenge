from dataloader.dataloader import DataLoader
import torch

def main():
    file_path = 'data/processed/es1dia_cln.csv'
    print(f"Testing PyTorch Integration with file: {file_path}")
    
    loader = DataLoader(file_path)
    
    # 1. Load and Split
    print("\n1. Loading and splitting data...")
    train_df, test_df = loader.split_data(test_size=0.20)
    train_df, test_df = loader.create_targets(train_df, test_df)
    
    # 2. Create Torch DataLoaders
    print("\n2. Creating Torch DataLoaders...")
    window_size = 10
    batch_size = 32
    
    train_loader = loader.get_torch_dataloader(train_df, window_size=window_size, batch_size=batch_size, shuffle=True)
    test_loader = loader.get_torch_dataloader(test_df, window_size=window_size, batch_size=batch_size, shuffle=False)
    
    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Test Loader Batches:  {len(test_loader)}")
    
    # 3. Verify Batch Structure
    print("\n3. Verifying Batch Structure (One Batch)...")
    
    # Get first batch
    x_past, x_current, y_reg, y_cls = next(iter(train_loader))
    
    print(f"x_past shape:    {x_past.shape}  (Batch, Window, Features)")
    print(f"x_current shape: {x_current.shape}  (Batch, Features)")
    print(f"y_reg shape:     {y_reg.shape}   (Batch)")
    print(f"y_cls shape:     {y_cls.shape}   (Batch)")
    
    # Check dimensions
    assert x_past.shape == (batch_size, window_size, 3) # ['log_return', 'vol_z', 'log_range']
    assert x_current.shape == (batch_size, 1)           # ['overnight_gap']
    assert y_reg.shape == (batch_size,)
    assert y_cls.shape == (batch_size,)
    
    print("\nVerification Successful!")
    print("Example Targets (Classification):", y_cls[:5].tolist())
    print("Example Targets (Regression):", y_reg[:5].tolist())

if __name__ == "__main__":
    main()
