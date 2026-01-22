from model.model import FiT
from dataloader.dataloader import DataLoader
import torch

def main():
    print("Testing FiT (Financial Transformer) Model Architecture...")
    
    # 1. Setup Dummy Data Config
    batch_size = 32
    window_size = 10
    n_past_features = 3    # ['log_return', 'vol_z', 'log_range']
    n_current_features = 1 # ['overnight_gap']
    d_model = 64
    
    # Dummy Inputs
    x_past = torch.randn(batch_size, window_size, n_past_features)
    x_current = torch.randn(batch_size, n_current_features)
    
    print("\nInput Shapes:")
    print(f"x_past:    {x_past.shape}")
    print(f"x_current: {x_current.shape}")
    
    # 2. Test Regression Mode
    print("\n--- Testing Regression Mode ---")
    model_reg = FiT(
        n_past_features=n_past_features,
        n_current_features=n_current_features,
        window_size=window_size,
        d_model=d_model,
        task_type='regression'
    )
    
    output_reg = model_reg(x_past, x_current)
    print(f"Regression Output Shape: {output_reg.shape}")
    assert output_reg.shape == (batch_size, 1)
    
    # 3. Test Classification Mode
    print("\n--- Testing Classification Mode ---")
    model_cls = FiT(
        n_past_features=n_past_features,
        n_current_features=n_current_features,
        window_size=window_size,
        d_model=d_model,
        task_type='classification'
    )
    
    output_cls = model_cls(x_past, x_current)
    print(f"Classification Output Shape: {output_cls.shape}")
    assert output_cls.shape == (batch_size, 3)
    
    print("\nFiT Architecture Verification Successful!")

if __name__ == "__main__":
    main()
