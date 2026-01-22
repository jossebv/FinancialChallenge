from dataloader.dataloader import DataLoader
import pandas as pd
import sys

def main():
    file_path = 'data/processed/es1dia_cln.csv'
    print(f"Testing DataLoader with file: {file_path}")
    
    loader = DataLoader(file_path)
    
    try:
        df = loader.load_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)
        
    print("\nSplitting data (Time-series)...")
    train_df, test_df = loader.split_data(test_size=0.20)
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape:  {test_df.shape}")
    
    print("\nCreating targets (Fit on Train, Apply to Test)...")
    train_df, test_df = loader.create_targets(train_df, test_df)
    
    loader.verify_data(train_df, test_df)
    
    # Check date ranges
    print("\nDate Ranges:")
    print(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
