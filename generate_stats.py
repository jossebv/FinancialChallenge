from dataloader.dataloader import DataLoader
import pandas as pd
import sys

def main():
    file_path = 'data/processed/es1dia_cln.csv'
    output_file = 'data_stats.txt'
    
    loader = DataLoader(file_path)
    
    # Validation/Loading
    try:
        loader.load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Split and Target Creation
    train_df, test_df = loader.split_data(test_size=0.20)
    train_df, test_df = loader.create_targets(train_df, test_df)

    # Generate Report
    with open(output_file, 'w') as f:
        f.write("SP500 Data Statistics Report\n")
        f.write("============================\n\n")
        
        # 1. Split Info
        f.write("1. Train/Test Split (Strict Time-Series)\n")
        f.write("----------------------------------------\n")
        f.write(f"Train Set: {len(train_df)} samples\n")
        f.write(f"  Range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}\n")
        f.write(f"Test Set:  {len(test_df)} samples\n")
        f.write(f"  Range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}\n")
        f.write(f"Total:     {len(train_df) + len(test_df)} samples\n\n")
        
        # 2. Class Distribution
        f.write("2. Class Distribution (0=Bearish, 1=Neutral, 2=Bullish)\n")
        f.write("-------------------------------------------------------\n")
        f.write("Train Set (Reference - Balanced):\n")
        f.write(train_df['target_class'].value_counts(normalize=True).sort_index().to_string())
        f.write("\n\n")
        
        f.write("Test Set (Realistic - Imbalanced):\n")
        f.write(test_df['target_class'].value_counts(normalize=True).sort_index().to_string())
        f.write("\n\n")
        
        # 3. Thresholds
        f.write("3. Classification Thresholds (Derived from Train)\n")
        f.write("-------------------------------------------------\n")
        low = train_df['intraday_return'].quantile(0.33)
        high = train_df['intraday_return'].quantile(0.66)
        f.write(f"Bearish Threshold (< 33%): {low:.6f}\n")
        f.write(f"Bullish Threshold (> 66%): {high:.6f}\n")

    print(f"Statistics written to {output_file}")

if __name__ == "__main__":
    main()
