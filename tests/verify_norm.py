from dataloader.dataloader import DataLoader
import numpy as np

def main():
    file_path = 'data/processed/es1dia_cln.csv'
    print(f"Testing Data Normalization with file: {file_path}")
    
    loader = DataLoader(file_path)
    
    # 1. Load, Split, Create Targets
    print("\n1. Loading, Splitting and Target Creation...")
    train_df, test_df = loader.split_data(test_size=0.20)
    train_df, test_df = loader.create_targets(train_df, test_df)
    
    # Store original stats for comparison
    orig_train_mean_ret = train_df['log_return'].mean()
    orig_train_std_ret = train_df['log_return'].std()
    
    # 2. Scale Data
    print("\n2. Scaling Data (Features + Target)...")
    features_to_scale = ['log_return', 'vol_z', 'log_range', 'overnight_gap']
    train_df, test_df = loader.scale_data(train_df, test_df, features=features_to_scale, scale_target=True)
    
    # 3. Verify Normalization (Train set should be ~0 mean, ~1 std)
    print("\n3. Verification of Scaled Features (Train Set):")
    for feat in features_to_scale:
        mean = train_df[feat].mean()
        std = train_df[feat].std()
        print(f"  {feat}: Mean = {mean:.4f}, Std = {std:.4f}")
        # Assertions (relaxed tolerance due to ddof differences: sklearn=0, pandas=1)
        assert np.abs(mean) < 1e-5, f"{feat} Mean not 0"
        assert np.abs(std - 1.0) < 1e-3, f"{feat} Std not 1"

    print("\n4. Verification of Scaled Target (Train Set):")
    mean_target = train_df['intraday_return'].mean()
    std_target = train_df['intraday_return'].std()
    print(f"  intraday_return: Mean = {mean_target:.4f}, Std = {std_target:.4f}")
    assert np.abs(mean_target) < 1e-5
    assert np.abs(std_target - 1.0) < 1e-3
    
    # 4. Check Test Set (Should NOT be exactly 0/1, but close-ish assuming stationarity)
    print("\n5. Check Test Set (Transformed using Train stats):")
    for feat in features_to_scale:
        mean = test_df[feat].mean()
        std = test_df[feat].std()
        print(f"  {feat}: Mean = {mean:.4f}, Std = {std:.4f}")

    print("\nValidation Successful!")

if __name__ == "__main__":
    main()
