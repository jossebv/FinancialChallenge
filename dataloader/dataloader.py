import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from sklearn.preprocessing import StandardScaler

class SP500Dataset(Dataset):
    """
    PyTorch Dataset for SP500 data.
    
    Returns samples (X_past, X_current, y_reg, y_cls) where:
    - X_past: Features from the past W days.
    - X_current: Features for the current day T (e.g., overnight_gap).
    - y_reg: Intraday return for day T.
    - y_cls: Classification target for day T.
    """
    def __init__(self, df, window_size=10, 
                 past_features=['log_return', 'vol_z', 'log_range'], 
                 current_features=['overnight_gap']):
        """
        Args:
            df (pd.DataFrame): Dataframe containing the data.
            window_size (int): Number of past days to include.
            past_features (list): List of feature names for past context.
            current_features (list): List of feature names for current context.
        """
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.past_features = past_features
        self.current_features = current_features
        
        # Pre-convert to tensors to speed up dataloading
        self.data_past = torch.tensor(self.df[past_features].values, dtype=torch.float32)
        self.data_current = torch.tensor(self.df[current_features].values, dtype=torch.float32)
        self.target_reg = torch.tensor(self.df['intraday_return'].values, dtype=torch.float32)
        self.target_cls = torch.tensor(self.df['target_class'].values, dtype=torch.long)

    def __len__(self):
        # We need W days of history, so we start at index W
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # idx 0 corresponds to prediction for day W (using 0 to W-1 as history)
        # Target day T index
        idx_t = idx + self.window_size
        
        # Past Context: [T-W : T]
        x_past = self.data_past[idx : idx_t]
        
        # Current Context: [T]
        x_current = self.data_current[idx_t]
        
        # Targets: [T]
        y_reg = self.target_reg[idx_t]
        y_cls = self.target_cls[idx_t]
        
        return x_past, x_current, y_reg, y_cls

class DataLoader:
    """
    DataLoader class for processing SP500 data for regression and classification.
    """
    def __init__(self, file_path):
        """
        Initialize the DataLoader.
        
        Args:
            file_path (str): Path to the CSV file containing the data.
        """
        self.file_path = file_path
        self.df = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def load_data(self):
        """
        Load data from CSV file and parse timestamps.
        """
        try:
            self.df = pd.read_csv(self.file_path)
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            return self.df
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            raise

    def create_features(self):
        """
        Create features.
        Currently ensures basic feature availability.
        Target creation is handled separately.
        """
        if self.df is None:
            self.load_data()
        pass

    def split_data(self, test_size=0.2):
        """
        Split data into train and test sets using strict time-series split.
        
        Args:
            test_size (float): Fraction of data to use for testing (default 0.2).
            
        Returns:
            tuple: (train_df, test_df)
        """
        if self.df is None:
            self.load_data()
            
        # Ensure data is sorted by date
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        cutpoint = int(len(self.df) * (1 - test_size))
        train_df = self.df.iloc[:cutpoint].copy()
        test_df = self.df.iloc[cutpoint:].copy()
        
        return train_df, test_df

    def create_targets(self, train_df, test_df):
        """
        Create regression and classification targets for train and test sets.
        
        1. Regression Target: Intraday Log Return = ln(Close / Open)
        2. Classification Target: Bearish (0) / Neutral (1) / Bullish (2)
           
        CRITICAL: Thresholds for classification are calculated ONLY on train_df
        to avoid look-ahead bias. These fixed thresholds are then applied to test_df.
        """
        # --- Regression Target ---
        # Calculate for both sets independent of each other
        train_df['intraday_return'] = np.log(train_df['close'] / train_df['open'])
        test_df['intraday_return'] = np.log(test_df['close'] / test_df['open'])

        # --- Classification Target ---
        # Calculate thresholds strictly on TRAIN data
        low_threshold = train_df['intraday_return'].quantile(0.33)
        high_threshold = train_df['intraday_return'].quantile(0.66)
        
        print(f"Classification Thresholds (Calculated on TRAIN only): < {low_threshold:.6f} (Bearish), > {high_threshold:.6f} (Bullish)")

        # Helper function to apply labeling
        def apply_labeling(df, low, high):
             conditions = [
                (df['intraday_return'] <= low),
                (df['intraday_return'] > low) & (df['intraday_return'] <= high),
                (df['intraday_return'] > high)
            ]
             return np.select(conditions, [0, 1, 2], default=1)

        # Apply to Train
        train_df['target_class'] = apply_labeling(train_df, low_threshold, high_threshold)
        
        # Apply strict thresholds to Test
        test_df['target_class'] = apply_labeling(test_df, low_threshold, high_threshold)
        
        return train_df, test_df

    def scale_data(self, train_df, test_df, 
                   features=['log_return', 'vol_z', 'log_range', 'overnight_gap'],
                   scale_target=True):
        """
        Scale features and optionally the regression target using StandardScaler.
        Fits on TRAIN, transforms TRAIN and TEST.
        
        Args:
            train_df, test_df: dataframes
            features: list of columns to scale
            scale_target: whether to scale 'intraday_return'
            
        Returns:
            train_df, test_df (with scaled columns)
        """
        # Fit on Train Features
        self.feature_scaler.fit(train_df[features])
        
        # Transform Features
        train_df[features] = self.feature_scaler.transform(train_df[features])
        test_df[features] = self.feature_scaler.transform(test_df[features])
        
        if scale_target and 'intraday_return' in train_df.columns:
            # Reshape for scalar scaler requirements
            y_train = train_df['intraday_return'].values.reshape(-1, 1)
            y_test = test_df['intraday_return'].values.reshape(-1, 1)
            
            self.target_scaler.fit(y_train)
            
            train_df['intraday_return'] = self.target_scaler.transform(y_train).flatten()
            test_df['intraday_return'] = self.target_scaler.transform(y_test).flatten()
            
            print(f"Target Scaler Mean: {self.target_scaler.mean_[0]:.6f}, Scale: {self.target_scaler.scale_[0]:.6f}")

        return train_df, test_df

    def get_torch_dataloader(self, df, window_size=10, batch_size=32, shuffle=False, 
                             past_features=None, current_features=None):
        """
        Create a PyTorch DataLoader from a dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe with targets already created.
            window_size (int): Window size for past context.
            batch_size (int): Batch size for the dataloader.
            shuffle (boolean): Whether to shuffle the data.
            past_features (list): List of past features names.
            current_features (list): List of current features names.
            
        Returns:
            torch.utils.data.DataLoader: PyTorch DataLoader.
        """
        # Dictionary unpacking or direct passing
        kwargs = {}
        if past_features is not None:
            kwargs['past_features'] = past_features
        if current_features is not None:
            kwargs['current_features'] = current_features
            
        dataset = SP500Dataset(df, window_size=window_size, **kwargs)
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def verify_data(self, train_df, test_df):
        """
        Verify data integrity and class balance for both sets.
        """
        print("\n--- Verification ---")
        
        # Check 'log_return' vs 'intraday_return' for both
        for name, df in [("Train", train_df), ("Test", test_df)]:
            if 'log_return' in df.columns:
                diff = (df['log_return'] - df['intraday_return']).abs().mean()
                print(f"[{name}] Mean Abs Diff (log_return vs intraday_return): {diff:.6f}")

        # Check Class Balance
        print("\nClass Distribution (Train - should be balanced):")
        print(train_df['target_class'].value_counts(normalize=True).sort_index())
        
        print("\nClass Distribution (Test - realistic distribution):")
        print(test_df['target_class'].value_counts(normalize=True).sort_index())
