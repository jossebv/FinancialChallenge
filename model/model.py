import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FiT(nn.Module):
    """
    FiT: Financial Transformer
    
    Architecture:
    1. Tokenizer (Conv1d): Projects past features into d_model dimensional space.
    2. Transformer Backbone: Processes temporal dependencies.
    3. Fusion: Integrates current market context (e.g. overnight gap).
    4. Head: Task-specific output (Regression or Classification).
    """
    def __init__(self, n_past_features, n_current_features, window_size, 
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 dropout=0.1, task_type='regression', kernel_size=1):
        """
        Args:
           n_past_features (int): Number of features in the past window sequence.
           n_current_features (int): Number of features in the current context.
           window_size (int): Length of the past window.
           d_model (int): Latent dimension.
           nhead (int): Number of attention heads.
           num_layers (int): Number of transformer encoder layers.
           dim_feedforward (int): Feedforward network dimension.
           dropout (float): Dropout rate.
           task_type (str): 'regression' for return prediction, 'classification' for regime.
           kernel_size (int): Kernel size for the Conv1d tokenizer.
        """
        super(FiT, self).__init__()
        self.task_type = task_type
        self.d_model = d_model

        # 1. Tokenizer (Conv1d)
        # Input: (Batch, Features, Window) -> Output: (Batch, d_model, Window)
        # padding='same' ensures output length == input length (Window)
        # Note: kernel_size must be odd for padding='same' usually, but PyTorch handles it if stride=1
        self.tokenizer = nn.Conv1d(
            in_channels=n_past_features, 
            out_channels=d_model, 
            kernel_size=kernel_size,
            padding='same' 
        )
        
        # 2. Transformer Backbone
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size + 10)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # Important: Input shape is (Batch, Seq_Len, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 3. Fusion
        # Project current context to d_model to make it compatible
        self.current_projector = nn.Linear(n_current_features, d_model)
        
        # 4. Heads
        fusion_dim = d_model * 2 # Concatenation strategy
        
        if task_type == 'regression':
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 1)
            )
        elif task_type == 'classification':
             self.head = nn.Sequential(
                nn.Linear(fusion_dim, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 3) # Bearish, Neutral, Bullish
            )
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

    def forward(self, x_past, x_current):
        """
        Args:
            x_past: (Batch, Window, Past Features)
            x_current: (Batch, Current Features)
        """
        # --- 1. Tokenization ---
        # Conv1d expects (Batch, Channels, Length), so we transpose x_past
        x_past_t = x_past.transpose(1, 2) # (B, F_past, W)
        tokens = self.tokenizer(x_past_t) # (B, d_model, W)
        
        # Transpose back for Transformer (Batch, Seq, Dim)
        tokens = tokens.transpose(1, 2)   # (B, W, d_model)
        
        # --- 2. Transformer ---
        # Add PE (expects permuted inputs inside PositionalEncoding if not batch_first?)
        # Standard implementation expects [Seq, Batch, Dim], but we set batch_first=True in EncoderLayer.
        # However, custom PositionalEncoding might expect [Seq, Batch, Dim].
        # Let's adjust PositionalEncoding usage.
        
        # Standard PE usually expects [Seq, Batch, Dim]. 
        # My implementation is: x + self.pe[:x.size(0), :] -> slice on dim 0.
        # So I need to transpose tokens to [Seq, Batch, Dim] for PE, then transpose back?
        # Or fix PE to handle batch_first=True. 
        # Let's use simple scaling + PE.
        
        tokens = tokens * math.sqrt(self.d_model)
        # Permute for PE: (Seq, Batch, Dim)
        tokens = tokens.permute(1, 0, 2)
        tokens = self.pos_encoder(tokens)
        # Permute back: (Batch, Seq, Dim)
        tokens = tokens.permute(1, 0, 2)
        
        # Encoder
        past_rep = self.transformer_encoder(tokens) # (B, W, d_model)
        
        # Pool (Global Average Pooling over time)
        past_rep_pooled = past_rep.mean(dim=1)      # (B, d_model)
        
        # --- 3. Fusion ---
        current_rep = self.current_projector(x_current) # (B, d_model)
        
        # Concatenate
        fusion_rep = torch.cat([past_rep_pooled, current_rep], dim=1) # (B, d_model * 2)
        
        # --- 4. Prediction ---
        output = self.head(fusion_rep)
        
        return output
