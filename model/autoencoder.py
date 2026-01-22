import torch
import torch.nn as nn
import math
from model.model import FiT

class FiTAE(FiT):
    """
    FiT Autoencoder (FiTAE) for Unsupervised clustering.
    Inherits from FiT to reuse the Encoder backbone.
    Adds a Decoder to reconstruct the past market window.
    """
    def __init__(self, n_past_features, n_current_features, window_size, 
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 dropout=0.1, kernel_size=1):
        
        # Initialize FiT with 'regression' just to pass the check, we won't use self.head
        super(FiTAE, self).__init__(
            n_past_features, n_current_features, window_size, 
            d_model, nhead, num_layers, dim_feedforward, 
            dropout, task_type='regression', kernel_size=kernel_size
        )
        
        self.n_past_features = n_past_features
        self.window_size = window_size
        
        # Decoder
        # Input: Latent Z (d_model * 2) -> Output: Reconstructed Window (Window * PastFeatures)
        # We use a simple MLP decoder for now. Could be an LSTM or Deconv.
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, window_size * n_past_features)
        )
        
    def forward(self, x_past, x_current):
        """
        Returns:
            recon: Reconstructed x_past (Batch, Window, Features)
            embedding: Latent representation (Batch, d_model * 2)
        """
        # --- 1. Tokenization (Same as FiT) ---
        x_past_t = x_past.transpose(1, 2) # (B, F_past, W)
        tokens = self.tokenizer(x_past_t) # (B, d_model, W)
        tokens = tokens.transpose(1, 2)   # (B, W, d_model)
        
        # --- 2. Transformer (Same as FiT) ---
        tokens = tokens * math.sqrt(self.d_model)
        tokens = tokens.permute(1, 0, 2) # (Seq, Batch, Dim)
        tokens = self.pos_encoder(tokens)
        tokens = tokens.permute(1, 0, 2) # (Batch, Seq, Dim)
        
        past_rep = self.transformer_encoder(tokens) # (B, W, d_model)
        past_rep_pooled = past_rep.mean(dim=1)      # (B, d_model)
        
        # --- 3. Fusion (Same as FiT) ---
        current_rep = self.current_projector(x_current) # (B, d_model)
        fusion_rep = torch.cat([past_rep_pooled, current_rep], dim=1) # (B, d_model * 2)
        
        # --- 4. Decoding (New) ---
        recon_flat = self.decoder(fusion_rep) # (B, W * F)
        recon = recon_flat.view(x_past.size(0), self.window_size, self.n_past_features)
        
        return recon, fusion_rep
