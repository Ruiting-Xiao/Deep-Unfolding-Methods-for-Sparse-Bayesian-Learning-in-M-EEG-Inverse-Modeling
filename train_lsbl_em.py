#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Source Localization using L-SBL with Pre-normalized Dataset (train + valA + valB)
======================================================================================

This script implements a Learned Sparse Bayesian Learning (L-SBL) neural network
for EEG source localization. The model learns to transform the classical EM algorithm
into trainable neural network layers through algorithm unrolling.

Key Features:
-------------
1. Mode: EM algorithm with two-term weighted update
   - γ_new = w1_pos * term1 + w2_pos * term2
   - w1, w2 are ensured to be non-negative using Softplus activation
   
2. Loss Functions:
   - MSE: Mean Squared Error
   - MAE: Mean Absolute Error (L1)
   - Adaptive: (1-λ)*MSE_normalized + λ*MAE_normalized with learnable λ
   
3. Data Structure:
   - Uses pre-normalized datasets (train + valA + valB)
   - train: uses global_G_norm_train.npy
   - valA: uses global_G_norm_valA.npy (same head model as train, independent file)
   - valB: uses global_G_norm_valB.npy (validation head model)
   
4. Training Strategy:
   - Layer-wise training approach
   - Pass 1: Train only the current layer
   - Pass 3: Fine-tune all layers up to current layer
   - Dynamic epoch allocation based on layer depth:
     * Layers 0-24: 3 epochs
     * Layers 25-49: 3 epochs  
     * Layers 50+: 3 epochs
   - Weight copying from previous layer during initialization
   
5. Memory Management:
   - Sub-batch processing to reduce GPU memory usage
   - Supports large batch files by splitting into smaller sub-batches

Architecture:
-------------
The L-SBL network consists of T layers, each containing:
- E-step: Posterior mean and variance estimation using Cholesky decomposition
- M-step: Hyperparameter (gamma) update using weighted EM terms

Author: [Original Author]
Date: 2024
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ================================
# Command Line Arguments
# ================================
parser = argparse.ArgumentParser(description='L-SBL EEG Source Localization Training')
parser.add_argument('-dF', '--data_folder', type=str, default='./normalized_data',
                    help="Path to pre-normalized data folder")
parser.add_argument('-T', '--T', type=int, default=100, 
                    help="Number of L-SBL layers")
parser.add_argument('-batch_size', '--batch_size', type=int, default=128,
                    help="Batch size used when generating .npy files")
parser.add_argument('-sub_batch_size', '--sub_batch_size', type=int, default=128,
                    help="Actual training batch size (split samples from .npy files)")
parser.add_argument('-num_workers', '--num_workers', type=int, default=2,
                    help="Number of data loading workers (recommend 0 for stable memory)")
parser.add_argument('-lr', '--lr', type=float, default=0.0001, 
                    help="Learning rate")
parser.add_argument('-loss_type', '--loss_type', type=str, default='mse',
                    choices=['mse', 'mae', 'adaptive'],
                    help="Loss function type: 'mse', 'mae', or 'adaptive'")
parser.add_argument('-init_lambda', '--init_lambda', type=float, default=0.5,
                    help="Initial lambda for adaptive loss (default: 0.5)")
parser.add_argument('-momentum', '--momentum', type=float, default=0.99,
                    help="Momentum for moving average in adaptive loss (default: 0.99)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================
# Dataset Class
# ================================
class NormalizedEEGDataset(Dataset):
    """
    Dataset class for loading pre-normalized EEG data.
    
    Each batch consists of 4 files:
        - X_norm: Normalized source signals
        - Y_norm: Normalized sensor measurements
        - Lambda_norm: Normalized noise covariance
        - M_norm: Normalization factors
        
    G_norm and G_norm_const are loaded globally once (not per batch).
    
    Parameters:
    -----------
    data_folder : str
        Path to the folder containing pre-normalized data
    split : str
        Dataset split - one of {'train', 'valA', 'valB'}
    g_tag : str
        Tag for G matrix files - corresponds to split:
        - 'train' → global_G_norm_train.npy
        - 'valA'  → global_G_norm_valA.npy
        - 'valB'  → global_G_norm_valB.npy
    
    Attributes:
    -----------
    n_batches : int
        Number of batch files in this split
    batch_size : int
        Number of samples per batch file
    M : int
        Number of EEG sensors
    N : int
        Number of source locations
    L : int
        Number of time points
    G_norm : torch.Tensor
        Normalized lead field matrix, shape (1, M, N)
    G_norm_const : torch.Tensor
        Normalization constant for G, shape (1, 1, 1)
    """
    
    def __init__(self, data_folder, split='train', g_tag='train'):
        self.data_folder = data_folder
        self.split = split
        self.g_tag = g_tag
        
        # Load metadata
        metadata_path = os.path.join(data_folder, 'metadata.npz')
        md = np.load(metadata_path)
        
        # Set batch count and prefix based on split
        if split == 'train':
            self.n_batches = int(md["n_train_batches"])
            self.prefix = 'train'
        elif split == 'valA':
            self.n_batches = int(md["n_valA_batches"])
            self.prefix = 'valA'
        elif split == 'valB':
            self.n_batches = int(md["n_valB_batches"])
            self.prefix = 'valB'
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.batch_size = int(md["batch_size"])
        self.M = int(md["n_sensors"])
        self.N = int(md["n_sources"])
        self.L = int(md["n_time"])
        
        print(f"\nLoading {split} dataset: {self.n_batches} batches (prefix = {self.prefix})")
        
        # Load global G_norm and G_norm_const based on g_tag
        print(f"  Loading global G_norm for tag = {g_tag} ...")
        
        if g_tag == 'train':
            g_name = "global_G_norm_train.npy"
            g_const_name = "global_G_norm_const_train.npy"
        elif g_tag == 'valA':
            g_name = "global_G_norm_valA.npy"
            g_const_name = "global_G_norm_const_valA.npy"
        elif g_tag == 'valB':
            g_name = "global_G_norm_valB.npy"
            g_const_name = "global_G_norm_const_valB.npy"
        else:
            raise ValueError(f"Unknown g_tag: {g_tag}")
        
        self.G_norm = torch.from_numpy(
            np.load(os.path.join(data_folder, g_name))
        ).float().to(device)  # shape (M, N)
        
        self.G_norm_const = torch.from_numpy(
            np.load(os.path.join(data_folder, g_const_name))
        ).float().to(device)  # shape (1,) or scalar
        
        # Adjust dimensions
        self.G_norm = self.G_norm.unsqueeze(0)  # (1, M, N)
        self.G_norm_const = self.G_norm_const.view(1, 1, 1)  # (1, 1, 1)
        
        self.file_indices = list(range(self.n_batches))
    
    def __len__(self):
        """
        Returns the number of batch files in the dataset.
        
        Returns:
        --------
        int
            Number of batches
        """
        return self.n_batches
    
    def _load_file(self, idx):
        """
        Load a single batch file.
        
        Parameters:
        -----------
        idx : int
            Batch file index
            
        Returns:
        --------
        tuple of torch.Tensor
            (X_norm, Y_norm, Lambda_norm, M_norm)
        """
        base = os.path.join(self.data_folder, f"{self.prefix}")
        
        X_norm = np.load(f"{base}_X_norm_{idx:04d}.npy", mmap_mode='r')
        Y_norm = np.load(f"{base}_Y_norm_{idx:04d}.npy", mmap_mode='r')
        Lambda_norm = np.load(f"{base}_Lambda_norm_{idx:04d}.npy", mmap_mode='r')
        M_norm = np.load(f"{base}_M_norm_{idx:04d}.npy", mmap_mode='r')
        
        return (torch.from_numpy(X_norm).float(),
                torch.from_numpy(Y_norm).float(),
                torch.from_numpy(Lambda_norm).float(),
                torch.from_numpy(M_norm).float())
    
    def __getitem__(self, idx):
        """
        Get a batch by index.
        
        Parameters:
        -----------
        idx : int
            Batch index
            
        Returns:
        --------
        tuple of torch.Tensor
            (X_norm, Y_norm, Lambda_norm, M_norm)
        """
        return self._load_file(idx)


# ================================
# Adaptive Mixed Loss Function
# ================================
class AdaptiveMixedLoss(nn.Module):
    """
    Adaptive mixed loss function: (1-λ)*MSE_normalized + λ*MAE_normalized
    
    The weight λ is learnable and constrained to [0, 1] using sigmoid activation.
    MSE and MAE are normalized to the same scale using exponential moving averages.
    
    Parameters:
    -----------
    initial_lambda : float, optional
        Initial value for λ (default: 0.5)
    momentum : float, optional
        Momentum for exponential moving average (default: 0.99)
        
    Attributes:
    -----------
    lambda_logit : nn.Parameter
        Learnable parameter for λ (before sigmoid)
    mse_scale : torch.Tensor
        Moving average scale for MSE normalization
    mae_scale : torch.Tensor
        Moving average scale for MAE normalization
    initialized : torch.Tensor
        Flag indicating if scales have been initialized
    """
    
    def __init__(self, initial_lambda=0.5, momentum=0.99):
        super().__init__()
        
        # Initialize lambda_logit such that sigmoid(lambda_logit) = initial_lambda
        if initial_lambda <= 0 or initial_lambda >= 1:
            initial_lambda = 0.5
        initial_logit = np.log(initial_lambda / (1 - initial_lambda))
        self.lambda_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        self.momentum = momentum
        
        # Buffers for moving averages (not model parameters)
        self.register_buffer('mse_scale', torch.ones(1))
        self.register_buffer('mae_scale', torch.ones(1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))
    
    def get_lambda(self):
        """
        Get current λ value (after sigmoid mapping to [0, 1]).
        
        Returns:
        --------
        torch.Tensor
            Current λ value in range [0, 1]
        """
        return torch.sigmoid(self.lambda_logit)
    
    def forward(self, outputs, targets):
        """
        Compute adaptive mixed loss.
        
        Parameters:
        -----------
        outputs : torch.Tensor
            Model predictions
        targets : torch.Tensor
            Ground truth targets
            
        Returns:
        --------
        loss : torch.Tensor
            Combined loss value
        mse : torch.Tensor
            Raw MSE value
        mae : torch.Tensor
            Raw MAE value
        lambda_val : torch.Tensor
            Current λ value
        mse_normalized : torch.Tensor
            Normalized MSE value
        mae_normalized : torch.Tensor
            Normalized MAE value
        """
        mse = self.mse_loss(outputs, targets)
        mae = self.mae_loss(outputs, targets)
        
        # Update moving averages
        with torch.no_grad():
            if not self.initialized:
                # First initialization
                self.mse_scale.copy_(mse.detach())
                self.mae_scale.copy_(mae.detach())
                self.initialized.fill_(True)
            else:
                # Exponential moving average update
                self.mse_scale.mul_(self.momentum).add_(mse.detach(), alpha=1 - self.momentum)
                self.mae_scale.mul_(self.momentum).add_(mae.detach(), alpha=1 - self.momentum)
        
        # Normalize to same scale
        mse_normalized = mse / (self.mse_scale + 1e-8)
        mae_normalized = mae / (self.mae_scale + 1e-8)
        
        lambda_val = self.get_lambda()
        
        # Compute weighted combination
        loss = (1 - lambda_val) * mse_normalized + lambda_val * mae_normalized
        
        return loss, mse, mae, lambda_val, mse_normalized, mae_normalized


# ================================
# Parameter Statistics Printing
# ================================
def print_layer_parameters(model, layer_idx):
    """
    Print parameter statistics for a specific layer.
    
    Displays statistics for both raw weights (w1_raw, w2_raw) and their
    positive versions after Softplus activation (w1_pos, w2_pos).
    
    Parameters:
    -----------
    model : LSBLNetwork
        The L-SBL network model
    layer_idx : int
        Index of the layer to print statistics for
        
    Returns:
    --------
    None
    """
    if layer_idx >= len(model.sbl_layers):
        return
    
    layer = model.sbl_layers[layer_idx]
    
    with torch.no_grad():
        # Get raw weight parameters (unconstrained)
        w1_raw = layer.gamma_update.weight_params[0].cpu().numpy()
        w2_raw = layer.gamma_update.weight_params[1].cpu().numpy()
        
        # Compute positive weights after softplus
        w1_pos = F.softplus(layer.gamma_update.weight_params[0]).cpu().numpy()
        w2_pos = F.softplus(layer.gamma_update.weight_params[1]).cpu().numpy()
        
        print(f"\n{'=' * 80}")
        print(f"Layer {layer_idx} Parameter Statistics (EM Two-Term Weighting - Softplus on weights):")
        print(f"{'=' * 80}")
        
        print(f"\nw1_raw - Raw weight for term1 (γ² * mean(A²)) (shape: {w1_raw.shape}):")
        print(f"  Mean:   {np.mean(w1_raw):.8f}")
        print(f"  Min:    {np.min(w1_raw):.8f}")
        print(f"  Max:    {np.max(w1_raw):.8f}")
        print(f"  Median: {np.median(w1_raw):.8f}")
        print(f"  Std:    {np.std(w1_raw):.8f}")
        print(f"  Negative count: {np.sum(w1_raw < 0)} / {w1_raw.size}")
        
        print(f"\nw1_pos - Positive weight (softplus(w1_raw)) (shape: {w1_pos.shape}):")
        print(f"  Mean:   {np.mean(w1_pos):.8f}")
        print(f"  Min:    {np.min(w1_pos):.8f}")
        print(f"  Max:    {np.max(w1_pos):.8f}")
        print(f"  Median: {np.median(w1_pos):.8f}")
        print(f"  Std:    {np.std(w1_pos):.8f}")
        
        print(f"\nw2_raw - Raw weight for term2 (γ * (1 - γ * trace_term)) (shape: {w2_raw.shape}):")
        print(f"  Mean:   {np.mean(w2_raw):.8f}")
        print(f"  Min:    {np.min(w2_raw):.8f}")
        print(f"  Max:    {np.max(w2_raw):.8f}")
        print(f"  Median: {np.median(w2_raw):.8f}")
        print(f"  Std:    {np.std(w2_raw):.8f}")
        print(f"  Negative count: {np.sum(w2_raw < 0)} / {w2_raw.size}")
        
        print(f"\nw2_pos - Positive weight (softplus(w2_raw)) (shape: {w2_pos.shape}):")
        print(f"  Mean:   {np.mean(w2_pos):.8f}")
        print(f"  Min:    {np.min(w2_pos):.8f}")
        print(f"  Max:    {np.max(w2_pos):.8f}")
        print(f"  Median: {np.median(w2_pos):.8f}")
        print(f"  Std:    {np.std(w2_pos):.8f}")
        
        print(f"\nNote: γ_new = softplus(w1_raw) * term1 + softplus(w2_raw) * term2")
        print(f"      Softplus on weights ensures non-negative contributions")
        
        print(f"{'=' * 80}\n")


# ================================
# Neural Network Model Components
# ================================
class MuEstimateLayer(nn.Module):
    """
    E-step layer: Compute posterior mean, variance, A matrix, and trace term
    using Cholesky decomposition.
    
    This layer implements the expectation step of the EM algorithm, computing
    the posterior distribution q(x|y, γ) for the source signals given the
    sensor measurements and current hyperparameters.
    
    Parameters:
    -----------
    InpSize : int
        Input dimension (number of sensors M)
    OutSize : int
        Output dimension (number of sources N)
    L : int
        Number of time points
    reg_lambda : float, optional
        Regularization parameter (default: 1e-6)
        
    Attributes:
    -----------
    InpSize : int
        Number of sensors
    OutSize : int
        Number of sources
    L : int
        Number of time points
    reg_lambda : float
        Regularization parameter
    """
    
    def __init__(self, InpSize, OutSize, L, reg_lambda=1e-6):
        super(MuEstimateLayer, self).__init__()
        self.InpSize = InpSize
        self.OutSize = OutSize
        self.L = L
        self.reg_lambda = reg_lambda
    
    def forward(self, y, gamma, Lambda, Lmat):
        """
        Forward pass for posterior estimation.
        
        Computes:
        - CM = G * diag(gamma) * G^T + Lambda
        - CMinvG = CM^{-1} * G  (using Cholesky decomposition)
        - A = G^T * CM^{-1} * y
        - mu = diag(gamma) * A
        - Sigma_X_diag = gamma - gamma^2 * trace_term
        
        Parameters:
        -----------
        y : torch.Tensor
            Sensor measurements, shape (B, M, L)
        gamma : torch.Tensor
            Hyperparameters, shape (B, N)
        Lambda : torch.Tensor
            Noise covariance, shape (B, M, M)
        Lmat : torch.Tensor
            Lead field matrix, shape (B, M, N)
            
        Returns:
        --------
        mu : torch.Tensor
            Posterior mean, shape (B, N, L)
        Sigma_X_diag : torch.Tensor
            Posterior variance (diagonal), shape (B, N)
        A : torch.Tensor
            Intermediate matrix G^T * CM^{-1} * y, shape (B, N, L)
        trace_term : torch.Tensor
            Trace term sum(G * CMinvG, dim=1), shape (B, N)
        """
        B, M, L = y.shape
        _, _, N = Lmat.shape
        eps = torch.finfo(gamma.dtype).eps
        
        # CM = G * diag(gamma) * G^T + Lambda
        gamma_expanded = gamma.unsqueeze(1)  # (B, 1, N)
        L_Gamma = Lmat * gamma_expanded  # (B, M, N)
        CM = torch.bmm(L_Gamma, Lmat.transpose(1, 2))
        CM = CM + Lambda
        
        # Ensure symmetry
        CM = (CM + CM.transpose(1, 2)) / 2
        
        # Solve linear system using Cholesky decomposition
        try:
            # First attempt: direct Cholesky
            L_chol = torch.linalg.cholesky(CM)
            CMinvG = torch.cholesky_solve(Lmat, L_chol)
            
        except RuntimeError as e1:
            # Print detailed error for first failure
            print("\n[Cholesky ERROR 1] Cholesky failed on CM:")
            print("  → Error:", str(e1))
            print("  → CM stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(
                CM.min().item(), CM.max().item(), CM.mean().item()))
            diag = CM.diagonal(dim1=1, dim2=2)
            print("  → CM diag stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(
                diag.min().item(), diag.max().item(), diag.mean().item()))
            
            # Second attempt: add jitter
            jitter = 1e-10
            eye = torch.eye(CM.size(-1), device=CM.device).unsqueeze(0)
            CM_reg = CM + jitter * eye
            
            try:
                L_chol = torch.linalg.cholesky(CM_reg)
                CMinvG = torch.cholesky_solve(Lmat, L_chol)
                
            except RuntimeError as e2:
                # Print detailed error for second failure
                print("\n[Cholesky ERROR 2] Even after adding jitter, Cholesky failed:")
                print("  → Error:", str(e2))
                print("  → CM_reg stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(
                    CM_reg.min().item(), CM_reg.max().item(), CM_reg.mean().item()))
                diag_reg = CM_reg.diagonal(dim1=1, dim2=2)
                print("  → CM_reg diag stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(
                    diag_reg.min().item(), diag_reg.max().item(), diag_reg.mean().item()))
                print("  → Fallback to SVD...\n")
                
                # Fallback to SVD
                U, S, Vh = torch.linalg.svd(CM, full_matrices=False)
                S_inv = 1.0 / (S.unsqueeze(1) + eps)
                CMinv = torch.bmm(U * S_inv, U.transpose(1, 2))
                CMinvG = torch.bmm(CMinv, Lmat)
        
        # A = G^T * CM^{-1} * y
        A = torch.bmm(CMinvG.transpose(1, 2), y)  # (B, N, L)
        
        # mu = gamma * A
        mu = gamma.unsqueeze(2) * A
        
        # trace_term = sum(G * CMinvG, dim=1)
        trace_term = torch.sum(Lmat * CMinvG, dim=1)  # (B, N)
        
        # Sigma_X_diag = gamma - gamma^2 * trace_term
        Sigma_X_diag = gamma - gamma * gamma * trace_term
        
        # Prevent negative variance
        Sigma_X_diag = torch.clamp(Sigma_X_diag, min=eps)
        
        return mu, Sigma_X_diag, A, trace_term


class DiagonalGammaUpdate(nn.Module):
    """
    M-step: EM algorithm two-term weighted gamma update with Softplus on weights.
    
    Computes gamma update as:
        γ_new = softplus(w1_raw) * term1 + softplus(w2_raw) * term2
    
    where:
        term1 = γ² * mean(A²)
        term2 = γ * (1 - γ * trace_term)
    
    w1_raw and w2_raw are unconstrained real-valued parameters.
    Softplus ensures their contributions are non-negative.
    
    Parameters:
    -----------
    OutSize : int
        Number of sources (N)
        
    Attributes:
    -----------
    OutSize : int
        Number of sources
    weight_params : nn.Parameter
        Learnable weights, shape (2, N)
        Initialized such that softplus(w_raw) = 1.0
    """
    
    def __init__(self, OutSize):
        super().__init__()
        self.OutSize = OutSize
        
        # Initialize weights such that softplus(w_raw) = 1.0
        # softplus(x) = log(1 + exp(x))
        # To get softplus(x) = 1: log(1 + exp(x)) = 1
        # 1 + exp(x) = e, exp(x) = e - 1, x = log(e - 1) ≈ 0.541
        init_val = np.log(np.e - 1)
        self.weight_params = nn.Parameter(torch.ones(2, OutSize) * init_val)
    
    def forward(self, gamma, A, trace_term):
        """
        Forward pass for gamma update.
        
        Parameters:
        -----------
        gamma : torch.Tensor
            Current gamma values, shape (B, N)
        A : torch.Tensor
            Matrix G^T * CM^{-1} * y, shape (B, N, L)
        trace_term : torch.Tensor
            Trace term sum(G * CMinvG, dim=1), shape (B, N)
            
        Returns:
        --------
        gamma_new : torch.Tensor
            Updated gamma values, shape (B, N)
        """
        eps = torch.finfo(gamma.dtype).eps
        
        # Compute mean(A²)
        A_squared_mean = torch.mean(A * A, dim=2)  # (B, N)
        
        # ========================================
        # EM algorithm two terms
        # ========================================
        # term1 = γ² * mean(A²)
        term1 = gamma * gamma * A_squared_mean
        
        # term2 = γ * (1 - γ * trace_term)
        term2 = gamma * (1 - gamma * trace_term)
        
        # ========================================
        # Apply softplus to ensure non-negative weights
        # ========================================
        w1_pos = F.softplus(self.weight_params[0])  # (N,)
        w2_pos = F.softplus(self.weight_params[1])  # (N,)
        
        # Weighted combination
        gamma_new = w1_pos * term1 + w2_pos * term2
        
        # ========================================
        # Diagnostic: Check for NaN/Inf
        # ========================================
        if torch.isnan(gamma_new).any() or torch.isinf(gamma_new).any():
            print("\n" + "!" * 80)
            print("⚠️  WARNING: NaN or Inf detected in gamma update!")
            print("!" * 80)
            print(f"w1_raw stats: min={self.weight_params[0].min().item():.6e}, "
                  f"max={self.weight_params[0].max().item():.6e}, "
                  f"mean={self.weight_params[0].mean().item():.6e}")
            print(f"w2_raw stats: min={self.weight_params[1].min().item():.6e}, "
                  f"max={self.weight_params[1].max().item():.6e}, "
                  f"mean={self.weight_params[1].mean().item():.6e}")
            print(f"w1_pos stats: min={w1_pos.min().item():.6e}, "
                  f"max={w1_pos.max().item():.6e}, "
                  f"mean={w1_pos.mean().item():.6e}")
            print(f"w2_pos stats: min={w2_pos.min().item():.6e}, "
                  f"max={w2_pos.max().item():.6e}, "
                  f"mean={w2_pos.mean().item():.6e}")
            print(f"term1 stats: min={term1.min().item():.6e}, max={term1.max().item():.6e}")
            print(f"term2 stats: min={term2.min().item():.6e}, max={term2.max().item():.6e}")
            print(f"gamma_new stats: min={gamma_new.min().item():.6e}, max={gamma_new.max().item():.6e}")
            print(f"NaN count in gamma_new: {torch.isnan(gamma_new).sum().item()} / {gamma_new.numel()}")
            print(f"Inf count in gamma_new: {torch.isinf(gamma_new).sum().item()} / {gamma_new.numel()}")
            print("!" * 80 + "\n")
        
        return gamma_new


class SBLLayer(nn.Module):
    """
    Single SBL layer combining E-step and M-step.
    
    Parameters:
    -----------
    OutSize : int
        Number of sources (N)
    InpSize : int
        Number of sensors (M)
    L : int
        Number of time points
        
    Attributes:
    -----------
    OutSize : int
        Number of sources
    InpSize : int
        Number of sensors
    L : int
        Number of time points
    mu_estimate : MuEstimateLayer
        E-step layer
    gamma_update : DiagonalGammaUpdate
        M-step layer
    """
    
    def __init__(self, OutSize, InpSize, L):
        super(SBLLayer, self).__init__()
        self.OutSize = OutSize
        self.InpSize = InpSize
        self.L = L
        
        self.mu_estimate = MuEstimateLayer(InpSize, OutSize, L)
        self.gamma_update = DiagonalGammaUpdate(OutSize)
    
    def forward(self, y, gamma, Lambda, Lmat):
        """
        Forward pass through one SBL iteration.
        
        Parameters:
        -----------
        y : torch.Tensor
            Sensor measurements, shape (B, M, L)
        gamma : torch.Tensor
            Hyperparameters, shape (B, N)
        Lambda : torch.Tensor
            Noise covariance, shape (B, M, M)
        Lmat : torch.Tensor
            Lead field matrix, shape (B, M, N)
            
        Returns:
        --------
        mu : torch.Tensor
            Posterior mean, shape (B, N, L)
        Sigma_X_diag : torch.Tensor
            Posterior variance, shape (B, N)
        gamma_new : torch.Tensor
            Updated gamma, shape (B, N)
        """
        mu, Sigma_X_diag, A, trace_term = self.mu_estimate(y, gamma, Lambda, Lmat)
        gamma_new = self.gamma_update(gamma, A, trace_term)
        return mu, Sigma_X_diag, gamma_new


class LSBLNetwork(nn.Module):
    """
    Complete L-SBL network with T layers.
    
    Each layer performs one iteration of the EM algorithm with learned
    weight parameters for the gamma update.
    
    Parameters:
    -----------
    InpSize : int
        Number of sensors (M)
    OutSize : int
        Number of sources (N)
    L : int
        Number of time points
    T : int
        Number of L-SBL layers
        
    Attributes:
    -----------
    InpSize : int
        Number of sensors
    OutSize : int
        Number of sources
    L : int
        Number of time points
    T : int
        Number of layers
    sbl_layers : nn.ModuleList
        List of SBL layers
    """
    
    def __init__(self, InpSize, OutSize, L, T):
        super().__init__()
        self.InpSize = InpSize
        self.OutSize = OutSize
        self.L = L
        self.T = T
        
        self.sbl_layers = nn.ModuleList([
            SBLLayer(OutSize, InpSize, L) for _ in range(T)
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights such that softplus(w_raw) = 1.0 for all layers.
        
        Returns:
        --------
        None
        """
        init_val = np.log(np.e - 1)  # ≈ 0.541
        for layer in self.sbl_layers:
            with torch.no_grad():
                layer.gamma_update.weight_params.fill_(init_val)
    
    def forward(self, y, gamma_0, Lambda, Lmat, num_layers=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        y : torch.Tensor
            Sensor measurements, shape (B, M, L)
        gamma_0 : torch.Tensor
            Initial gamma values, shape (B, N)
        Lambda : torch.Tensor
            Noise covariance, shape (B, M, M)
        Lmat : torch.Tensor
            Lead field matrix, shape (B, M, N)
        num_layers : int, optional
            Number of layers to use (default: all layers)
            
        Returns:
        --------
        mu_final : torch.Tensor
            Final posterior mean estimate, shape (B, N, L)
        """
        if num_layers is None:
            num_layers = self.T
        
        gamma = gamma_0
        
        # Iterate through layers
        for i in range(num_layers):
            mu, Sigma_X_diag, gamma = self.sbl_layers[i](y, gamma, Lambda, Lmat)
        
        # Compute final mu using last gamma
        if num_layers > 0:
            mu_final, _, _, _ = self.sbl_layers[num_layers - 1].mu_estimate(y, gamma, Lambda, Lmat)
            return mu_final
        else:
            mu_final, _, _, _ = self.sbl_layers[0].mu_estimate(y, gamma_0, Lambda, Lmat)
            return mu_final


# ================================
# Training and Evaluation Functions
# ================================
def train_one_epoch(model, dataloader, optimizer, criterion, device, num_layers,
                    loss_type, show_debug=False, sub_batch_size=64):
    """
    Train the model for one epoch.
    
    Processes data in sub-batches to reduce memory usage. Each batch file
    is split into smaller sub-batches for training.
    
    Parameters:
    -----------
    model : LSBLNetwork
        The L-SBL network model
    dataloader : DataLoader
        Data loader for training data
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates
    criterion : callable or nn.Module
        Loss function
    device : torch.device
        Device to run training on
    num_layers : int
        Number of layers to use in forward pass
    loss_type : str
        Type of loss function ('mse', 'mae', or 'adaptive')
    show_debug : bool, optional
        Whether to print debug information (default: False)
    sub_batch_size : int, optional
        Size of sub-batches for memory management (default: 64)
        
    Returns:
    --------
    avg_loss_norm : float
        Average normalized loss
    avg_mae_norm : float
        Average normalized MAE
    avg_mse_norm : float
        Average normalized MSE
    avg_mae_denorm : float
        Average denormalized MAE
    avg_mse_denorm : float
        Average denormalized MSE
    avg_lambda : float or None
        Average lambda value (for adaptive loss only)
    avg_mse_normalized : float or None
        Average normalized MSE component (for adaptive loss only)
    avg_mae_normalized : float or None
        Average normalized MAE component (for adaptive loss only)
    """
    model.train()
    total_loss_norm = 0.0
    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_denorm = 0.0
    total_mae_denorm = 0.0
    total_samples = 0
    
    # For tracking adaptive loss statistics
    lambda_values = []
    mse_norm_values = []
    mae_norm_values = []
    
    for batch_idx, batch in enumerate(dataloader):
        X_norm, Y_norm, Lambda_norm, M_norm = [
            x.squeeze(0).to(device) for x in batch
        ]
        
        B = X_norm.shape[0]  # Original batch size
        N = X_norm.shape[1]
        
        dataset = dataloader.dataset
        G_norm = dataset.G_norm  # (1, M, N)
        G_norm_const = dataset.G_norm_const
        
        # ========= Split into sub-batches =========
        num_sub_batches = (B + sub_batch_size - 1) // sub_batch_size  # Ceiling division
        
        for sub_idx in range(num_sub_batches):
            start_idx = sub_idx * sub_batch_size
            end_idx = min(start_idx + sub_batch_size, B)
            
            # Extract current sub-batch
            X_sub = X_norm[start_idx:end_idx]
            Y_sub = Y_norm[start_idx:end_idx]
            Lambda_sub = Lambda_norm[start_idx:end_idx]
            M_sub = M_norm[start_idx:end_idx]
            
            B_sub = X_sub.shape[0]
            
            # Expand G_norm to current sub-batch size
            G_norm_sub = G_norm.expand(B_sub, -1, -1)
            Gamma_batch = torch.ones(B_sub, N, device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass (normalized space)
            outputs_norm = model(Y_sub, Gamma_batch, Lambda_sub, G_norm_sub, num_layers=num_layers)
            
            # Compute loss
            if loss_type == 'adaptive':
                loss_norm, mse_raw, mae_raw, lambda_val, mse_normalized, mae_normalized = criterion(outputs_norm, X_sub)
                lambda_values.append(lambda_val.item())
                mse_norm_values.append(mse_normalized.item())
                mae_norm_values.append(mae_normalized.item())
            else:
                loss_norm, _, _, _ = criterion(outputs_norm, X_sub)
                mse_raw = None
                mae_raw = None
                lambda_val = None
            
            # Normalized space MSE and MAE (for monitoring)
            with torch.no_grad():
                if mse_raw is None:
                    mse_norm = torch.mean((outputs_norm - X_sub) ** 2)
                    mae_norm = torch.mean(torch.abs(outputs_norm - X_sub))
                else:
                    mse_norm = mse_raw
                    mae_norm = mae_raw
                
                # Denormalize
                denorm_factor = (M_sub.sqrt().view(B_sub, 1, 1) / G_norm_const)
                outputs_denorm = outputs_norm * denorm_factor
                X_denorm = X_sub * denorm_factor
                
                # Denormalized space MSE and MAE
                mse_denorm = torch.mean((outputs_denorm - X_denorm) ** 2)
                mae_denorm = torch.mean(torch.abs(outputs_denorm - X_denorm))
            
            # Debug output (only for first batch, first sub-batch)
            if batch_idx == 0 and sub_idx == 0 and show_debug:
                print(f"\n  [Debug Info - Batch 0, Sub-batch 0/{num_sub_batches}]")
                print(f"    Original batch size: {B}, Sub-batch size: {B_sub}")
                print(f"    {'─' * 60}")
                with torch.no_grad():
                    print(f"\n    Data Range (Denormalized Space):")
                    print(f"      X_denorm:       mean={X_denorm.abs().mean().item():.6f}, "
                          f"max={X_denorm.abs().max().item():.6f}")
                    print(f"      outputs_denorm: mean={outputs_denorm.abs().mean().item():.6f}, "
                          f"max={outputs_denorm.abs().max().item():.6f}")
                    print(f"      denorm_factor:  mean={denorm_factor.mean().item():.6e}, "
                          f"min={denorm_factor.min().item():.6e}, "
                          f"max={denorm_factor.max().item():.6e}")
                
                print(f"\n    Loss Comparison:")
                print(f"      Loss type: {loss_type}")
                
                if loss_type == 'adaptive' and lambda_val is not None:
                    print(f"\n      Adaptive Loss Details:")
                    print(f"        Current λ: {lambda_val.item():.6f}")
                    print(f"        MSE scale: {criterion.mse_scale.item():.6e}")
                    print(f"        MAE scale: {criterion.mae_scale.item():.6e}")
                    print(f"        MSE (raw):        {mse_norm.item():.8f}")
                    print(f"        MAE (raw):        {mae_norm.item():.8f}")
                    print(f"        MSE (normalized): {mse_normalized.item():.8f}")
                    print(f"        MAE (normalized): {mae_normalized.item():.8f}")
                    print(f"        Mixed loss:       {loss_norm.item():.8f}  ← (1-λ)*MSE_norm + λ*MAE_norm")
                    print(f"        Ratio MSE/MAE:    {(mse_norm.item() / (mae_norm.item() + 1e-10)):.2f}x")
                elif loss_type == 'mse':
                    print(f"        MSE:  {mse_norm.item():.8f}  ← used for backprop")
                elif loss_type == 'mae':
                    print(f"        MAE:  {mae_norm.item():.8f}  ← used for backprop")
                
                print(f"\n      Denormalized space:")
                print(f"        MAE (L1): {mae_denorm.item():.8f}")
                print(f"        MSE:      {mse_denorm.item():.8f}")
                
                k = denorm_factor.mean().item()
                k2 = k ** 2
                print(f"\n      Scaling factors:")
                print(f"        k (for MAE): {k:.6e}")
                print(f"        k² (for MSE): {k2:.6e}")
            
            # Backward pass
            loss_norm.backward()
            
            # Gradient checking (only for first batch, first sub-batch)
            if batch_idx == 0 and sub_idx == 0 and show_debug:
                target_layer = num_layers - 1
                if target_layer < len(model.sbl_layers):
                    layer = model.sbl_layers[target_layer]
                    if layer.gamma_update.weight_params.grad is None:
                        print(f"\n    ⚠️ WARNING: No gradient for Layer {target_layer} weight_params!")
                    else:
                        grad_norm = layer.gamma_update.weight_params.grad.abs().mean().item()
                        grad_max = layer.gamma_update.weight_params.grad.abs().max().item()
                        print(f"\n    ✓ Layer {target_layer} weight_params gradient:")
                        print(f"      Mean: {grad_norm:.6e}, Max: {grad_max:.6e}")
                
                # If using adaptive loss, show lambda gradient
                if loss_type == 'adaptive' and criterion.lambda_logit.grad is not None:
                    lambda_grad = criterion.lambda_logit.grad.item()
                    print(f"    ✓ Lambda gradient: {lambda_grad:.6e}")
                
                print(f"    {'─' * 60}\n")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss_norm += loss_norm.item() * B_sub
            total_mse_norm += mse_norm.item() * B_sub
            total_mae_norm += mae_norm.item() * B_sub
            total_mse_denorm += mse_denorm.item() * B_sub
            total_mae_denorm += mae_denorm.item() * B_sub
            total_samples += B_sub
            
            # Clean up memory
            del X_sub, Y_sub, Lambda_sub, M_sub, G_norm_sub
            del outputs_norm, loss_norm, mse_norm, mae_norm
            del mse_denorm, mae_denorm, denorm_factor
        
        # Clean up outer batch variables
        del X_norm, Y_norm, Lambda_norm, M_norm
    
    avg_loss_norm = total_loss_norm / total_samples
    avg_mae_norm = total_mae_norm / total_samples
    avg_mse_norm = total_mse_norm / total_samples
    avg_mae_denorm = total_mae_denorm / total_samples
    avg_mse_denorm = total_mse_denorm / total_samples
    
    avg_lambda = np.mean(lambda_values) if lambda_values else None
    avg_mse_normalized = np.mean(mse_norm_values) if mse_norm_values else None
    avg_mae_normalized = np.mean(mae_norm_values) if mae_norm_values else None
    
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / 1024 ** 2, "MB")
    print("Max Memory Reserved:", torch.cuda.max_memory_reserved() / 1024 ** 2, "MB")
    
    return (avg_loss_norm, avg_mae_norm, avg_mse_norm, avg_mae_denorm, avg_mse_denorm,
            avg_lambda, avg_mse_normalized, avg_mae_normalized)


def evaluate_model(model, dataloader, device, num_layers):
    """
    Evaluate model performance on a given dataset.
    
    Computes both normalized and denormalized MSE and MAE metrics.
    
    Parameters:
    -----------
    model : LSBLNetwork
        The L-SBL network model
    dataloader : DataLoader
        Data loader for evaluation data
    device : torch.device
        Device to run evaluation on
    num_layers : int
        Number of layers to use in forward pass
        
    Returns:
    --------
    MSE_norm : float
        Mean squared error in normalized space
    MAE_norm : float
        Mean absolute error in normalized space
    MSE_denorm : float
        Mean squared error in denormalized (original) space
    MAE_denorm : float
        Mean absolute error in denormalized (original) space
    """
    model.eval()
    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_denorm = 0.0
    total_mae_denorm = 0.0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X_norm, Y_norm, Lambda_norm, M_norm = [x.squeeze(0).to(device) for x in batch]
            
            B = X_norm.shape[0]
            
            dataset = dataloader.dataset
            G_norm = dataset.G_norm.expand(B, -1, -1)
            G_norm_const = dataset.G_norm_const
            
            Gamma = torch.ones(B, X_norm.shape[1], device=device)
            
            out = model(Y_norm, Gamma, Lambda_norm, G_norm, num_layers=num_layers)
            
            # Normalized space
            mse_norm = torch.mean((out - X_norm) ** 2)
            mae_norm = torch.mean(torch.abs(out - X_norm))
            
            # Denormalize
            denorm_factor = (M_norm.sqrt().view(B, 1, 1) / G_norm_const)
            mse_denorm = torch.mean(((out - X_norm) * denorm_factor) ** 2)
            mae_denorm = torch.mean(torch.abs((out - X_norm) * denorm_factor))
            
            total_mse_norm += mse_norm.item() * B
            total_mae_norm += mae_norm.item() * B
            total_mse_denorm += mse_denorm.item() * B
            total_mae_denorm += mae_denorm.item() * B
            total += B
    
    return (total_mse_norm / total, total_mae_norm / total,
            total_mse_denorm / total, total_mae_denorm / total)


# ================================
# Weight Copying and Verification
# ================================
def copy_layer_weights(source_layer_idx, target_layer_idx, model):
    """
    Copy weights from source layer to target layer.
    
    Used to initialize new layers with weights from the previous layer,
    providing a warm start for training.
    
    Parameters:
    -----------
    source_layer_idx : int
        Index of source layer to copy from
    target_layer_idx : int
        Index of target layer to copy to
    model : LSBLNetwork
        The L-SBL network model
        
    Returns:
    --------
    bool
        True if copy was successful, False otherwise
    """
    if source_layer_idx >= len(model.sbl_layers) or target_layer_idx >= len(model.sbl_layers):
        print(f"  ⚠️ Layer index out of range: source={source_layer_idx}, target={target_layer_idx}")
        return False
    
    source_layer = model.sbl_layers[source_layer_idx]
    target_layer = model.sbl_layers[target_layer_idx]
    
    with torch.no_grad():
        # Copy weight_params (w1_raw, w2_raw)
        target_layer.gamma_update.weight_params.copy_(
            source_layer.gamma_update.weight_params
        )
    
    return True


def verify_weight_copy(source_layer_idx, target_layer_idx, model):
    """
    Verify that weights were successfully copied between layers.
    
    Checks if the weights are numerically identical and reports
    any differences.
    
    Parameters:
    -----------
    source_layer_idx : int
        Index of source layer
    target_layer_idx : int
        Index of target layer
    model : LSBLNetwork
        The L-SBL network model
        
    Returns:
    --------
    bool
        True if weights are identical (within tolerance), False otherwise
    """
    source_layer = model.sbl_layers[source_layer_idx]
    target_layer = model.sbl_layers[target_layer_idx]
    
    with torch.no_grad():
        source_weights = source_layer.gamma_update.weight_params
        target_weights = target_layer.gamma_update.weight_params
        
        # Check if completely identical
        is_equal = torch.allclose(source_weights, target_weights, rtol=1e-9, atol=1e-9)
        
        # Compute differences
        max_diff = (source_weights - target_weights).abs().max().item()
        mean_diff = (source_weights - target_weights).abs().mean().item()
        
        print(f"\n  Weight Copy Verification (Layer {source_layer_idx} → Layer {target_layer_idx}):")
        print(f"    Completely identical: {'✓ YES' if is_equal else '✗ NO'}")
        print(f"    Max difference: {max_diff:.2e}")
        print(f"    Mean difference: {mean_diff:.2e}")
        
        if is_equal:
            print(f"    ✓ Weight copy successful!")
        else:
            print(f"    ⚠️ Weights differ (possibly due to numerical precision)")
        
        # Display statistics
        print(f"\n  Source layer (Layer {source_layer_idx}) weight statistics:")
        print(f"    w1_raw: mean={source_weights[0].mean():.6e}, std={source_weights[0].std():.6e}")
        print(f"    w2_raw: mean={source_weights[1].mean():.6e}, std={source_weights[1].std():.6e}")
        
        print(f"\n  Target layer (Layer {target_layer_idx}) weight statistics:")
        print(f"    w1_raw: mean={target_weights[0].mean():.6e}, std={target_weights[0].std():.6e}")
        print(f"    w2_raw: mean={target_weights[1].mean():.6e}, std={target_weights[1].std():.6e}")
        
        return is_equal


# ================================
# Main Training Loop
# ================================

if __name__ == "__main__":
    maximum_lsbl_layers = 300
    config, unparsed = parser.parse_known_args()
    
    print('=' * 80)
    print('EEG Source Localization using L-SBL - EM Algorithm Two-Term Weighting')
    print('Using Softplus on w1, w2 to ensure non-negativity (no bias)')
    print('γ_new = softplus(w1_raw) * term1 + softplus(w2_raw) * term2')
    print('Initial values: softplus(w1_raw) = softplus(w2_raw) = 1.0')
    print(f'Loss function: {config.loss_type.upper()}')
    if config.loss_type == 'adaptive':
        print(f'  Adaptive mixed loss: (1-λ)*MSE_normalized + λ*MAE_normalized')
        print(f'  Initial λ = {config.init_lambda}, momentum = {config.momentum}')
    print('=' * 80)
    print(f'Data folder: {config.data_folder}')
    print(f'Number of L-SBL layers: {config.T}')
    print(f'Batch size for .npy generation: {config.batch_size}')
    print(f'Training sub-batch size (actual GPU batch): {config.sub_batch_size}')
    print(f'Learning rate: {config.lr}')
    print(f'Device: {device}')
    print('=' * 80)
    
    # Load metadata
    metadata_path = os.path.join(config.data_folder, 'metadata.npz')
    if not os.path.exists(metadata_path):
        raise RuntimeError(f'Metadata not found: {metadata_path}')
    
    metadata = np.load(metadata_path)
    
    # Check if data is normalized
    if 'normalized' not in metadata or not metadata['normalized']:
        raise RuntimeError(
            f'\nError: Data folder does not contain normalized data!\n'
            f'Please run: python preprocess_normalized_data.py -iF <original_data_path> -oF {config.data_folder}'
        )
    
    M = int(metadata['n_sensors'])
    N = int(metadata['n_sources'])
    L = int(metadata['n_time'])
    
    print(f'\nData Information:')
    print(f'  Sensors (M): {M}')
    print(f'  Sources (N): {N}')
    print(f'  Time points (L): {L}')
    print(f'  Training samples:   {metadata["n_train"]}')
    print(f'  Validation A samples (valA, same head as train): {metadata["n_valA"]}')
    print(f'  Validation B samples (valB, val head):           {metadata["n_valB"]}')
    print(f'  Data status: ✓ Normalized')
    
    if config.T > maximum_lsbl_layers:
        print(f'Warning: Maximum number of layers cannot exceed {maximum_lsbl_layers}')
        config.T = maximum_lsbl_layers
    
    # Create datasets
    print('\nCreating datasets...')
    train_dataset = NormalizedEEGDataset(config.data_folder, split='train', g_tag='train')
    valA_dataset = NormalizedEEGDataset(config.data_folder, split='valA', g_tag='valA')
    valB_dataset = NormalizedEEGDataset(config.data_folder, split='valB', g_tag='valB')
    
    # DataLoaders: one .npy file per batch, outer batch_size=1
    effective_num_workers = 0
    effective_pin_memory = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory
    )
    
    valA_loader = DataLoader(
        valA_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory
    )
    
    valB_loader = DataLoader(
        valB_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory
    )
    
    # Parameter settings
    InpSize = M
    OutSize = N
    T = config.T
    
    # ================================
    # Training Parameters - Dynamic epoch allocation by layer depth
    # ================================
    def get_epochs_for_layer(layer_idx):
        """
        Return the number of epochs for a given layer based on depth.
        
        Parameters:
        -----------
        layer_idx : int
            Index of the layer
            
        Returns:
        --------
        int
            Number of epochs to train this layer
        """
        if layer_idx < 25:
            return 3
        elif layer_idx < 50:
            return 3
        else:
            return 3
    
    # Create dynamic iTr and iTr2 arrays
    iTr = np.array([get_epochs_for_layer(i) for i in range(T)], dtype='int32')
    iTr2 = np.array([get_epochs_for_layer(i) for i in range(T)], dtype='int32')
    
    print("\nTraining Schedule:")
    print(f"  Layers 0-24:  {get_epochs_for_layer(0)} epochs per layer (Pass 1 & Pass 3)")
    print(f"  Layers 25-49: {get_epochs_for_layer(25)} epochs per layer (Pass 1 & Pass 3)")
    print(f"  Layers 50+:   {get_epochs_for_layer(50)} epochs per layer (Pass 1 & Pass 3)")
    
    # Loss matrices: track both MSE and MAE separately
    # Pass 1 - valA
    LossMatrix1_mse_norm_valA = np.zeros(T)
    LossMatrix1_mae_norm_valA = np.zeros(T)
    LossMatrix1_mse_denorm_valA = np.zeros(T)
    LossMatrix1_mae_denorm_valA = np.zeros(T)
    
    # Pass 1 - valB
    LossMatrix1_mse_norm_valB = np.zeros(T)
    LossMatrix1_mae_norm_valB = np.zeros(T)
    LossMatrix1_mse_denorm_valB = np.zeros(T)
    LossMatrix1_mae_denorm_valB = np.zeros(T)
    
    # Pass 3 - valA
    LossMatrix3_mse_norm_valA = np.zeros(T)
    LossMatrix3_mae_norm_valA = np.zeros(T)
    LossMatrix3_mse_denorm_valA = np.zeros(T)
    LossMatrix3_mae_denorm_valA = np.zeros(T)
    
    # Pass 3 - valB
    LossMatrix3_mse_norm_valB = np.zeros(T)
    LossMatrix3_mae_norm_valB = np.zeros(T)
    LossMatrix3_mse_denorm_valB = np.zeros(T)
    LossMatrix3_mae_denorm_valB = np.zeros(T)
    
    lr = config.lr
    
    ### Loss Function Selection ###
    if config.loss_type == 'mse':
        print("\nInitializing loss function: MSE")
        criterion_base = nn.MSELoss()
        
        def criterion(outputs, targets):
            return criterion_base(outputs, targets), None, None, None
    elif config.loss_type == 'mae':
        print("\nInitializing loss function: MAE (L1)")
        criterion_base = nn.L1Loss()
        
        def criterion(outputs, targets):
            return criterion_base(outputs, targets), None, None, None
    elif config.loss_type == 'adaptive':
        print(f"\nInitializing loss function: Adaptive Mixed Loss")
        print(f"  (1-λ)*MSE_normalized + λ*MAE_normalized")
        print(f"  Initial λ = {config.init_lambda}, momentum = {config.momentum}")
        criterion = AdaptiveMixedLoss(initial_lambda=config.init_lambda, momentum=config.momentum).to(device)
    else:
        raise ValueError(f"Unknown loss_type: {config.loss_type}")
    ### Loss Function Selection ###
    
    # Create model
    print('\nInitializing model (EM Two-Term Weighting, Softplus on weights, no bias)...')
    model = LSBLNetwork(InpSize, OutSize, L, T).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    layer_params = sum(p.numel() for p in model.sbl_layers[0].parameters())
    print(f"Single layer parameters: {layer_params} (weight_params: 2*N)")
    
    # Layer-wise training
    layerIndex = np.arange(0, T).tolist()
    
    for layer in layerIndex:
        print("\n" + "=" * 80)
        print(f"Training Layer: {layer}")
        print(f"Epochs for this layer: Pass1={iTr[layer]}, Pass3={iTr2[layer]}")
        print("=" * 80)
        
        # ========== Pass 1 ==========
        print("\n" + "-" * 80)
        print("Pass 1: Training current layer only")
        print("-" * 80)
        
        if layer > 0:
            checkpoint_path = f"g{layer - 1}.pth"
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
                # Load criterion state if using adaptive loss
                if config.loss_type == 'adaptive' and 'criterion_state_dict' in checkpoint:
                    criterion.load_state_dict(checkpoint['criterion_state_dict'])
                    print(f"  Loaded criterion state, current λ = {criterion.get_lambda().item():.6f}")
                
                # ========================================
                # NEW: Copy weights from previous layer
                # ========================================
                print(f"\nCopying weights: Layer {layer - 1} → Layer {layer}")
                copy_success = copy_layer_weights(layer - 1, layer, model)
                
                if copy_success:
                    # Verify copy was successful
                    verify_weight_copy(layer - 1, layer, model)
                else:
                    print(f"  ⚠️ Weight copy failed!")
                
                print("\nTesting performance after loading...")
                
                # Evaluate - returns 4 values: (MSE_norm, MAE_norm, MSE_denorm, MAE_denorm)
                mse_old_norm_A, mae_old_norm_A, mse_old_denorm_A, mae_old_denorm_A = \
                    evaluate_model(model, valA_loader, device, num_layers=layer)
                mse_new_norm_A, mae_new_norm_A, mse_new_denorm_A, mae_new_denorm_A = \
                    evaluate_model(model, valA_loader, device, num_layers=layer + 1)
                
                mse_old_norm_B, mae_old_norm_B, mse_old_denorm_B, mae_old_denorm_B = \
                    evaluate_model(model, valB_loader, device, num_layers=layer)
                mse_new_norm_B, mae_new_norm_B, mse_new_denorm_B, mae_new_denorm_B = \
                    evaluate_model(model, valB_loader, device, num_layers=layer + 1)
                
                print(f"\n[valA] Performance Comparison:")
                print(f"  With {layer} layers:")
                print(f"    MSE (Denorm): {mse_old_denorm_A:.8f}")
                print(f"    MAE (Denorm): {mae_old_denorm_A:.8f}")
                print(f"  With {layer + 1} layers:")
                print(f"    MSE (Denorm): {mse_new_denorm_A:.8f}")
                print(f"    MAE (Denorm): {mae_new_denorm_A:.8f}")
                
                improvement_mse_A = mse_old_denorm_A - mse_new_denorm_A
                improvement_mae_A = mae_old_denorm_A - mae_new_denorm_A
                pct_mse_A = (improvement_mse_A / mse_old_denorm_A) * 100 if mse_old_denorm_A > 0 else 0
                pct_mae_A = (improvement_mae_A / mae_old_denorm_A) * 100 if mae_old_denorm_A > 0 else 0
                print(f"  Improvement:")
                print(f"    MSE: {improvement_mse_A:.8f} ({pct_mse_A:.2f}%)")
                print(f"    MAE: {improvement_mae_A:.8f} ({pct_mae_A:.2f}%)")
                
                print(f"\n[valB] Performance Comparison:")
                print(f"  With {layer} layers:")
                print(f"    MSE (Denorm): {mse_old_denorm_B:.8f}")
                print(f"    MAE (Denorm): {mae_old_denorm_B:.8f}")
                print(f"  With {layer + 1} layers:")
                print(f"    MSE (Denorm): {mse_new_denorm_B:.8f}")
                print(f"    MAE (Denorm): {mae_new_denorm_B:.8f}")
                
                improvement_mse_B = mse_old_denorm_B - mse_new_denorm_B
                improvement_mae_B = mae_old_denorm_B - mae_new_denorm_B
                pct_mse_B = (improvement_mse_B / mse_old_denorm_B) * 100 if mse_old_denorm_B > 0 else 0
                pct_mae_B = (improvement_mae_B / mae_old_denorm_B) * 100 if mae_old_denorm_B > 0 else 0
                print(f"  Improvement:")
                print(f"    MSE: {improvement_mse_B:.10f} ({pct_mse_B:.5f}%)")
                print(f"    MAE: {improvement_mae_B:.10f} ({pct_mae_B:.5f}%)")
            
            else:
                print(f"Checkpoint not found: {checkpoint_path}")
                print("Starting from scratch for this layer")
        
        # Freeze previous layers
        for i in range(layer):
            for param in model.sbl_layers[i].parameters():
                param.requires_grad = False
        
        # Unfreeze current layer
        if layer < len(model.sbl_layers):
            for param in model.sbl_layers[layer].parameters():
                param.requires_grad = True
        
        # Learning rate adjustment
        if layer >= 4:
            lr1 = lr
        else:
            lr1 = lr
        
        print(f"\nLearning Rate (Pass 1): {lr1}")
        
        # Create optimizer parameter list
        optimizer_params = [
            {'params': filter(lambda p: p.requires_grad, model.parameters())}
        ]
        
        # If using adaptive loss, add criterion parameters
        if config.loss_type == 'adaptive':
            optimizer_params.append({'params': criterion.parameters()})
        
        optimizer = optim.Adam(
            optimizer_params,
            lr=lr1, betas=(0.9, 0.999), eps=1e-07,
            amsgrad=(layer >= 4)
        )
        
        # Pass 1 training
        for epoch in range(iTr[layer]):
            show_debug = (epoch == 0)
            (train_loss, train_mae_norm, train_mse_norm, train_mae_denorm, train_mse_denorm,
             train_lambda, train_mse_normalized, train_mae_normalized) = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, num_layers=layer + 1, loss_type=config.loss_type,
                show_debug=show_debug, sub_batch_size=config.sub_batch_size
            )
            
            print(f"\nEpoch {epoch + 1}/{iTr[layer]}, Layer {layer}, Pass 1")
            print(f"  Train Loss:")
            if config.loss_type == 'adaptive':
                print(f"    Mixed Loss:   {train_loss:.6f}  ← (1-λ)*MSE_norm + λ*MAE_norm")
                if train_lambda is not None:
                    print(f"    Current λ:    {train_lambda:.6f}")
                    if train_mse_normalized is not None and train_mae_normalized is not None:
                        print(f"    MSE (normalized): {train_mse_normalized:.6f}")
                        print(f"    MAE (normalized): {train_mae_normalized:.6f}")
                print(f"    MSE (raw):    {train_mse_norm:.8f}")
                print(f"    MAE (raw):    {train_mae_norm:.8f}")
            print(f"    MAE (Denorm): {train_mae_denorm:.6f}")
            print(f"    MSE (Denorm): {train_mse_denorm:.6f}")
            
            if (epoch + 1) % 10 == 0 or epoch == iTr[layer] - 1:
                valA_mse_norm, valA_mae_norm, valA_mse_denorm, valA_mae_denorm = \
                    evaluate_model(model, valA_loader, device, num_layers=layer + 1)
                valB_mse_norm, valB_mae_norm, valB_mse_denorm, valB_mae_denorm = \
                    evaluate_model(model, valB_loader, device, num_layers=layer + 1)
                
                # Save to matrices
                LossMatrix1_mse_norm_valA[layer] = valA_mse_norm
                LossMatrix1_mae_norm_valA[layer] = valA_mae_norm
                LossMatrix1_mse_denorm_valA[layer] = valA_mse_denorm
                LossMatrix1_mae_denorm_valA[layer] = valA_mae_denorm
                
                LossMatrix1_mse_norm_valB[layer] = valB_mse_norm
                LossMatrix1_mae_norm_valB[layer] = valB_mae_norm
                LossMatrix1_mse_denorm_valB[layer] = valB_mse_denorm
                LossMatrix1_mae_denorm_valB[layer] = valB_mae_denorm
                
                print(f"  [valA] Validation Loss:")
                print(f"    MSE (Denorm): {valA_mse_denorm:.6f}")
                print(f"    MAE (Denorm): {valA_mae_denorm:.6f}")
                print(f"  [valB] Validation Loss:")
                print(f"    MSE (Denorm): {valB_mse_denorm:.6f}")
                print(f"    MAE (Denorm): {valB_mae_denorm:.6f}")
                print(f"  Loss Vector1 valA:")
                print(f"    MSE (Denorm): {LossMatrix1_mse_denorm_valA[:layer + 1]}")
                print(f"    MAE (Denorm): {LossMatrix1_mae_denorm_valA[:layer + 1]}")
                print(f"  Loss Vector1 valB:")
                print(f"    MSE (Denorm): {LossMatrix1_mse_denorm_valB[:layer + 1]}")
                print(f"    MAE (Denorm): {LossMatrix1_mae_denorm_valB[:layer + 1]}")
                
                # Save checkpoint
                checkpoint_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'mse_norm_valA': valA_mse_norm,
                    'mae_norm_valA': valA_mae_norm,
                    'mse_denorm_valA': valA_mse_denorm,
                    'mae_denorm_valA': valA_mae_denorm,
                    'mse_norm_valB': valB_mse_norm,
                    'mae_norm_valB': valB_mae_norm,
                    'mse_denorm_valB': valB_mse_denorm,
                    'mae_denorm_valB': valB_mae_denorm,
                    'layer': layer,
                    'loss_type': config.loss_type
                }
                
                # Save criterion state if using adaptive loss
                if config.loss_type == 'adaptive':
                    checkpoint_dict['criterion_state_dict'] = criterion.state_dict()
                    checkpoint_dict['lambda_value'] = criterion.get_lambda().item()
                
                torch.save(checkpoint_dict, f"g{layer}.pth")
        
        # Evaluate and save after Pass 1 if not already done
        if iTr[layer] % 10 != 0:
            valA_mse_norm, valA_mae_norm, valA_mse_denorm, valA_mae_denorm = \
                evaluate_model(model, valA_loader, device, num_layers=layer + 1)
            valB_mse_norm, valB_mae_norm, valB_mse_denorm, valB_mae_denorm = \
                evaluate_model(model, valB_loader, device, num_layers=layer + 1)
            
            LossMatrix1_mse_norm_valA[layer] = valA_mse_norm
            LossMatrix1_mae_norm_valA[layer] = valA_mae_norm
            LossMatrix1_mse_denorm_valA[layer] = valA_mse_denorm
            LossMatrix1_mae_denorm_valA[layer] = valA_mae_denorm
            
            LossMatrix1_mse_norm_valB[layer] = valB_mse_norm
            LossMatrix1_mae_norm_valB[layer] = valB_mae_norm
            LossMatrix1_mse_denorm_valB[layer] = valB_mse_denorm
            LossMatrix1_mae_denorm_valB[layer] = valB_mae_denorm
            
            print(f"\n{'=' * 80}")
            print(f"Pass 1 Completed - Layer {layer}")
            print(f"  [valA] Final Validation Loss:")
            print(f"    MSE (Denorm): {valA_mse_denorm:.6f}")
            print(f"    MAE (Denorm): {valA_mae_denorm:.6f}")
            print(f"  [valB] Final Validation Loss:")
            print(f"    MSE (Denorm): {valB_mse_denorm:.6f}")
            print(f"    MAE (Denorm): {valB_mae_denorm:.6f}")
            print(f"{'=' * 80}")
            
            # Save checkpoint
            checkpoint_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': iTr[layer],
                'mse_norm_valA': valA_mse_norm,
                'mae_norm_valA': valA_mae_norm,
                'mse_denorm_valA': valA_mse_denorm,
                'mae_denorm_valA': valA_mae_denorm,
                'mse_norm_valB': valB_mse_norm,
                'mae_norm_valB': valB_mae_norm,
                'mse_denorm_valB': valB_mse_denorm,
                'mae_denorm_valB': valB_mae_denorm,
                'layer': layer,
                'loss_type': config.loss_type
            }
            
            if config.loss_type == 'adaptive':
                checkpoint_dict['criterion_state_dict'] = criterion.state_dict()
                checkpoint_dict['lambda_value'] = criterion.get_lambda().item()
            
            torch.save(checkpoint_dict, f"g{layer}.pth")
        
        print_layer_parameters(model, layer)
        
        # ========== Pass 3 ==========
        print("\n" + "-" * 80)
        print("Pass 3: Fine-tune all layers")
        print("-" * 80)
        
        if os.path.exists(f"g{layer}.pth"):
            checkpoint = torch.load(f"g{layer}.pth", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if config.loss_type == 'adaptive' and 'criterion_state_dict' in checkpoint:
                criterion.load_state_dict(checkpoint['criterion_state_dict'])
                print(f"  Loaded criterion state, current λ = {criterion.get_lambda().item():.6f}")
        
        # Unfreeze all layers
        for i in range(layer + 1):
            if i < len(model.sbl_layers):
                for param in model.sbl_layers[i].parameters():
                    param.requires_grad = True
        
        lr1 = lr
        print(f"Learning Rate (Pass 3): {lr1}")
        
        # Create optimizer parameter list
        optimizer_params = [
            {'params': filter(lambda p: p.requires_grad, model.parameters())}
        ]
        
        if config.loss_type == 'adaptive':
            optimizer_params.append({'params': criterion.parameters()})
        
        optimizer = optim.Adam(
            optimizer_params,
            lr=lr1, betas=(0.9, 0.999), eps=1e-07, amsgrad=True
        )
        
        for epoch in range(iTr2[layer]):
            show_debug = (epoch == 0)
            (train_loss, train_mae_norm, train_mse_norm, train_mae_denorm, train_mse_denorm,
             train_lambda, train_mse_normalized, train_mae_normalized) = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, num_layers=layer + 1, loss_type=config.loss_type,
                show_debug=show_debug, sub_batch_size=config.sub_batch_size
            )
            
            print(f"\nEpoch {epoch + 1}/{iTr2[layer]}, Layer {layer}, Pass 3")
            print(f"  Train Loss:")
            if config.loss_type == 'adaptive':
                print(f"    Mixed Loss:   {train_loss:.6f}  ← (1-λ)*MSE_norm + λ*MAE_norm")
                if train_lambda is not None:
                    print(f"    Current λ:    {train_lambda:.6f}")
                    if train_mse_normalized is not None and train_mae_normalized is not None:
                        print(f"    MSE (normalized): {train_mse_normalized:.6f}")
                        print(f"    MAE (normalized): {train_mae_normalized:.6f}")
                print(f"    MSE (raw):    {train_mse_norm:.8f}")
                print(f"    MAE (raw):    {train_mae_norm:.8f}")
            print(f"    MAE (Denorm): {train_mae_denorm:.6f}")
            print(f"    MSE (Denorm): {train_mse_denorm:.6f}")
            
            if (epoch + 1) % 10 == 0 or epoch == iTr2[layer] - 1:
                valA_mse_norm, valA_mae_norm, valA_mse_denorm, valA_mae_denorm = \
                    evaluate_model(model, valA_loader, device, num_layers=layer + 1)
                valB_mse_norm, valB_mae_norm, valB_mse_denorm, valB_mae_denorm = \
                    evaluate_model(model, valB_loader, device, num_layers=layer + 1)
                
                LossMatrix3_mse_norm_valA[layer] = valA_mse_norm
                LossMatrix3_mae_norm_valA[layer] = valA_mae_norm
                LossMatrix3_mse_denorm_valA[layer] = valA_mse_denorm
                LossMatrix3_mae_denorm_valA[layer] = valA_mae_denorm
                
                LossMatrix3_mse_norm_valB[layer] = valB_mse_norm
                LossMatrix3_mae_norm_valB[layer] = valB_mae_norm
                LossMatrix3_mse_denorm_valB[layer] = valB_mse_denorm
                LossMatrix3_mae_denorm_valB[layer] = valB_mae_denorm
                
                print(f"  [valA] Validation Loss:")
                print(f"    MSE (Denorm): {valA_mse_denorm:.6f}")
                print(f"    MAE (Denorm): {valA_mae_denorm:.6f}")
                print(f"  [valB] Validation Loss:")
                print(f"    MSE (Denorm): {valB_mse_denorm:.6f}")
                print(f"    MAE (Denorm): {valB_mae_denorm:.6f}")
                print(f"  Loss Vector3 valA:")
                print(f"    MSE (Denorm): {LossMatrix3_mse_denorm_valA[:layer + 1]}")
                print(f"    MAE (Denorm): {LossMatrix3_mae_denorm_valA[:layer + 1]}")
                print(f"  Loss Vector3 valB:")
                print(f"    MSE (Denorm): {LossMatrix3_mse_denorm_valB[:layer + 1]}")
                print(f"    MAE (Denorm): {LossMatrix3_mae_denorm_valB[:layer + 1]}")
                
                # Save checkpoint
                checkpoint_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'mse_norm_valA': valA_mse_norm,
                    'mae_norm_valA': valA_mae_norm,
                    'mse_denorm_valA': valA_mse_denorm,
                    'mae_denorm_valA': valA_mae_denorm,
                    'mse_norm_valB': valB_mse_norm,
                    'mae_norm_valB': valB_mae_norm,
                    'mse_denorm_valB': valB_mse_denorm,
                    'mae_denorm_valB': valB_mae_denorm,
                    'layer': layer,
                    'loss_type': config.loss_type
                }
                
                if config.loss_type == 'adaptive':
                    checkpoint_dict['criterion_state_dict'] = criterion.state_dict()
                    checkpoint_dict['lambda_value'] = criterion.get_lambda().item()
                
                torch.save(checkpoint_dict, f"g{layer}.pth")
        
        if iTr2[layer] % 10 != 0:
            valA_mse_norm, valA_mae_norm, valA_mse_denorm, valA_mae_denorm = \
                evaluate_model(model, valA_loader, device, num_layers=layer + 1)
            valB_mse_norm, valB_mae_norm, valB_mse_denorm, valB_mae_denorm = \
                evaluate_model(model, valB_loader, device, num_layers=layer + 1)
            
            LossMatrix3_mse_norm_valA[layer] = valA_mse_norm
            LossMatrix3_mae_norm_valA[layer] = valA_mae_norm
            LossMatrix3_mse_denorm_valA[layer] = valA_mse_denorm
            LossMatrix3_mae_denorm_valA[layer] = valA_mae_denorm
            
            LossMatrix3_mse_norm_valB[layer] = valB_mse_norm
            LossMatrix3_mae_norm_valB[layer] = valB_mae_norm
            LossMatrix3_mse_denorm_valB[layer] = valB_mse_denorm
            LossMatrix3_mae_denorm_valB[layer] = valB_mae_denorm
            
            print(f"\n{'=' * 80}")
            print(f"Pass 3 Completed - Layer {layer}")
            print(f"  [valA] Final Validation Loss:")
            print(f"    MSE (Denorm): {valA_mse_denorm:.6f}")
            print(f"    MAE (Denorm): {valA_mae_denorm:.6f}")
            print(f"  [valB] Final Validation Loss:")
            print(f"    MSE (Denorm): {valB_mse_denorm:.6f}")
            print(f"    MAE (Denorm): {valB_mae_denorm:.6f}")
            print(f"{'=' * 80}")
            
            # Save checkpoint
            checkpoint_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': iTr2[layer],
                'mse_norm_valA': valA_mse_norm,
                'mae_norm_valA': valA_mae_norm,
                'mse_denorm_valA': valA_mse_denorm,
                'mae_denorm_valA': valA_mae_denorm,
                'mse_norm_valB': valB_mse_norm,
                'mae_norm_valB': valB_mae_norm,
                'mse_denorm_valB': valB_mse_denorm,
                'mae_denorm_valB': valB_mae_denorm,
                'layer': layer,
                'loss_type': config.loss_type
            }
            
            if config.loss_type == 'adaptive':
                checkpoint_dict['criterion_state_dict'] = criterion.state_dict()
                checkpoint_dict['lambda_value'] = criterion.get_lambda().item()
            
            torch.save(checkpoint_dict, f"g{layer}.pth")
        
        print_layer_parameters(model, layer)
        
        # If using adaptive loss, print final lambda value
        if config.loss_type == 'adaptive':
            final_lambda = criterion.get_lambda().item()
            print(f"\nFinal λ value after Layer {layer}: {final_lambda:.6f}")
    
    # Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    
    if config.loss_type == 'adaptive':
        final_lambda = criterion.get_lambda().item()
        print(f"\nFinal adaptive λ value: {final_lambda:.6f}")
        print(f"  This means the loss was: {(1 - final_lambda):.4f}*MSE_normalized + {final_lambda:.4f}*MAE_normalized")
    
    print(f"\n{'=' * 80}")
    print("Final Loss Summary")
    print(f"{'=' * 80}")
    
    print(f"\nPass 1 Results - valA (Same head as training):")
    print(f"  MSE (Denorm): {LossMatrix1_mse_denorm_valA}")
    print(f"  MAE (Denorm): {LossMatrix1_mae_denorm_valA}")
    
    print(f"\nPass 1 Results - valB (Differenthead):")
    print(f"  MSE (Denorm): {LossMatrix1_mse_denorm_valB}")
    print(f"  MAE (Denorm): {LossMatrix1_mae_denorm_valB}")
    print(f"\nPass 3 Results - valA (Same head as training):")
    print(f"  MSE (Denorm): {LossMatrix3_mse_denorm_valA}")
    print(f"  MAE (Denorm): {LossMatrix3_mae_denorm_valA}")

    print(f"\nPass 3 Results - valB (Different head):")
    print(f"  MSE (Denorm): {LossMatrix3_mse_denorm_valB}")
    print(f"  MAE (Denorm): {LossMatrix3_mae_denorm_valB}")

    print("=" * 80)

    # ========================================
    # Save validation loss matrices to .npz file
    # ========================================
    loss_filename = f"validation_losses_{config.loss_type}.npz"

    print(f"\nSaving validation losses to file: {loss_filename}")

    np.savez(
        loss_filename,
        # Pass 1 - valA
        LossMatrix1_mse_norm_valA=LossMatrix1_mse_norm_valA,
        LossMatrix1_mae_norm_valA=LossMatrix1_mae_norm_valA,
        LossMatrix1_mse_denorm_valA=LossMatrix1_mse_denorm_valA,
        LossMatrix1_mae_denorm_valA=LossMatrix1_mae_denorm_valA,
        # Pass 1 - valB
        LossMatrix1_mse_norm_valB=LossMatrix1_mse_norm_valB,
        LossMatrix1_mae_norm_valB=LossMatrix1_mae_norm_valB,
        LossMatrix1_mse_denorm_valB=LossMatrix1_mse_denorm_valB,
        LossMatrix1_mae_denorm_valB=LossMatrix1_mae_denorm_valB,
        # Pass 3 - valA
        LossMatrix3_mse_norm_valA=LossMatrix3_mse_norm_valA,
        LossMatrix3_mae_norm_valA=LossMatrix3_mae_norm_valA,
        LossMatrix3_mse_denorm_valA=LossMatrix3_mse_denorm_valA,
        LossMatrix3_mae_denorm_valA=LossMatrix3_mae_denorm_valA,
        # Pass 3 - valB
        LossMatrix3_mse_norm_valB=LossMatrix3_mse_norm_valB,
        LossMatrix3_mae_norm_valB=LossMatrix3_mae_norm_valB,
        LossMatrix3_mse_denorm_valB=LossMatrix3_mse_denorm_valB,
        LossMatrix3_mae_denorm_valB=LossMatrix3_mae_denorm_valB,
        # Metadata
        loss_type=config.loss_type,
        n_layers=T,
        init_lambda=config.init_lambda if config.loss_type == 'adaptive' else None,
        final_lambda=criterion.get_lambda().item() if config.loss_type == 'adaptive' else None
    )

    print(f"✓ Validation losses saved")
    print(f"  File: {loss_filename}")
    print(f"  Contents:")
    print(f"    - Pass 1 (valA): MSE_norm, MAE_norm, MSE_denorm, MAE_denorm")
    print(f"    - Pass 1 (valB): MSE_norm, MAE_norm, MSE_denorm, MAE_denorm")
    print(f"    - Pass 3 (valA): MSE_norm, MAE_norm, MSE_denorm, MAE_denorm")
    print(f"    - Pass 3 (valB): MSE_norm, MAE_norm, MSE_denorm, MAE_denorm")

    print("\n" + "=" * 80)
    print("All tasks completed successfully!")
    print("=" * 80)