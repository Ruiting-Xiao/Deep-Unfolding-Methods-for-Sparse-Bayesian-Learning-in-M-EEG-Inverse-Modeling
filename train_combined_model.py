#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ruiting Xiao
"""
EEG Source Localization Training Script using L-SBL (Learned Sparse Bayesian Learning)
=======================================================================================

Mode 4: Combined Gamma Update (Weighted Combination of MacKay, Modified MacKay, and EM)

This script implements layer-wise training of L-SBL networks for EEG inverse problem solving.
It uses pre-normalized datasets with three splits:
- train:  Uses global_G_norm_train.npy (training head model)
- valA:   Uses global_G_norm_valA.npy (same head model as train, for distribution match)
- valB:   Uses global_G_norm_valB.npy (validation head model, for generalization test)

Training Strategy:
-----------------
- Layer-wise progressive training
- Pass 1: Train only the current layer
- Pass 3: Fine-tune all layers up to the current layer
- Preserves all debug/gradient checking/checkpoint behavior

Key Features:
------------
- Simple normalization for weights: ensures w1, w2, w3 > 0 and w1 + w2 + w3 = 1
- Uses L1 loss for training while monitoring both MSE and MAE
- No bias terms
- Sub-batch processing to reduce GPU memory pressure

Model Architecture:
------------------
Each SBL layer combines three gamma update rules:
1. MacKay:          γ² * mean(A²) / (γ * trace_term)
2. Modified MacKay: γ * sqrt(mean(A²)) / sqrt(trace_term)
3. EM:              γ² * mean(A²) + γ * (1 - γ * trace_term)

Final update: γ_new = w1 * γ_mode1 + w2 * γ_mode2 + w3 * γ_mode3

"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Command line argument parser
parser = argparse.ArgumentParser(description='L-SBL Mode 4 Training with Combined Updates')
parser.add_argument('-dF', '--data_folder', type=str, default='./normalized_data',
                    help="Path to pre-normalized data folder")
parser.add_argument('-T', '--T', type=int, default=100, 
                    help="Number of LSBL layers")
parser.add_argument('-batch_size', '--batch_size', type=int, default=128,
                    help="Batch size used during .npy generation")
parser.add_argument('-sub_batch_size', '--sub_batch_size', type=int, default=128,
                    help="Actual batch size during training (splits .npy samples into sub-batches)")
parser.add_argument('-num_workers', '--num_workers', type=int, default=2,
                    help="Number of data loading workers (0 recommended for memory stability)")
parser.add_argument('-lr', '--lr', type=float, default=0.0001, 
                    help="Learning rate")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================
# Dataset Classes
# ================================
class NormalizedEEGDataset(Dataset):
    """
    Dataset for pre-normalized EEG data with three splits: train/valA/valB.
    
    Each batch consists of 4 files:
    - X_norm: Normalized source signals
    - Y_norm: Normalized sensor measurements
    - Lambda_norm: Normalized noise covariance
    - M_norm: Normalization constants
    
    Global lead field (G_norm) and normalization constant (G_norm_const) are loaded once.
    
    Parameters
    ----------
    data_folder : str
        Path to normalized data directory
    split : str
        Dataset split: 'train', 'valA', or 'valB'
    g_tag : str
        Lead field tag: 'train', 'valA', or 'valB'
        Maps to corresponding global_G_norm_{tag}.npy file
    
    Attributes
    ----------
    G_norm : torch.Tensor
        Normalized lead field matrix, shape (1, M, N)
    G_norm_const : torch.Tensor
        Lead field normalization constant, shape (1, 1, 1)
    n_batches : int
        Number of batch files in this split
    M : int
        Number of sensors
    N : int
        Number of sources
    L : int
        Number of time points
    """

    def __init__(self, data_folder, split='train', g_tag='train'):
        self.data_folder = data_folder
        self.split = split
        self.g_tag = g_tag

        # Load metadata
        metadata_path = os.path.join(data_folder, 'metadata.npz')
        md = np.load(metadata_path)

        # Determine batch count based on split
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

        # Reshape for broadcasting
        self.G_norm = self.G_norm.unsqueeze(0)            # (1, M, N)
        self.G_norm_const = self.G_norm_const.view(1, 1, 1)  # (1, 1, 1)

        self.file_indices = list(range(self.n_batches))

    def __len__(self):
        """Return number of batch files."""
        return self.n_batches

    def _load_file(self, idx):
        """
        Load a single batch file.
        
        Parameters
        ----------
        idx : int
            Batch file index
        
        Returns
        -------
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
        """Get batch data by index."""
        return self._load_file(idx)


# ================================
# Utility Functions
# ================================
def print_layer_parameters(model, layer_idx):
    """
    Print parameter statistics for a specific layer (Mode 4: Simple Normalization).
    
    Displays statistics for the three gamma update weights (w1, w2, w3) including:
    - Mean, min, max, median, standard deviation
    - Weight sum verification (should equal 1.0)
    
    Parameters
    ----------
    model : LSBLNetwork
        The L-SBL model
    layer_idx : int
        Index of the layer to inspect
    """
    if layer_idx >= len(model.sbl_layers):
        return

    layer = model.sbl_layers[layer_idx]

    with torch.no_grad():
        # Get normalized weights
        weights = layer.gamma_update.get_weights()  # (3, N)
        w1 = weights[0].cpu().numpy()
        w2 = weights[1].cpu().numpy()
        w3 = weights[2].cpu().numpy()

        print(f"\n{'=' * 80}")
        print(f"Layer {layer_idx} Parameter Statistics (Mode 4 - Simple Normalization):")
        print(f"{'=' * 80}")

        print(f"\nw1 - MacKay weight (shape: {w1.shape}):")
        print(f"  Mean:   {np.mean(w1):.8f}")
        print(f"  Min:    {np.min(w1):.8f}")
        print(f"  Max:    {np.max(w1):.8f}")
        print(f"  Median: {np.median(w1):.8f}")
        print(f"  Std:    {np.std(w1):.8f}")

        print(f"\nw2 - Modified MacKay weight (shape: {w2.shape}):")
        print(f"  Mean:   {np.mean(w2):.8f}")
        print(f"  Min:    {np.min(w2):.8f}")
        print(f"  Max:    {np.max(w2):.8f}")
        print(f"  Median: {np.median(w2):.8f}")
        print(f"  Std:    {np.std(w2):.8f}")

        print(f"\nw3 - EM weight (shape: {w3.shape}):")
        print(f"  Mean:   {np.mean(w3):.8f}")
        print(f"  Min:    {np.min(w3):.8f}")
        print(f"  Max:    {np.max(w3):.8f}")
        print(f"  Median: {np.median(w3):.8f}")
        print(f"  Std:    {np.std(w3):.8f}")

        # Verify weight sum equals 1
        weight_sum = w1 + w2 + w3
        print(f"\nWeight sum verification (should be 1.0):")
        print(f"  Mean:   {np.mean(weight_sum):.8f}")
        print(f"  Min:    {np.min(weight_sum):.8f}")
        print(f"  Max:    {np.max(weight_sum):.8f}")

        print(f"{'=' * 80}\n")


# ================================
# Neural Network Layers
# ================================
class MuEstimateLayer(nn.Module):
    """
    E-step: Compute posterior mean, variance, A, and trace_term using Cholesky decomposition.
    
    This layer implements the expectation step of the SBL algorithm, computing the posterior
    distribution of source signals given sensor measurements.
    
    Parameters
    ----------
    InpSize : int
        Number of sensors (M)
    OutSize : int
        Number of sources (N)
    L : int
        Number of time points
    reg_lambda : float
        Regularization parameter for numerical stability (default: 1e-6)
    """

    def __init__(self, InpSize, OutSize, L, reg_lambda=1e-6):
        super(MuEstimateLayer, self).__init__()
        self.InpSize = InpSize
        self.OutSize = OutSize
        self.L = L
        self.reg_lambda = reg_lambda

    def forward(self, y, gamma, Lambda, Lmat):
        """
        Compute E-step quantities.
        
        Parameters
        ----------
        y : torch.Tensor
            Sensor measurements, shape (B, M, L)
        gamma : torch.Tensor
            Source variances (hyperparameters), shape (B, N)
        Lambda : torch.Tensor
            Noise covariance matrix, shape (B, M, M)
        Lmat : torch.Tensor
            Lead field matrix, shape (B, M, N)
        
        Returns
        -------
        mu : torch.Tensor
            Posterior mean of sources, shape (B, N, L)
        Sigma_X_diag : torch.Tensor
            Diagonal of posterior covariance, shape (B, N)
        A : torch.Tensor
            Intermediate quantity CMinvG.T @ y, shape (B, N, L)
        trace_term : torch.Tensor
            Trace term sum(G * CMinvG, dim=1), shape (B, N)
        
        Notes
        -----
        Computes CM = G * diag(gamma) * G^T + Lambda
        Solves linear system using Cholesky decomposition with fallback to SVD
        """
        B, M, L = y.shape
        _, _, N = Lmat.shape
        eps = torch.finfo(gamma.dtype).eps

        # Compute CM = G * diag(gamma) * G^T + Lambda
        gamma_expanded = gamma.unsqueeze(1)       # (B, 1, N)
        L_Gamma = Lmat * gamma_expanded          # (B, M, N)
        CM = torch.bmm(L_Gamma, Lmat.transpose(1, 2))
        CM = CM + Lambda

        # Ensure symmetry
        CM = (CM + CM.transpose(1, 2)) / 2

        # Solve linear system using Cholesky decomposition
        try:
            # First attempt: Direct Cholesky
            L_chol = torch.linalg.cholesky(CM)
            CMinvG = torch.cholesky_solve(Lmat, L_chol)

        except RuntimeError as e1:
            # Print first failure details
            print("\n[Cholesky ERROR 1] Cholesky failed on CM:")
            print("  → Error:", str(e1))
            print("  → CM stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(
                CM.min().item(), CM.max().item(), CM.mean().item()))
            diag = CM.diagonal(dim1=1, dim2=2)
            print("  → CM diag stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(
                diag.min().item(), diag.max().item(), diag.mean().item()))

            # Second attempt: Add jitter
            jitter = 1e-10
            eye = torch.eye(CM.size(-1), device=CM.device).unsqueeze(0)
            CM_reg = CM + jitter * eye

            try:
                L_chol = torch.linalg.cholesky(CM_reg)
                CMinvG = torch.cholesky_solve(Lmat, L_chol)

            except RuntimeError as e2:
                # Print second failure details
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

        # Compute A = CMinvG^T @ y
        A = torch.bmm(CMinvG.transpose(1, 2), y)  # (B, N, L)

        # Compute posterior mean: mu = gamma * A
        mu = gamma.unsqueeze(2) * A

        # Compute trace_term = sum(G * CMinvG, dim=1)
        trace_term = torch.sum(Lmat * CMinvG, dim=1)  # (B, N)

        # Compute posterior variance diagonal: Sigma_X_diag = gamma - gamma² * trace_term
        Sigma_X_diag = gamma - gamma * gamma * trace_term

        # Prevent negative variance
        Sigma_X_diag = torch.clamp(Sigma_X_diag, min=eps)

        return mu, Sigma_X_diag, A, trace_term


class DiagonalGammaUpdate(nn.Module):
    """
    M-step: Learnable combined gamma update (Mode 4).
    
    Combines three gamma update rules with learnable weights:
    1. MacKay:          γ² * mean(A²) / (γ * trace_term)
    2. Modified MacKay: γ * sqrt(mean(A²)) / sqrt(trace_term)
    3. EM:              γ² * mean(A²) + γ * (1 - γ * trace_term)
    
    Final update: γ_new = w1 * γ_mode1 + w2 * γ_mode2 + w3 * γ_mode3
    
    Uses simple normalization to ensure:
    - w1, w2, w3 > 0
    - w1 + w2 + w3 = 1
    
    Parameters
    ----------
    OutSize : int
        Number of sources (N)
    
    Attributes
    ----------
    weight_params : nn.Parameter
        Learnable weight parameters, shape (3, N)
        Initialized to 1/3 for equal weighting
    """

    def __init__(self, OutSize):
        super().__init__()
        self.OutSize = OutSize
        # Initialize weights to 1/3 (equal weighting)
        self.weight_params = nn.Parameter(torch.ones(3, OutSize) / 3)

    def get_weights(self):
        """
        Get normalized weights ensuring w1 + w2 + w3 = 1 and all positive.
        
        Returns
        -------
        torch.Tensor
            Normalized weights, shape (3, N)
        """
        # Use abs to ensure non-negative
        abs_weights = torch.abs(self.weight_params)
        # Normalize to sum to 1
        return abs_weights / (abs_weights.sum(dim=0, keepdim=True) + 1e-10)

    def forward(self, gamma, A, trace_term):
        """
        Compute combined gamma update.
        
        Parameters
        ----------
        gamma : torch.Tensor
            Current gamma values, shape (B, N)
        A : torch.Tensor
            Intermediate quantity CMinvG.T @ y, shape (B, N, L)
        trace_term : torch.Tensor
            Trace term sum(G * CMinvG, dim=1), shape (B, N)
        
        Returns
        -------
        gamma_new : torch.Tensor
            Updated gamma values, shape (B, N)
        """
        eps = torch.finfo(gamma.dtype).eps

        # Compute mean(A²)
        A_squared_mean = torch.mean(A * A, dim=2)  # (B, N)

        # Mode 1: MacKay
        # γ² * mean(A²) / (γ * trace_term)
        numer1 = gamma * gamma * A_squared_mean
        denom1 = gamma * trace_term
        gamma_mode1 = numer1 / (denom1 + eps)

        # Mode 2: Modified MacKay
        # γ * sqrt(mean(A²)) / sqrt(trace_term)
        numer2 = gamma * torch.sqrt(A_squared_mean + eps)
        denom2 = torch.sqrt(trace_term + eps)
        gamma_mode2 = numer2 / (denom2 + eps)

        # Mode 3: EM
        # γ² * mean(A²) + γ * (1 - γ * trace_term)
        gamma_mode3 = gamma * gamma * A_squared_mean + gamma * (1 - gamma * trace_term)

        # Get normalized weights and compute weighted combination
        weights = self.get_weights()  # (3, N)
        w1 = weights[0]  # (N,)
        w2 = weights[1]  # (N,)
        w3 = weights[2]  # (N,)

        gamma_new = w1 * gamma_mode1 + w2 * gamma_mode2 + w3 * gamma_mode3

        return gamma_new


class SBLLayer(nn.Module):
    """
    Single SBL layer: E-step + M-step.
    
    Combines posterior estimation (E-step) and hyperparameter update (M-step)
    into a single differentiable layer.
    
    Parameters
    ----------
    OutSize : int
        Number of sources (N)
    InpSize : int
        Number of sensors (M)
    L : int
        Number of time points
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
        Forward pass through SBL layer.
        
        Parameters
        ----------
        y : torch.Tensor
            Sensor measurements, shape (B, M, L)
        gamma : torch.Tensor
            Source variances, shape (B, N)
        Lambda : torch.Tensor
            Noise covariance, shape (B, M, M)
        Lmat : torch.Tensor
            Lead field matrix, shape (B, M, N)
        
        Returns
        -------
        mu : torch.Tensor
            Posterior mean, shape (B, N, L)
        Sigma_X_diag : torch.Tensor
            Posterior variance diagonal, shape (B, N)
        gamma_new : torch.Tensor
            Updated gamma, shape (B, N)
        """
        mu, Sigma_X_diag, A, trace_term = self.mu_estimate(y, gamma, Lambda, Lmat)
        gamma_new = self.gamma_update(gamma, A, trace_term)
        return mu, Sigma_X_diag, gamma_new


class LSBLNetwork(nn.Module):
    """
    Complete L-SBL network (Mode 4: Combined Update).
    
    Stacks multiple SBL layers for iterative refinement of source estimates.
    
    Parameters
    ----------
    InpSize : int
        Number of sensors (M)
    OutSize : int
        Number of sources (N)
    L : int
        Number of time points
    T : int
        Number of SBL layers
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
        """Initialize all layers with equal weight combination: w1 = w2 = w3 = 1/3."""
        for layer in self.sbl_layers:
            with torch.no_grad():
                layer.gamma_update.weight_params.fill_(1.0 / 3.0)

    def forward(self, y, gamma_0, Lambda, Lmat, num_layers=None):
        """
        Forward pass through network.
        
        Parameters
        ----------
        y : torch.Tensor
            Sensor measurements, shape (B, M, L)
        gamma_0 : torch.Tensor
            Initial gamma (typically ones), shape (B, N)
        Lambda : torch.Tensor
            Noise covariance, shape (B, M, M)
        Lmat : torch.Tensor
            Lead field matrix, shape (B, M, N)
        num_layers : int, optional
            Number of layers to use (default: all T layers)
        
        Returns
        -------
        mu_final : torch.Tensor
            Final source estimate, shape (B, N, L)
        """
        if num_layers is None:
            num_layers = self.T

        gamma = gamma_0

        # Iteratively update gamma through layers
        for i in range(num_layers):
            mu, Sigma_X_diag, gamma = self.sbl_layers[i](y, gamma, Lambda, Lmat)

        # Compute final mu with updated gamma
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
                   show_debug=False, sub_batch_size=64):
    """
    Train model for one epoch with sub-batch processing.
    
    Parameters
    ----------
    model : LSBLNetwork
        The L-SBL model to train
    dataloader : DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates
    criterion : callable
        Loss function (L1 loss)
    device : torch.device
        Device for computation
    num_layers : int
        Number of layers to use in forward pass
    show_debug : bool
        Whether to print debug information (default: False)
    sub_batch_size : int
        Size of sub-batches for memory efficiency (default: 64)
    
    Returns
    -------
    avg_mae_norm : float
        Average MAE in normalized space
    avg_mse_norm : float
        Average MSE in normalized space
    avg_mae_denorm : float
        Average MAE in denormalized space
    avg_mse_denorm : float
        Average MSE in denormalized space
    
    Notes
    -----
    - Uses L1 loss for training
    - Monitors both MSE and MAE
    - Splits large batches into sub-batches to reduce GPU memory usage
    - Gradient clipping applied with max_norm=1.0
    """
    model.train()
    total_loss_norm = 0.0
    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_denorm = 0.0
    total_mae_denorm = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        X_norm, Y_norm, Lambda_norm, M_norm = [
            x.squeeze(0).to(device) for x in batch
        ]

        B = X_norm.shape[0]  # Original batch size (e.g., 128)
        N = X_norm.shape[1]

        dataset = dataloader.dataset
        G_norm = dataset.G_norm  # (1, M, N)
        G_norm_const = dataset.G_norm_const

        # Split into sub-batches for memory efficiency
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

            # Normalized space loss (L1 for training)
            loss_norm = criterion(outputs_norm, X_sub)

            # Compute metrics in normalized space (for monitoring)
            with torch.no_grad():
                mse_norm = torch.mean((outputs_norm - X_sub) ** 2)
                mae_norm = torch.mean(torch.abs(outputs_norm - X_sub))

                # Denormalize for real-space metrics
                denorm_factor = (M_sub.sqrt().view(B_sub, 1, 1) / G_norm_const)
                outputs_denorm = outputs_norm * denorm_factor
                X_denorm = X_sub * denorm_factor
                
                # Denormalized space MSE and MAE
                mse_denorm = torch.mean((outputs_denorm - X_denorm) ** 2)
                mae_denorm = torch.mean(torch.abs(outputs_denorm - X_denorm))

            # Debug output (only first batch, first sub-batch)
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
                print(f"      Normalized space:")
                print(f"        MAE (L1): {mae_norm.item():.8f}  ← Used for backprop")
                print(f"        MSE:      {mse_norm.item():.8f}  ← For monitoring")
                print(f"      Denormalized space:")
                print(f"        MAE (L1): {mae_denorm.item():.8f}")
                print(f"        MSE:      {mse_denorm.item():.8f}")

                k = denorm_factor.mean().item()
                k2 = k ** 2
                print(f"\n      Scaling factors:")
                print(f"        k (for MAE): {k:.6e}")
                print(f"        k² (for MSE): {k2:.6e}")

            # Backward pass (using L1 loss)
            loss_norm.backward()

            # Gradient checking (only first batch, first sub-batch)
            if batch_idx == 0 and sub_idx == 0 and show_debug:
                target_layer = num_layers - 1
                if target_layer < len(model.sbl_layers):
                    layer = model.sbl_layers[target_layer]
                    if layer.gamma_update.weight_params.grad is None:
                        print(f"\n    ⚠️ WARNING: No gradient for Layer {target_layer}!")
                    else:
                        grad_norm = layer.gamma_update.weight_params.grad.abs().mean().item()
                        grad_max = layer.gamma_update.weight_params.grad.abs().max().item()
                        print(f"\n    ✓ Layer {target_layer} gradient statistics:")
                        print(f"      Mean: {grad_norm:.6e}")
                        print(f"      Max:  {grad_max:.6e}")
                print(f"    {'─' * 60}\n")

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate statistics
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

    # Compute averages
    avg_mae_norm = total_mae_norm / total_samples
    avg_mse_norm = total_mse_norm / total_samples
    avg_mae_denorm = total_mae_denorm / total_samples
    avg_mse_denorm = total_mse_denorm / total_samples

    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / 1024 ** 2, "MB")
    print("Max Memory Reserved:", torch.cuda.max_memory_reserved() / 1024 ** 2, "MB")
    
    return avg_mae_norm, avg_mse_norm, avg_mae_denorm, avg_mse_denorm


def evaluate_model(model, dataloader, device, num_layers):
    """
    Evaluate model on a given dataset (train/valA/valB).
    
    Parameters
    ----------
    model : LSBLNetwork
        The L-SBL model to evaluate
    dataloader : DataLoader
        Data loader for evaluation
    device : torch.device
        Device for computation
    num_layers : int
        Number of layers to use
    
    Returns
    -------
    tuple of float
        (MSE_norm, MAE_norm, MSE_denorm, MAE_denorm)
        Metrics in both normalized and denormalized spaces
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

            # Normalized space metrics
            mse_norm = torch.mean((out - X_norm) ** 2)
            mae_norm = torch.mean(torch.abs(out - X_norm))

            # Denormalized space metrics
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
# Main Training Pipeline
# ================================

if __name__ == "__main__":
    maximum_lsbl_layers = 200
    config, unparsed = parser.parse_known_args()

    print('=' * 80)
    print('EEG Source Localization using L-SBL Mode 4')
    print('Combined Update: MacKay + Modified MacKay + EM')
    print('Simple normalization constraint: w1, w2, w3 > 0 and w1 + w2 + w3 = 1')
    print('Training with L1 loss, monitoring both MSE and MAE')
    print('=' * 80)
    print(f'Data folder: {config.data_folder}')
    print(f'Number of LSBL layers: {config.T}')
    print(f'Batch size in .npy files: {config.batch_size}')
    print(f'Sub-batch size for training (actual GPU batch): {config.sub_batch_size}')
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
            f'Please run: python preprocess_normalized_data.py -iF <raw_data_path> -oF {config.data_folder}'
        )

    M = int(metadata['n_sensors'])
    N = int(metadata['n_sources'])
    L = int(metadata['n_time'])

    print(f'\nData Information:')
    print(f'  Sensors (M): {M}')
    print(f'  Sources (N): {N}')
    print(f'  Time points (L): {L}')
    print(f'  Training samples:   {metadata["n_train"]}')
    print(f'  Validation A samples (same head as train): {metadata["n_valA"]}')
    print(f'  Validation B samples (different head):     {metadata["n_valB"]}')
    print(f'  Data status: ✓ Normalized')

    if config.T > maximum_lsbl_layers:
        print(f'Warning: Maximum number of layers cannot exceed {maximum_lsbl_layers}')
        config.T = maximum_lsbl_layers

    # Create datasets
    print('\nCreating datasets...')
    train_dataset = NormalizedEEGDataset(config.data_folder, split='train', g_tag='train')
    valA_dataset = NormalizedEEGDataset(config.data_folder, split='valA', g_tag='valA')
    valB_dataset = NormalizedEEGDataset(config.data_folder, split='valB', g_tag='valB')

    # DataLoaders: batch_size=1 at outer level (each .npy file is one batch)
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

    # Model parameters
    InpSize = M
    OutSize = N
    T = config.T

    # Training parameters
    iTr = np.ones([1, T], dtype='int32') * 3   # Pass 1: epochs per layer
    iTr2 = np.ones([1, T], dtype='int32') * 3  # Pass 3: fine-tune epochs

    iTr[0, 0] = 3
    iTr2[0, 0] = 3
    if T > 1:
        iTr2[0, 1] = 3

    iTr = iTr.reshape(T,)
    iTr2 = iTr2.reshape(T,)

    # Loss matrices for tracking (MSE and MAE separately)
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

    # Define loss functions
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    def criterion(outputs, targets):
        """Combined loss function: L1 + 10*MSE"""
        l1 = criterion_l1(outputs, targets)
        return l1

    # Create model
    print('\nInitializing model (Mode 4: Combined Update with Simple Normalization)...')
    model = LSBLNetwork(InpSize, OutSize, L, T).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    layer_params = sum(p.numel() for p in model.sbl_layers[0].parameters())
    print(f"Parameters per layer: {layer_params} (weight_params: 3*N)")

    # Layer-wise training
    layerIndex = np.arange(0, T).tolist()

    for layer in layerIndex:
        print("\n" + "=" * 80)
        print(f"Training Layer: {layer}")
        print("=" * 80)

        # ========== Pass 1: Train current layer only ==========
        print("\n" + "-" * 80)
        print("Pass 1: Training current layer only")
        print("-" * 80)

        # Load previous checkpoint if exists
        if layer > 0:
            checkpoint_path = f"g{layer - 1}.pth"
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

                print("\nTesting performance after loading...")

                # Evaluate on both validation sets
                mse_old_norm_A, mae_old_norm_A, mse_old_denorm_A, mae_old_denorm_A = \
                    evaluate_model(model, valA_loader, device, num_layers=layer)
                mse_new_norm_A, mae_new_norm_A, mse_new_denorm_A, mae_new_denorm_A = \
                    evaluate_model(model, valA_loader, device, num_layers=layer + 1)

                mse_old_norm_B, mae_old_norm_B, mse_old_denorm_B, mae_old_denorm_B = \
                    evaluate_model(model, valB_loader, device, num_layers=layer)
                mse_new_norm_B, mae_new_norm_B, mse_new_denorm_B, mae_new_denorm_B = \
                    evaluate_model(model, valB_loader, device, num_layers=layer + 1)

                # Print valA comparison
                print(f"\n[valA] Performance Comparison:")
                print(f"  With {layer} layers:")
                print(f"    MSE (Denorm): {mse_old_denorm_A:.8f}")
                print(f"    MAE (Denorm): {mae_old_denorm_A:.8f}")
                print(f"  With {layer+1} layers:")
                print(f"    MSE (Denorm): {mse_new_denorm_A:.8f}")
                print(f"    MAE (Denorm): {mae_new_denorm_A:.8f}")
                
                improvement_mse_A = mse_old_denorm_A - mse_new_denorm_A
                improvement_mae_A = mae_old_denorm_A - mae_new_denorm_A
                pct_mse_A = (improvement_mse_A / mse_old_denorm_A) * 100 if mse_old_denorm_A > 0 else 0
                pct_mae_A = (improvement_mae_A / mae_old_denorm_A) * 100 if mae_old_denorm_A > 0 else 0
                print(f"  Improvement:")
                print(f"    MSE: {improvement_mse_A:.8f} ({pct_mse_A:.2f}%)")
                print(f"    MAE: {improvement_mae_A:.8f} ({pct_mae_A:.2f}%)")

                # Print valB comparison
                print(f"\n[valB] Performance Comparison:")
                print(f"  With {layer} layers:")
                print(f"    MSE (Denorm): {mse_old_denorm_B:.8f}")
                print(f"    MAE (Denorm): {mae_old_denorm_B:.8f}")
                print(f"  With {layer+1} layers:")
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
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr1, betas=(0.9, 0.999), eps=1e-07,
            amsgrad=(layer >= 4)
        )

        # Pass 1 training loop
        for epoch in range(iTr[layer]):
            show_debug = (epoch == 0)
            train_mae_norm, train_mse_norm, train_mae_denorm, train_mse_denorm = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, num_layers=layer + 1, show_debug=show_debug, sub_batch_size=config.sub_batch_size 
            )

            print(f"\nEpoch {epoch + 1}/{iTr[layer]}, Layer {layer}, Pass 1")
            print(f"  Train Loss:")
            print(f"    MAE (Denorm): {train_mae_denorm:.6f}  ← Training metric (L1)")
            print(f"    MSE (Denorm): {train_mse_denorm:.6f}")

            # Periodic evaluation
            if (epoch + 1) % 10 == 0 or epoch == iTr[layer] - 1:
                valA_mse_norm, valA_mae_norm, valA_mse_denorm, valA_mae_denorm = \
                    evaluate_model(model, valA_loader, device, num_layers=layer + 1)
                valB_mse_norm, valB_mae_norm, valB_mse_denorm, valB_mae_denorm = \
                    evaluate_model(model, valB_loader, device, num_layers=layer + 1)

                # Save to loss matrices
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
                torch.save({
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
                    'layer': layer
                }, f"g{layer}.pth")

        # Final evaluation for Pass 1 if not already done
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

            torch.save({
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
                'layer': layer
            }, f"g{layer}.pth")

        print_layer_parameters(model, layer)

        # ========== Pass 3: Fine-tune all layers ==========
        print("\n" + "-" * 80)
        print("Pass 3: Fine-tune all layers")
        print("-" * 80)

        # Reload checkpoint
        if os.path.exists(f"g{layer}.pth"):
            checkpoint = torch.load(f"g{layer}.pth", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Unfreeze all layers up to current
        for i in range(layer + 1):
            if i < len(model.sbl_layers):
                for param in model.sbl_layers[i].parameters():
                    param.requires_grad = True

        lr1 = lr
        print(f"Learning Rate (Pass 3): {lr1}")

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr1, betas=(0.9, 0.999), eps=1e-07, amsgrad=True
        )

        # Pass 3 training loop
        for epoch in range(iTr2[layer]):
            show_debug = (epoch == 0)
            train_mae_norm, train_mse_norm, train_mae_denorm, train_mse_denorm = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, num_layers=layer + 1, show_debug=show_debug, sub_batch_size=config.sub_batch_size 
            )

            print(f"\nEpoch {epoch + 1}/{iTr2[layer]}, Layer {layer}, Pass 3")
            print(f"  Train Loss:")
            print(f"    MAE (Denorm): {train_mae_denorm:.6f}  ← Training metric (L1)")
            print(f"    MSE (Denorm): {train_mse_denorm:.6f}")

            # Periodic evaluation
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

                torch.save({
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
                    'layer': layer
                }, f"g{layer}.pth")

        # Final evaluation for Pass 3 if not already done
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

            torch.save({
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
                'layer': layer
            }, f"g{layer}.pth")

        print_layer_parameters(model, layer)

    # Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    
    print(f"\n{'=' * 80}")
    print("Final Loss Summary")
    print(f"{'=' * 80}")
    
    print(f"\nPass 1 Results - valA (Same head as training):")
    print(f"  MSE (Denorm): {LossMatrix1_mse_denorm_valA}")
    print(f"  MAE (Denorm): {LossMatrix1_mae_denorm_valA}")
    
    print(f"\nPass 1 Results - valB (Different head):")
    print(f"  MSE (Denorm): {LossMatrix1_mse_denorm_valB}")
    print(f"  MAE (Denorm): {LossMatrix1_mae_denorm_valB}")
    
    print(f"\nPass 3 Results - valA (Same head as training):")
    print(f"  MSE (Denorm): {LossMatrix3_mse_denorm_valA}")
    print(f"  MAE (Denorm): {LossMatrix3_mae_denorm_valA}")
    
    print(f"\nPass 3 Results - valB (Different head):")
    print(f"  MSE (Denorm): {LossMatrix3_mse_denorm_valB}")
    print(f"  MAE (Denorm): {LossMatrix3_mae_denorm_valB}")
    
    print("=" * 80)
