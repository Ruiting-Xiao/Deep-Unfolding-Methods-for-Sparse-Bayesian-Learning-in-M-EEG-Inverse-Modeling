#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate an EM-based neural network on the TRAINING dataset.

This script is designed to be run in a notebook or as a standalone script.
Hyperparameters and paths are configured directly in this file.

Main purpose:
- Load a trained LSBL (Learned SBL) network checkpoint.
- Run inference for a certain number of EM/SBL iterations.
- Compute multiple per-sample metrics (MSE/MAE/RMSE/RMAE/EMD/Sparsity/Active sources/Sigma).
- Save aggregated statistics to an .npz file.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import ot
from scipy.spatial.distance import cdist

# Use GPU if available, otherwise fallback to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Earth Mover's Distance (EMD) computation
# =============================================================================
def compute_emd_batch(y_pred, y_true, source_locations):
    """
    Compute EMD (Earth Mover's Distance) per sample in a batch.
    This is the "raw" EMD version: the input magnitudes are normalized to sum to 1,
    but the EMD output is not further normalized by any maximum distance.

    Idea:
    - For each sample, convert each source time-series into a scalar "mass"
      using L2 norm across time: mass_i = ||x_i(t)||_2.
    - Keep only non-zero masses (to avoid degenerate distributions).
    - Build pairwise Euclidean distance matrix between predicted non-zero sources
      and true non-zero sources using source_locations.
    - Use POT's ot.emd2 to compute the transportation cost.

    Args:
        y_pred (torch.Tensor): Predicted source activity, shape (B, N, L).
        y_true (torch.Tensor): Ground-truth source activity, shape (B, N, L).
        source_locations (np.ndarray or array-like):
            Source coordinates for each of N sources, shape (N, D),
            usually D=3 for 3D coordinates.

    Returns:
        np.ndarray:
            EMD values for each sample, shape (B,).
            If either predicted or true mass distribution is empty for a sample,
            returns np.inf for that sample.
    """
    B, N, L = y_pred.shape

    # Convert time-series to per-source magnitude (mass) via L2 norm over time.
    temp_pred = torch.norm(y_pred, p=2, dim=2)  # (B, N)
    temp_true = torch.norm(y_true, p=2, dim=2)  # (B, N)

    # Move to CPU numpy for POT + scipy distance computations.
    temp_pred_np = temp_pred.detach().cpu().numpy().astype(np.float64)
    temp_true_np = temp_true.detach().cpu().numpy().astype(np.float64)

    emd_values = []
    eps_np = np.finfo(np.float64).eps

    for b in range(B):
        # Keep only non-zero mass sources to reduce cost and avoid empty support issues.
        pred_mask = temp_pred_np[b] != 0
        true_mask = temp_true_np[b] != 0

        a = temp_pred_np[b][pred_mask]
        b_vals = temp_true_np[b][true_mask]

        # If either distribution is empty, EMD is undefined -> set to inf.
        if len(a) == 0 or len(b_vals) == 0:
            emd_values.append(np.inf)
            continue

        pred_locs = source_locations[pred_mask]
        true_locs = source_locations[true_mask]

        # Ground cost matrix: Euclidean distance between source coordinates.
        M = cdist(pred_locs, true_locs, metric='euclidean')

        # Normalize masses to sum to 1 (discrete probability distributions).
        a_norm = a / (a.sum() + eps_np)
        b_norm = b_vals / (b_vals.sum() + eps_np)

        # emd2 returns the transportation cost (scalar).
        emd_val = ot.emd2(a_norm, b_norm, M)
        emd_values.append(emd_val)

    return np.array(emd_values)


# =============================================================================
# Loss module definition (optional / standalone)
# =============================================================================
class SparsityLossModule(nn.Module):
    """
    A simple sparsity-promoting loss module.

    This module computes mean absolute value over all entries.
    It can be used as an L1-like regularization term for source estimates.

    Input:
        y_pred: (B, N, L) or any tensor shape

    Output:
        scalar tensor: mean(|y_pred|)
    """
    def forward(self, y_pred):
        return torch.mean(torch.abs(y_pred))


# =============================================================================
# Network building blocks: E-step and M-step layers
# =============================================================================
class MuEstimateLayer(nn.Module):
    """
    E-step layer:
    Given gamma, compute posterior mean mu and posterior variance diagonal Sigma_X_diag.

    In SBL/EM formulation, we have:
        C = L * Gamma * L^T + Lambda
    where Gamma = diag(gamma) is source covariance (diagonal here),
    Lambda is noise covariance/precision related matrix (given by dataset).

    This layer computes:
        C^{-1} L
        A = (C^{-1} L)^T y
        mu = Gamma * A
        Sigma_X_diag = diag( Gamma - Gamma * L^T C^{-1} L * Gamma )
                     = gamma - gamma^2 * trace_term

    Notes:
    - This implementation treats Gamma as diagonal with entries gamma.
    - Uses Cholesky solve primarily for numerical stability and speed.
    - If Cholesky fails, adds jitter; if still fails, falls back to SVD-based inversion.

    Args (constructor):
        InpSize (int): typically M (number of sensors) (kept for interface compatibility).
        OutSize (int): typically N (number of sources) (kept for interface compatibility).
        L (int): number of time points.
        reg_lambda (float): reserved; can be used for additional regularization if needed.

    Forward Inputs:
        y (torch.Tensor):           (B, M, L) observed EEG
        gamma (torch.Tensor):       (B, N) current diagonal Gamma entries
        Lambda (torch.Tensor):      (B, M, M) noise covariance/precision matrix (depends on your setup)
        Lmat (torch.Tensor):        (B, M, N) leadfield/forward matrix

    Returns:
        mu (torch.Tensor):          (B, N, L) posterior mean of sources
        Sigma_X_diag (torch.Tensor):(B, N) diagonal entries of posterior covariance
        A (torch.Tensor):           (B, N, L) equals (C^{-1} L)^T y
        trace_term (torch.Tensor):  (B, N) equals sum over sensors of (L * C^{-1} L), per source
    """
    def __init__(self, InpSize, OutSize, L, reg_lambda=1e-6):
        super(MuEstimateLayer, self).__init__()
        self.InpSize = InpSize
        self.OutSize = OutSize
        self.L = L
        self.reg_lambda = reg_lambda

    def forward(self, y, gamma, Lambda, Lmat):
        """
        See class docstring for full description.

        Shapes:
            y:      (B, M, L)
            gamma:  (B, N)
            Lambda: (B, M, M)
            Lmat:   (B, M, N)
        """
        B, M, L = y.shape
        _, _, N = Lmat.shape
        eps = torch.finfo(gamma.dtype).eps

        # Expand gamma to broadcast along sensor dimension and form L * Gamma
        # gamma_expanded: (B, 1, N)
        gamma_expanded = gamma.unsqueeze(1)
        # L_Gamma: (B, M, N)
        L_Gamma = Lmat * gamma_expanded

        # Compute C = L Gamma L^T + Lambda, shape (B, M, M)
        CM = torch.bmm(L_Gamma, Lmat.transpose(1, 2))
        CM = CM + Lambda

        # Symmetrize to reduce numerical asymmetry from finite precision.
        CM = (CM + CM.transpose(1, 2)) / 2

        # Solve C^{-1} L efficiently:
        # Prefer Cholesky. If it fails, add a tiny jitter and retry;
        # if still fails, fall back to SVD inversion.
        try:
            L_chol = torch.linalg.cholesky(CM)
            CMinvG = torch.cholesky_solve(Lmat, L_chol)  # (B, M, N)
        except RuntimeError:
            jitter = 1e-10
            eye = torch.eye(CM.size(-1), device=CM.device).unsqueeze(0)
            CM_reg = CM + jitter * eye
            try:
                L_chol = torch.linalg.cholesky(CM_reg)
                CMinvG = torch.cholesky_solve(Lmat, L_chol)
            except RuntimeError:
                # Fallback: SVD-based inverse (slower, but more robust).
                U, S, Vh = torch.linalg.svd(CM, full_matrices=False)
                S_inv = 1.0 / (S.unsqueeze(1) + eps)
                CMinv = torch.bmm(U * S_inv, U.transpose(1, 2))
                CMinvG = torch.bmm(CMinv, Lmat)

        # A = (C^{-1} L)^T y, shape (B, N, L)
        A = torch.bmm(CMinvG.transpose(1, 2), y)

        # Posterior mean: mu = Gamma * A, where Gamma is diagonal => elementwise
        mu = gamma.unsqueeze(2) * A  # (B, N, L)

        # trace_term per source: sum_m L(m,n) * (C^{-1}L)(m,n)
        # shape (B, N)
        trace_term = torch.sum(Lmat * CMinvG, dim=1)

        # Posterior covariance diagonal:
        # Sigma_X_diag = gamma - gamma^2 * trace_term
        Sigma_X_diag = gamma - gamma * gamma * trace_term
        Sigma_X_diag = torch.clamp(Sigma_X_diag, min=eps)

        return mu, Sigma_X_diag, A, trace_term


class DiagonalGammaUpdate(nn.Module):
    """
    M-step layer:
    Update gamma using a weighted combination of two EM-derived terms.
    The weights are learned and constrained to be non-negative via Softplus.

    term1 = gamma^2 * mean_t( A^2 )
    term2 = gamma * (1 - gamma * trace_term)

    gamma_new = w1 * term1 + w2 * term2

    where w1, w2 are per-source learned weights (size N each), enforced >= 0.

    Constructor Args:
        OutSize (int): N, number of sources.

    Forward Inputs:
        gamma (torch.Tensor):      (B, N) current gamma
        A (torch.Tensor):          (B, N, L) (C^{-1}L)^T y
        trace_term (torch.Tensor): (B, N) sum(L * C^{-1}L) across sensors

    Returns:
        gamma_new (torch.Tensor):  (B, N) updated gamma
    """
    def __init__(self, OutSize):
        super().__init__()
        self.OutSize = OutSize

        # Initialize softplus^{-1}(1) so that softplus(init_val) ≈ 1
        init_val = np.log(np.e - 1)
        self.weight_params = nn.Parameter(torch.ones(2, OutSize) * init_val)

    def forward(self, gamma, A, trace_term):
        eps = torch.finfo(gamma.dtype).eps

        # mean over time dimension: (B, N)
        A_squared_mean = torch.mean(A * A, dim=2)

        # Two EM terms
        term1 = gamma * gamma * A_squared_mean
        term2 = gamma * (1 - gamma * trace_term)

        # Non-negative weights (N,)
        w1_pos = F.softplus(self.weight_params[0])
        w2_pos = F.softplus(self.weight_params[1])

        # Weighted update (broadcast over batch)
        gamma_new = w1_pos * term1 + w2_pos * term2

        # Diagnostics: detect numerical instability
        if torch.isnan(gamma_new).any() or torch.isinf(gamma_new).any():
            print("\n" + "!" * 80)
            print("⚠️  WARNING: NaN or Inf detected in gamma update!")
            print("!" * 80)
            print(f"w1_raw stats: min={self.weight_params[0].min().item():.6e}, max={self.weight_params[0].max().item():.6e}, mean={self.weight_params[0].mean().item():.6e}")
            print(f"w2_raw stats: min={self.weight_params[1].min().item():.6e}, max={self.weight_params[1].max().item():.6e}, mean={self.weight_params[1].mean().item():.6e}")
            print(f"w1_pos stats: min={w1_pos.min().item():.6e}, max={w1_pos.max().item():.6e}, mean={w1_pos.mean().item():.6e}")
            print(f"w2_pos stats: min={w2_pos.min().item():.6e}, max={w2_pos.max().item():.6e}, mean={w2_pos.mean().item():.6e}")
            print(f"term1 stats: min={term1.min().item():.6e}, max={term1.max().item():.6e}")
            print(f"term2 stats: min={term2.min().item():.6e}, max={term2.max().item():.6e}")
            print(f"gamma_new stats: min={gamma_new.min().item():.6e}, max={gamma_new.max().item():.6e}")
            print(f"NaN count in gamma_new: {torch.isnan(gamma_new).sum().item()} / {gamma_new.numel()}")
            print(f"Inf count in gamma_new: {torch.isinf(gamma_new).sum().item()} / {gamma_new.numel()}")
            print("!" * 80 + "\n")

        return gamma_new


class SBLLayer(nn.Module):
    """
    One SBL iteration layer (one EM iteration):
    - E-step: compute mu and Sigma_X_diag given gamma
    - M-step: update gamma given (A, trace_term)

    Constructor Args:
        OutSize (int): N, number of sources
        InpSize (int): M, number of sensors
        L (int): number of time points
        is_trainable (bool): kept for compatibility (e.g., freeze/unfreeze in training)

    Forward Inputs:
        y:      (B, M, L)
        gamma:  (B, N)
        Lambda: (B, M, M)
        Lmat:   (B, M, N)

    Returns:
        mu:            (B, N, L)
        Sigma_X_diag:  (B, N)
        gamma_new:     (B, N)
    """
    def __init__(self, OutSize, InpSize, L, is_trainable=True):
        super(SBLLayer, self).__init__()
        self.OutSize = OutSize
        self.InpSize = InpSize
        self.L = L
        self.is_trainable = is_trainable

        self.mu_estimate = MuEstimateLayer(InpSize, OutSize, L)
        self.gamma_update = DiagonalGammaUpdate(OutSize)

    def forward(self, y, gamma, Lambda, Lmat):
        """Run one full EM/SBL iteration."""
        mu, Sigma_X_diag, A, trace_term = self.mu_estimate(y, gamma, Lambda, Lmat)
        gamma_new = self.gamma_update(gamma, A, trace_term)
        return mu, Sigma_X_diag, gamma_new


class LSBLNetwork(nn.Module):
    """
    Learned SBL (L-SBL) network:
    Stack multiple SBLLayer blocks to perform T EM/SBL iterations.

    Constructor Args:
        InpSize (int): M, number of sensors
        OutSize (int): N, number of sources
        L (int): number of time points
        T (int): number of stacked layers / iterations
        is_trainable (bool): kept for compatibility; can be used to control training behavior

    Forward Inputs:
        y:          (B, M, L) observed EEG
        gamma_0:    (B, N) initial gamma (often initialized to ones)
        Lambda:     (B, M, M) noise matrix
        Lmat:       (B, M, N) leadfield
        num_layers: int or None, number of layers to run (default: T)

    Returns:
        mu_final:            (B, N, L) posterior mean after final iteration
        Sigma_X_diag_final:  (B, N) posterior variance diagonal after final iteration
    """
    def __init__(self, InpSize, OutSize, L, T, is_trainable=True):
        super().__init__()
        self.InpSize = InpSize
        self.OutSize = OutSize
        self.L = L
        self.T = T
        self.is_trainable = is_trainable

        self.sbl_layers = nn.ModuleList([
            SBLLayer(OutSize, InpSize, L, is_trainable) for _ in range(T)
        ])

    def forward(self, y, gamma_0, Lambda, Lmat, num_layers=None):
        """
        Forward pass: run multiple EM/SBL iterations.

        Note:
        - This code runs `num_layers` updates of gamma, then performs one final E-step
          to output mu and Sigma_X_diag corresponding to the final gamma.
        """
        if num_layers is None:
            num_layers = self.T

        gamma = gamma_0

        # Iteratively update gamma via stacked layers
        for i in range(num_layers):
            mu, Sigma_X_diag, gamma = self.sbl_layers[i](y, gamma, Lambda, Lmat)

        # One final E-step to obtain output mu/Sigma under the final gamma
        if num_layers > 0:
            mu_final, Sigma_X_diag_final, _, _ = self.sbl_layers[num_layers - 1].mu_estimate(
                y, gamma, Lambda, Lmat
            )
        else:
            mu_final, Sigma_X_diag_final, _, _ = self.sbl_layers[0].mu_estimate(
                y, gamma_0, Lambda, Lmat
            )

        return mu_final, Sigma_X_diag_final


# =============================================================================
# Dataset loading
# =============================================================================
class NormalizedEEGDataset:
    """
    Load a normalized EEG training dataset from disk.

    Expected files:
        metadata.npz                      : contains M, N, L, batch_size, n_train_batches, etc.
        global_G_norm_train.npy           : normalized leadfield matrix (M, N)
        global_G_norm_const_train.npy     : scalar normalization constant (may be saved as array([value]))

        train_X_norm_XXXX.npy             : normalized ground-truth sources (B, N, L)
        train_Y_norm_XXXX.npy             : normalized sensor observations (B, M, L)
        train_Lambda_norm_XXXX.npy        : normalized noise matrix (B, M, M)
        train_M_norm_XXXX.npy             : per-sample normalization factor (B,) or compatible shape

    Notes:
    - This dataset class provides a simple `load_batch(idx)` interface.
    - It does not implement PyTorch Dataset/Loader; it is intended for file-based batch loading.
    """
    def __init__(self, data_folder):
        self.data_folder = data_folder

        metadata_path = os.path.join(data_folder, 'metadata.npz')
        md = np.load(metadata_path)

        self.M = int(md["n_sensors"])
        self.N = int(md["n_sources"])
        self.L = int(md["n_time"])
        self.batch_size = int(md["batch_size"])
        self.n_batches = int(md["n_train_batches"])

        # This class is for the training set; prefix fixed to 'train'
        self.prefix = 'train'

        # Training uses train-specific normalized leadfield and constant
        g_name = "global_G_norm_train.npy"
        g_const_name = "global_G_norm_const_train.npy"

        self.G_norm = np.load(os.path.join(data_folder, g_name))
        g_const_array = np.load(os.path.join(data_folder, g_const_name))

        # global_G_norm_const might be stored as array([value]) -> extract scalar carefully
        self.G_norm_const = g_const_array.item() if g_const_array.size == 1 else g_const_array[0].item()

    def load_batch(self, idx):
        """
        Load one batch by its index.

        Args:
            idx (int): batch index (0 ... n_batches-1)

        Returns:
            X_norm (np.ndarray):      (B, N, L) normalized ground-truth sources
            Y_norm (np.ndarray):      (B, M, L) normalized observations
            Lambda_norm (np.ndarray): (B, M, M) normalized noise matrix
            M_norm (np.ndarray):      (B,) or compatible, used for de-normalization factor
        """
        base = os.path.join(self.data_folder, f"{self.prefix}")
        X_norm = np.load(f"{base}_X_norm_{idx:04d}.npy")
        Y_norm = np.load(f"{base}_Y_norm_{idx:04d}.npy")
        Lambda_norm = np.load(f"{base}_Lambda_norm_{idx:04d}.npy")
        M_norm = np.load(f"{base}_M_norm_{idx:04d}.npy")
        return X_norm, Y_norm, Lambda_norm, M_norm


def evaluate_dataset(model, dataset, device, n_iterations,
                     source_locations, activation_threshold=1e-8):
    """
    Evaluate a dataset and compute per-sample metrics, plus overall mean/std.

    Workflow per batch:
    1) Load normalized data (X_norm, Y_norm, Lambda_norm, M_norm)
    2) Build leadfield batch from dataset.G_norm (broadcast to B)
    3) Initialize Gamma (gamma_0) as ones
    4) Run model inference with `n_iterations` layers (EM iterations)
    5) De-normalize mu and X using `denorm_factor`
    6) Apply activation threshold to mu (hard thresholding)
    7) Compute metrics per sample

    Metrics:
        - MSE: mean((mu - X)^2)
        - MAE: mean(|mu - X|)
        - RMSE: relative L2 squared error: ||mu-X||_2^2 / (||X||_2^2 + eps)
        - RMAE: relative L1 error: ||mu-X||_1 / (||X||_1 + eps)
        - EMD: earth mover's distance on per-source L2-mass distributions
        - Sparsity: mean(|mu|)
        - N_active: number of active sources (RMS over time > threshold)
        - sigma_diag: sqrt(mean(Sigma_X_diag)) per sample

    Args:
        model (LSBLNetwork): trained network
        dataset (NormalizedEEGDataset): dataset loader
        device (torch.device): cpu/cuda
        n_iterations (int): number of layers/iterations to run
        source_locations (np.ndarray): (N, D) coordinates for sources used in EMD
        activation_threshold (float): threshold for considering activity non-zero

    Returns:
        dict: metrics dictionary with structure:
              metrics[name] = {'mean': float, 'std': float, 'values': list}
    """
    model.eval()

    # Preload G_norm onto GPU once (M, N)
    G_norm_base = torch.from_numpy(dataset.G_norm).float().to(device)
    G_norm_const_torch = torch.tensor(dataset.G_norm_const).float().to(device)  # scalar

    # Collect all per-sample metric values across the whole dataset
    all_mse, all_mae, all_rmse, all_rmae = [], [], [], []
    all_emd, all_sparsity, all_n_active, all_sigma = [], [], [], []

    with torch.no_grad():
        for batch_idx in tqdm(range(dataset.n_batches), desc="  Evaluating batches"):
            X_norm, Y_norm, Lambda_norm, M_norm = dataset.load_batch(batch_idx)

            B = X_norm.shape[0]
            X_norm_torch = torch.from_numpy(X_norm).float().to(device)
            Y_norm_torch = torch.from_numpy(Y_norm).float().to(device)
            Lambda_norm_torch = torch.from_numpy(Lambda_norm).float().to(device)
            M_norm_torch = torch.from_numpy(M_norm).float().to(device)

            # Broadcast leadfield to batch: (B, M, N)
            G_norm_torch = G_norm_base.unsqueeze(0).expand(B, -1, -1)

            # Initial gamma (Gamma diagonal entries)
            Gamma = torch.ones(B, dataset.N, device=device)

            # De-normalization factor:
            # denorm_factor: (B, 1, 1) broadcast to (B, N, L)
            denorm_factor = (M_norm_torch.sqrt().view(B, 1, 1) / G_norm_const_torch.view(1, 1, 1))
            X_denorm = X_norm_torch * denorm_factor

            # Forward inference
            mu_final, Sigma_X_diag_final = model(
                Y_norm_torch, Gamma, Lambda_norm_torch, G_norm_torch,
                num_layers=n_iterations
            )

            # De-normalize and apply thresholding (hard mask)
            mu_denorm = mu_final * denorm_factor
            mu_denorm_thresholded = torch.where(
                torch.abs(mu_denorm) >= activation_threshold,
                mu_denorm,
                torch.zeros_like(mu_denorm)
            )

            eps = torch.finfo(mu_denorm.dtype).eps

            # ----------------------------
            # Per-sample metrics
            # ----------------------------
            mse_per_sample = torch.mean((mu_denorm_thresholded - X_denorm) ** 2, dim=(1, 2)).cpu().numpy()
            mae_per_sample = torch.mean(torch.abs(mu_denorm_thresholded - X_denorm), dim=(1, 2)).cpu().numpy()

            # Relative L2 squared error (named RMSE here in your code)
            diff_l2_squared = torch.sum((mu_denorm_thresholded - X_denorm) ** 2, dim=(1, 2))
            true_l2_squared = torch.sum(X_denorm ** 2, dim=(1, 2))
            rmse_per_sample = (diff_l2_squared / (true_l2_squared + eps)).cpu().numpy()

            # Relative L1 error
            diff_l1 = torch.sum(torch.abs(mu_denorm_thresholded - X_denorm), dim=(1, 2))
            true_l1 = torch.sum(torch.abs(X_denorm), dim=(1, 2))
            rmae_per_sample = (diff_l1 / (true_l1 + eps)).cpu().numpy()

            # EMD (uses numpy + POT)
            emd_per_sample = compute_emd_batch(mu_denorm_thresholded, X_denorm, source_locations)

            # Sparsity proxy: mean absolute value
            sparsity_per_sample = torch.mean(torch.abs(mu_denorm_thresholded), dim=(1, 2)).cpu().numpy()

            # Active sources count:
            # RMS over time per source -> active if > threshold
            rms_per_source = torch.sqrt(torch.mean(mu_denorm_thresholded ** 2, dim=2))  # (B, N)
            active_sources = (rms_per_source > activation_threshold).float()
            n_active_per_sample = torch.sum(active_sources, dim=1).cpu().numpy()

            # Sigma diagonal summary: sqrt(mean diag elements))
            sigma_per_sample = torch.sqrt(torch.mean(Sigma_X_diag_final, dim=1)).cpu().numpy()

            # Append to global lists
            all_mse.extend(mse_per_sample)
            all_mae.extend(mae_per_sample)
            all_rmse.extend(rmse_per_sample)
            all_rmae.extend(rmae_per_sample)
            all_emd.extend(emd_per_sample)
            all_sparsity.extend(sparsity_per_sample)
            all_n_active.extend(n_active_per_sample)
            all_sigma.extend(sigma_per_sample)

    metrics = {
        'mse': {'mean': np.mean(all_mse), 'std': np.std(all_mse), 'values': all_mse},
        'mae': {'mean': np.mean(all_mae), 'std': np.std(all_mae), 'values': all_mae},
        'rmse': {'mean': np.mean(all_rmse), 'std': np.std(all_rmse), 'values': all_rmse},
        'rmae': {'mean': np.mean(all_rmae), 'std': np.std(all_rmae), 'values': all_rmae},
        'emd': {'mean': np.mean(all_emd), 'std': np.std(all_emd), 'values': all_emd},
        'sparsity': {'mean': np.mean(all_sparsity), 'std': np.std(all_sparsity), 'values': all_sparsity},
        'n_active': {'mean': np.mean(all_n_active), 'std': np.std(all_n_active), 'values': all_n_active},
        'sigma_diag': {'mean': np.mean(all_sigma), 'std': np.std(all_sigma), 'values': all_sigma},
    }

    return metrics


def load_checkpoint_and_infer_layers(checkpoint_path, M, N, L, device):
    """
    Load a saved checkpoint and infer how many SBL layers (iterations) the model has.

    The number of layers is inferred by scanning the state_dict keys like:
        'sbl_layers.<idx>....'

    Args:
        checkpoint_path (str): path to .pth checkpoint
        M (int): number of sensors
        N (int): number of sources
        L (int): number of time points
        device (torch.device): cpu/cuda

    Returns:
        model (LSBLNetwork): instantiated model with inferred number of layers loaded
        num_layers (int): inferred layer count
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    max_layer_idx = -1
    for key in checkpoint['model_state_dict'].keys():
        if key.startswith('sbl_layers.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                max_layer_idx = max(max_layer_idx, layer_idx)

    num_layers = max_layer_idx + 1
    print(f"  Inferred number of layers: {num_layers}")

    # Build model with trainable weights (even for evaluation, this is fine)
    model = LSBLNetwork(M, N, L, num_layers, is_trainable=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("  Model weights loaded successfully!")

    return model, num_layers


def evaluate_neural_em_train(data_folder, checkpoint_path, activation_threshold=1e-8):
    """
    Evaluate EM neural network on the TRAINING dataset.

    This function:
    - Reads metadata
    - Loads source locations used for EMD
    - Loads model checkpoint and infers its iteration depth
    - Evaluates the training dataset
    - Prints formatted results
    - Saves everything into ./output/evaluation_results_EM_neural_train.npz

    Args:
        data_folder (str): folder containing normalized dataset files
        checkpoint_path (str): path to trained model checkpoint (.pth)
        activation_threshold (float): threshold for activity masking

    Returns:
        dict: a dictionary containing configuration and computed metrics
    """
    print("=" * 80)
    print("Evaluate EM Neural Network - Training Dataset - 8 core metrics")
    print("=" * 80)

    metadata = np.load(os.path.join(data_folder, "metadata.npz"))
    M = int(metadata['n_sensors'])
    N = int(metadata['n_sources'])
    L = int(metadata['n_time'])

    print(f"\nData dimensions: M={M}, N={N}, L={L}")
    print(f"Activation threshold: {activation_threshold}")

    # Load source locations for training set (used by EMD).
    train_locs_path = "./data/fsaverage/-ico3surfFixedSourceLocstrain.npy"
    train_locs = np.load(train_locs_path)

    # Load model
    print("\n" + "=" * 80)
    print("Loading EM Neural Network")
    print("=" * 80)

    model, num_layers = load_checkpoint_and_infer_layers(checkpoint_path, M, N, L, device)

    # Evaluate training set
    print("\n" + "=" * 80)
    print("Evaluating training dataset")
    print("=" * 80)

    train_dataset = NormalizedEEGDataset(data_folder)
    train_metrics = evaluate_dataset(
        model, train_dataset, device, num_layers,
        train_locs, activation_threshold
    )

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation results (Mean ± Std across samples):")
    print("=" * 80)

    print("\nTraining set:")
    print("-" * 160)
    print(f"{'Metric':<20} {'MSE':<20} {'MAE':<20} {'RMSE(L2norm)':<20} {'RMAE(L1norm)':<20} "
          f"{'EMD':<20} {'Sparsity':<20} {'N_Active':<15} {'Sigma':<20}")
    print("-" * 160)

    v = train_metrics
    print(f"{'EM Neural Net':<20} "
          f"{v['mse']['mean']:.2e}±{v['mse']['std']:.2e}  "
          f"{v['mae']['mean']:.2e}±{v['mae']['std']:.2e}  "
          f"{v['rmse']['mean']:.2e}±{v['rmse']['std']:.2e}  "
          f"{v['rmae']['mean']:.2e}±{v['rmae']['std']:.2e}  "
          f"{v['emd']['mean']:.2e}±{v['emd']['std']:.2e}  "
          f"{v['sparsity']['mean']:.2e}±{v['sparsity']['std']:.2e}  "
          f"{v['n_active']['mean']:.1f}±{v['n_active']['std']:.1f}  "
          f"{v['sigma_diag']['mean']:.2e}±{v['sigma_diag']['std']:.2e}")

    print("\n" + "=" * 80)
    print("Note: Mean and Std are computed across all training samples.")
    print("=" * 80)

    # Save results
    os.makedirs('./output', exist_ok=True)
    save_results = {
        'method': 'EM neural network',
        'dataset': 'training set',
        'n_iterations': num_layers,
        'checkpoint': checkpoint_path,
        'activation_threshold': activation_threshold,
        'train': train_metrics,
    }

    output_file = './output/evaluation_results_EM_neural_train.npz'
    np.savez(output_file, **save_results)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)

    return save_results


if __name__ == "__main__":
    # ==================== Configuration ====================

    data_folder = './normalized_data'   # Folder containing normalized dataset files
    checkpoint_path = "EM_model.pth"    # Path to the EM neural network checkpoint
    activation_threshold = 1e-6         # Threshold for active source masking

    # ==================== Run evaluation ====================
    print("=" * 80)
    print("Evaluation configuration:")
    print(f"  Data folder: {data_folder}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Threshold:   {activation_threshold}")
    print("=" * 80)
    print()

    results = evaluate_neural_em_train(
        data_folder=data_folder,
        checkpoint_path=checkpoint_path,
        activation_threshold=activation_threshold
    )