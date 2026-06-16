#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ruiting Xiao
"""
Evaluate the Combined Model (Mode 4) on the TRAINING dataset.

This script is suitable for running in a notebook or as a standalone script.
All paths and hyperparameters are configured directly in this file.

Main purpose:
- Load a trained Combined Model checkpoint (Mode 4 gamma update).
- Run inference for a certain number of EM/SBL iterations (layers).
- Compute 8 core metrics on the training set.
- Save aggregated statistics and per-sample values to an .npz file.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import ot
from scipy.spatial.distance import cdist

# Use GPU if available, otherwise fall back to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================
# EMD computation
# ================================
def compute_emd_batch(y_pred, y_true, source_locations):
    """
    Compute EMD (Earth Mover's Distance) per sample for a batch.

    This is the "raw" EMD version:
    - For each sample, each source time-series is converted into a nonnegative "mass"
      using the L2 norm across time: mass_i = ||x_i(t)||_2.
    - Mass vectors are normalized to sum to 1 (discrete probability distributions).
    - The ground cost is Euclidean distance between source coordinates.
    - The final EMD value is the optimal transport cost returned by `ot.emd2`.

    Args:
        y_pred (torch.Tensor): Predicted sources, shape (B, N, L).
        y_true (torch.Tensor): Ground-truth sources, shape (B, N, L).
        source_locations (np.ndarray or array-like):
            Coordinates of all N sources, shape (N, D), typically D=3.

    Returns:
        np.ndarray:
            EMD values of length B (one value per sample).
            If the predicted or true distribution has no nonzero support for a sample,
            the returned value for that sample is np.inf.
    """
    B, N, L = y_pred.shape

    # Convert time-series to per-source magnitudes (masses): (B, N)
    temp_pred = torch.norm(y_pred, p=2, dim=2)  # (B, N)
    temp_true = torch.norm(y_true, p=2, dim=2)  # (B, N)

    # Convert to numpy for POT/scipy functions
    temp_pred_np = temp_pred.detach().cpu().numpy().astype(np.float64)
    temp_true_np = temp_true.detach().cpu().numpy().astype(np.float64)

    emd_values = []
    eps_np = np.finfo(np.float64).eps

    for b in range(B):
        # Keep only non-zero mass sources to avoid empty supports
        pred_mask = temp_pred_np[b] != 0
        true_mask = temp_true_np[b] != 0

        a = temp_pred_np[b][pred_mask]
        b_vals = temp_true_np[b][true_mask]

        # If either side has no support, EMD is undefined -> set to infinity
        if len(a) == 0 or len(b_vals) == 0:
            emd_values.append(np.inf)
            continue

        pred_locs = source_locations[pred_mask]
        true_locs = source_locations[true_mask]

        # Ground cost matrix: Euclidean distances between active predicted/true sources
        M = cdist(pred_locs, true_locs, metric='euclidean')

        # Normalize masses to sum to 1
        a_norm = a / (a.sum() + eps_np)
        b_norm = b_vals / (b_vals.sum() + eps_np)

        # Optimal transport cost
        emd_val = ot.emd2(a_norm, b_norm, M)
        emd_values.append(emd_val)

    return np.array(emd_values)


# ================================
# Neural network layers
# ================================
class MuEstimateLayer(nn.Module):
    """
    E-step layer (posterior update):

    Given:
        - observations y
        - current diagonal gamma (source hyperparameters)
        - noise matrix Lambda
        - leadfield matrix Lmat

    Compute:
        C = Lmat * diag(gamma) * Lmat^T + Lambda
        C^{-1}Lmat  (via Cholesky solve when possible)

        A = (C^{-1}Lmat)^T y
        mu = diag(gamma) A
        Sigma_X_diag = gamma - gamma^2 * trace_term
        where trace_term = sum_m Lmat(m,n) * (C^{-1}Lmat)(m,n)

    Notes:
    - Uses Cholesky decomposition for numerical stability.
    - If Cholesky fails, adds a small jitter and retries.
    - If it still fails, falls back to SVD-based inversion.

    Args (constructor):
        InpSize (int): Typically M (number of sensors). Kept for interface compatibility.
        OutSize (int): Typically N (number of sources). Kept for interface compatibility.
        L (int): Number of time points.
        reg_lambda (float): Reserved for potential extra regularization (not used in computation).

    Forward Args:
        y (torch.Tensor):      (B, M, L) observed EEG
        gamma (torch.Tensor):  (B, N) current gamma (diagonal of Gamma)
        Lambda (torch.Tensor): (B, M, M) noise matrix
        Lmat (torch.Tensor):   (B, M, N) leadfield matrix

    Returns:
        mu (torch.Tensor):            (B, N, L) posterior mean of sources
        Sigma_X_diag (torch.Tensor):  (B, N) diagonal entries of posterior covariance
        A (torch.Tensor):             (B, N, L) equals (C^{-1}Lmat)^T y
        trace_term (torch.Tensor):    (B, N) equals sum(Lmat * (C^{-1}Lmat), dim=1)
    """
    def __init__(self, InpSize, OutSize, L, reg_lambda=1e-6):
        super(MuEstimateLayer, self).__init__()
        self.InpSize = InpSize
        self.OutSize = OutSize
        self.L = L
        self.reg_lambda = reg_lambda

    def forward(self, y, gamma, Lambda, Lmat):
        """
        Run one E-step computation.

        Shapes:
            y:      (B, M, L)
            gamma:  (B, N)
            Lambda: (B, M, M)
            Lmat:   (B, M, N)
        """
        B, M, L = y.shape
        _, _, N = Lmat.shape
        eps = torch.finfo(gamma.dtype).eps

        # Form Lmat * Gamma (Gamma is diagonal => elementwise scaling on source dimension)
        gamma_expanded = gamma.unsqueeze(1)  # (B, 1, N)
        L_Gamma = Lmat * gamma_expanded      # (B, M, N)

        # C = L Gamma L^T + Lambda
        CM = torch.bmm(L_Gamma, Lmat.transpose(1, 2))  # (B, M, M)
        CM = CM + Lambda
        CM = (CM + CM.transpose(1, 2)) / 2  # enforce symmetry numerically

        # Solve C^{-1} Lmat
        try:
            L_chol = torch.linalg.cholesky(CM)
            CMinvG = torch.cholesky_solve(Lmat, L_chol)  # (B, M, N)
        except RuntimeError:
            # If Cholesky fails, add jitter and retry
            jitter = 1e-10
            eye = torch.eye(CM.size(-1), device=CM.device).unsqueeze(0)
            CM_reg = CM + jitter * eye
            try:
                L_chol = torch.linalg.cholesky(CM_reg)
                CMinvG = torch.cholesky_solve(Lmat, L_chol)
            except RuntimeError:
                # Fallback: SVD-based inverse (more robust but slower)
                U, S, Vh = torch.linalg.svd(CM, full_matrices=False)
                S_inv = 1.0 / (S.unsqueeze(1) + eps)
                CMinv = torch.bmm(U * S_inv, U.transpose(1, 2))
                CMinvG = torch.bmm(CMinv, Lmat)

        # A = (C^{-1}Lmat)^T y
        A = torch.bmm(CMinvG.transpose(1, 2), y)  # (B, N, L)

        # Posterior mean mu = Gamma * A (Gamma diagonal => elementwise scaling)
        mu = gamma.unsqueeze(2) * A  # (B, N, L)

        # trace_term = sum_m Lmat(m,n) * (C^{-1}Lmat)(m,n)
        trace_term = torch.sum(Lmat * CMinvG, dim=1)  # (B, N)

        # Posterior covariance diagonal: Sigma_X_diag = gamma - gamma^2 * trace_term
        Sigma_X_diag = gamma - gamma * gamma * trace_term
        Sigma_X_diag = torch.clamp(Sigma_X_diag, min=eps)

        return mu, Sigma_X_diag, A, trace_term


class DiagonalGammaUpdateMode4(nn.Module):
    """
    Combined Model (Mode 4): weighted combination of three gamma update rules.

    The model computes three candidate updates and combines them using learned weights:
        - Mode 1: MacKay update
        - Mode 2: Modified MacKay update
        - Mode 3: EM update

    Weights are learned per source (shape (3, N)) and normalized across the 3 modes
    for each source using absolute value + sum normalization.

    Args (constructor):
        OutSize (int): N, number of sources.

    Forward Args:
        gamma (torch.Tensor):      (B, N) current gamma
        A (torch.Tensor):          (B, N, L) equals (C^{-1}Lmat)^T y
        trace_term (torch.Tensor): (B, N) equals sum(Lmat * (C^{-1}Lmat), dim=1)

    Returns:
        gamma_new (torch.Tensor):  (B, N) updated gamma after weighted combination
    """
    def __init__(self, OutSize):
        super().__init__()
        self.OutSize = OutSize

        # Initialize weights equally (before normalization)
        self.weight_params = nn.Parameter(torch.ones(3, OutSize) / 3)

    def get_weights(self):
        """
        Convert raw weight parameters into normalized nonnegative weights per source.

        Steps:
        - Take absolute value to ensure nonnegativity.
        - Normalize across the 3 modes for each source (sum to 1 over dim=0).

        Returns:
            torch.Tensor: weights of shape (3, N), where weights[:, n] sums to 1.
        """
        eps = torch.finfo(self.weight_params.dtype).eps
        abs_weights = torch.abs(self.weight_params)
        return abs_weights / (abs_weights.sum(dim=0, keepdim=True) + eps)

    def forward(self, gamma, A, trace_term):
        """
        Perform Mode-4 gamma update.

        Args:
            gamma: (B, N)
            A: (B, N, L)
            trace_term: (B, N)

        Returns:
            gamma_new: (B, N)
        """
        eps = torch.finfo(gamma.dtype).eps
        A_squared_mean = torch.mean(A * A, dim=2)  # (B, N)

        # ----------------------------
        # Mode 1: MacKay update
        # gamma = gamma^2 * E[A^2] / (gamma * trace_term)
        # ----------------------------
        numer1 = gamma * gamma * A_squared_mean
        denom1 = gamma * trace_term
        gamma_mode1 = numer1 / (denom1 + eps)

        # ----------------------------
        # Mode 2: Modified MacKay update
        # gamma = gamma * sqrt(E[A^2]) / sqrt(trace_term)
        # ----------------------------
        numer2 = gamma * torch.sqrt(A_squared_mean + eps)
        denom2 = torch.sqrt(trace_term + eps)
        gamma_mode2 = numer2 / (denom2 + eps)

        # ----------------------------
        # Mode 3: EM update
        # gamma = gamma^2 * E[A^2] + gamma * (1 - gamma * trace_term)
        # ----------------------------
        gamma_mode3 = gamma * gamma * A_squared_mean + gamma * (1 - gamma * trace_term)

        # Weighted combination (weights are per-source)
        weights = self.get_weights()
        w1, w2, w3 = weights[0], weights[1], weights[2]
        gamma_new = w1 * gamma_mode1 + w2 * gamma_mode2 + w3 * gamma_mode3

        return gamma_new


class SBLLayer(nn.Module):
    """
    One iteration layer of the (learned) SBL procedure for Mode 4.

    This layer contains:
        - E-step: MuEstimateLayer
        - M-step: DiagonalGammaUpdateMode4

    Args (constructor):
        OutSize (int): N, number of sources
        InpSize (int): M, number of sensors
        L (int): number of time points

    Forward Args:
        y:      (B, M, L) observed EEG
        gamma:  (B, N) current gamma
        Lambda: (B, M, M) noise matrix
        Lmat:   (B, M, N) leadfield

    Returns:
        mu:           (B, N, L) posterior mean
        Sigma_X_diag: (B, N) posterior covariance diagonal
        gamma_new:    (B, N) updated gamma
    """
    def __init__(self, OutSize, InpSize, L):
        super(SBLLayer, self).__init__()
        self.OutSize = OutSize
        self.InpSize = InpSize
        self.L = L

        self.mu_estimate = MuEstimateLayer(InpSize, OutSize, L)
        self.gamma_update = DiagonalGammaUpdateMode4(OutSize)

    def forward(self, y, gamma, Lambda, Lmat):
        """
        Run one complete SBL iteration (E-step + M-step).
        """
        mu, Sigma_X_diag, A, trace_term = self.mu_estimate(y, gamma, Lambda, Lmat)
        gamma_new = self.gamma_update(gamma, A, trace_term)
        return mu, Sigma_X_diag, gamma_new


class LSBLNetwork(nn.Module):
    """
    L-SBL network (Mode 4):
    Stack T SBLLayer blocks and run iterative inference.

    Args (constructor):
        InpSize (int): M, number of sensors
        OutSize (int): N, number of sources
        L (int): number of time points
        T (int): number of stacked layers / iterations

    Forward Args:
        y:          (B, M, L)
        gamma_0:    (B, N) initial gamma (commonly ones)
        Lambda:     (B, M, M)
        Lmat:       (B, M, N)
        num_layers: number of layers to execute (default uses all T)

    Returns:
        mu_final:            (B, N, L)
        Sigma_X_diag_final:  (B, N)
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

    def forward(self, y, gamma_0, Lambda, Lmat, num_layers=None):
        """
        Forward pass: run `num_layers` iterative updates of gamma.

        Note:
        - After finishing gamma updates, we run one final E-step to output mu/Sigma
          corresponding to the final gamma.
        """
        if num_layers is None:
            num_layers = self.T

        gamma = gamma_0

        # Iterative inference (gamma updates)
        for i in range(num_layers):
            mu, Sigma_X_diag, gamma = self.sbl_layers[i](y, gamma, Lambda, Lmat)

        # Final E-step for output
        if num_layers > 0:
            mu_final, Sigma_X_diag_final, _, _ = self.sbl_layers[num_layers - 1].mu_estimate(
                y, gamma, Lambda, Lmat
            )
        else:
            mu_final, Sigma_X_diag_final, _, _ = self.sbl_layers[0].mu_estimate(
                y, gamma_0, Lambda, Lmat
            )

        return mu_final, Sigma_X_diag_final


# ================================
# Dataset loading
# ================================
class NormalizedEEGDataset:
    """
    Load the normalized EEG TRAINING dataset from disk.

    Expected files in `data_folder`:
        metadata.npz:
            - n_sensors, n_sources, n_time, batch_size, n_train_batches, etc.
        global_G_norm_train.npy:
            - normalized leadfield matrix, shape (M, N)
        global_G_norm_const_train.npy:
            - scalar normalization constant (sometimes saved as array([value]))

        train_X_norm_XXXX.npy:
            - normalized ground-truth sources, shape (B, N, L)
        train_Y_norm_XXXX.npy:
            - normalized observations, shape (B, M, L)
        train_Lambda_norm_XXXX.npy:
            - normalized noise matrix, shape (B, M, M)
        train_M_norm_XXXX.npy:
            - per-sample scaling factor, shape (B,) or compatible

    This class is a lightweight batch loader (not a torch Dataset).
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

        # Training set prefix
        self.prefix = 'train'

        # Training set uses train-specific normalized leadfield and constant
        g_name = "global_G_norm_train.npy"
        g_const_name = "global_G_norm_const_train.npy"

        self.G_norm = np.load(os.path.join(data_folder, g_name))
        g_const_array = np.load(os.path.join(data_folder, g_const_name))
        self.G_norm_const = g_const_array.item() if g_const_array.size == 1 else g_const_array[0].item()

    def load_batch(self, idx):
        """
        Load one batch by index.

        Args:
            idx (int): batch index (0 ... n_batches-1)

        Returns:
            X_norm (np.ndarray):      (B, N, L) normalized ground-truth sources
            Y_norm (np.ndarray):      (B, M, L) normalized observations
            Lambda_norm (np.ndarray): (B, M, M) normalized noise matrix
            M_norm (np.ndarray):      (B,) or compatible, used for de-normalization
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
    Evaluate the training dataset and return aggregated metrics.

    For each batch:
        1) Load normalized X/Y/Lambda/M_norm
        2) Broadcast G_norm to batch
        3) Initialize gamma as ones
        4) Run forward inference for `n_iterations`
        5) De-normalize mu and X using denorm_factor
        6) Threshold mu for activity sparsification
        7) Compute per-sample metrics

    Metrics:
        - MSE
        - MAE
        - RMSE (here: relative L2 squared error, not the classic sqrt(MSE))
        - RMAE (relative L1 error)
        - EMD
        - Sparsity (mean absolute value of thresholded mu)
        - N_Active (count of active sources based on RMS over time)
        - Sigma (sqrt(mean of posterior covariance diagonal))

    Args:
        model (LSBLNetwork): trained network
        dataset (NormalizedEEGDataset): dataset loader
        device (torch.device): cpu/cuda
        n_iterations (int): number of iterations/layers to run
        source_locations (np.ndarray): (N, D) coordinates for EMD
        activation_threshold (float): threshold for considering activity non-zero

    Returns:
        dict: metrics with mean/std and per-sample values:
              metrics[name] = {'mean': float, 'std': float, 'values': list}
    """
    model.eval()

    # Preload leadfield to GPU once
    G_norm_base = torch.from_numpy(dataset.G_norm).float().to(device)
    G_norm_const_torch = torch.tensor(dataset.G_norm_const).float().to(device)

    # Collect per-sample metrics across the whole dataset
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

            # Broadcast G_norm to (B, M, N)
            G_norm_torch = G_norm_base.unsqueeze(0).expand(B, -1, -1)

            # Initialize gamma (diagonal Gamma entries)
            Gamma = torch.ones(B, dataset.N, device=device)

            # De-normalization factor (broadcastable to (B, N, L))
            denorm_factor = (M_norm_torch.sqrt().view(B, 1, 1) / G_norm_const_torch.view(1, 1, 1))
            X_denorm = X_norm_torch * denorm_factor

            # Forward inference
            mu_final, Sigma_X_diag_final = model(
                Y_norm_torch, Gamma, Lambda_norm_torch, G_norm_torch,
                num_layers=n_iterations
            )

            # De-normalize mu and apply thresholding
            mu_denorm = mu_final * denorm_factor
            mu_denorm_thresholded = torch.where(
                torch.abs(mu_denorm) >= activation_threshold,
                mu_denorm,
                torch.zeros_like(mu_denorm)
            )

            eps = torch.finfo(mu_denorm.dtype).eps

            # Per-sample MSE
            mse_per_sample = torch.mean((mu_denorm_thresholded - X_denorm) ** 2, dim=(1, 2)).cpu().numpy()

            # Per-sample MAE
            mae_per_sample = torch.mean(torch.abs(mu_denorm_thresholded - X_denorm), dim=(1, 2)).cpu().numpy()

            # "RMSE" in this code: relative L2 squared error
            diff_l2_squared = torch.sum((mu_denorm_thresholded - X_denorm) ** 2, dim=(1, 2))
            true_l2_squared = torch.sum(X_denorm ** 2, dim=(1, 2))
            rmse_per_sample = (diff_l2_squared / (true_l2_squared + eps)).cpu().numpy()

            # RMAE: relative L1 error
            diff_l1 = torch.sum(torch.abs(mu_denorm_thresholded - X_denorm), dim=(1, 2))
            true_l1 = torch.sum(torch.abs(X_denorm), dim=(1, 2))
            rmae_per_sample = (diff_l1 / (true_l1 + eps)).cpu().numpy()

            # EMD
            emd_per_sample = compute_emd_batch(mu_denorm_thresholded, X_denorm, source_locations)

            # Sparsity proxy: mean absolute value of thresholded mu
            sparsity_per_sample = torch.mean(torch.abs(mu_denorm_thresholded), dim=(1, 2)).cpu().numpy()

            # Active sources: RMS over time per source
            rms_per_source = torch.sqrt(torch.mean(mu_denorm_thresholded ** 2, dim=2))
            active_sources = (rms_per_source > activation_threshold).float()
            n_active_per_sample = torch.sum(active_sources, dim=1).cpu().numpy()

            # Sigma summary: sqrt(mean diagonal posterior variance)
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

    # Aggregate statistics across all samples
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
    Load a checkpoint and infer the number of stacked SBL layers (iterations).

    The number of layers is inferred by scanning state_dict keys of the form:
        "sbl_layers.<idx>...."

    Args:
        checkpoint_path (str): path to model checkpoint (.pth)
        M (int): number of sensors
        N (int): number of sources
        L (int): number of time points
        device (torch.device): cpu/cuda

    Returns:
        model (LSBLNetwork): instantiated model with weights loaded
        num_layers (int): inferred number of layers in the checkpoint
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer number of layers by finding the maximum layer index in state_dict keys
    max_layer_idx = -1
    for key in checkpoint['model_state_dict'].keys():
        if key.startswith('sbl_layers.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                max_layer_idx = max(max_layer_idx, layer_idx)

    num_layers = max_layer_idx + 1
    print(f"  Inferred number of layers: {num_layers}")

    # Create the Combined Model (Mode 4) network with the inferred depth
    model = LSBLNetwork(M, N, L, num_layers).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"  Model weights loaded successfully!")

    return model, num_layers


def evaluate_combined_model_train(data_folder, checkpoint_path, activation_threshold=1e-8):
    """
    Evaluate the Combined Model (Mode 4) on the training dataset.

    This function:
        - Loads metadata (M, N, L)
        - Loads training source locations (used for EMD)
        - Loads the model checkpoint and infers network depth
        - Evaluates the training dataset and prints 8 metrics
        - Saves results to ./output/evaluation_results_combined_model_train.npz

    Args:
        data_folder (str): path to the normalized data folder
        checkpoint_path (str): path to the Combined Model checkpoint (.pth)
        activation_threshold (float): threshold for considering a source "active"

    Returns:
        dict: configuration and evaluation metrics (mean/std and per-sample values)
    """
    print("="*80)
    print("Evaluate Combined Model (Mode 4) - Training Dataset - 8 core metrics")
    print("="*80)

    # Load metadata
    metadata = np.load(os.path.join(data_folder, "metadata.npz"))
    M = int(metadata['n_sensors'])
    N = int(metadata['n_sources'])
    L = int(metadata['n_time'])

    print(f"\nData dimensions: M={M}, N={N}, L={L}")
    print(f"Threshold: {activation_threshold}")

    # Load source locations for the training set (used by EMD)
    train_locs_path = "./data/fsaverage/-ico3surfFixedSourceLocstrain.npy"
    train_locs = np.load(train_locs_path)

    # Load model
    print("\n" + "="*80)
    print("Loading Combined Model")
    print("="*80)

    model, num_layers = load_checkpoint_and_infer_layers(checkpoint_path, M, N, L, device)

    # Evaluate training set
    print("\n" + "="*80)
    print("Evaluating training dataset")
    print("="*80)
    train_dataset = NormalizedEEGDataset(data_folder)
    train_metrics = evaluate_dataset(
        model, train_dataset, device, num_layers,
        train_locs, activation_threshold
    )

    # Print results
    print("\n" + "="*80)
    print("Evaluation results (Mean ± Std across samples):")
    print("="*80)

    print("\nTraining set results:")
    print("-"*160)
    print(f"{'Metric':<20} {'MSE':<20} {'MAE':<20} {'RMSE(L2norm)':<20} {'RMAE(L1norm)':<20} "
          f"{'EMD':<20} {'Sparsity':<20} {'N_Active':<15} {'Sigma':<20}")
    print("-"*160)

    v = train_metrics
    print(f"{'Combined Model':<20} "
          f"{v['mse']['mean']:.2e}±{v['mse']['std']:.2e}  "
          f"{v['mae']['mean']:.2e}±{v['mae']['std']:.2e}  "
          f"{v['rmse']['mean']:.2e}±{v['rmse']['std']:.2e}  "
          f"{v['rmae']['mean']:.2e}±{v['rmae']['std']:.2e}  "
          f"{v['emd']['mean']:.2e}±{v['emd']['std']:.2e}  "
          f"{v['sparsity']['mean']:.2e}±{v['sparsity']['std']:.2e}  "
          f"{v['n_active']['mean']:.1f}±{v['n_active']['std']:.1f}  "
          f"{v['sigma_diag']['mean']:.2e}±{v['sigma_diag']['std']:.2e}")

    print("\n" + "="*80)
    print("Note: Mean and Std are computed across training samples.")
    print("="*80)

    # Save results
    os.makedirs('./output', exist_ok=True)
    save_results = {
        'method': 'Combined Model (Mode 4)',
        'dataset': 'training set',
        'n_iterations': num_layers,
        'checkpoint': checkpoint_path,
        'activation_threshold': activation_threshold,
        'train': train_metrics,
    }

    output_file = './output/evaluation_results_combined_model_train.npz'
    np.savez(output_file, **save_results)
    print(f"\nResults saved: {output_file}")

    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80)

    return save_results


if __name__ == "__main__":
    # ==================== Configuration ====================

    data_folder = './normalized_data'           # Folder containing normalized dataset files
    checkpoint_path = "combined_model.pth"      # Path to the Combined Model checkpoint
    activation_threshold = 1e-6                 # Threshold for active source masking

    # ==================== Run evaluation ====================
    print("="*80)
    print("Evaluation configuration:")
    print(f"  Data folder: {data_folder}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Threshold: {activation_threshold}")
    print("="*80)
    print()

    results = evaluate_combined_model_train(
        data_folder=data_folder,
        checkpoint_path=checkpoint_path,
        activation_threshold=activation_threshold
    )
