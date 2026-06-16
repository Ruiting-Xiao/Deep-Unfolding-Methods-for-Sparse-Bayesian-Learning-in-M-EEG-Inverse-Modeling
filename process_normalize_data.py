#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Normalization Preprocessing Script
========================================

This script normalizes EEG datasets for training and validation. It handles three 
dataset splits with appropriate lead field normalization:

1. Training set: Uses training lead field for normalization
2. Validation set A: Uses training lead field (distribution match with training)
3. Validation set B: Uses validation lead field (generalization test)

Normalization Strategy:
----------------------
For each sample, the script normalizes:
- Source signals (X): Scaled by sqrt(M_norm) / G_norm_const
- Sensor measurements (Y): Scaled by sqrt(M_norm)
- Noise covariance (Lambda): Scaled by M_norm

where:
- M_norm = ||Y @ Y^T||_F (Frobenius norm of measurement covariance)
- G_norm_const = ||G||_inf (infinity norm of lead field matrix)

This normalization ensures numerical stability and consistent scaling across samples
while preserving the physical relationships in the forward model.

Output Files:
------------
For each dataset (train/valA/valB):
- {prefix}_X_norm_*.npy: Normalized source signals
- {prefix}_Y_norm_*.npy: Normalized sensor measurements
- {prefix}_Lambda_norm_*.npy: Normalized noise covariances
- {prefix}_M_norm_*.npy: Normalization constants (M_norm values)
- global_G_norm_{prefix}.npy: Normalized lead field matrix
- global_G_norm_const_{prefix}.npy: Lead field normalization constant


"""

import os
import numpy as np
from tqdm import tqdm


# ============================================================
# Configuration: Input/Output Paths
# ============================================================
input_folder = './generated_data'
output_folder = './normalized_data'
leadfield_folder = './data'


# ============================================================
# Normalization Functions
# ============================================================
def normalize_single_sample(Y, Lambda, M_norm, G_norm_const):
    """
    Normalize a single sample's sensor measurements and noise covariance.
    
    Parameters
    ----------
    Y : np.ndarray
        Sensor measurements, shape (n_sensors, n_time)
    Lambda : np.ndarray
        Noise covariance matrix, shape (n_sensors, n_sensors)
    M_norm : float
        Frobenius norm of Y @ Y^T, used for scaling
    G_norm_const : float
        Lead field normalization constant (infinity norm of G)
    
    Returns
    -------
    Y_norm : np.ndarray
        Normalized sensor measurements, shape (n_sensors, n_time)
    Lambda_norm : np.ndarray
        Normalized noise covariance, shape (n_sensors, n_sensors)
    
    Notes
    -----
    Normalization formulas:
    - Y_norm = Y / sqrt(M_norm)
    - Lambda_norm = Lambda / M_norm
    """
    Y = Y.astype(np.float64)
    Lambda = Lambda.astype(np.float64)
    Y_norm = Y / np.sqrt(M_norm)
    Lambda_norm = Lambda / M_norm
    return Y_norm, Lambda_norm


def process_batch_file(prefix, file_idx, G_norm_const_global, output_folder, batch_size):
    """
    Process and normalize a single batch file.
    
    Parameters
    ----------
    prefix : str
        Dataset prefix ('train', 'valA', or 'valB')
    file_idx : int
        Batch file index (0-based)
    G_norm_const_global : float
        Global lead field normalization constant
    output_folder : str
        Directory path for saving normalized outputs
    batch_size : int
        Expected batch size (for validation)
    
    Returns
    -------
    processed_count : int
        Number of samples successfully processed in this batch
    
    Notes
    -----
    For each sample in the batch:
    1. Computes M_norm = ||Y @ Y^T||_F
    2. Normalizes Y and Lambda using M_norm
    3. Normalizes X using both M_norm and G_norm_const
    4. Saves normalized data to separate .npy files
    """
    X_path = os.path.join(input_folder, f"{prefix}_X_{file_idx:04d}.npy")
    Y_path = os.path.join(input_folder, f"{prefix}_Y_{file_idx:04d}.npy")
    Lambda_path = os.path.join(input_folder, f"{prefix}_Lambda_{file_idx:04d}.npy")

    # Check if all required files exist
    if not (os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(Lambda_path)):
        print(f"Warning: Missing npy file for batch {file_idx}")
        return 0

    # Load batch data
    X = np.load(X_path)
    Y = np.load(Y_path)
    Lambda = np.load(Lambda_path)
    B = X.shape[0]

    X_norm_list, Y_norm_list, Lambda_norm_list, M_norm_list = [], [], [], []

    # Process each sample in batch
    for i in range(B):
        # Compute M_norm: Frobenius norm of measurement covariance
        YYT = Y[i] @ Y[i].T
        M_norm = np.linalg.norm(YYT, ord="fro")
        if M_norm < 1e-10:
            M_norm = 1.0

        # Normalize Y and Lambda
        Y_norm, Lambda_norm = normalize_single_sample(Y[i], Lambda[i], M_norm, G_norm_const_global)
        
        # Normalize X (source signals)
        X_norm = X[i] / (np.sqrt(M_norm) / G_norm_const_global)

        X_norm_list.append(X_norm)
        Y_norm_list.append(Y_norm)
        Lambda_norm_list.append(Lambda_norm)
        M_norm_list.append(M_norm)

    # Save normalized batch
    np.save(os.path.join(output_folder, f"{prefix}_X_norm_{file_idx:04d}.npy"),
            np.array(X_norm_list, dtype=np.float32))
    np.save(os.path.join(output_folder, f"{prefix}_Y_norm_{file_idx:04d}.npy"),
            np.array(Y_norm_list, dtype=np.float32))
    np.save(os.path.join(output_folder, f"{prefix}_Lambda_norm_{file_idx:04d}.npy"),
            np.array(Lambda_norm_list, dtype=np.float32))
    np.save(os.path.join(output_folder, f"{prefix}_M_norm_{file_idx:04d}.npy"),
            np.array(M_norm_list, dtype=np.float32))

    return B


# ============================================================
# Main Processing Pipeline
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Data Normalization: train / valA / valB (with three G_norm outputs)")
    print("=" * 80)

    os.makedirs(output_folder, exist_ok=True)

    # Load metadata from generated dataset
    md = np.load(os.path.join(input_folder, "metadata.npz"))
    M = int(md["n_sensors"])
    N = int(md["n_sources"])
    L = int(md["n_time"])
    batch_size = int(md["batch_size"])
    n_train_batches = int(md["n_train_batches"])
    n_valA_batches = int(md["n_valA_batches"])
    n_valB_batches = int(md["n_valB_batches"])

    # =====================================================================
    # Step 1: Process train + valA with training lead field
    # =====================================================================
    print("\n##### Loading TRAIN Lead Field for train + valA #####")
    G_train = np.load(os.path.join(leadfield_folder, "fsaverage/-ico3surfFixedLeadFieldtrain.npy")).astype(np.float64)
    if G_train.ndim == 3:
        G_train = G_train.reshape(M, -1)

    # Compute normalization constant (infinity norm)
    G_norm_const_train = np.linalg.norm(G_train, ord=np.inf)
    if G_norm_const_train < 1e-10:
        G_norm_const_train = 1.0
    G_norm_train = G_train / G_norm_const_train

    # Save normalized lead field for train and valA (shared)
    np.save(os.path.join(output_folder, "global_G_norm_train.npy"), G_norm_train.astype(np.float32))
    np.save(os.path.join(output_folder, "global_G_norm_const_train.npy"),
            np.array([G_norm_const_train], dtype=np.float32))
    np.save(os.path.join(output_folder, "global_G_norm_valA.npy"), G_norm_train.astype(np.float32))
    np.save(os.path.join(output_folder, "global_G_norm_const_valA.npy"),
            np.array([G_norm_const_train], dtype=np.float32))
    print("Saved global_G_norm_train.npy / global_G_norm_valA.npy")

    # =====================================================================
    # Step 2: Process valB with validation lead field
    # =====================================================================
    print("\n##### Loading VAL Lead Field for valB #####")
    G_val = np.load(os.path.join(leadfield_folder, "fsaverage/-ico3surfFixedLeadFieldval.npy")).astype(np.float64)
    if G_val.ndim == 3:
        G_val = G_val.reshape(M, -1)

    # Compute normalization constant for validation lead field
    G_norm_const_val = np.linalg.norm(G_val, ord=np.inf)
    if G_norm_const_val < 1e-10:
        G_norm_const_val = 1.0
    G_norm_val = G_val / G_norm_const_val

    # Save normalized lead field for valB
    np.save(os.path.join(output_folder, "global_G_norm_valB.npy"), G_norm_val.astype(np.float32))
    np.save(os.path.join(output_folder, "global_G_norm_const_valB.npy"),
            np.array([G_norm_const_val], dtype=np.float32))
    print("Saved global_G_norm_valB.npy")

    # =====================================================================
    # Step 3: Process training dataset
    # =====================================================================
    print("\nProcessing TRAIN ...")
    train_N = 0
    for i in tqdm(range(n_train_batches), desc="Training batches"):
        train_N += process_batch_file("train", i, G_norm_const_train, output_folder, batch_size)
    print(f"Train completed: {train_N} samples")

    # =====================================================================
    # Step 4: Process validation set A (using train G_norm_const)
    # =====================================================================
    print("\nProcessing valA (with train G_norm_const) ...")
    valA_N = 0
    for i in tqdm(range(n_valA_batches), desc="Val-A batches"):
        valA_N += process_batch_file("valA", i, G_norm_const_train, output_folder, batch_size)
    print(f"valA completed: {valA_N} samples")

    # =====================================================================
    # Step 5: Process validation set B (using val G_norm_const)
    # =====================================================================
    print("\nProcessing valB (with val G_norm_const) ...")
    valB_N = 0
    for i in tqdm(range(n_valB_batches), desc="Val-B batches"):
        valB_N += process_batch_file("valB", i, G_norm_const_val, output_folder, batch_size)
    print(f"valB completed: {valB_N} samples")

    # =====================================================================
    # Step 6: Save metadata
    # =====================================================================
    np.savez(
        os.path.join(output_folder, "metadata.npz"),
        n_sensors=M,
        n_sources=N,
        n_time=L,
        batch_size=batch_size,
        n_train_batches=n_train_batches,
        n_valA_batches=n_valA_batches,
        n_valB_batches=n_valB_batches,
        n_train=train_N,
        n_valA=valA_N,
        n_valB=valB_N,
        normalized=True,
    )

    print("=" * 80)
    print("Normalization Complete: train + valA + valB (with three G_norm outputs)")
    print("=" * 80)
    print(f"Normalized data saved to: {output_folder}")
    print(f"  Training samples: {train_N}")
    print(f"  Validation A samples: {valA_N}")
    print(f"  Validation B samples: {valB_N}")
    print("=" * 80)

#%%