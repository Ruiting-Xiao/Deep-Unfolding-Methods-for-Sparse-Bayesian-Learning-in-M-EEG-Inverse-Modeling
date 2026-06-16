# generate_dataset_fast.py
"""
EEG Dataset Generation Script
==============================

This script generates synthetic EEG datasets for training and validation of source 
localization models. It creates three dataset splits:
- Training set: Generated using training head model
- Validation set A: Generated using training head model (distribution match)
- Validation set B: Generated using validation head model (generalization test)

The generation follows the forward model: y = L @ x + n
where:
- L: Lead field matrix (normalized)
- x: Source signals with spatial spread
- n: Sensor white noise (parameterized by SNR)

Key Features:
- On-the-fly generation with configurable parameters
- GPU acceleration support
- Batch-wise file storage for memory efficiency
- Reproducible with seed control

Author: Based on Marco's implementation
"""

import os
import math
import json
import numpy as np
import torch
from typing import Tuple, Dict, Optional
from tqdm import tqdm

from scipy.stats import norm
from scipy.signal import butter, filtfilt
from multiprocessing import Pool


class HeadModel:
    """
    Minimalist head model containing only normalized lead field and source locations.
    """

    def __init__(self,
                 leadfield_norm_path: str,
                 source_locs_path: str,
                 device: str = 'cpu'):
        """
        Initialize head model from saved files.
        
        Parameters
        ----------
        leadfield_norm_path : str
            Path to normalized lead field matrix (.npy file)
        source_locs_path : str
            Path to source locations (.npy file)
        device : str
            Device for tensor operations ('cpu' or 'cuda'). Default: 'cpu'
        """
        # Load data
        self.L = torch.Tensor(np.load(leadfield_norm_path)).float()
        self.source_locs = torch.Tensor(np.load(source_locs_path)).float()

        # Derive parameters from L
        self.n_sensors = self.L.shape[0]
        self.n_sources = self.L.shape[1]
        self.n_orient = self.L.shape[2]
        self.L_flat = self.L.flatten(1)

        # Move to specified device
        self.device = device
        self.to(device)

        print(f"Head model loaded:")
        print(f"  Sensors: {self.n_sensors}")
        print(f"  Sources: {self.n_sources}")
        print(f"  Orientations: {self.n_orient}")
        print(f"  Device: {self.device}")

        # Verify normalization
        L_2d = self.L.reshape(self.n_sensors, -1)
        col_norms = torch.norm(L_2d, dim=0)
        print(f"  Lead Field column norms: min={col_norms.min():.6f}, max={col_norms.max():.6f}")

    def to(self, device: str):
        """
        Move model data to specified device.
        
        Parameters
        ----------
        device : str
            Target device ('cpu' or 'cuda')
        
        Returns
        -------
        self : HeadModel
            Returns self for method chaining
        """
        self.device = device
        self.L = self.L.to(device)
        self.L_flat = self.L_flat.to(device)
        self.source_locs = self.source_locs.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection: y = L @ x
        
        Parameters
        ----------
        x : torch.Tensor
            Source signals, shape (batch, time, n_sources * n_orient) or 
            (batch, time, n_sources, n_orient)
        
        Returns
        -------
        y : torch.Tensor
            Sensor measurements, shape (batch, time, n_sensors)
        """
        if len(x.shape) == 4:
            x = x.flatten(2)
        return torch.einsum("BTD, MD -> BTM", x, self.L_flat)

    def get_model_info(self) -> Dict:
        """
        Get summary of model information.
        
        Returns
        -------
        info : dict
            Dictionary containing model dimensions and device info
        """
        return {
            "n_sensors": self.n_sensors,
            "n_sources": self.n_sources,
            "n_orient": self.n_orient,
            "device": self.device,
            "L_shape": tuple(self.L.shape),
            "source_locs_shape": tuple(self.source_locs.shape),
        }


class EEGDataGenerator:
    """
    EEG data generator: y = L @ x + n
    Generates sensor measurements with white noise only.
    """

    def __init__(self,
                 head_model,
                 min_snr: float = 0,
                 max_snr: float = 5,
                 min_sources: int = 5,
                 max_sources: int = 20,
                 min_spread: float = 0.001,
                 max_spread: float = 0.001,
                 time_steps: int = 1,
                 fs: int = 100):
        """
        Initialize EEG data generator with configurable parameters.
        
        Parameters
        ----------
        head_model : HeadModel
            HeadModel instance containing lead field and source locations
        min_snr : float
            Minimum SNR in dB. Default: 0
        max_snr : float
            Maximum SNR in dB. Default: 5
        min_sources : int
            Minimum number of active sources. Default: 5
        max_sources : int
            Maximum number of active sources. Default: 20
        min_spread : float
            Minimum spatial spread (std) for source distribution. Default: 0.001
        max_spread : float
            Maximum spatial spread (std) for source distribution. Default: 0.001
        time_steps : int
            Number of time points. Default: 1
        fs : int
            Sampling frequency in Hz. Default: 100
        """
        self.head_model = head_model
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.time_steps = time_steps
        self.fs = fs

    def generate_sample(self,
                        seed: Optional[int] = None,
                        device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Generate a single sample: y = L @ x + n
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        device : str
            Device for tensor operations ('cpu' or 'cuda'). Default: 'cpu'
        
        Returns
        -------
        sample : dict
            Dictionary containing:
            - 'x': Source signals, shape (n_sources, time) or (n_sources, time, n_orient)
            - 'y': Sensor measurements, shape (n_sensors, time)
            - 'Lambda': Noise covariance matrix, shape (n_sensors, n_sensors)
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 1. Generate clean source signal x_clean (time, n_sources, n_orient)
        x_clean = self._generate_source_signal()
        x_clean = torch.Tensor(x_clean).float().to(device)

        # 2. Forward project clean signal
        y_clean = self.head_model.forward(x_clean.unsqueeze(0)).squeeze(0)  # (time, n_sensors)

        # 3. Calculate SNR and noise standard deviation
        snr = np.random.rand() * (self.max_snr - self.min_snr) + self.min_snr
        signal_std = torch.std(y_clean)
        noise_std = signal_std / (10 ** (snr / 20))

        # 4. Generate sensor white noise
        y_white_noise = torch.randn_like(y_clean) * noise_std

        # 5. Synthesize final sensor signal
        y = y_clean + y_white_noise

        # 6. Noise covariance matrix
        Lambda = (noise_std ** 2) * torch.eye(self.head_model.n_sensors, device=device)

        # 7. Source signal is the clean signal
        x = x_clean

        # Transpose output format
        x = x.permute(1, 0, 2)
        if self.head_model.n_orient == 1:
            x = x.squeeze(-1)
        y = y.permute(1, 0)

        return {
            "x": x,  # (n_sources, time)
            "y": y,  # (n_sensors, time)
            "Lambda": Lambda,  # (n_sensors, n_sensors)
        }

    def _generate_source_signal(self) -> np.ndarray:
        """
        Generate clean source signal components with spatial spread.
        
        Returns
        -------
        x_clean : np.ndarray
            Clean source signals, shape (time, n_sources, n_orient)
        """
        # Random number of active sources
        if self.min_sources == 0:
            n_active = max(np.random.randint(int(-0.5 * self.max_sources),
                                             self.max_sources + 1), 1)
        else:
            n_active = np.random.randint(self.min_sources, self.max_sources + 1)

        # Random source centers
        n_sources = self.head_model.n_sources
        source_centers = np.random.randint(0, n_sources, n_active)

        # Random spatial spread standard deviations
        std_sources = np.random.rand(n_active) * (self.max_spread - self.min_spread) + self.min_spread

        # Generate spatial distribution
        sources_spatial = []
        source_locs = self.head_model.source_locs.cpu().numpy()

        for i in range(n_active):
            center_loc = source_locs[source_centers[i]]
            distances = np.linalg.norm(center_loc[None] - source_locs, axis=1)
            source_amp = norm.pdf(distances, loc=0, scale=std_sources[i])
            source_amp = source_amp / np.linalg.norm(source_amp)

            # Fixed orientation: n_orient = 1
            source_amp_3D = source_amp[:, None]
            sources_spatial.append(source_amp_3D)

        sources_spatial = np.stack(sources_spatial, axis=-1)  # (n_sources, n_orient, n_active)

        # Generate time series
        source_timeseries = self._generate_time_series(n_active)  # (n_active, time)

        # Spatiotemporal fusion
        x_clean = np.einsum("LOS, ST -> TLO", sources_spatial, source_timeseries)

        return x_clean

    def _generate_time_series(self, n_sources: int) -> np.ndarray:
        """
        Generate time series for active sources with frequency band filtering.
        
        Parameters
        ----------
        n_sources : int
            Number of time series to generate
        
        Returns
        -------
        signals : np.ndarray
            Time series array, shape (n_sources, time_steps)
        """
        if self.time_steps == 1:
            return np.random.randn(n_sources, 1)

        signals = []
        freq_bands = {
            'mu': (8, 12),
            'beta': (12, 30),
            'alpha': (8, 13),
            'theta': (4, 8),
            'delta': (0.5, 4),
        }

        for i in range(n_sources):
            band = freq_bands[np.random.choice(list(freq_bands.keys()))]
            base_signal = np.random.randn(self.time_steps)
            b, a = butter(4, [f / (self.fs / 2) for f in band], btype='band')
            filtered_signal = filtfilt(b, a, base_signal)
            normalized_signal = filtered_signal / np.linalg.norm(filtered_signal)
            signals.append(normalized_signal)

        return np.stack(signals, axis=0)

    def generate_batch(self,
                       batch_size: int,
                       device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Generate a batch of samples.
        
        Parameters
        ----------
        batch_size : int
            Number of samples in batch
        device : str
            Device for tensor operations. Default: 'cpu'
        
        Returns
        -------
        batch : dict
            Dictionary containing:
            - 'x': Batched source signals, shape (batch_size, n_sources, time)
            - 'y': Batched sensor measurements, shape (batch_size, n_sensors, time)
            - 'Lambda': Batched noise covariances, shape (batch_size, n_sensors, n_sensors)
        """
        samples = [self.generate_sample(device=device) for _ in range(batch_size)]

        batch = {
            "x": torch.stack([s["x"] for s in samples]),
            "y": torch.stack([s["y"] for s in samples]),
            "Lambda": torch.stack([s["Lambda"] for s in samples]),
        }

        return batch


def main():
    """
    Main function to generate complete EEG dataset including:
    - Training set with training head model
    - Validation set A with training head model
    - Validation set B with validation head model
    
    All parameters are hardcoded in this function for easy modification.
    Generated data is saved in batch-wise .npy files along with metadata.
    """
    import os
    import numpy as np
    import torch
    from tqdm import tqdm

    # ----------------------------------------------------
    # Configuration: All parameters hardcoded here
    # ----------------------------------------------------
    data_folder = "./generated_data"

    # Train head model paths
    train_leadfield_path = "data/fsaverage/-ico3surfFixedLeadFieldtrain.npy"
    train_sourcelocs_path = "data/fsaverage/-ico3surfFixedSourceLocstrain.npy"

    # Val head model paths
    val_leadfield_path = "data/fsaverage/-ico3surfFixedLeadFieldval.npy"
    val_sourcelocs_path = "data/fsaverage/-ico3surfFixedSourceLocsval.npy"

    # Dataset sizes
    n_train = 100000
    n_val = 1000
    batch_size = 128
    n_time = 100

    # Source generation parameters
    n_active_min = 5
    n_active_max = 20
    snr_min = 5
    snr_max = 30
    std_min = 0.001
    std_max = 0.005
    fs = 100

    # Random seed and device
    seed = 42
    device = "cuda"
    # ----------------------------------------------------

    os.makedirs(data_folder, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ===================================================================
    # 1. Load TRAIN head model and generate training set
    # ===================================================================
    print("\n===== Loading TRAIN Head Model =====")
    train_head = HeadModel(train_leadfield_path, train_sourcelocs_path, device)

    train_gen = EEGDataGenerator(
        head_model=train_head,
        min_snr=snr_min, max_snr=snr_max,
        min_sources=n_active_min, max_sources=n_active_max,
        min_spread=std_min, max_spread=std_max,
        time_steps=n_time, fs=fs
    )

    n_train_batches = (n_train + batch_size - 1) // batch_size
    print(f"\n===== Generating Training Set ({n_train_batches} batches, {n_train} samples) =====")

    for bi in tqdm(range(n_train_batches), desc="Training batches"):
        # Set batch-level seed for reproducibility
        batch_seed = seed + bi * 10000
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)

        batch = train_gen.generate_batch(batch_size, device=device)
        X = batch["x"].cpu().numpy()
        Y = batch["y"].cpu().numpy()
        Lambda = batch["Lambda"].cpu().numpy()

        np.save(os.path.join(data_folder, f"train_X_{bi:04d}.npy"), X)
        np.save(os.path.join(data_folder, f"train_Y_{bi:04d}.npy"), Y)
        np.save(os.path.join(data_folder, f"train_Lambda_{bi:04d}.npy"), Lambda)

    # ===================================================================
    # 2. Generate VAL_A dataset (using TRAIN head model)
    # ===================================================================
    print(f"\n===== Generating Validation Set A (using TRAIN head model, {n_val} samples) =====")
    n_valA_batches = (n_val + batch_size - 1) // batch_size

    for bi in tqdm(range(n_valA_batches), desc="Val-A batches"):
        batch_seed = seed + 1000000 + bi * 10000
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)

        batch = train_gen.generate_batch(batch_size, device=device)
        X = batch["x"].cpu().numpy()
        Y = batch["y"].cpu().numpy()
        Lambda = batch["Lambda"].cpu().numpy()

        np.save(os.path.join(data_folder, f"valA_X_{bi:04d}.npy"), X)
        np.save(os.path.join(data_folder, f"valA_Y_{bi:04d}.npy"), Y)
        np.save(os.path.join(data_folder, f"valA_Lambda_{bi:04d}.npy"), Lambda)

    # ===================================================================
    # 3. Generate VAL_B dataset (using VAL head model)
    # ===================================================================
    print("\n===== Loading VAL Head Model =====")
    val_head = HeadModel(val_leadfield_path, val_sourcelocs_path, device)

    val_gen = EEGDataGenerator(
        head_model=val_head,
        min_snr=snr_min, max_snr=snr_max,
        min_sources=n_active_min, max_sources=n_active_max,
        min_spread=std_min, max_spread=std_max,
        time_steps=n_time, fs=fs
    )

    n_valB_batches = (n_val + batch_size - 1) // batch_size
    print(f"\n===== Generating Validation Set B (using VAL head model, {n_val} samples) =====")

    for bi in tqdm(range(n_valB_batches), desc="Val-B batches"):
        batch_seed = seed + 2000000 + bi * 10000
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)

        batch = val_gen.generate_batch(batch_size, device=device)
        X = batch["x"].cpu().numpy()
        Y = batch["y"].cpu().numpy()
        Lambda = batch["Lambda"].cpu().numpy()

        np.save(os.path.join(data_folder, f"valB_X_{bi:04d}.npy"), X)
        np.save(os.path.join(data_folder, f"valB_Y_{bi:04d}.npy"), Y)
        np.save(os.path.join(data_folder, f"valB_Lambda_{bi:04d}.npy"), Lambda)

    # ===================================================================
    # 4. Save metadata
    # ===================================================================
    metadata = {
        'n_train': n_train,
        'n_valA': n_val,
        'n_valB': n_val,
        'batch_size': batch_size,
        'n_train_batches': n_train_batches,
        'n_valA_batches': n_valA_batches,
        'n_valB_batches': n_valB_batches,
        'n_sensors': train_head.n_sensors,
        'n_sources': train_head.n_sources,
        'n_time': n_time,
    }

    np.savez(os.path.join(data_folder, "metadata.npz"), **metadata)
    print("\n✅ Metadata saved.")
    print("\n🎉 Dataset generation complete!")
    print(f"   Total samples: {n_train + 2 * n_val}")
    print(f"   Location: {data_folder}/")


if __name__ == "__main__":
    main()