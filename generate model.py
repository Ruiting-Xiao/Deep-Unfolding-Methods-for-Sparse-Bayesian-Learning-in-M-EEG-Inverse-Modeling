#%%
"""
EEG Forward Model Generation Script
====================================

This script is based on Marco's original code (with some unused code sections removed).

Main Functionality:
------------------
Generates four essential files for EEG source localization:
1. Training set Lead Field matrix
2. Validation set Lead Field matrix  
3. Training set Source Locations
4. Validation set Source Locations

Author: Ruiting Xiao
Author: Based on Marco's implementation
"""

import numpy as np
import scipy.linalg
import mne
import os

def get_info(kind='easycap-M10', sfreq=1000):
    """
    Create a generic MNE Info object with standard montage.
    
    Parameters
    ----------
    kind : str
        Montage type. For available options, see:
        https://mne.tools/stable/generated/mne.channels.make_standard_montage.html
        Default: 'easycap-M10'
    sfreq : float
        Sampling frequency in Hz. Default: 1000
    
    Returns
    -------
    info : mne.Info
        MNE Info object containing channel information and montage
    """
    montage = mne.channels.make_standard_montage(kind)
    info = mne.create_info(montage.ch_names, sfreq, ch_types=['eeg'] * len(montage.ch_names), verbose=0)
    info.set_montage(montage)
    return info


def mkfilt_eloreta_v2(L, regu=0.05):
    """
    Compute eLORETA inverse operator using iterative algorithm.
    
    Based on: R.D. Pascual-Marqui: Discrete, 3D distributed, linear imaging methods 
    of electric neuronal activity. Part 1: exact, zero error localization. 
    arXiv:0710.3341 [math-ph], 2007-October-17, http://arxiv.org/pdf/0710.3341
    
    Parameters
    ----------
    L : ndarray, shape (n_channels, n_sources, n_orientations)
        Lead field matrix
    regu : float
        Regularization parameter. Default: 0.05
    
    Returns
    -------
    A : ndarray, shape (n_channels, n_sources, n_orientations)
        eLORETA inverse operator
    """
    nchan, ng, ndum = L.shape
    LL = np.zeros((nchan, ndum, ng))
    for i in range(ndum):
        LL[:, i, :] = L[:, :, i]
    LL = np.reshape(LL, (nchan, ndum * ng), order='F')

    u0 = np.eye(nchan)
    W = np.reshape(np.tile(np.eye(ndum), (1, ng)), (ndum, ndum, ng), order='F')
    Winv = np.zeros((ndum, ndum, ng))
    winvkt = np.zeros((ng * ndum, nchan))
    kont = 0
    kk = 0
    while kont == 0:
        kk += 1
        for i in range(ng):
            Winv[:, :, i] = np.linalg.inv(W[:, :, i] + np.trace(W[:, :, i]) / (ndum * 1e6))
        for i in range(ng):
            winvkt[ndum * i:ndum * (i + 1), :] = np.dot(Winv[:, :, i], LL[:, ndum * i:ndum * (i + 1)].conj().T)
        kwinvkt = np.dot(LL, winvkt)
        alpha = regu * np.trace(kwinvkt) / nchan
        M = np.linalg.inv(kwinvkt + alpha * u0)
        ux, sx, vx = np.linalg.svd(kwinvkt)
        for i in range(ng):
            Lloc = L[:, i, :]
            Wold = np.copy(W)
            W[:, :, i] = np.real(scipy.linalg.sqrtm(np.dot(Lloc.conj().T, np.dot(M, Lloc))))
        reldef = np.linalg.norm(W.flatten() - Wold.flatten()) / np.linalg.norm(Wold.flatten())
        if kk > 20 or reldef < 1e-6:
            kont = 1

    ktm = np.dot(LL.conj().T, M)
    A = np.zeros((nchan, ng, ndum))
    for i in range(ng):
        A[:, i, :] = np.dot(Winv[:, :, i], ktm[ndum * i:ndum * (i + 1), :]).conj().T

    return A


def generete_necessary_data(subject, grid_size=14, dataset="", mode="train", subjects_dir="./data",
                            vol_source_space=False,
                            cap_kind="easycap-M10",
                            fixed_orientation=True,
                            ico="ico3"):
    """
    Generate necessary data for EEG source localization including forward model,
    inverse operator, and spatial information.
    
    Parameters
    ----------
    subject : str
        Subject name (e.g., 'fsaverage' for template brain)
    grid_size : int
        Grid spacing for volumetric source space in mm. Default: 14
    dataset : str
        Dataset name for file naming. Default: ""
    mode : str
        Mode identifier ('train' or 'val') affecting BEM conductivity. Default: "train"
    subjects_dir : str
        FreeSurfer subjects directory path. Default: "./data"
    vol_source_space : bool
        If True, use volumetric source space; if False, use surface source space. Default: False
    cap_kind : str
        EEG cap montage type. Default: "easycap-M10"
    fixed_orientation : bool
        If True, constrain dipoles to cortical surface normal. Default: True
    ico : str
        Icosahedron subdivision level for surface source space (e.g., 'ico3'). Default: "ico3"
    
    Returns
    -------
    train_data : dict
        Dictionary containing:
        - 'LeadField': ndarray, forward matrix (n_channels, n_sources, n_orientations)
        - 'PseudoInv': ndarray, eLORETA inverse operator
        - 'grid_loc': ndarray, source locations (n_sources, 3)
        - 'sens_loc': ndarray, sensor locations (n_channels, 3)
        - 'source_inds': ndarray, vertex indices for sources
        - 'source_mask': ndarray or None, volumetric source mask
        - 'morph': ndarray or None, morphing matrix to fsaverage
        - 'time_steps': int, number of time samples for dataset
        - 'fs': int, sampling frequency
    
    Notes
    -----
    Saves multiple files:
    - Complete head model dictionary: *HeadInformation{mode}-fwd.npy
    - Lead field matrix: *LeadField{mode}.npy
    - Source locations: *SourceLocs{mode}.npy
    """
    # Create subject directory if it doesn't exist
    if not os.path.exists(subjects_dir + "/" + subject):
        os.makedirs(subjects_dir + "/" + subject)
    
    # Setup transformation matrix
    if subject == "fsaverage":
        fs_dir = mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
        trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    else:
        trans = subjects_dir + "/" + subject + "/HeadTransform.fif"
    
    # Create info object with EEG montage
    info = get_info(cap_kind)
    n_sensors = len(info.ch_names)
    
    # Setup source space (volumetric or surface)
    if vol_source_space:
        src = mne.setup_volume_source_space(subject, pos=grid_size,
                                            subjects_dir=subjects_dir, verbose=False)
        src_type = "vol"
    else:
        src = mne.setup_source_space(
            subject=subject,
            spacing=ico,
            subjects_dir=subjects_dir,
            add_dist=False,
            verbose=False,
        )
        src_type = "surf"

    # Determine source type string based on orientation constraint
    if fixed_orientation:
        src_type += "Fixed"
    else:
        src_type += "Free"
    
    # Dataset-specific time step mapping
    time_dict = {"Zhou2016": 1251,
                 "Weibo2014": 801,
                 "BNCI2015_004": 1793,
                 "Cho2017": 1537,
                 "PhysionetMI": 481,
                 "": 1}
    
    # Load or create BEM solution
    try:
        bem = mne.read_bem_solution(f"{subjects_dir}/{subject}/BEM{mode}.fif")
    except:
        # Different conductivity for train vs validation
        if mode == "train":
            conductivity = [0.3, 0.006, 0.3]
        else:
            conductivity = [0.332, 0.0113, 0.332]
        bem_init = mne.make_bem_model(subject, subjects_dir=subjects_dir, conductivity=conductivity, verbose=False)
        bem = mne.make_bem_solution(bem_init, verbose=False)
        mne.write_bem_solution(f"{subjects_dir}/{subject}/BEM{mode}.fif", bem, verbose=False)

    # Load or create forward solution
    fwd_path = f"{subjects_dir}/{subject}/{dataset}-{ico if not vol_source_space else grid_size}{src_type}forward{mode}-fwd.fif"
    try:
        fwd = mne.read_forward_solution(fwd_path)
    except:
        fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                        bem=bem, eeg=True, meg=False, mindist=5.0, n_jobs=1, verbose=False)
        mne.write_forward_solution(fwd_path, fwd)

    # Convert to fixed orientation if requested
    if fixed_orientation:
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=True)
        print("Using fixed orientation forward model")

    # Extract sensor positions and forward matrix
    sens_pos = np.array(list(info.get_montage()._get_ch_pos().values()))
    fwd_matrix = fwd["sol"]["data"]
    fwd_matrix = fwd_matrix.reshape(n_sensors, -1, 3 if vol_source_space else 1)

    # Compute eLORETA inverse operator
    pinv = mkfilt_eloreta_v2(fwd_matrix)

    # Create morphing matrix if not using fsaverage
    if subject != "fsaverage":
        fs_dict = get_mne_src_fwd_inv("train")
        morph = mne.compute_source_morph(
            src=fs_dict["fwd"]["src"],
            subject_from="fsaverage",
            subject_to=subject,
            subjects_dir=subjects_dir,
            src_to=fwd["src"],
        )
        morph.compute_vol_morph_mat()
        morph_matrix = morph.vol_morph_mat
    else:
        morph_matrix = None

    # Get source indices
    if vol_source_space:
        source_inds = fwd["src"][0]["vertno"]
    else:
        source_inds = np.arange(fwd_matrix.shape[1])
    
    # Get source locations
    source_locs = fwd["source_rr"]
    
    # Build complete data dictionary
    train_data = {
        "LeadField": fwd_matrix,
        "PseudoInv": pinv,
        "grid_loc": source_locs,
        "sens_loc": sens_pos,
        "source_inds": source_inds,
        "source_mask": fwd["src"][0]["inuse"].reshape(fwd["src"][0]["shape"]) if vol_source_space else None,
        "morph": morph_matrix,
        "time_steps": time_dict.get(dataset, 1),
        "fs": int(info["sfreq"]),
    }
    
    # Generate file name prefix
    file_prefix = f"{subjects_dir}/{subject}/{dataset}-{ico if not vol_source_space else grid_size}{src_type}"
    
    # Save complete data dictionary
    np.save(f"{file_prefix}HeadInformation{mode}-fwd.npy", train_data)
    print(f"Saved complete head model to: {file_prefix}HeadInformation{mode}-fwd.npy")
    
    # Save lead field matrix separately
    np.save(f"{file_prefix}LeadField{mode}.npy", fwd_matrix)
    print(f"Saved Lead Field matrix to: {file_prefix}LeadField{mode}.npy")
    
    # Save source locations separately
    np.save(f"{file_prefix}SourceLocs{mode}.npy", source_locs)
    print(f"Saved source locations to: {file_prefix}SourceLocs{mode}.npy")
    
    # Print summary
    print(f"Summary:")
    print(f"  Lead Field shape: {fwd_matrix.shape}")
    print(f"  Source locations shape: {source_locs.shape}")
    print(f"  Number of sensors: {n_sensors}")
    print(f"  Number of sources: {fwd_matrix.shape[1]}")
    
    return train_data

#%%
# Generate the necessary files for fsaverage template brain
for mode in ["train", "val"]:
    generete_necessary_data("fsaverage", grid_size=14, mode=mode,
                            vol_source_space=False,
                            fixed_orientation=True,
                            cap_kind="biosemi64",
                            ico="ico3")
#%%
