"""Helper function for RDM creation"""
import numpy as np
import pandas as pd
from os.path import join as pjoin
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from nilearn.masking import apply_mask, intersect_masks, unmask
from utils import pearsonr_nd, get_hrflib, spearman_brown, match_scale, apply_mask_smoothed, regress_out, r2_ndarr
#create a function for extracting the masks and reading the mask
nsessions=12
def union_masker(func):
    #creating the decorator function for reading the mask and 
    #returning the union mask
    #creating the session names
    sessions = [f'things{i:02d}' for i in range(1, nsessions+1)]
    bidsroot='/DATA1/satwick22/Documents/fMRI/thingsmri'#SET
    subject='01'
    subj_prepdir=pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{subject}')
    def wrapper(subject,subj_prepdir=subj_prepdir,sessions=sessions):
        masks=func(subject,subj_prepdir,sessions)
        union_mask=intersect_masks(masks, threshold=0)
        return union_mask
    return wrapper
@union_masker
def get_masks(subject,subj_prepdir,sessions):
    masks = [
        pjoin(subj_prepdir, f'ses-{sesname}', 'func',
              f'sub-{subject}_ses-{sesname}_task-things_run-{run_i + 1:02d}_space-T1w_desc-brain_mask.nii.gz')
        for sesname in sessions for run_i in range(10)
    ]
    for mask in masks:
        assert os.path.exists(mask), f'\nmask not found:\n{mask}\n'
    return masks
#creating the function for reading the beta weights and smootehning the same
def load_betas(
        sub: str,
        mask: str,
        bidsroot: str = '/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_',
        betas_derivname: str = 'betas_run01/unregularized',
        ses_i=None,
        smoothing=0.,
        dtype=np.single,
) -> np.ndarray:
    if not smoothing:
        smoothing = None
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname, f'sub-{sub}')
    betafiles = [
        pjoin(betasdir, f'ses-things{ses_i:02d}', f'sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_betas.nii.gz')
        for run_i in range(1, 11)
    ]
    for b in betafiles:
        assert os.path.exists(b)
    with Parallel(n_jobs=-1) as parallel:
        betas_l = parallel(
            delayed(apply_mask_smoothed)(bf, mask, smoothing, dtype)
            for bf in tqdm(betafiles, desc='loading betas')
        )
    betas = np.vstack(betas_l)
    return betas  # shape (ntrials, nvoxel)

def beta_driver(session):
    #extracting the union mask
    sessions = [f'things{i:02d}' for i in range(1, nsessions)]
    subject='01'#CHANGE
    bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_'
    subj_prepdir=pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{subject}')
    union_mask=get_masks(subject)
    betas_derivname='betas_run01/unregularized'#CHANGE
    #loading the beta weights
    betas=load_betas(subject,union_mask,bidsroot,betas_derivname,session)
    return betas
if __name__=='__main__':
    beta_driver('things01')