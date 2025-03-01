"""
Essential foor RDM creation
Contains helper functions for running RDM creation
"""
import glob
import os
import random
import time
from itertools import combinations
from os.path import join as pjoin
from shutil import copyfile

import numpy as np
import pandas as pd
import json
#from fracridge import FracRidgeRegressor
from joblib import Parallel, delayed
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import load_img
from nilearn.masking import unmask, apply_mask
from numpy.random import normal
from scipy.linalg import block_diag
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from nilearn.masking import apply_mask, intersect_masks, unmask
#from glm import THINGSGLM, df_to_boxcar_design, get_nuisance_df
from utils import pearsonr_nd, get_hrflib, spearman_brown, match_scale, apply_mask_smoothed, regress_out, r2_ndarr
from rdm_helper_rois import mask_driver2
from nilearn import image


#definining the functions

def union_masker(func):
    #creating the decorator function for reading the mask and 
    #returning the union mask
    #creating the session names
    sessions = [f'things{i:02d}' for i in range(1, 13)]
    bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_'
    subject='01'
    subj_prepdir=pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{subject}')
    def wrapper(subject,subj_prepdir,sessions):
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
    betasdir = pjoin(bidsroot, 'beta_derivatives', betas_derivname, f'sub-{sub}')
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

def beta_driver(session,betas_derivname):
    #extracting the union mask
    sessions = [f'things{i:02d}' for i in range(1, 13)]
    subject='01'#CHANGE
    bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_'
    subj_prepdir=pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{subject}')
    union_mask=get_masks(subject,subj_prepdir,sessions)

    #loading the beta weights
    betas=load_betas(subject,union_mask,bidsroot,betas_derivname,session)
    #print(betas.shape)
    return betas

def beta_driver2(session,betas_derivname,transform_mask=False,
                bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_',subject='01'):
    sessions = [f'things{i:02d}' for i in range(1, 13)]
    subj_prepdir=pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{subject}')
    union_mask=get_masks(subject,subj_prepdir,sessions)
    if transform_mask:
        #read an roi mask
        _,_,mask_img_left,_=mask_driver2(subject,'EBA',transform=not(transform_mask),return_masks=True)
        #transform the union mask to the roi mask  
        union_mask=image.resample_to_img(union_mask,mask_img_left,interpolation='nearest')
    #loading the beta weights 
    betas=load_betas(subject,union_mask,bidsroot,betas_derivname,session)
    return betas 
    



def get_beta_weights_subject(
        sub: str = '01',
        bidsroot: str = '/LOCAL/ocontier/thingsmri/bids',
        betas_dir: str = '/LOCAL/ocontier/thingsmri/bids/derivatives/betas',
        betas_derivname: str = 'betas_run01/unregularized',
transform=False) -> tuple:
    #pjoin(betas_basedir, f'sub-{sub}')#creating betas basedir for subvject 
    betas,condname_list = {},[]#specifying empty list for betas
    nsessions=12#SET
    for ses_i in tqdm(range(nsessions), desc='Loading betas of repeated stimuli'):
        sesdir = pjoin(betas_dir, f'ses-things{ses_i + 1:02d}')#specifying session directory for betas
        rawdir = pjoin(bidsroot, 'rawdata', f'sub-{sub}', f'ses-things{ses_i + 1:02d}', 'func')#specifying raw data directory for fmri data
        event_tsvs = [
            pjoin(rawdir, f'sub-{sub}_ses-things{ses_i + 1:02d}_task-things_run-{run_i + 1:02d}_events.tsv')
            for run_i in range(10)
        ]#creating the event tsvs file path for each run[ in the raw data directory]
        cond_tsvs = [
            pjoin(sesdir, f'sub-{sub}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_conditions.tsv')
            for run_i in range(10)
        ]#creating the condition tsvs file path for each run [in the beta directory]
        conds_series = pd.concat([pd.read_csv(tsv, sep='\t')['image_filename'] for tsv in cond_tsvs])#extracting the conditions for each run and adding to list
        conds=conds_series.to_numpy(dtype=str)#converting the conditions to numpy array
        #conds=pd.concat([pd.read_csv(tsv, sep='\t')['image_filename'].str.replace(r'\/.*','_ses_'+str(ses_i+1)) for tsv in cond_tsvs]).sort_values().to_numpy()
        event_dfs = pd.concat([pd.read_csv(tsv, sep='\t') for tsv in event_tsvs])#extracting the events for each run and adding to list
        #event_dfs=pd.concat([pd.read_csv(tsv, sep='\t') for tsv in{self.ds.target_sessions}_tsvs]).sort_values(by='file_path',ignore_index=True)
        #if ses_i == 0:#if session is 0 i.e beginning of the for loop extract the names for the repeated conditions
        expcondnames = event_dfs[event_dfs['trial_type'] == 'exp']['file_path'].to_numpy(dtype=str)#extracting the file path for the test trials
        #extracting the experimental condition names
        run_niis = [pjoin(sesdir, f'sub-{sub}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz')
                    for run_i in range(10)]#creating nifti file paths for each set of beta weights
        ses_betas=beta_driver2(ses_i+1,betas_derivname,transform_mask=transform,bidsroot=bidsroot)
        #vectorized alternative
        expcond_is=np.where(np.repeat(conds.reshape(-1,conds.shape[0]),expcondnames.shape[0],axis=0)==expcondnames.reshape(expcondnames.shape[0],-1))[1]
        exp_betas = ses_betas[expcond_is]#extracting the beta weights for the repeated conditions
        cond_names=conds[expcond_is]#extracting the condition names for the experiment condition
        #sorting the beta weights and condition names
        exp_betas=exp_betas[np.argsort(cond_names)]
        cond_names_sorted=conds_series.str.replace(r'\/.*','_ses_'+str(ses_i+1)).to_numpy(dtype=str)[expcond_is][np.argsort(cond_names)]
        #flattening the beta weights
        betas['ses_'+str(ses_i+1)]=exp_betas#adding the beta weights to the list[in the end list will contain all the beta weights for all the repeated conditions across all sessions]
        condname_list.append(cond_names_sorted)#adding the condition names to the list
    return betas, condname_list#returning the beta weights and the condition names/




def custom_concat(ndarray_list:list)->np.ndarray:
    """custom concatenation of numpy arrays i.e interleaving the arrays
    Args:A list of 2D numpy arrays or 1D arrays
    Returns: A numpy array : 2D if the input arrays are 2D, 1D if the input arrays are 1D
    Assumptions: All the arrays in the list have the same shape across the second dimension"""
    #checking if the ndarrays are 1D 
    if ndarray_list[0].ndim==1:
        #reshaping into 2D vectors
        ndarray_list=[arr.reshape(-1,1) for arr in ndarray_list]
        one_d=True
    else:one_d=False
    #dstacking the arrays
    arr=np.dstack(ndarray_list)
    #trasposing the 2nd and 3rd axes
    arrT=np.moveaxis(arr,1,-1)
    #reshaping the array
    arr_concat=arrT.reshape(-1,arrT.shape[-1])
    if one_d:return arr_concat.squeeze()
    else:return arr_concat


def save_list(list_to_save:list,save_path:str,file_name:str):
    """saves a list to a .json file"""
    with open(pjoin(save_path,file_name+'.json'),'w') as f:
        json.dump(list_to_save,f)
def save_ndarray_list(list_to_save:list,save_path:str,file_name:str):
    """saves a list of numpy arrays to a .npz file"""
    np.savez_compressed(pjoin(save_path,file_name+'.npz'),*list_to_save)
def save_ndarray_dict(dict_to_save:dict,save_path:str,file_name:str):
    """saves a dictionary of numpy arrays to a .npz file"""
    np.savez_compressed(pjoin(save_path,file_name+'.npz'),**dict_to_save)
def save_ndarray(ndarray_to_save:np.ndarray,save_path:str,file_name:str):
    """saves a numpy array to a .npy file"""
    np.save(pjoin(save_path,file_name+'.npy'),ndarray_to_save)