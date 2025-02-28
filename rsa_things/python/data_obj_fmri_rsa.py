"""
Creates RDMs frmo betas
"""
# relevant imports
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset


 # now create a dataset object
des = {'session': 1, 'subj': 1}
nCond,nVox=5,10
measurements = np.random.randn(nCond,nVox)
obs_des = {'conds': np.array(['cond_' + str(x) for x in np.arange(nCond)])}
chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
#obs_des = {'conds': np.array(['cond_' + str(x) for x in np.arange(1,nCond+1)])} # indices␣˓ →from 1
#chn_des = {'conds': np.array(['voxel' + str(x) for x in np.arange(1,nVox+1)])} # indices␣˓ →from 1
data = rsd.Dataset(measurements=measurements,
descriptors=des,
obs_descriptors=obs_des,
channel_descriptors=chn_des)


#subsetting the data by condition
# select a subset of the dataset: select data only from conditions 0:2
sub_data = data.subset_obs(by='conds', value=['cond_'+str(i) for i in range(2)])
print(sub_data)


import glob
import os
import random
import time
from itertools import combinations
from os.path import join as pjoin
from shutil import copyfile

import numpy as np
import pandas as pd
from fracridge import FracRidgeRegressor
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

from glm import THINGSGLM, df_to_boxcar_design, get_nuisance_df
from utils import pearsonr_nd, get_hrflib, spearman_brown, match_scale, apply_mask_smoothed, regress_out, r2_ndarr



#definining the functions



def get_beta_weights_subject(
        sub: str = '01',
        #bidsroot: str = '/LOCAL/ocontier/thingsmri/bids',
        betas_basedir: str = '/LOCAL/ocontier/thingsmri/bids/derivatives/betas',
) -> tuple:
    betas_dir = pjoin(betas_basedir, f'sub-{sub}')#creating betas basedir for subvject 
    betas = []#specifying empty list for betas
    for ses_i in tqdm(range(12), desc='Loading betas of repeated stimuli'):
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
        conds = np.hstack([pd.read_csv(tsv, sep='\t')['image_filename'].to_numpy() for tsv in cond_tsvs])#extracting the conditions for each run and adding to list
        event_dfs = [pd.read_csv(tsv, sep='\t') for tsv in event_tsvs]#extracting the events for each run and adding to list
        if ses_i == 0:#if session is 0 i.e beginning of the for loop extract the names for the repeated conditions
            repcondnames = np.unique(np.hstack(
                [df[df['trial_type'] == 'test']['file_path'].to_numpy(dtype=str)
                 for df in event_dfs]#extracting the file path for the test trials
            ))#extracting the repeated condition names
        run_niis = [pjoin(sesdir, f'sub-{sub}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz')
                    for run_i in range(10)]#creating nifti file paths for each set of beta weights
        ses_betas = np.moveaxis(
            np.concatenate([load_img(ni).get_fdata(dtype=np.single) for ni in run_niis], axis=3),
            -1, 0
        )#loading the data into a numpy array and concatenating the data for all images across all runs in a session
        repcond_is = np.hstack([np.where(conds == repcond) for repcond in repcondnames]).squeeze()#extracting the indices for the repeated conditions
        #vectorized alternative
        #np.where(np.repeat(conds.reshape(-1,conds.shape[0]),100,axis=0)==repcondnames.reshape(repcondnames.shape[0],-1))[1]
        rep_betas = ses_betas[repcond_is]#extracting the beta weights for the repeated conditions
        betas.append(rep_betas)#adding the beta weights to the list[in the end list will contain all the beta weights for all the repeated conditions across all sessions]
    betas = np.moveaxis((np.moveaxis(np.stack(betas), 0, -1)), 0, -1)  # shape (voxX, voxY, nRepetitions, nStimuli)
    return betas, run_niis[0]
