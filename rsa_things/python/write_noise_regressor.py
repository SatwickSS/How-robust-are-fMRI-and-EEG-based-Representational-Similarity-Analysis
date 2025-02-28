#function for creating noise regressor based on the specified thresholds



import os
from os import pardir
from os.path import abspath
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
from matplotlib.gridspec import GridSpec
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import load_img, threshold_img, math_img, resample_to_img
from nilearn.masking import intersect_masks
from nipype.interfaces.fsl.maths import TemporalFilter
from nipype.interfaces.fsl.model import MELODIC
from nipype.interfaces.fsl.preprocess import SUSAN
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import MapNode, Workflow, Node
from numpy.random import choice as rndchoice
from scipy.ndimage.morphology import binary_erosion
from scipy.signal import periodogram
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from dataset import ThingsMRIdataset
from utils import calc_hfc, get_render3_cmap


def write_noise_tsvs(
        outdir,
        melodic_features_tsv,
        bidsroot,
        thresholds=dict(edgefrac=.225, hfc=.4),
        combine_thresholds='any',
):
    # load component features
    feature_df = pd.read_csv(melodic_features_tsv, sep='\t')
    # mask by thresholds
    masks = np.vstack([
        np.array(feature_df[feature] > threshold)
        for feature, threshold in thresholds.items()
    ]).T
    if combine_thresholds == 'any':
        combined_mask = np.any(masks, axis=1)
    elif combine_thresholds == 'all':
        combined_mask = np.all(masks, axis=1)
    else:
        raise ValueError('combine_thresholds must be "all" or "any"')
    feature_df['noise'] = 0
    feature_df.loc[combined_mask, 'noise'] = 1
    # only take rows for noise components
    noise_df = feature_df.loc[feature_df['noise'] == 1,]
    # Get time series for each noise component and make out file names

    def _get_comp_ts(row, bidsroot=bidsroot):
        melodic_basedir = pjoin(bidsroot, 'derivatives', 'melodic_run3', 'runwise', 'space-T1w',
                                f"sub-{row['subject']:02d}")
        melodic_outdir = pjoin(
            melodic_basedir, f"ses-{row['session']}",
            f"sub-{row['subject']:02d}_ses-{row['session']}_task-{row['task']}_run-{row['run']:02d}_melodic"
        )
        mixmat = np.loadtxt(pjoin(melodic_outdir, 'melodic_mix'))
        return mixmat[:, row['comp_i']]

    def _get_out_txt(row, outdir=outdir):
        out_base=f"sub-{row['subject']:02d}/ses-{row['session']}/" 
        if not os.path.exists(pjoin(outdir,out_base)):
            os.makedirs(pjoin(outdir,out_base)) 
        #out_txt_basename = f"sub-{row['subject']:02d}/ses-{row['session']}/sub-{row['subject']:02d}_ses-{row['session']}_task-{row['task']}_run-{row['run']:02d}.txt"
        out_txt_basename = pjoin(out_base,f"sub-{row['subject']:02d}_ses-{row['session']}_task-{row['task']}_run-{row['run']:02d}.txt")
        return pjoin(outdir, out_txt_basename)

    timeseries = noise_df.apply(_get_comp_ts, axis=1)
    noise_df['comp_ts'] = timeseries
    out_txts = noise_df.apply(_get_out_txt, axis=1)
    noise_df['out_txt'] = out_txts
    # save to file
    for out_txt in tqdm(noise_df.out_txt.unique(), desc='saving to file'):
        outdir = pjoin(*out_txt.split('/')[:-1])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        run_df = noise_df[noise_df['out_txt'] == out_txt]
        print(f'found {len(run_df)} noise components for this run')
        noise_arr = np.vstack(run_df.comp_ts.to_list()).T
        np.savetxt(out_txt, noise_arr)
    return None

#calling the function
write_noise_tsvs(outdir='/DATA1/satwick22/Documents/fMRI/derivatives/noise_reg',melodic_features_tsv='/DATA1/satwick22/Documents/fMRI/thingsmri_/derivatives/melodic_features_run3/melodic_correlations_space-T1w.tsv',bidsroot='/DATA1/satwick22/Documents/fMRI/thingsmri_')
