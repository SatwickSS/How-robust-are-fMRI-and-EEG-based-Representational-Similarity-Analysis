#! /usr/env/python

import copy
import glob
import io
import json
import os
import subprocess
import sys
from os.path import abspath
from os.path import join as pjoin
from os.path import pardir

import numpy as np
import nilearn
import pandas as pd

#import cmasher as cmr#CRITICAL
from scipy.io import loadmat
import requests
import scipy
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from nilearn.image import resample_to_img, math_img, smooth_img
from nilearn.masking import intersect_masks, apply_mask
from nipype.interfaces.ants.resampling import ApplyTransforms
from scipy.signal import periodogram
from scipy.stats import sem
import matplotlib.pylab as pl
from sklearn.linear_model import LinearRegression




def psc(a: np.array, timeaxis: int = 0) -> np.array:
    """rescale array with fmri data to percent signal change (relative to the mean of each voxel time series)"""
    return 100 * ((a / a.mean(axis=timeaxis)) - 1)



def ci_array(a, confidence=.95, alongax=0):
    """Returns tuple of upper and lower CI for mean along some axis in multidimensional array"""
    m, se = np.mean(a), sem(a, axis=alongax)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., a.shape[alongax] - 1)
    return m - h, m + h


def spearman_brown(corrs: np.ndarray):
    """Spearman-Brown correction for split-half reliability. <corrs> can be any shape."""
    return (2 * corrs) / (1 + corrs)

def apply_mask_smoothed(nifti, mask, smoothing=0., dtype=np.single):
    return apply_mask(smooth_img(nifti, smoothing), mask, dtype=dtype)

def match_scale(original: np.ndarray, target: np.ndarray, alongax: int = 0) -> np.ndarray:
    """
    Rescale 'original' to match the mean and variance of 'target'.
    The result will be the values of 'original', distributed with  mean and variance of 'target'.
    Both input arrays can have any shape, 'alongax' specifies the axis along which mean and SD is considered.
    """
    # matched = zscore(original, axis=alongax)
    # matched *= target.std(axis=alongax)
    # matched += target.mean(axis=alongax)
    matched = (original / original.std(axis=alongax)) * target.std(axis=alongax)
    return matched
    
def pearsonr_nd(arr1: np.ndarray, arr2: np.ndarray, alongax: int = 0) -> np.ndarray:
    """
    Pearson correlation between respective variables in two arrays.
    arr1 and arr2 are 2d arrays. Rows correspond to observations, columns to variables.
    Returns:
        correlations: np.ndarray (shape nvariables)
    """
    # center each feature
    arr1_c = arr1 - arr1.mean(axis=alongax)
    arr2_c = arr2 - arr2.mean(axis=alongax)
    # get sum of products for each voxel (numerator)
    numerators = np.sum(arr1_c * arr2_c, axis=alongax)
    # denominator
    arr1_sds = np.sqrt(np.sum(arr1_c ** 2, axis=alongax))
    arr2_sds = np.sqrt(np.sum(arr2_c ** 2, axis=alongax))
    denominators = arr1_sds * arr2_sds
    return numerators / denominators


def r2_ndarr(x, y, alongax=-1):
    """Calculate the coefficient of determination in y explained by x"""
    ssres = np.nansum(np.square(y - x), axis=alongax)
    sstot = np.nansum(np.square(
        y - np.expand_dims(y.mean(axis=alongax), axis=alongax)
    ), axis=alongax)
    return 100 * (1 - np.nan_to_num(ssres) / np.nan_to_num(sstot))


def df_from_url(url: str, sep: str, header):
    s = requests.get(url).content
    return pd.read_csv(
        io.StringIO(s.decode('utf-8')),
        sep=sep, header=header
    )
    
def get_hrflib(
        hrflib_url: str,
        rescale_amplitude: bool = True,
        resample_to_tr: bool = False,
        tr: float = 1.5,
        dtype=np.single,
) -> np.ndarray:
    """
    Get HRF library from Kendrick Kay's github repository.
    optionally rescale amplitudes of all HRFs to 1 (recommended) and resample to a specific TR (not recommended).
    """
    while True:
        try:
            hrflib = np.array(df_from_url(hrflib_url, sep='\t', header=None))
            break
        except Exception as e:
            continue
    if resample_to_tr:  # resample to our TR
        sampleinds = np.arange(0, hrflib.shape[0], tr * 10, dtype=np.int16)
        hrflib = hrflib[sampleinds, :]
    if rescale_amplitude:  # rescale all HRFs to a peak amplitude of 1
        hrflib = hrflib / np.max(hrflib, axis=0)
    return hrflib.astype(dtype)



def regress_out(
        x: np.ndarray, y: np.ndarray, dtype=np.single,
        lr_kws: dict = dict(copy_X=True, fit_intercept=True, normalize=True, n_jobs=-1)
) -> np.ndarray:
    reg = LinearRegression(**lr_kws)
    reg.fit(x, y)
    resid = y - reg.predict(x)#fit x to the model 
    return resid.astype(dtype)#predict y from the fitted model and take the residuals 
