#/home/satwick22/miniconda3/envs/fmri/bin/python
"""
Performs ICA Melodic
"""

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


def grab_data(subject,bidsroot,fmriprepdir,derivdir,space='func_preproc'):
    ds = ThingsMRIdataset(bidsroot)#creating the datastructure from the BIDS format directory strcuture of the raw data
    bidsobjs = ds.layout.get(subject=subject, suffix='bold', extension='.nii.gz')#extracting a list of files with the keywords as mentioned 
    space_str = {'T1w': '_space-T1w', 'func_preproc': ''}#specifiying space dictionary for mapping the space to the file names
    assert space in space_str #checking if the space is in the space_str dictionary
    
    
    #things to specify for boldfiles : fmriprep output directory, subject
    boldfiles = [abspath(pjoin(
            fmriprepdir, f"ses-{bidsobj.entities['session']}", 'func',
            f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_run-{bidsobj.entities['run']}{space_str[space]}_desc-preproc_bold.nii.gz")) 
            if bidsobj.entities['task'] != 'rest' else 
            abspath(pjoin(
            fmriprepdir, f"ses-{bidsobj.entities['session']}", 'func',
            f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_acq-reversePE{space_str[space]}_desc-preproc_bold.nii.gz")) 
            for bidsobj in bidsobjs ]#making a list of all the preprocessed bold files
    

    #need to specify the fmriprep output directiory and subject for extracting the mask filenames properly
    masks = [abspath(pjoin(
            fmriprepdir, f"ses-{bidsobj.entities['session']}", 'func',
            f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_run-{bidsobj.entities['run']}{space_str[space]}_desc-brain_mask.nii.gz")) 
            if bidsobj.entities['task'] != 'rest' else 
            abspath(pjoin(
            fmriprepdir, f"ses-{bidsobj.entities['session']}", 'func',
            f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_acq-reversePE{space_str[space]}_desc-brain_mask.nii.gz")) 
            for bidsobj in bidsobjs ]#making a list of all the preprocessed brain masks
    
    
    #specify a deriv directory and subject for the output directory mapping
    outdirs = [abspath(pjoin(
            derivdir, f'space-{space}', f'sub-{subject}', f"ses-{bidsobj.entities['session']}",
            f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_run-{bidsobj.entities['run']}_melodic"))
            if bidsobj.entities['task']!='rest' else 
            abspath(pjoin(
            derivdir, f'space-{space}', f'sub-{subject}', f"ses-{bidsobj.entities['session']}",
            f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_acq-reversePE_melodic"))
            for bidsobj in bidsobjs]#making a list of the path for the output files
    import re
    #changing all the run values like run-X to run-0X if X i from [1,9]
    boldfiles=list(map(lambda boldfile:re.sub(r'run-([1-9]_)', r'run-0\1', boldfile),boldfiles))
    masks=list(map(lambda mask:re.sub(r'run-([1-9]_)', r'run-0\1', mask),masks))
    outdirs=list(map(lambda outdir:re.sub(r'run-([1-9]_)', r'run-0\1', outdir),outdirs))
    
    for o in outdirs:#creating the output directory if one already does not exist
        if not os.path.exists(o):
            os.makedirs(o)
    
    
    
    return boldfiles, masks, outdirs#returns the three list with the filenames as mentioend above





def calc_susan_thresh(boldfile, maskfile, timeax=0, median_factor=.75):
    """
    Calculate the median value within brainmask and multiply with fixed factor to get an estimate of the contrast
    between background and brain for FSL's SUSAN.
    """
    from nilearn.masking import apply_mask
    import numpy as np
    data = apply_mask(boldfile, maskfile)
    med = np.median(data.mean(axis=timeax))
    del data  # prevent memory overuse
    return med * median_factor


def run_melodic_wf(
        subject, bidsroot, fmriprepdir,
        space='func_preproc',
        approach: str = 'runwise',
        fwhm: float = 4., hpf_sec: float = 120.,
        derivname: str = 'melodic_run_n1',#SET
        nprocs: int = 16, melodic_mem: float = 64.,
        tr: float = 1.5, try_last_n_runs: bool or int = False,
):
    """
    Run Melodic on the preprocessed functional images.
    Besides fwhm and hpf for additional preprocessing, user can choose reference space (func_preproc or T1w) and
    approach (runwise or concat). Note that concat requires data to be in T1w space.

    Example:
        import sys
        subject = sys.argv[1]
        bidsroot = abspath(pjoin(pardir, pardir, pardir))
        fmriprepdir = pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{subject}')
        run_melodic_wf(
            subject, bidsroot, fmriprepdir, approach='runwise', space='T1w',
        )
    """
    wf_bdir = pjoin(bidsroot, 'melodicwf_wdir_nrun1', f'space-{space}', f'sub-{subject}')#creating the working directory for melodic ica
    derivdir = pjoin(bidsroot, 'derivatives', derivname, approach)#*
    for d in [wf_bdir, derivdir]:#creating the directories if they do not exist
        if not os.path.exists(d):
            os.makedirs(d)
    boldfiles, masks, outdirs = grab_data(subject, bidsroot, fmriprepdir, derivdir, space=space)#getting the files structrure for melodiccd  
    if try_last_n_runs:#extracts the last n runs of the data only [disabled for our analysis as we have multiple sessions]
        boldfiles = boldfiles[-try_last_n_runs:]
        masks = masks[-try_last_n_runs:]
        outdirs = outdirs[-try_last_n_runs:]

    wf = Workflow(name='melodicwf',
                  base_dir=wf_bdir)#creating the execution pipeline controller
    calcthresh = MapNode(Function(function=calc_susan_thresh,
                                  input_names=['boldfile', 'maskfile'], output_names=['smooth_thresh']),
                         name='calcthresh', iterfield=['boldfile', 'maskfile'])#creating the iterable interface objects' wrapper  for the nodes ofr calc_thresh

    calcthresh.inputs.boldfile = boldfiles #mapping the input files to the iterable interface objects
    calcthresh.inputs.maskfile = masks #mapping the input files to the iterable interface objects
    susan = MapNode(SUSAN(fwhm=fwhm), iterfield=['in_file', 'brightness_threshold'], name='susan') # ... for smoothing the data
    susan.inputs.in_file = boldfiles #mapping the input files to the iterable interface objects
    tfilt = MapNode(TemporalFilter(highpass_sigma=float(hpf_sec / tr)), iterfield=['in_file'], name='tfilt')# ... for high pass filter 
    calcthresh.inputs.maskfile = masks #mapping the input files to the iterable interface objects
    if approach == 'runwise':
        melodic = MapNode(
            MELODIC(tr_sec=tr, out_all=True, no_bet=True, report=True), iterfield=['in_files', 'mask', 'out_dir'],
            name='melodic_runwise')
        melodic.inputs.mask = masks #maping mask files 
        melodic.inputs.out_dir = outdirs # mapping the output directiory for the melodic
    elif approach == 'concat':
        melodic = Node(MELODIC(tr_sec=tr, out_all=True, no_bet=True, report=True, approach='concat', args='--debug'),
                       name='melodic_concat', mem_gb=melodic_mem, n_procs=nprocs)
        outdir = abspath(pjoin(derivdir, f'space-{space}', f'sub-{subject}', 'concat'))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        umask_img = intersect_masks(masks, threshold=0)
        umask_img.to_filename(pjoin(wf_bdir, 'umask.nii.gz'))
        melodic.inputs.out_dir = outdir
        melodic.terminal_output = 'stream'
        melodic.inputs.mask = pjoin(wf_bdir, 'umask.nii.gz')
    else:
        raise ValueError(f'"approach" must be in ["runwise", "concat"]')
    wf.connect([
        (calcthresh, susan, [('smooth_thresh', 'brightness_threshold')]),
        (susan, tfilt, [('smoothed_file', 'in_file')]),
        (tfilt, melodic, [('out_file', 'in_files')]),
    ])# creating the workflow by sending in the node order and the connections between them
    wf.run(plugin='MultiProc', plugin_args=dict(n_procs=nprocs)) #executing the pipeline 





if __name__ == '__main__':
    import os
    #setting fsl environment variables
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    os.environ['FSLDIR'] = '/home/satwick22/fsl'
    #make fsl executable available in the path
    os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']
    import sys
    space='func_preproc'
    subject = '01'
    bidsroot = '/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/'
    fmriprepdir = '/DATA/satwick22/Documents/fMRI/fMRI_processing/bids/derivatives/fmriprep/sub-01/'
    run_melodic_wf(
        subject, bidsroot, fmriprepdir, approach='runwise', space='T1w',
        )