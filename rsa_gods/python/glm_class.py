
import copy
import os
import sys
import warnings
from os.path import join as pjoin
from os.path import pardir
#from memory_profiler import profile
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import load_img, concat_imgs
from nilearn.masking import apply_mask, intersect_masks, unmask
from scipy.linalg import block_diag
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(pjoin(pardir))





def load_masked(bold_file, mask, rescale='psc', dtype=np.single):
    if rescale == 'psc':
        return np.nan_to_num(psc(apply_mask(bold_file, mask, dtype=dtype)))
    elif rescale == 'z':
        return np.nan_to_num(zscore(apply_mask(bold_file, mask, dtype=dtype), nan_policy='omit', axis=0))
    elif rescale == 'center':
        data = np.nan_to_num(apply_mask(bold_file, mask, dtype=dtype))
        data -= data.mean(axis=0)
    else:
        return apply_mask(bold_file, mask, dtype=dtype)

from dataset import GodsMRIdataset
class GODGLM(object):
    """
    Parent class for different GLMs to run on the god mri dataset,
    mostly handling IO.
    """

    def __init__(self,
                 bidsroot: str,
                 subject: str,
                 out_deriv_name: str = 'glm',
                 noiseregs: list = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                    'framewise_displacement'],
                 acompcors: bool or int = 10,
                 include_all_aroma: bool = False,
                 # include_manual_ica: bool = False,
                 hrf_model: str or None = 'spm + derivative',
                 noise_model: str = 'ols',
                 high_pass: float = .01,
                 sigscale_nilearn: bool or int or tuple = False,
                 standardize: bool = True,
                 verbosity: int = 3,
                 nruns_perses: int = 10,
                 nprocs: int = 1,
                 lowmem=False,
                 ntrs: int = 178,
                 tr: float = 3,
                 drift_model: str = 'cosine',
                 drift_order: int = 4,
                 fwhm: bool or None = None,
                 overwrite: bool = False,
                 stc_reftime: float = 1.5,
                 target_session='perceptionTest',
                 task='perception',
                 prepdir='fmriprep_run-04',
                 ):
        self.bidsroot = os.path.abspath(bidsroot)
        self.include_all_aroma = include_all_aroma
        #self.include_manual_ica = include_manual_ica
        self.subject = subject
        self.out_deriv_name = out_deriv_name
        self.verbosity = verbosity
        self.lowmem = lowmem
        self.nprocs = nprocs
        self.acompcors = acompcors
        self.tr = tr
        self.ntrs = ntrs
        self.nruns_perses = nruns_perses
        self.high_pass = high_pass
        self.hrf_model = hrf_model
        #self.noise_model = noise_model
        self.drift_model = drift_model
        self.drift_order = drift_order
        self.sigscale_nilearn = sigscale_nilearn
        self.standardize = standardize
        self.fwhm = fwhm
        self.stc_reftime = stc_reftime  # fmriprep interpolates to mean of all slice times
        self.overwrite = overwrite
        self.task=task
        self.target_session=target_session
        self.ds = GodsMRIdataset(self.bidsroot,ses_type=self.target_session,sub_id=self.subject)
        self.n_sessions = len(self.ds.target_sessions)
        self.nruns_perses_={ses_id: len(self.ds.layout.get(subject=self.subject,session=ses_id,return_type='id',target='run')) for ses_id in self.ds.target_sessions}
        self.nruns_total = sum([len(self.ds.layout.get(subject=self.subject,session=ses_id,return_type='id',target='run')) for ses_id in self.ds.target_sessions]) 
        self.subj_prepdir = pjoin(bidsroot, 'derivatives', prepdir, f'sub-{self.subject}')
        self.subj_outdir = pjoin(bidsroot, 'derivatives', out_deriv_name, f'sub-{self.subject}')
        #self.icalabelled_dir ='/DATA/satwick22/Documents/fMRI/fMRI_processing/trash/noise_reg/sub-01' 
        if not os.path.exists(self.subj_outdir):
            os.makedirs(self.subj_outdir)
        if acompcors:
            noiseregs += [f'a_comp_cor_{i:02}' for i in range(self.acompcors)]
        self.noiseregs = noiseregs
        self.frame_times_tr = np.arange(0, self.ntrs * self.tr, self.tr) + self.stc_reftime
        # get image dimensions
        example_img = load_img(self.ds.layout.get(session='perceptionTest01', extension='nii.gz',
                                                  suffix='bold', subject=self.subject)[0].path)
        self.nx, self.ny, self.nz, self.ntrs = example_img.shape
        self.n_samples_total, self.nvox_masked, self.union_mask = None, None, None

    def _get_events_files(self):
        event_files=[]
        for sesname in self.ds.target_sessions:
            for run in range(1,self.nruns_perses_[sesname]+1):
                file_name=pjoin(self.bidsroot,'rawdata', f'sub-{self.subject}', f'ses-{sesname}', 'func',f'sub-{self.subject}_ses-{sesname}_task-{self.task}_run-{run:02d}_events.tsv')
                event_files.append(file_name)
        return event_files

    def _get_bold_files(self):
        bold_files=[]
        for sesname in self.ds.target_sessions:
            for run in range(1,self.nruns_perses_[sesname]+1):
                file_name=pjoin(self.subj_prepdir, f'ses-{sesname}', 'func',f'sub-{self.subject}_ses-{sesname}_task-{self.task}_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz')
                bold_files.append(file_name)
        return bold_files

    def _get_masks(self):
        masks=[]
        for sesname in self.ds.target_sessions:
            for run in range(1,self.nruns_perses_[sesname]+1):
                file_name=pjoin(self.subj_prepdir, f'ses-{sesname}', 'func',f'sub-{self.subject}_ses-{sesname}_task-{self.task}_run-{run:02d}_space-T1w_desc-brain_mask.nii.gz')
                masks.append(file_name)
        return masks

    def _get_nuisance_tsvs(self):
        nuisance_tsvs=[]
        for sesname in self.ds.target_sessions:
            for run in range(1,self.nruns_perses_[sesname]+1):
                file_name=pjoin(self.subj_prepdir, f'ses-{sesname}', 'func',f'sub-{self.subject}_ses-{sesname}_task-{self.task}_run-{run:02d}_desc-confounds_timeseries.tsv')
                nuisance_tsvs.append(file_name)
        return nuisance_tsvs

    def _get_ica_txts(self):
        ica_txts=[]
        for sesname in self.ds.target_sessions:
              for run in range(1,11):
                file_name=pjoin(self.subj_prepdir, f'ses-{sesname}', 'func',f'sub-{self.subject}_ses-{sesname}_task-{self.task}_run-{run:02d}_desc-ica_timeseries.txt')
                if os.path.exists(file_name):
                     ica_txts.append(file_name)
        return ica_txts

    def get_inputs(self):
        event_files = self._get_events_files()
        bold_files = self._get_bold_files()
        masks = self._get_masks()
        nuisance_tsvs = self._get_nuisance_tsvs()
        assert len(event_files) == len(bold_files) == len(nuisance_tsvs) == len(
            masks), f'\ninputs have unequal length\n'
        return event_files, bold_files, nuisance_tsvs, masks

    def add_union_mask(self, masks):
        """Create a union mask based on the run-wise brain masks"""
        self.union_mask = intersect_masks(masks, threshold=0)#taking the union of the masks
    #@profile
    def vstack_data_masked(self, bold_files, rescale_runwise='psc', rescale_global='off', dtype=np.single):
        import psutil
        #current_process = psutil.Process()
        #subproc_before = set([p.pid for p in current_process.children(recursive=True)])
        #paralleize the processes using multiprocessing
        #arrs = Pool(5).map(load_masked, bold_files, self.union_mask, rescale_runwise, dtype)
        #arrs=[load_masked(bf, self.union_mask, rescale_runwise, dtype) for bf in bold_files]
        with Parallel(n_jobs=20) as parallel:
            arrs = parallel(
            delayed(load_masked)(bf, self.union_mask, rescale_runwise, dtype) for bf in tqdm(bold_files, desc='Intra Batch Progress'))
        #loading the masked data
        #subproc_after = set([p.pid for p in current_process.children(recursive=True)])
        #for subproc in subproc_after - subproc_before:
        #    print('Killing process with pid {}'.format(subproc))
        #    psutil.Process(subproc).terminate()
        #converting the bold signal if required [last option(else) is the default signal formal]
        if rescale_global == 'psc':#converting to percent signal change 
            data = np.nan_to_num(psc(np.vstack(arrs)))#nan_to_num replaces nan with 0 and inf with large finite numbers after percent signal change conversion
        elif rescale_global == 'z':
            data = np.nan_to_num(zscore(np.vstack(arrs), nan_policy='omit', axis=0))
        elif rescale_global=='off':
            data = np.vstack(arrs)
        elif rescale_global=='re-center':
            data = np.vstack(arrs)
            data -= data.mean(axis=0)
        self.nvox_masked = data.shape[1]#number of voxels left after applying the brain mask
        return data.astype(dtype)#returning the data for all the runs across all the sessions

    def load_data_concat_volumes(self, bold_files):
        print('concatinating bold files')
        bold_imgs = [nib.Nifti2Image.from_image(load_img(b)) for b in tqdm(bold_files, 'loading nifti files')]
        return concat_imgs(bold_imgs, verbose=self.verbosity)

    def init_glm(self, mask):
        print(f'instantiating model with nprocs: {self.nprocs}')
        return FirstLevelModel(
            minimize_memory=self.lowmem, mask_img=mask, verbose=self.verbosity, noise_model=self.noise_model,
            t_r=self.tr, standardize=self.standardize, signal_scaling=self.sigscale_nilearn, n_jobs=self.nprocs,
            smoothing_fwhm=self.fwhm,
        )