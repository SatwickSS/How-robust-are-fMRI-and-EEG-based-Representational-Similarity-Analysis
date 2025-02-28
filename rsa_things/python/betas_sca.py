
"""
Perform SCA using specifications list
"""
import gc
import glob
import os
import random
import time
from itertools import combinations
from os.path import join as pjoin
from shutil import copyfile
import pickle 
import numpy as np
import pandas as pd
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
from scipy import stats

from nilearn import image, plotting
from nilearn.masking import intersect_masks

from glm import THINGSGLM, df_to_boxcar_design, get_nuisance_df
from utils import pearsonr_nd, get_hrflib, spearman_brown, match_scale, apply_mask_smoothed, regress_out, r2_ndarr
from rdm_builder_helper import save_ndarray

class SingleTrialBetas(THINGSGLM):
    """
    Calculate single trial response estimates for the THINGS-fMRI dataset.
    """

    def __init__(
            self,
            bidsroot: str ,#have to provide with the bidsroot directory
            subject: str ,#have to provide the subject ID
            out_deriv_name: str = 'betas_vol_run_00',#check before each run of the constructor 
            usecache: bool = True,
            tuning_procedure: str = 'stepwise',
            standardize_noiseregs: bool = True,
            standardize_trialegs: bool = True,
            on_residuals_data: bool = True,
            batched_data_loading: bool = True,
            on_residuals_design: bool = True,
            cv_scheme: str = 'loo',
            perf_metric: str = 'l2',
            assume_hrf: int or False = False,
            match_scale_runwise: bool = False,
            use_spearman_brown: bool = False,
            unregularized_targets: bool = True,
            hrflib_url: str = 'https://raw.githubusercontent.com/kendrickkay/GLMdenoise/master/utilities'
                              '/getcanonicalhrflibrary.tsv',
            rescale_hrflib_amplitude: bool = True,
            hrflib_resolution: float = .1,
            overfit_hrf_model: str = 'onoff',
            fracs: np.ndarray = np.hstack([np.arange(.1, .91, .05), np.arange(.91, 1.01, .01)]),
            fmriprep_noiseregs: list = [],
            fmriprep_compcors: bool or int = 0,
            aroma_regressors: bool = False,
            manual_ica_regressors: bool = True,
            drift_model: str = 'polynomial',
            poly_order: int = 4,
            high_pass: float = None,
            rescale_runwise_data: str = 'z',
            zscore_data_sessionwise: bool = False,
            stims_per_ses: int = 920,
            out_dtype=np.double,
            n_parallel_hrf: int = 50,
            n_parallel_repeated_betas: int = 6,
            n_parallel_splithalf: int = 15,
            mcnc_nsig: int = 1000, mcnc_nmes: int = 1, mcnc_njobs: int = 50, mcnc_ddof: int = 0,
    ):
        super().__init__(
            sigscale_nilearn=False, noise_model='ols', hrf_model=None, bidsroot=bidsroot, subject=subject,
            out_deriv_name=out_deriv_name, noiseregs=fmriprep_noiseregs, acompcors=fmriprep_compcors,
            include_all_aroma=aroma_regressors,
            high_pass=high_pass, drift_model=drift_model, drift_order=poly_order,
        )
        self.manual_ica_regressors = manual_ica_regressors
        self.usecache = usecache
        self.on_residuals_data = on_residuals_data
        self.on_residuals_design = on_residuals_design
        self.batched_data_loading = batched_data_loading
        self.zscore_data_sessionwise = zscore_data_sessionwise
        assert tuning_procedure in ['stepwise', 'combined']
        self.tuning_procedure = tuning_procedure
        assert cv_scheme in ['splithalf', 'mcnc', 'loo', 'unregularized']
        self.cv_scheme = cv_scheme
        assert perf_metric in ['correlation', 'l1', 'l2']
        self.perf_metric = perf_metric
        self.match_scale_runwise = match_scale_runwise
        self.use_spearman_brown = use_spearman_brown
        self.standardize_noiseregs = standardize_noiseregs
        self.standardize_trialegs = standardize_trialegs
        self.n_parallel_repeated_betas = n_parallel_repeated_betas
        self.n_parallel_splithalf = n_parallel_splithalf
        self.kf = KFold(n_splits=self.n_sessions)
        self.stims_per_ses = stims_per_ses
        self.hrflib_url = hrflib_url
        self.hrflib = get_hrflib(self.hrflib_url)
        self.nsamples_hrf, self.nhrfs = self.hrflib.shape
        assert fracs[-1] == 1.  # make sure we include OLS
        self.fracs = fracs
        self.unregularized_targets = unregularized_targets
        self.nfracs = len(self.fracs)
        assert rescale_runwise_data in ['z', 'psc', 'center']  # don't allow uncentered data
        self.rescale_runwise_data = rescale_runwise_data
        assert overfit_hrf_model in ['onoff', 'single-trial']
        self.overfit_hrf_model = overfit_hrf_model
        self.assume_hrf = assume_hrf  # picked 10 as canonical hrf
        self.n_parallel_hrf = n_parallel_hrf
        self.hrflib_resolution = hrflib_resolution
        self.rescale_hrflib_amplitude = rescale_hrflib_amplitude
        self.microtime_factor = int(self.tr / self.hrflib_resolution)  # should be 15 in our case
        self.frame_times_microtime = np.arange(0, self.ntrs * self.tr, self.hrflib_resolution) + self.stc_reftime
        self.frf = FracRidgeRegressor(fracs=1., fit_intercept=False, normalize=False)
        # mcnc settings
        self.mcnc_n_sig, self.mcnc_n_mes = mcnc_nsig, mcnc_nmes
        self.mcnc_njobs = mcnc_njobs
        self.mcnc_ddof = mcnc_ddof
        # directories and files
        self.workdirbase='/DATA/satwick22/Documents/fMRI/fMRI_processing/python_work_dir'
        self.workdir = pjoin(self.workdirbase, 'betas_py')
        self.outdirbase='/DATA1/satwick22/Documents/fMRI/thingsmri'#SET
        self.outdir = pjoin(self.outdirbase, 'derivatives', self.out_deriv_name, f'sub-{self.subject}')
        self.best_hrf_nii=pjoin(self.outdir,'best_hrf_inds.nii.gz')
        self.best_hrf_nii_base = pjoin(self.outdir, 'best_hrf_inds')
        self.best_frac_inds_nii = pjoin(self.outdir, 'best_frac_inds.nii.gz')
        self.max_performance_nii = pjoin(self.outdir, 'max_performance.nii.gz')
        self.best_fracs_nii = pjoin(self.outdir, 'best_fracs.nii.gz')
        self.out_dtype = out_dtype
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        #additional class attributes
        self.roi_hrf_fit=False#attrubute which indicates is overfiiting is done voxelwise or roiwise
        self.save_hrf_ind=True#attribute which decides whether to save the best hrfs or not #SET
        self.save_frac_ind=True#attribute which decides whether to save the best fracs or not #SET
        self.best_hrf_inds_by_reg={}
        self.best_hrf_inds_by_roi={}
        self.best_frac_inds_by_roi={}
        self.chunk_scores=None#paramter to skip the hrf fits multiple times
        self.chunksize_prev=0
        self.best_hrf_inds_by=None
        self.best_frac_inds_by=None
        self.fit_betas=True
        self.assume_alpha=None
        self.shuffle_data=False
        #use dictionary comprehension to initialize 
        self.betas_session={i+1:np.array([]) for i in range(self.n_sessions)}
        self.betas_concat={i+1:np.array([]) for i in range(self.n_sessions)}
        self.condnames_concat={i+1:np.array([]) for i in range(self.n_sessions)}
        self.condnames_sortind={i+1:np.array([]) for i in range(self.n_sessions)}
    def flush_sample(self):
        #flushing the output of the previous sample
        self.chunk_scores=None
        self.chunksize_prev=0
        self.condnames_concat={i+1:np.array([]) for i in range(self.n_sessions)}
        self.condnames_sortind={i+1:np.array([]) for i in range(self.n_sessions)}
    def flush_specification(self):
        self.betas_concat={i+1:np.array([]) for i in range(self.n_sessions)}
        self.best_hrf_inds_by_roi={}
        self.best_frac_inds_by_roi={}
    def flush_roi(self):
        self.betas_session={i+1:np.array([]) for i in range(self.n_sessions)}
    def change_out_dir(self,out_deriv_name='betas_run___'):
        self.out_deriv_name=out_deriv_name
        self.outdir = pjoin(self.outdirbase, 'derivatives', self.out_deriv_name, f'sub-{self.subject}')
        self.best_hrf_nii = pjoin(self.outdir, 'best_hrf_inds.nii.gz')
        self.best_frac_inds_nii = pjoin(self.outdir, 'best_frac_inds.nii.gz')
        self.max_performance_nii = pjoin(self.outdir, 'max_performance.nii.gz')
        self.best_fracs_nii = pjoin(self.outdir, 'best_fracs.nii.gz')
        #self.out_dtype = self.out_dtype
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
    def _make_design_df(self, event_file):
        """
        Make data frame containing specifying each trial as a separate condition.
        Also returns the list of condition names, and condition names for the repeated trials.
        """
        design_df = pd.read_csv(event_file, sep='\t')[['duration', 'onset', 'file_path', 'trial_type']]
        rep_condnames = design_df.loc[design_df.trial_type == 'test', 'file_path'].to_numpy()
        design_df = design_df.drop(columns='trial_type')
        design_df = design_df.rename(columns={'file_path': 'trial_type'})
        design_df = design_df.sort_values(by='onset', ignore_index=True)
        return design_df, rep_condnames

    def _onoff_df_from_design_df(self, design_df):
        """Take a single trial design data frame and turn it into an onoff design data frame"""
        onoff_df = design_df.copy(deep=True)
        onoff_df['trial_type'] = 'onoff'
        return onoff_df
    def make_design_dfs_shuffled(self, event_files):
        """
        Given our event files, give us the design data frames and condition names per run
        as well as the list of unique names (across all runs) of repeated conditions.
        """
        designs_zipped = [self._make_design_df(ef) for ef in tqdm(event_files, desc='reading event files')]
        design_dfs, rep_condnames_runs = list(zip(*designs_zipped))
        #omit the catch trials using str contains 
        design_dfs=[design_dfs[i][~design_dfs[i].loc[:,'trial_type'].str.contains('catch')] for i in range(len(design_dfs))]
        #shuffle the design dfs
        #iterate through the design dfs and shuffle the rows
        for i in range(len(design_dfs)):
            trial_col=design_dfs[i].loc[:,'trial_type']
            #shuffle the trial column
            shuffle_trial_col=np.random.permutation(trial_col)
            #replace the trial column with the shuffled trial column
            design_dfs[i].loc[:,'trial_type']=shuffle_trial_col
        rep_condnames = np.unique(np.hstack(rep_condnames_runs))
        onoff_dfs = [self._onoff_df_from_design_df(ddf) for ddf in design_dfs]
        return design_dfs, onoff_dfs, rep_condnames
    def make_design_dfs(self, event_files):
        """
        Given our event files, give us the design data frames and condition names per run
        as well as the list of unique names (across all runs) of repeated conditions.
        """
        designs_zipped = [self._make_design_df(ef) for ef in tqdm(event_files, desc='reading event files')]
        design_dfs, rep_condnames_runs = list(zip(*designs_zipped))
        #omitting the catch trials
        design_dfs=[design_dfs[i][~design_dfs[i].loc[:,'trial_type'].str.contains('catch')] for i in range(len(design_dfs))]
        rep_condnames = np.unique(np.hstack(rep_condnames_runs))
        onoff_dfs = [self._onoff_df_from_design_df(ddf) for ddf in design_dfs]
        return design_dfs, onoff_dfs, rep_condnames

    def convolve_designmat(self, designmat, rescale_hrflib_amplitude=True):
        """
        Convolve a boxcar design matrix with each hrf in self.hrflib.
        Returns array of shape (nhrfs, ntrs, ntrials)
        """
        convolved_ups = np.zeros(
            shape=(self.nhrfs, self.ntrs * self.microtime_factor, designmat.shape[1]))
        for hrf_i in range(self.nhrfs):
            conv_thishrf_ups = np.apply_along_axis(
                lambda m: np.convolve(m, self.hrflib[:, hrf_i], mode='full'),
                arr=designmat, axis=0
            )[:self.ntrs * self.microtime_factor, :]
            if rescale_hrflib_amplitude:
                conv_thishrf_ups = np.nan_to_num(conv_thishrf_ups / conv_thishrf_ups.max(axis=0))
            convolved_ups[hrf_i] = conv_thishrf_ups
        convolved_designmat = convolved_ups[:, ::self.microtime_factor, :]
        return convolved_designmat

    def make_designs(self, event_files, normalize_convolved_regressors: bool = False):
        """
        make convolved design matrices for each event file, also return condition names per run
        and list of unique names of repeated stimuli (across all runs)
        """
        if self.shuffle_data:
            design_dfs, onoff_dfs, rep_condnames = self.make_design_dfs_shuffled(event_files)#gets the design dfs and onoff dfs,where onoff dfs are just design_dfs with onoff on trial type column
        else:
            design_dfs, onoff_dfs, rep_condnames = self.make_design_dfs(event_files)
        with Parallel(n_jobs=-1) as parallel:
            designmats = parallel(
                delayed(df_to_boxcar_design)(df, self.frame_times_microtime)
                for df in tqdm(design_dfs, 'making single-trail design matrices')
            )#returns the design matrix with each condition as a separate column with values in rows being time points and correspondingly 1/0 for each timepoint
        with Parallel(n_jobs=-1) as parallel:
            onoffmats = parallel(
                delayed(df_to_boxcar_design)(df, self.frame_times_microtime)
                for df in tqdm(onoff_dfs, 'making onoff design matrices')
            )
        # nilearn sorts conditions alphabetically, hence get them from designmat instead of dataframe
        condnames_runs = [dm.columns for dm in designmats]
        convolved_designs = [
            self.convolve_designmat(designmat, rescale_hrflib_amplitude=self.rescale_hrflib_amplitude)
            for designmat in tqdm(
                designmats,
                desc='Convolving single-trial designs for each run with HRF library', position=0, leave=True
            )
        ]
        convolved_onoff = [
            self.convolve_designmat(onoffmat, rescale_hrflib_amplitude=self.rescale_hrflib_amplitude)
            for onoffmat in tqdm(
                onoffmats,
                desc='Convolving onoff designs for each run with HRF library', position=0, leave=True
            )
        ]
        if normalize_convolved_regressors:
            convolved_designs = [zscore(arr, axis=1) for arr in convolved_designs]
            convolved_onoff = [zscore(arr, axis=1) for arr in convolved_onoff]
        return convolved_designs, convolved_onoff, condnames_runs, rep_condnames

    def make_noise_mat(self, nuisance_tsv, ica_txt=None, add_constant=False):
        """
        Make design matrix for noise regressors obtained from fmripreps nuisance tsv files
        and/or our manually classified ICs.
        """
        nuisance_df = get_nuisance_df(self.noiseregs, nuisance_tsv, include_all_aroma=self.include_all_aroma)#returns a nuisance_df based on the nuisance.tsv returned by fmriprep [here it is empty]
        if ica_txt:
            ica_arr = np.loadtxt(ica_txt)
            nuisance_df = pd.DataFrame(
                np.hstack([nuisance_df.to_numpy(), ica_arr]),
                columns=[f'noisereg-{i}' for i in range(nuisance_df.shape[1] + ica_arr.shape[1])]
            )#concatenates the nuisance_df with the ica_arr
        dropcols = [] if add_constant else ['constant']
        return make_first_level_design_matrix(
            frame_times=self.frame_times_tr, add_regs=nuisance_df,
            hrf_model=None, drift_model=self.drift_model, drift_order=self.drift_order,
            high_pass=self.high_pass, events=None,
        ).drop(columns=dropcols)
    

    def regress_out_noise_runwise(self, noise_mats, data, zscore_residuals=True):
        """
        Regress the noise matrices out of our data separately for each session (to save memory).
        Original data is overwritten.
        """
        # fit intercept only if data was not rescaled runwise
        fitint = True if self.rescale_runwise_data in ['off', 'psc'] else False
        for run_i in tqdm(range(self.nruns_total), desc='Regressing out noise runwise'):
            start_sample, stop_sample = run_i * self.ntrs, run_i * self.ntrs + self.ntrs
            data_run = data[start_sample:stop_sample]
            data_filtered = regress_out(noise_mats[run_i], data_run,
                                        lr_kws=dict(copy_X=False, fit_intercept=fitint, normalize=False, n_jobs=-1))
            # overwrite raw data
            if zscore_residuals:
                data[start_sample:stop_sample] = np.nan_to_num(zscore(data_filtered, axis=0))
            else:
                data[start_sample:stop_sample] = np.nan_to_num(data_filtered)
        return data

    def orthogonalize_designmats(self, convolved_designs, convolved_onoff, noise_mats):
        """
        Predict the design regressors with the noise regressors and only keep the residuals.
        """
        convolved_designs = [
            np.stack([regress_out(noisemat, designmat[hrf_i]) for hrf_i in range(self.nhrfs)])
            for noisemat, designmat in zip(noise_mats, convolved_designs)
        ]
        convolved_onoff = [
            np.stack([regress_out(noisemat, designmat[hrf_i]) for hrf_i in range(self.nhrfs)])
            for noisemat, designmat in zip(noise_mats, convolved_onoff)
        ]
        return convolved_designs, convolved_onoff

    def overfit_hrf(self, data, convolved_designs, noise_mats=[], chunksize_runs=1):
        """
        Find best HRF per voxel measured by best within-sample r-squared with a single-trial design
        and no regularization. Noise regressors are optional and can be left out if data was cleaned beforehand.
        """
        n_chunks = int(self.nruns_total / chunksize_runs)
        start = time.time()
        # chunk inputs for parallelization
        print(f'HRF overfitting with {self.n_parallel_hrf} chunks in parallel')
        start_is = [chunk_i * chunksize_runs for chunk_i in range(n_chunks)]
        stop_is = [start_i + chunksize_runs for start_i in start_is]
        datachunks = [data[start_i * self.ntrs:stop_i * self.ntrs] for start_i, stop_i in zip(start_is, stop_is)]
        designchunks = [convolved_designs[start_i:stop_i] for start_i, stop_i in zip(start_is, stop_is)]
        if not noise_mats:
            noisechunks = [[]] * n_chunks
        else:
            noisechunks = [noise_mats[start_i:stop_i] for start_i, stop_i in zip(start_is, stop_is)]
        # Fit to each chunk
        with Parallel(n_jobs=-1) as parallel:
            chunk_scores = parallel(
                delayed(overfit_hrf_to_chunk)(data_, designs_, noise_mats_, self.nvox_masked, self.nhrfs)
                for data_, designs_, noise_mats_ in zip(datachunks, designchunks, noisechunks)
            )
        # aggregate
        #chunkscores 
        mean_r2s = np.nanmean(np.stack(chunk_scores), axis=0)
        #iterate through the chunkscores and get r2s for each session and each rois
        #extracting the meanr2s for the roi voxels
        mean_r2s_roi=mean_r2s[:,self.source_mask_index]
        #getting the average of the voxels for each voxel
        mean_r2s_roi_avg=np.nanmean(mean_r2s_roi,axis=1)
        #extracting the best hrf for the roi
        best_hrf_inds_roi=np.argmax(mean_r2s_roi_avg,axis=0)
        best_hrf_inds = np.argmax(mean_r2s, axis=0)
        #replacing the values of the roi voxels with the hrf index for the regionwise best
        best_hrf_inds[self.target_mask_index]=best_hrf_inds_roi
        #saving the best fitting hrf for each roi
        self.best_hrf_inds_by_reg[self.reg_id]+=[best_hrf_inds_roi]
        print(f'HRF overfitting completed in {(time.time() - start) / 60:.1f} minutes')
        return best_hrf_inds, mean_r2s
    def overfit_hrf_per_session(self, data, convolved_designs, noise_mats=[], chunksize_runs=1):
        """
        Find best HRF per voxelper session measured by best within-sample r-squared with a single-trial design
        and no regularization. Noise regressors are optional and can be left out if data was cleaned beforehand.
        """
        n_chunks = int(self.nruns_total / chunksize_runs)
        start = time.time()
        # chunk inputs for parallelization
        print(f'HRF overfitting with {self.n_parallel_hrf} chunks in parallel')
        start_is = [chunk_i * chunksize_runs for chunk_i in range(n_chunks)]
        stop_is = [start_i + chunksize_runs for start_i in start_is]
        datachunks = [data[start_i * self.ntrs:stop_i * self.ntrs] for start_i, stop_i in zip(start_is, stop_is)]
        designchunks = [convolved_designs[start_i:stop_i] for start_i, stop_i in zip(start_is, stop_is)]
        if not noise_mats:
            noisechunks = [[]] * n_chunks
        else:
            noisechunks = [noise_mats[start_i:stop_i] for start_i, stop_i in zip(start_is, stop_is)]
        # Fit to each chunk
        if not(self.chunk_scores) and self.chunksize_prev!=chunksize_runs:
            with Parallel(n_jobs=-1) as parallel:
                chunk_scores = parallel(
                    delayed(overfit_hrf_to_chunk)(data_, designs_, noise_mats_, self.nvox_masked, self.nhrfs)
                    for data_, designs_, noise_mats_ in zip(datachunks, designchunks, noisechunks)
                )
            if chunk_scores:
                self.chunksize_prev=chunksize_runs
                self.chunk_scores=chunk_scores
            else:
                self.chunksize_prev=0
                self.chunk_scores=None
        else:
            chunk_scores=self.chunk_scores

        # aggregate
        #initialize the array for storing the best hrf index for each session
        best_hrf_inds_sessions=[]
        mean_r2s_sessions=[]
        if self.roi_hrf_fit:
            self.best_hrf_inds_by_reg[self.reg_id]={}
        #iterate through n_sessions get the r2s for each session
        for ses_ind in range(self.n_sessions):
            #get the runs for the session
            start_run=ses_ind*self.nruns_perses
            stop_run=start_run+self.nruns_perses
            #extract the chunk_scores for the session
            mean_r2s_ses=np.nanmean(np.stack(chunk_scores[start_run:stop_run]),axis=0)
            if self.roi_hrf_fit:
                mean_r2s_roi=mean_r2s_ses[:,self.source_mask_index]
                #getting the average of the voxels for each voxel
                mean_r2s_roi_avg=np.nanmean(mean_r2s_roi,axis=1)
                #extracting the best hrf for the roi
                best_hrf_inds_roi=np.argmax(mean_r2s_roi_avg,axis=0)
                best_hrf_inds = np.argmax(mean_r2s_ses, axis=0)
                #replacing the values of the roi voxels with the hrf index for the regionwise best
                best_hrf_inds[self.target_mask_index]=best_hrf_inds_roi 
                #saving the best fitting hrf for each roi
                self.best_hrf_inds_by_reg[self.reg_id][f'ses{ses_ind+1}']=best_hrf_inds_roi
                if self.best_hrf_by_mode:
                    best_hrf_inds_roi=stats.mode(np.array([self.best_hrf_inds_by_reg[self.reg_id][f'ses{ses_ind+1}'] for ses_ind in range(self.n_sessions)]))
                    best_hrf_inds

            else:
                best_hrf_inds = np.argmax(mean_r2s_ses, axis=0)
            best_hrf_inds_sessions.append(best_hrf_inds)
            mean_r2s_sessions.append(mean_r2s_ses)
        
        
        #alternate code for roi based overfitting

        print(f'HRF overfitting completed in {(time.time() - start) / 60:.1f} minutes')
        return best_hrf_inds_sessions, mean_r2s_sessions
    def mask_iter(self,mask_dir='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_roi/rois/category_localizer'):
        def mask_transform(roi_mask):
            #resampling the mask in the union mask template
            roi_mask=image.resample_to_img(roi_mask,self.union_mask,interpolation='nearest')
            #taking intersect of the two masks
            #self.roi_mask=intersect_masks([self.roi_mask,self.union_mask],threshold=1,connected=False)
            #get the index of the union masks
            self.union_mask_index=np.where(self.union_mask.get_fdata().flatten()==1)[0]
            roi_mask_index=np.where(roi_mask.get_fdata().flatten()==1)[0]#getting the index of the roi mask from the union mask index
            #getting the index of the roi mask from the union mask index
            roi_mask_intersected_index=np.where(roi_mask_index[:,None]==self.union_mask_index[None,:])[1]
            return roi_mask_intersected_index
        roi_list=['EBA','OFA','PPA','RSC','TOS','FFA','LOC']
        sub_dir_base = pjoin(mask_dir, f'sub-{self.subject}')
        roi_path_dict={'EBA':'body_parcels','FFA':'face_parcels','OFA':'face_parcels','LOC':'object_parcels','PPA':'scene_parcels','STS':'face_parcels','RSC':'scene_parcels','TOS':'scene_parcels'}
        for reg_id in roi_list:
            sub_dir=pjoin(sub_dir_base,roi_path_dict[reg_id])
            mask_img_left = image.load_img(pjoin(sub_dir, f'sub-{self.subject}_l{reg_id}.nii.gz'))
            mask_img_right = image.load_img(pjoin(sub_dir, f'sub-{self.subject}_r{reg_id}.nii.gz'))
            roi_mask_image_path=pjoin(sub_dir,f'sub-{self.subject}_{reg_id}.nii.gz')
            #check if path exists
            if os.path.exists(roi_mask_image_path):
                roi_mask_img=image.load_img(roi_mask_image_path)
            else:
                roi_mask_l_data=mask_img_left.get_fdata()
                #convert the mask image to a numpy array
                roi_mask_r_data=mask_img_right.get_fdata()
                #add the numpy arrays
                roi_mask_data=roi_mask_l_data+roi_mask_r_data
                #convert the numpy array to a nifti image
                roi_mask_img=image.new_img_like(mask_img_left,roi_mask_data)
            #transform the mask to the union mask template and get the indexes
            roi_mask_l_intersected_index,roi_mask_r_intersected_index,roi_mask_intersected_index=tuple(map(mask_transform,[mask_img_left,mask_img_right,roi_mask_img]))
            yield roi_mask_l_intersected_index,roi_mask_r_intersected_index,roi_mask_intersected_index,reg_id
    def overfit_hrf_rois(self, data, convolved_designs, noise_mats=[], chunksize_runs=1,method='mean'):
        """Finds the best HRF parameter per rois based on highest mean R2(by two methods) 
        Either using the best mean R2 score over all session
        or By using the mode of the best hrf index across the sessions """
        
        n_chunks = int(self.nruns_total / chunksize_runs)
        start = time.time()
        # chunk inputs for parallelization
        print(f'HRF overfitting with {self.n_parallel_hrf} chunks in parallel')
        start_is = [chunk_i * chunksize_runs for chunk_i in range(n_chunks)]
        stop_is = [start_i + chunksize_runs for start_i in start_is]
        datachunks = [data[start_i * self.ntrs:stop_i * self.ntrs] for start_i, stop_i in zip(start_is, stop_is)]
        designchunks = [convolved_designs[start_i:stop_i] for start_i, stop_i in zip(start_is, stop_is)]
        if not noise_mats:
            noisechunks = [[]] * n_chunks
        else:
            noisechunks = [noise_mats[start_i:stop_i] for start_i, stop_i in zip(start_is, stop_is)]
        if not(self.chunk_scores) and self.chunksize_prev!=chunksize_runs:
            with Parallel(n_jobs=-1) as parallel:
                chunk_scores = parallel(
                    delayed(overfit_hrf_to_chunk)(data_, designs_, noise_mats_, self.nvox_masked, self.nhrfs)
                    for data_, designs_, noise_mats_ in zip(datachunks, designchunks, noisechunks)
                )
            if chunk_scores:
                self.chunksize_prev=chunksize_runs
                self.chunk_scores=chunk_scores
            else:
                self.chunksize_prev=0
                self.chunk_scores=None
        else:
            chunk_scores=self.chunk_scores
        # aggregate
        #getting the chunk_scores for n_sessions
        chunk_scores_sessions=chunk_scores[:self.nruns_perses*self.n_sessions]
        mean_r2s=np.nanmean(np.stack(chunk_scores_sessions), axis=0)
        if method == 'mean':
            #extracting the mean r2s for the roi
            mean_r2s_roi=mean_r2s[:,self.source_mask_index]
            #getting the average of the voxels of the roi
            r2s_avg_roi=np.nanmean(mean_r2s_roi,axis=1)
            #extracting the best hrf for the roi
            best_hrf_inds_roi=np.argmax(r2s_avg_roi,axis=0)
        elif method == 'mode':
            #iterating through the chunks and extracting the best hrf index
            # for the roi for each session
            best_hrf_inds_by_ses=[]
            for ses_ind in range(self.n_sessions):#iterating throgh the session
                #get the runs for the session
                start_run=ses_ind*self.nruns_perses
                stop_run=start_run+self.nruns_perses
                #extract the chunk_scores for the session
                mean_r2s_ses=np.nanmean(np.stack(chunk_scores_sessions[start_run:stop_run]),axis=0)
                #extract the roi from the mean r2 ses
                mean_r2s_roi=mean_r2s_ses[:,self.source_mask_index]
                #getting the average of the voxels of the roi
                r2s_avg_roi=np.nanmean(mean_r2s_roi,axis=1)
                #extracting the best hrf for the roi
                best_hrf_inds_roi_ses=np.argmax(r2s_avg_roi,axis=0)
                #extracting the best hrf for the session and add the same to the list
                best_hrf_inds_by_ses.append(best_hrf_inds_roi_ses)
            #getting the mode of the best hrf index across the sessions
            best_hrf_inds_roi,best_hrf_inds_roi_count=stats.mode(np.array(best_hrf_inds_by_ses))
        #saving the voxelwise best hrf for the brain mask
        best_hrf_inds = np.argmax(mean_r2s, axis=0)
        #overwriting the hrf values for the voxels of the roi
        best_hrf_inds[self.target_mask_index]=best_hrf_inds_roi
        #saving the best hrf index for the roi
        self.best_hrf_inds_by_roi[self.reg_id]=np.arange(1,self.nhrfs+1)[best_hrf_inds_roi]
        print(f'HRF overfitting completed in {(time.time() - start) / 60:.1f} minutes')
        return best_hrf_inds, mean_r2s

    def get_repeated_betas(self, data, convolved_designs, noise_mats, condnames_runs, rep_condnames, hrf_inds):
        """
        Fit single trial GLM to each session and return the beta estimates of the repeatedly presented stimuli.
        """
        betas_per_session=[]
        for ses_i in tqdm(range(self.n_sessions), desc='Getting repeated stimuli runwise', leave=True):
            startrun, stoprun = ses_i * self.nruns_perses, ses_i * self.nruns_perses + self.nruns_perses
            startsample, stopsample = startrun * self.ntrs, stoprun * self.ntrs
            data_ses = data[startsample:stopsample]
            designs_ses = convolved_designs[startrun:stoprun]
            noisemats_ses = noise_mats[startrun:stoprun] if noise_mats else []
            condnames_ses = np.hstack(condnames_runs[startrun:stoprun])
            repis_ses = np.hstack([np.where(condnames_ses == repcond) for repcond in rep_condnames]).squeeze()
            # _ = get_betas_run(0, 0, data_ses, designs_ses, noisemats_ses, hrf_inds, self.fracs, self.nvox_masked)
            #current_process = psutil.Process()
            #subproc_before = set([p.pid for p in current_process.children(recursive=True)])
            with Parallel(n_jobs=-1) as parallel:
                betas_per_run = parallel(
                    delayed(get_betas_run)(  # TODO: could be parallelized more elegantly
                        ses_i, run_i, data_ses, designs_ses, noisemats_ses, hrf_inds, self.fracs, self.nvox_masked,
                    )
                    for run_i in range(self.nruns_perses)
                )
            # saving the betas per run into the list betas_per_session
            betas_per_session.append(np.concatenate(betas_per_run, axis=0)[repis_ses])
            #subproc_after = set([p.pid for p in current_process.children(recursive=True)])
            #for subproc in subproc_after - subproc_before:
            #    print('Killing process with pid {}'.format(subproc))
            #    psutil.Process(subproc).terminate()
            #cleaaring out the betas_per_run list
            del betas_per_run
            gc.collect()
            #saving the betas per run into the workdir
            #np.savez_compressed(pjoin(self.workdir, f'betas_per_run_ses-things{ses_i+1:2d}.npz'), betas_per_run)
        #betas_per_session=[]
        #for ses_i in range(self.n_sessions):
        #    betas_per_run = np.load(pjoin(self.workdir, f'betas_per_run_ses-things{ses_i+1:02d}.npz'))
        #    betas_per_session.append(np.concatenate(betas_per_run, axis=0)[repis_ses])
        return betas_per_session
    def splithalf(self, betas_per_session, n_splits: int = 0):
        """
        Calculate mean split-half correlation across sessions in order to evaluate best alpha fraction.
        Within each split, beta estimates of respective stimuli will be averaged.
        """
        combs = combinations(np.arange(self.n_sessions), int(self.n_sessions / 2))
        if n_splits:
            combs = list(combs)
            random.shuffle(combs)
            combs = combs[:n_splits]
        with Parallel(n_jobs=-1) as parallel:
            split_performances = parallel(
                delayed(eval_split_comb)(comb_i, comb, betas_per_session, self.nfracs,
                                         self.use_spearman_brown, self.unregularized_targets, self.perf_metric)
                for comb_i, comb in enumerate(combs)
            )
        performances = np.mean(np.stack(split_performances), axis=0)
        return performances
    def loov2(self, betas_per_session):
        """Performs the same leave-one-out Cross Validation 
        Returns the performance scores for each session's repeated stimulus
        instead of averaging it out across the sessions"""
        performance_folds = []
        for test_i in range(self.n_sessions):
            performance_folds.append(eval_loo(test_i, betas_per_session, self.nfracs, self.unregularized_targets, self.perf_metric))
        #with Parallel(n_jobs=self.n_parallel_splithalf) as parallel:
        #    performance_folds = parallel(
        #        delayed(eval_loo)(test_i, betas_per_session, self.nfracs, self.unregularized_targets, self.perf_metric)
        #        for test_i in range(self.n_sessions)
        #    )
        #performances = np.mean(np.stack(performance_folds), axis=0)
        return performance_folds
    def loo(self, betas_per_session):
        """
        Leave-one-out cross validation
        """
        performance_folds = []
        for test_i in range(self.n_sessions):
            performance_folds.append(eval_loo(test_i, betas_per_session, self.nfracs, self.unregularized_targets, self.perf_metric))
        #with Parallel(n_jobs=self.n_parallel_splithalf) as parallel:
        #    performance_folds = parallel(
        #        delayed(eval_loo)(test_i, betas_per_session, self.nfracs, self.unregularized_targets, self.perf_metric)
        #        for test_i in range(self.n_sessions)
        #    )
        performances = np.mean(np.stack(performance_folds), axis=0)
        return performances

    def combined_tuning(self, data, convolved_designs, condnames_runs, rep_condnames, noise_mats):
        """
        Find best pair of hyperparameters by searching through all combinations.
        """
        print('running LOO for all HRFs and alpha fractions')
        performances_hrfs = []  # one for each HRF
        for hrf_i in tqdm(range(self.nhrfs), desc='running LOO for all HRFs and alpha fractions', leave=True):
            hrf_inds = np.full(shape=self.nvox_masked, fill_value=hrf_i, dtype=int)
            print(f'getting repeated betas for HRF {hrf_i}')
            betas_per_session = self.get_repeated_betas(
                data, convolved_designs, noise_mats, condnames_runs, rep_condnames, hrf_inds
            )
            print(f'running {self.tuning_procedure} for HRF {hrf_i}')
            if self.cv_scheme == 'mcnc':
                performances = self.mcnc(betas_per_session)
            elif self.cv_scheme == 'splithalf':
                performances = self.splithalf(betas_per_session)
            elif self.cv_scheme == 'loo':
                performances = self.loo(betas_per_session)
            performances_hrfs.append(performances)
        performances_hrfs = np.stack(performances_hrfs)  # will have shape (hrf, fracs, nvox)???
        performances_combined = performances_hrfs.reshape((self.nhrfs * self.nfracs, self.nvox_masked))
        best_hrf_inds, best_frac_inds = np.unravel_index(np.argmax(performances_combined, axis=0),
                                                         shape=(self.nhrfs, self.nfracs))
        max_performance = np.max(performances_combined, axis=0)
        return best_hrf_inds, best_frac_inds, max_performance

    def mcnc(self, betas_per_session):
        """
        Use Monte Carlo noise ceiling to determine best alpha fraction for each voxel.
        """
        # stack over sessions
        betas_stacked = np.stack(betas_per_session).T.swapaxes(2, 0).swapaxes(2, 1).swapaxes(2, 3)
        del betas_per_session
        ncs_per_frac = []
        for frac_i in tqdm(range(self.nfracs), desc='MCNC for different fracs'):
            betas_frac = betas_stacked[frac_i]
            # estimate signal and noise distribution
            mn = betas_frac.mean(axis=1)
            se = betas_frac.var(axis=1, ddof=self.mcnc_ddof)
            noisevar = se.mean(axis=1)
            noisesd = np.sqrt(noisevar)
            overallmn = mn.mean(axis=1)
            overallvar = np.var(mn, axis=1, ddof=self.mcnc_ddof)
            signalsd = np.sqrt(
                np.clip(overallvar - noisevar, 0, None)
            )
            # repeat for sampling convenience
            overallmn = np.repeat(overallmn[:, np.newaxis], 100, axis=-1)
            signalsd = np.repeat(signalsd[:, np.newaxis], 100, axis=-1)
            noisesd = np.repeat(noisesd[:, np.newaxis], 100, axis=-1)
            # run simulations in parallel
            r2s = Parallel(n_jobs=self.mcnc_njobs)(
                delayed(_run_job_sim_notnested)(overallmn, signalsd, noisesd) for _ in range(self.mcnc_n_sig)
            )
            r2s = np.stack(r2s, axis=-1)
            ncs_per_frac.append(np.median(r2s, axis=-1))
        performances = np.stack(ncs_per_frac)
        return performances

    def final_fit(self, data, convolved_designs, best_param_inds, condnames_runs, noise_mats):
        """
        Fit the model with best HRF and alpha fraction per voxel to obtain final single trial beta estimates.
        """
        for ses_i in tqdm(range(self.n_sessions), desc='sessions'):
            #if ses_i not in [2,3,7]:continue
            sesdir = pjoin(self.outdir, f'ses-things{ses_i + 1:02d}')
            if not os.path.exists(sesdir):
                os.makedirs(sesdir)
            for run_i in tqdm(range(self.nruns_perses), desc='runs'):
                # figure out indices
                flat_i = ses_i * self.nruns_perses + run_i
                startsample, stopsample = flat_i * self.ntrs, flat_i * self.ntrs + self.ntrs
                nconds = len(condnames_runs[flat_i])
                # iterate over voxel sets and populate our results array
                #betas=np.zeros(shape=self.nvox_target_masked)
                betas = np.zeros(shape=(self.nvox_masked, nconds))
                for param_i in tqdm(np.unique(best_param_inds), 'parameter combinations'):
                    hrf_i, frac_i = np.unravel_index(param_i, shape=(self.nhrfs, self.nfracs))
                    #taking only the masked voxel
                    voxel_inds=self.target_mask_index
                    #voxel_inds = np.where(best_param_inds == param_i)
                    data_sub = data[startsample:stopsample, voxel_inds[0]].squeeze()
                    design = convolved_designs[flat_i][hrf_i]
                    if noise_mats:
                        design = np.hstack([design, noise_mats[flat_i]])
                    if self.match_scale_runwise:
                        self.frf.fracs = [self.fracs[frac_i], 1.]
                        self.frf.fit(design, data_sub)
                        betas_thisparam = match_scale(self.frf.coef_[:nconds, 0], self.frf.coef_[:nconds, 1])
                    else:
                        self.frf.fracs = self.fracs[frac_i]
                        self.frf.fit(design, data_sub)
                        betas_thisparam = self.frf.coef_[:nconds]
                    betas[voxel_inds] = betas_thisparam.T
                #for this run we will store the betas and the condnames in a list
                if not np.any(self.betas_session[ses_i+1]):
                    self.betas_session[ses_i+1]=betas[voxel_inds]
                else:#to concatenate sessionwise data
                    self.betas_session[ses_i+1]=np.concatenate((self.betas_session[ses_i+1],betas[voxel_inds]),axis=1)
                #storing the condnames as well sessionwise
                if not np.any(self.condnames_concat[ses_i+1]):
                    self.condnames_concat[ses_i+1]=condnames_runs[flat_i]
                elif self.condnames_concat[ses_i+1].shape[0]!=(betas.shape[1]*self.nruns_perses):#if condnames are already appended for one roi 
                    self.condnames_concat[ses_i+1]=np.concatenate((self.condnames_concat[ses_i+1],condnames_runs[flat_i]),axis=0)
                # save betas and condition names for this run to file
                #betas_nii = pjoin(
                #    sesdir, f'sub-{self.subject}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz'
                #)
                #conds_tsv = betas_nii.replace('_betas.nii.gz', '_conditions.tsv')
                #betas_img = unmask(betas.T.astype(self.out_dtype), self.union_mask)
                #betas_img.to_filename(betas_nii)
                #pd.DataFrame(condnames_runs[flat_i]).to_csv(conds_tsv, sep='\t', header=['image_filename'])
    #@profile
    def data_reader(self):
        """Loads Noise Regressor and Data and removes noise"""

        _, bold_files, nuisance_tsvs, masks = self.get_inputs()#returns bold_files,.... in form of list 
        self.add_union_mask(masks)#get the masks based on intersection of all masks
        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [self.make_noise_mat(nuisance_tsv, ica_tsv)
                          for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)]#get the noise design matrix 
        else:
            noise_mats = [self.make_noise_mat(nuisance_tsv) for nuisance_tsv in nuisance_tsvs]
        #standardizing the design matrices
        if self.standardize_noiseregs:
            print('\nStandardizing noise regressors')
            noise_mats = [np.nan_to_num(zscore(m, axis=0)) for m in noise_mats]


        
        #print('\nLoading data\n')
        data = self.vstack_data_masked(bold_files, rescale_runwise=self.rescale_runwise_data)
        print('\nLoading data\n')
        #running the data creator from bold signal in batches
        batch_size=20
        if self.batched_data_loading:
            data = np.vstack([self.vstack_data_masked(bold_files[batch_i-batch_size:batch_i], rescale_runwise=self.rescale_runwise_data)
                                     for batch_i in [i for i in range(batch_size,len(bold_files)+1,batch_size)]])
        else:
            data = self.vstack_data_masked(bold_files, rescale_runwise=self.rescale_runwise_data)
        if self.on_residuals_data:
            #data=np.vstack([data,self.vstack_data_masked(bold_files[100:120], rescale_runwise=self.rescale_runwise_data)
            print('\nRegressing out noise from data\n')
            data = self.regress_out_noise_runwise(noise_mats, data, zscore_residuals=True)

        if self.on_residuals_data or self.on_residuals_design:
            print('\nOmitting noise regressors for model fitting since data and/or design was orthogonalized\n')
            noise_mats = []
        self.data=data

    def convolve_creator(self):
        """Creates design matrix and on-off design matrix"""

        event_files, _, nuisance_tsvs, masks = self.get_inputs()#returns bold_files,.... in form of list 
        self.add_union_mask(masks)#get the masks based on intersection of all masks

        convolved_designs, convolved_onoff, condnames_runs, rep_condnames = self.make_designs(event_files)#get the convolved design matrix and the convolved on off matrix along with the condition names

        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [self.make_noise_mat(nuisance_tsv, ica_tsv)
                          for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)]#get the noise design matrix 
        else:
            noise_mats = [self.make_noise_mat(nuisance_tsv) for nuisance_tsv in nuisance_tsvs]
        #standardizing the design matrices
        if self.standardize_noiseregs:
            print('\nStandardizing noise regressors')
            noise_mats = [np.nan_to_num(zscore(m, axis=0)) for m in noise_mats]
        if self.standardize_trialegs:
            print('\nStandardizing trial regressors')
            convolved_designs = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_designs]
            convolved_onoff = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_onoff]
            

        if self.on_residuals_design:
            print('\nRegressing out noise from design\n')
            convolved_designs, convolved_onoff = self.orthogonalize_designmats(
                convolved_designs, convolved_onoff, noise_mats
            )#orthogonalize the design matrix with the noise matrix[remove the noise regressors from the design matrix]
        if self.on_residuals_data or self.on_residuals_design:
            print('\nOmitting noise regressors for model fitting since data and/or design was orthogonalized\n')
            noise_mats = []
        self.convolved_designs=convolved_designs
        self.convolved_onoff=convolved_onoff
        self.condnames_runs=condnames_runs
        self.rep_condnames=rep_condnames
        self.noise_mats=noise_mats
    def data_loader(self):
        print('\nLoading design and noise regressors\n')
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()#returns bold_files,.... in form of list 
        self.add_union_mask(masks)#get the masks based on intersection of all masks

        convolved_designs, convolved_onoff, condnames_runs, rep_condnames = self.make_designs(event_files)#get the convolved design matrix and the convolved on off matrix along with the condition names

        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [self.make_noise_mat(nuisance_tsv, ica_tsv)
                          for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)]#get the noise design matrix 
        else:
            noise_mats = [self.make_noise_mat(nuisance_tsv) for nuisance_tsv in nuisance_tsvs]
        #standardizing the design matrices
        if self.standardize_noiseregs:
            print('\nStandardizing noise regressors')
            noise_mats = [np.nan_to_num(zscore(m, axis=0)) for m in noise_mats]
        if self.standardize_trialegs:
            print('\nStandardizing trial regressors')
            convolved_designs = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_designs]
            convolved_onoff = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_onoff]
            

        if self.on_residuals_design:
            print('\nRegressing out noise from design\n')
            convolved_designs, convolved_onoff = self.orthogonalize_designmats(
                convolved_designs, convolved_onoff, noise_mats
            )#orthogonalize the design matrix with the noise matrix[remove the noise regressors from the design matrix]

        print('\nLoading data\n')
        #running the data creator from bold signal in batches
        batch_size=20
        data = np.vstack([self.vstack_data_masked(bold_files[batch_i-batch_size:batch_i], rescale_runwise=self.rescale_runwise_data)
                                     for batch_i in [i for i in range(batch_size,len(bold_files)+1,batch_size)]])
        if self.on_residuals_data:
            #data=np.vstack([data,self.vstack_data_masked(bold_files[100:120], rescale_runwise=self.rescale_runwise_data)
            print('\nRegressing out noise from data\n')
            data = self.regress_out_noise_runwise(noise_mats, data, zscore_residuals=True)

        if self.on_residuals_data or self.on_residuals_design:
            print('\nOmitting noise regressors for model fitting since data and/or design was orthogonalized\n')
            noise_mats = []

        self.data=data
        self.convolved_designs=convolved_designs
        self.convolved_onoff=convolved_onoff
        self.condnames_runs=condnames_runs
        self.rep_condnames=rep_condnames
        self.noise_mats=noise_mats

    def hrf_overfit_roi(self):
        data=self.data
        convolved_designs=self.convolved_designs
        convolved_onoff=self.convolved_onoff
        condnames_runs=self.condnames_runs
        rep_condnames=self.rep_condnames
        noise_mats=self.noise_mats


        if self.tuning_procedure == 'combined':
            if self.usecache and os.path.exists(self.best_hrf_nii) and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading HRF indices and alpha fractions from pre-stored files')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
                _ = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                print('\nCombined tuning of HRF and regularization')
                best_hrf_inds, _ , max_performance = self.combined_tuning(
                    data, convolved_designs, condnames_runs, rep_condnames, noise_mats
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
                unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        elif self.tuning_procedure == 'stepwise':
            if self.assume_hrf:
                assert np.abs(self.hrf_index) < self.nhrfs
                best_hrf_inds = np.full(fill_value=self.hrf_index, shape=self.nvox_masked, dtype=int)
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
            else:
                print('\nOverfitting HRF\n')
                if self.overfit_hrf_model == 'onoff':
                    best_hrf_inds, _ = self.overfit_hrf_per_session(data, convolved_onoff, noise_mats)
                else:
                    best_hrf_inds, _ = self.overfit_hrf_per_session(data, convolved_designs, noise_mats)
                # save best HRF per voxel#INSERT
                self.best_hrf_inds=best_hrf_inds
                #calling the function to save the best hrf for each session
                if self.save_hrf_ind:
                    self.save_best_hrf()
    def save_best_hrf(self):
        """Function to save the best hrf for each session"""
        for ses_i,best_hrf_ind in enumerate(self.best_hrf_inds):
            #creating the directory for the session
            sesdir=pjoin(self.best_hrf_nii_base,f'ses-things{ses_i+1:02d}')
            #creating the directory if it does not exist[which it should not by default]
            if not os.path.exists(sesdir):
                os.makedirs(sesdir)
            #saving the best hrf for each session
            unmask(best_hrf_ind, self.union_mask).to_filename(pjoin(sesdir,f'sub-{self.subject}_ses-things{ses_i+1:02d}_best_hrf.nii.gz'))
    def hrf_overfit_by_session(self):
        data=self.data
        convolved_designs=self.convolved_designs
        convolved_onoff=self.convolved_onoff
        condnames_runs=self.condnames_runs
        rep_condnames=self.rep_condnames
        noise_mats=self.noise_mats


        if self.tuning_procedure == 'combined':
            if self.usecache and os.path.exists(self.best_hrf_nii) and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading HRF indices and alpha fractions from pre-stored files')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
                _ = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                print('\nCombined tuning of HRF and regularization')
                best_hrf_inds, _ , max_performance = self.combined_tuning(
                    data, convolved_designs, condnames_runs, rep_condnames, noise_mats
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
                unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        elif self.tuning_procedure == 'stepwise':
            if self.assume_hrf:
                assert np.abs(self.hrf_index) < self.nhrfs
                best_hrf_inds = np.full(fill_value=self.hrf_index, shape=self.nvox_masked, dtype=int)
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
            else:
                print('\nOverfitting HRF\n')
                if self.overfit_hrf_model == 'onoff':
                    best_hrf_inds, _ = self.overfit_hrf_per_session(data, convolved_onoff, noise_mats)
                else:
                    best_hrf_inds, _ = self.overfit_hrf_per_session(data, convolved_designs, noise_mats)
                # save best HRF per voxel#INSERT
                self.best_hrf_inds=best_hrf_inds
                #calling the function to save the best hrf for each session
                if self.save_hrf_ind:
                    self.save_best_hrf()
    def save_best_hrf(self):
        """Function to save the best hrf for each session"""
        for ses_i,best_hrf_ind in enumerate(self.best_hrf_inds):
            #creating the directory for the session
            sesdir=pjoin(self.best_hrf_nii_base,f'ses-things{ses_i+1:02d}')
            #creating the directory if it does not exist[which it should not by default]
            if not os.path.exists(sesdir):
                os.makedirs(sesdir)
            #saving the best hrf for each session
            unmask(best_hrf_ind, self.union_mask).to_filename(pjoin(sesdir,f'sub-{self.subject}_ses-things{ses_i+1:02d}_best_hrf.nii.gz'))
    def alpha_best_fit(self,performance_folds,method):
        """Finds the best fitting alpha for each roi"""
        best_alpha_param_per_ses=[]#stores the value for best alpha param for each session for the roi
        #stack the results for each session for the roi
        performances = np.mean(np.stack(performance_folds), axis=0)
        #check the method parameter:mean or mode
        if method=='mean':
            #extract the indices of roi 
            performance_roi=performances[:,self.source_mask_index]
            #take the mean of the performance score of the roi voxels
            performance_roi_avg=np.nanmean(performance_roi,axis=1)
            #extract the best alpha parameter for the roi
            best_frac_inds_roi=np.argmax(performance_roi_avg,axis=0)
        elif method=='mode':
            #create a data structure to store the best alpha param for each session for the roi
            best_frac_inds_roi_ses=[]
            #iterate over the sessions
            for ses_ind in range(self.n_sessions):
                #extract the performance scores for the session
                performance_ses=performance_folds[ses_ind]
                #extract the performance scores for the roi
                performance_ses_roi=performance_ses[:,self.source_mask_index]
                #take the mean of the performance score of the roi voxels
                performance_ses_roi_avg=np.nanmean(performance_ses_roi,axis=1)
                #extract the best alpha parameter for the roi
                temp_best_frac=np.argmax(performance_ses_roi_avg,axis=0)
                #add the best alpha param for the session into the list
                best_frac_inds_roi_ses.append(temp_best_frac)
            #get the mode of the best alpha param for the roi
            best_frac_inds_roi,best_frac_inds_roi_count=stats.mode(np.array(best_frac_inds_roi_ses))
        #get the best frac  indices for all the voxels
        best_frac_inds=np.argmax(performances,axis=0)
        #replace the roi voxels with the best frac indices for the roi
        best_frac_inds[self.target_mask_index]=best_frac_inds_roi
        #save the best frac indices for the roi
        self.best_frac_inds_by_roi[self.reg_id]=self.fracs[best_frac_inds_roi] 
        return best_frac_inds
    def overfit_parameter(self):
        #loading the data
        data=self.data
        convolved_designs=self.convolved_designs
        convolved_onoff=self.convolved_onoff
        condnames_runs=self.condnames_runs
        rep_condnames=self.rep_condnames
        noise_mats=self.noise_mats
        #choosing the correct tuning parameter based on the attribute
        #performing overfit of parameter based on the attributes
        if self.tuning_procedure == 'combined':
            if self.usecache and os.path.exists(self.best_hrf_nii) and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading HRF indices and alpha fractions from pre-stored files')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                print('\nCombined tuning of HRF and regularization')
                best_hrf_inds, best_frac_inds, max_performance = self.combined_tuning(
                    data, convolved_designs, condnames_runs, rep_condnames, noise_mats
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
                unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        elif self.tuning_procedure == 'stepwise':
            if self.assume_hrf:
                print(f'\nAssuming HRF with index {self.assume_hrf}, creating temporary file\n')
                assert np.abs(self.assume_hrf) < self.nhrfs
                best_hrf_inds = np.full(fill_value=self.assume_hrf, shape=self.nvox_masked, dtype=int)
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)

            if self.usecache and os.path.exists(self.best_hrf_nii):
                print('\nLoading HRF indices from pre-stored file\n')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
            else:
                print('\nOverfitting HRF\n')
                if self.overfit_hrf_model == 'onoff':
                    if self.best_hrf_inds_by:
                        best_hrf_inds, _ = self.overfit_hrf_rois(data, convolved_onoff, noise_mats,method=self.best_hrf_inds_by)
                    else:
                        best_hrf_inds, _ = self.overfit_hrf(data, convolved_onoff, noise_mats)
                    #return #POINT OF RETURN
                else:
                    if self.best_hrf_inds_by:
                        best_hrf_inds, _ = self.overfit_hrf_rois(data, convolved_designs, noise_mats,method=self.best_hrf_inds_by)
                    else:
                        best_hrf_inds, _ = self.overfit_hrf(data, convolved_designs, noise_mats)
                # save best HRF per voxel
                if self.save_hrf_ind:#SET the attribute if you want to save the best hrf parameter
                    unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
            # regularization
            if self.usecache and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading best alpha fractions from pre-stored file\n')
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            #add a elif block for using 0.1 as the alpha for all voxels i.e 0 as everywhere
            elif self.assume_alpha:
                print(f'\nAssuming alpha fraction {self.assume_alpha}, creating temporary file\n')
                best_frac_inds = np.full(fill_value=np.where(self.fracs == self.assume_alpha)[0][0],
                                         shape=self.nvox_masked, dtype=int)
                unmask(best_frac_inds, self.union_mask).to_filename(self.best_frac_inds_nii)
            else:
                if self.cv_scheme == 'unregularized':
                    print('\nSkipping cross-validation and calculating unregularized betas\n')
                    best_frac_inds = np.full(fill_value=self.nfracs - 1, shape=self.nvox_masked, dtype=int)
                else:
                    print('\nEstimating betas of repeated stimuli for each session and each alpha fraction\n')
                    betas_per_session = self.get_repeated_betas(
                        data, convolved_designs, noise_mats, condnames_runs, rep_condnames, best_hrf_inds
                    )
                    print(f'\nUsing {self.cv_scheme} to find best alpha fraction per voxel')
                    if self.cv_scheme == 'mcnc':
                        performances = self.mcnc(betas_per_session)
                    elif self.cv_scheme == 'splithalf':
                        print(f"Using unregularized targets = {self.unregularized_targets}")
                        performances = self.splithalf(betas_per_session)
                    elif self.cv_scheme == 'loo':
                        if self.best_frac_inds_by:
                            performances_folds = self.loov2(betas_per_session)#gets back the array of performances for each sessions
                            performances=np.mean(np.stack(performances_folds),axis=0)
                        else:
                            performances = self.loo(betas_per_session)

                        # best_frac_inds = np.argmax(performance, axis=0)
                        # max_performance = np.max(performance, axis=0)
                    max_performance = np.max(performances, axis=0)
                    if self.best_frac_inds_by:
                        best_frac_inds=self.alpha_best_fit(performances_folds,method=self.best_frac_inds_by)
                    else:
                        best_frac_inds = np.argmax(performances, axis=0)
                    unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        # save best regularization parameter
        if self.save_frac_ind:
            best_fracs = self.fracs[best_frac_inds]
            for arr, fname in zip(
                    [best_frac_inds, best_fracs],
                    [self.best_frac_inds_nii, self.best_fracs_nii]
            ):
                unmask(arr, self.union_mask).to_filename(fname)
        if self.fit_betas:
            print('\nFinal fit\n')
            #extracting only the best hrf indices for the roi
            best_hrf_inds_roi=best_hrf_inds[self.source_mask_index]
            #extracting only the best alpha indices for the roi
            best_frac_inds_roi=best_frac_inds[self.target_mask_index]
            best_param_inds = np.ravel_multi_index((best_hrf_inds_roi, best_frac_inds_roi), (self.nhrfs, self.nfracs))
            self.final_fit(data, convolved_designs, best_param_inds, condnames_runs, noise_mats)
            print('\nDone.\n')

    def hrf_alpha_overfit_rois(self):
        #assigning the variables to the class attributes
        data=self.data
        convolved_designs=self.convolved_designs
        convolved_onoff=self.convolved_onoff
        condnames_runs=self.condnames_runs
        rep_condnames=self.rep_condnames
        noise_mats=self.noise_mats
        if self.tuning_procedure == 'combined':
            if self.usecache and os.path.exists(self.best_hrf_nii) and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading HRF indices and alpha fractions from pre-stored files')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                print('\nCombined tuning of HRF and regularization')
                best_hrf_inds, best_frac_inds, max_performance = self.combined_tuning(
                    data, convolved_designs, condnames_runs, rep_condnames, noise_mats
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
                unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        elif self.tuning_procedure == 'stepwise':
            if self.assume_hrf:
                print(f'\nAssuming HRF with index {self.assume_hrf}, creating temporary file\n')
                assert np.abs(self.assume_hrf) < self.nhrfs
                best_hrf_inds = np.full(fill_value=self.assume_hrf, shape=self.nvox_masked, dtype=int)
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)

            if self.usecache and os.path.exists(self.best_hrf_nii):
                print('\nLoading HRF indices from pre-stored file\n')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
            else:
                print('\nOverfitting HRF\n')
                if self.overfit_hrf_model == 'onoff':
                    best_hrf_inds, _ = self.overfit_hrf_rois(data, convolved_onoff, noise_mats)
                    #return #POINT OF RETURN
                else:
                    best_hrf_inds, _ = self.overfit_hrf_rois(data, convolved_designs, noise_mats)
                # save best HRF per voxel
                if self.save_hrf_ind:
                    unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
            # regularization
            if self.usecache and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading best alpha fractions from pre-stored file\n')
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                if self.cv_scheme == 'unregularized':
                    print('\nSkipping cross-validation and calculating unregularized betas\n')
                    best_frac_inds = np.full(fill_value=self.nfracs - 1, shape=self.nvox_masked, dtype=int)
                else:
                    print('\nEstimating betas of repeated stimuli for each session and each alpha fraction\n')
                    betas_per_session = self.get_repeated_betas(
                        data, convolved_designs, noise_mats, condnames_runs, rep_condnames, best_hrf_inds
                    )
                    print(f'\nUsing {self.cv_scheme} to find best alpha fraction per voxel')
                    if self.cv_scheme == 'mcnc':
                        performances = self.mcnc(betas_per_session)
                    elif self.cv_scheme == 'splithalf':
                        print(f"Using unregularized targets = {self.unregularized_targets}")
                        performances = self.splithalf(betas_per_session)
                    elif self.cv_scheme == 'loo':
                        performances = self.loo(betas_per_session)
                        # best_frac_inds = np.argmax(performance, axis=0)
                        # max_performance = np.max(performance, axis=0)
                    # max_performance = np.max(performances, axis=0)
                    best_frac_inds = np.argmax(performances, axis=0)
                    unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        # save best regularization parameter
        best_fracs = self.fracs[best_frac_inds]
        for arr, fname in zip(
                [best_frac_inds, best_fracs],
                [self.best_frac_inds_nii, self.best_fracs_nii]
        ):
            unmask(arr, self.union_mask).to_filename(fname)
        if self.fit_betas:
            print('\nFinal fit\n')
            best_param_inds = np.ravel_multi_index((best_hrf_inds, best_frac_inds), (self.nhrfs, self.nfracs))
            self.final_fit(data, convolved_designs, best_param_inds, condnames_runs, noise_mats)
            print('\nDone.\n')

            
            


    def run(self):
        print('\nLoading design and noise regressors\n')
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()#returns bold_files,.... in form of list 
        self.add_union_mask(masks)#get the masks based on intersection of all masks

        convolved_designs, convolved_onoff, condnames_runs, rep_condnames = self.make_designs(event_files)#get the convolved design matrix and the convolved on off matrix along with the condition names

        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [self.make_noise_mat(nuisance_tsv, ica_tsv)
                          for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)]#get the noise design matrix 
        else:
            noise_mats = [self.make_noise_mat(nuisance_tsv) for nuisance_tsv in nuisance_tsvs]
        #standardizing the design matrices
        if self.standardize_noiseregs:
            print('\nStandardizing noise regressors')
            noise_mats = [np.nan_to_num(zscore(m, axis=0)) for m in noise_mats]
        if self.standardize_trialegs:
            print('\nStandardizing trial regressors')
            convolved_designs = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_designs]
            convolved_onoff = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_onoff]
            

        if self.on_residuals_design:
            print('\nRegressing out noise from design\n')
            convolved_designs, convolved_onoff = self.orthogonalize_designmats(
                convolved_designs, convolved_onoff, noise_mats
            )#orthogonalize the design matrix with the noise matrix[remove the noise regressors from the design matrix]

        print('\nLoading data\n')
        #running the data creator from bold signal in batches
        batch_size=20
        data = np.vstack([self.vstack_data_masked(bold_files[batch_i-batch_size:batch_i], rescale_runwise=self.rescale_runwise_data)
                                     for batch_i in [i for i in range(batch_size,len(bold_files)+1,batch_size)]])
        if self.on_residuals_data:
            #data=np.vstack([data,self.vstack_data_masked(bold_files[100:120], rescale_runwise=self.rescale_runwise_data)
            print('\nRegressing out noise from data\n')
            data = self.regress_out_noise_runwise(noise_mats, data, zscore_residuals=True)

        if self.on_residuals_data or self.on_residuals_design:
            print('\nOmitting noise regressors for model fitting since data and/or design was orthogonalized\n')
            noise_mats = []

        if self.tuning_procedure == 'combined':
            if self.usecache and os.path.exists(self.best_hrf_nii) and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading HRF indices and alpha fractions from pre-stored files')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                print('\nCombined tuning of HRF and regularization')
                best_hrf_inds, best_frac_inds, max_performance = self.combined_tuning(
                    data, convolved_designs, condnames_runs, rep_condnames, noise_mats
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
                unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        elif self.tuning_procedure == 'stepwise':
            if self.assume_hrf:
                print(f'\nAssuming HRF with index {self.assume_hrf}, creating temporary file\n')
                assert np.abs(self.assume_hrf) < self.nhrfs
                best_hrf_inds = np.full(fill_value=self.assume_hrf, shape=self.nvox_masked, dtype=int)
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)

            if self.usecache and os.path.exists(self.best_hrf_nii):
                print('\nLoading HRF indices from pre-stored file\n')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
            else:
                print('\nOverfitting HRF\n')
                if self.overfit_hrf_model == 'onoff':
                    best_hrf_inds, _ = self.overfit_hrf(data, convolved_onoff, noise_mats)
                else:
                    best_hrf_inds, _ = self.overfit_hrf(data, convolved_designs, noise_mats)
                # save best HRF per voxel
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
            # regularization
            if self.usecache and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading best alpha fractions from pre-stored file\n')
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                if self.cv_scheme == 'unregularized':
                    print('\nSkipping cross-validation and calculating unregularized betas\n')
                    best_frac_inds = np.full(fill_value=self.nfracs - 1, shape=self.nvox_masked, dtype=int)
                else:
                    print('\nEstimating betas of repeated stimuli for each session and each alpha fraction\n')
                    betas_per_session = self.get_repeated_betas(
                        data, convolved_designs, noise_mats, condnames_runs, rep_condnames, best_hrf_inds
                    )
                    print(f'\nUsing {self.cv_scheme} to find best alpha fraction per voxel')
                    if self.cv_scheme == 'mcnc':
                        performances = self.mcnc(betas_per_session)
                    elif self.cv_scheme == 'splithalf':
                        print(f"Using unregularized targets = {self.unregularized_targets}")
                        performances = self.splithalf(betas_per_session)
                    elif self.cv_scheme == 'loo':
                        performances = self.loo(betas_per_session)
                    best_frac_inds = np.argmax(performances, axis=0)
                    unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        # save best regularization parameter
        best_fracs = self.fracs[best_frac_inds]
        for arr, fname in zip(
                [best_frac_inds, best_fracs],
                [self.best_frac_inds_nii, self.best_fracs_nii]
        ):
            unmask(arr, self.union_mask).to_filename(fname)

        print('\nFinal fit\n')
        best_param_inds = np.ravel_multi_index((best_hrf_inds, best_frac_inds), (self.nhrfs, self.nfracs))
        self.final_fit(data, convolved_designs, best_param_inds, condnames_runs, noise_mats)
        print('\nDone.\n')
    def mask_transform(self):
        #resampling the mask in the union mask template
        self.roi_mask=image.resample_to_img(self.roi_mask,self.union_mask,interpolation='nearest')
        #taking intersect of the two masks
        self.roi_mask_hf=image.resample_to_img(self.roi_mask_hf,self.union_mask,interpolation='nearest')
        #get the index of the union masks
        self.union_mask_index=np.where(self.union_mask.get_fdata().flatten()==1)[0]
        self.roi_mask_index=np.where(self.roi_mask.get_fdata().flatten()==1)[0]#getting the index of the roi mask from the union mask index
        self.roi_mask_hf_index=np.where(self.roi_mask_hf.get_fdata().flatten()==1)[0]#getting the index of the roi mask from the union mask index
        #getting the index of the roi mask from the union mask index
        self.target_mask_index=np.where(self.roi_mask_index[:,None]==self.union_mask_index[None,:])[1]
        self.source_mask_index=np.where(self.roi_mask_hf_index[:,None]==self.union_mask_index[None,:])[1]
        self.nvox_target_masked=self.target_mask_index.shape[0]
    def mask_transform2(self):
        """Different in this version: combined mask not taken into account"""
        #resampling the roi mask into the union mask template
        #get the index of the intersection of the union mask and roi mask
        self.union_mask_index=np.where(self.union_mask.get_fdata().flatten()==1)[0]
        #get the index of the roi mask of the hf
        self.roi_mask_hf_index=np.where(self.roi_mask_hf.get_fdata().flatten()==1)[0]
        #get the intersection
        self.target_mask_index=np.where(self.union_mask_index[:,np.newaxis]==self.roi_mask_hf_index[np.newaxis,:])[0]
        self.source_mask_index=self.target_mask_index
    def run_combi(self):
        print('\nLoading design and noise regressors\n')
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()#returns bold_files,.... in form of list 
        self.add_union_mask(masks)#get the masks based on intersection of all masks
        convolved_designs, convolved_onoff, condnames_runs, rep_condnames = self.make_designs(event_files)#get the convolved design matrix and the convolved on off matrix along with the condition names

        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [self.make_noise_mat(nuisance_tsv, ica_tsv)
                          for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)]#get the noise design matrix 
        else:
            noise_mats = [self.make_noise_mat(nuisance_tsv) for nuisance_tsv in nuisance_tsvs]
        #standardizing the design matrices
        if self.standardize_noiseregs:
            print('\nStandardizing noise regressors')
            noise_mats = [np.nan_to_num(zscore(m, axis=0)) for m in noise_mats]
        if self.standardize_trialegs:
            print('\nStandardizing trial regressors')
            convolved_designs = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_designs]
            convolved_onoff = [np.nan_to_num(zscore(m, axis=1)) for m in convolved_onoff]
            

        if self.on_residuals_design:
            print('\nRegressing out noise from design\n')
            convolved_designs, convolved_onoff = self.orthogonalize_designmats(
                convolved_designs, convolved_onoff, noise_mats
            )#orthogonalize the design matrix with the noise matrix[remove the noise regressors from the design matrix]

        print('\nLoading data\n')
        #running the data creator from bold signal in batches
        batch_size=20
        data = np.vstack([self.vstack_data_masked(bold_files[batch_i-batch_size:batch_i], rescale_runwise=self.rescale_runwise_data)
                                     for batch_i in [i for i in range(batch_size,len(bold_files)+1,batch_size)]])
        if self.on_residuals_data:
            #data=np.vstack([data,self.vstack_data_masked(bold_files[100:120], rescale_runwise=self.rescale_runwise_data)
            print('\nRegressing out noise from data\n')
            data = self.regress_out_noise_runwise(noise_mats, data, zscore_residuals=True)

        if self.on_residuals_data or self.on_residuals_design:
            print('\nOmitting noise regressors for model fitting since data and/or design was orthogonalized\n')
            noise_mats = []
        #assigning the necesary variables as class attributes before returning
        self.data=data
        self.convolved_designs=convolved_designs
        self.convolved_onoff=convolved_onoff
        self.condnames_runs=condnames_runs
        self.rep_condnames=rep_condnames
        self.noise_mats=noise_mats
    def hrf_alpha_overfit(self):
        #assigning the variables to the class attributes
        data=self.data
        convolved_designs=self.convolved_designs
        convolved_onoff=self.convolved_onoff
        condnames_runs=self.condnames_runs
        rep_condnames=self.rep_condnames
        noise_mats=self.noise_mats
        if self.tuning_procedure == 'combined':
            if self.usecache and os.path.exists(self.best_hrf_nii) and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading HRF indices and alpha fractions from pre-stored files')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                print('\nCombined tuning of HRF and regularization')
                best_hrf_inds, best_frac_inds, max_performance = self.combined_tuning(
                    data, convolved_designs, condnames_runs, rep_condnames, noise_mats
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
                unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        elif self.tuning_procedure == 'stepwise':
            if self.assume_hrf:
                print(f'\nAssuming HRF with index {self.assume_hrf}, creating temporary file\n')
                assert np.abs(self.assume_hrf) < self.nhrfs
                best_hrf_inds = np.full(fill_value=self.assume_hrf, shape=self.nvox_masked, dtype=int)
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)

            if self.usecache and os.path.exists(self.best_hrf_nii):
                print('\nLoading HRF indices from pre-stored file\n')
                best_hrf_inds = apply_mask(self.best_hrf_nii, self.union_mask, dtype=int)
            else:
                print('\nOverfitting HRF\n')
                if self.overfit_hrf_model == 'onoff':
                    best_hrf_inds, _ = self.overfit_hrf(data, convolved_onoff, noise_mats)
                    #return #POINT OF RETURN
                else:
                    best_hrf_inds, _ = self.overfit_hrf(data, convolved_designs, noise_mats)
                self.best_hrf_inds=best_hrf_inds#here hrf inds is a list of best hrf indices for each session
                # save best HRF per voxel
            # regularization
            if self.usecache and os.path.exists(self.best_frac_inds_nii):
                print('\nLoading best alpha fractions from pre-stored file\n')
                best_frac_inds = apply_mask(self.best_frac_inds_nii, self.union_mask, dtype=int)
            else:
                if self.cv_scheme == 'unregularized':
                    print('\nSkipping cross-validation and calculating unregularized betas\n')
                    best_frac_inds = np.full(fill_value=self.nfracs - 1, shape=self.nvox_masked, dtype=int)
                else:
                    print('\nEstimating betas of repeated stimuli for each session and each alpha fraction\n')
                    betas_per_session = self.get_repeated_betas(
                        data, convolved_designs, noise_mats, condnames_runs, rep_condnames, best_hrf_inds
                    )
                    print(f'\nUsing {self.cv_scheme} to find best alpha fraction per voxel')
                    if self.cv_scheme == 'mcnc':
                        performances = self.mcnc(betas_per_session)
                    elif self.cv_scheme == 'splithalf':
                        print(f"Using unregularized targets = {self.unregularized_targets}")
                        performances = self.splithalf(betas_per_session)
                    elif self.cv_scheme == 'loo':
                        performances = self.loo(betas_per_session)
                    best_frac_inds = np.argmax(performances, axis=0)
                    unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)

        # save best regularization parameter
        best_fracs = self.fracs[best_frac_inds]
        for arr, fname in zip(
                [best_frac_inds, best_fracs],
                [self.best_frac_inds_nii, self.best_fracs_nii]
        ):
            unmask(arr, self.union_mask).to_filename(fname)

        print('\nFinal fit\n')
        best_param_inds = np.ravel_multi_index((best_hrf_inds, best_frac_inds), (self.nhrfs, self.nfracs))
        self.final_fit(data, convolved_designs, best_param_inds, condnames_runs, noise_mats)
        print('\nDone.\n')

def get_betas_run(
        ses_i, run_i, data_ses, designs_ses, noisemats_ses, hrf_inds, fracs, nvox_masked,
        nconds=92, ntrs=284,
        fr_kws=dict(fit_intercept=False, normalize=False)
):
    fr = FracRidgeRegressor(fracs=fracs, **fr_kws)
    betas = np.zeros(shape=(nconds, len(fracs), nvox_masked), dtype=np.single)
    for hrf_i in tqdm(np.unique(hrf_inds), desc=f'Getting repeated betas, run {run_i} session {ses_i}', leave=False):
        voxel_inds = np.where(hrf_inds == hrf_i)
        data_vox = data_ses[run_i * ntrs:run_i * ntrs + ntrs, voxel_inds[0]]
        design = designs_ses[run_i][hrf_i]
        if noisemats_ses:
            design = np.hstack([design, noisemats_ses[run_i]])
        fr.fit(design, data_vox)
        if len(voxel_inds[0]) == 1:  # special case if only one voxel has this HRF index
            betas[:, :, voxel_inds[0]] = fr.coef_[:nconds, :, None]
        else:
            betas[:, :, voxel_inds[0]] = fr.coef_[:nconds]
    return betas


def overfit_hrf_to_chunk(data, convolved_designs, noise_mats, nvoxmasked, nhrfs,
                         reghrf_kws=dict(fit_intercept=False, normalize=False, n_jobs=-1)):
    reghrf = LinearRegression(**reghrf_kws)
    scores = np.zeros(shape=(nhrfs, nvoxmasked))
    for hrf_i in tqdm(range(nhrfs), desc='Iterating over sub-chunks for HRF overfitting'):
        design_hrf = block_diag(*[rundesign[hrf_i] for rundesign in convolved_designs])
        if noise_mats:
            design_hrf = np.hstack([design_hrf, block_diag(*noise_mats)])
        reghrf.fit(design_hrf, data)
        scores[hrf_i] = r2_score(data, reghrf.predict(design_hrf), multioutput='raw_values')
    return scores


def eval_split_comb(comb_i, comb, betas_per_session, nfracs, use_spearman_brown=False,
                    unregularized_targets=True, metric='l2'):
    start = time.time()
    split_train = np.mean(np.stack([b for i, b in enumerate(betas_per_session) if i in comb]), axis=0)
    split_test = np.mean(np.stack([b for i, b in enumerate(betas_per_session) if i not in comb]), axis=0)
    # each element of betas_per_session is (nconds, len(fracs), nvox_masked)
    if metric == 'correlation':
        if unregularized_targets: performance = np.stack(
                [pearsonr_nd(split_train[:, frac_i, :], split_test[:, -1, :]) for frac_i in range(nfracs)])
        else:
            performance = np.stack(
                [pearsonr_nd(split_train[:, frac_i, :], split_test[:, frac_i, :]) for frac_i in range(nfracs)])
        if use_spearman_brown:
            performance = spearman_brown(performance)
    elif metric == 'l1':
        if unregularized_targets:
            err = np.stack(
                [np.abs(split_train[:, frac_i, :] - split_test[:, -1, :]).sum(axis=0) for frac_i in range(nfracs)])
        else:
            err = np.stack(
                [np.abs(split_train[:, frac_i, :] - split_test[:, frac_i, :]).sum(axis=0) for frac_i in range(nfracs)])
        performance = err * -1
    elif metric == 'l2':
        if unregularized_targets:
            err = np.stack(
                [np.square(np.abs(split_train[:, frac_i, :] - split_test[:, -1, :])).sum(axis=0) for frac_i in
                 range(nfracs)])
        else:
            err = np.stack(
                [np.square(np.abs(split_train[:, frac_i, :] - split_test[:, frac_i, :])).sum(axis=0) for frac_i in
                 range(nfracs)])
        performance = err * -1
    print(f'Finished split combination {comb_i} in {((time.time() - start) / 60.):.2f} minutes')
    return performance


def eval_loo(test_i, betas_per_session, nfracs, unregularized_targets, metric):
    # betas_per_session is list of len 12, each element is shape (nrepeated, nfracs, nvox)
    betas_test = betas_per_session[test_i]  # shape (nrepeated, nfracs, nvox)
    betas_train = np.mean(np.stack([b for sesi, b in enumerate(betas_per_session) if sesi != test_i]),
                          axis=0)  # shape (nrepeated, nfracs, nvox)
    if metric == 'correlation':
        if unregularized_targets:
            performance = np.stack(
                [pearsonr_nd(betas_train[:, frac_i, :], betas_test[:, -1, :]) for frac_i in range(nfracs)])
        else:
            performance = np.stack(
                [pearsonr_nd(betas_train[:, frac_i, :], betas_test[:, frac_i, :]) for frac_i in range(nfracs)])
    elif metric == 'l1':
        if unregularized_targets:
            err = np.stack(
                [np.abs(betas_train[:, frac_i, :] - betas_test[:, -1, :]).sum(axis=0) for frac_i in range(nfracs)])
        else:
            err = np.stack(
                [np.abs(betas_train[:, frac_i, :] - betas_test[:, frac_i, :]).sum(axis=0) for frac_i in range(nfracs)])
        performance = err * -1
    elif metric == 'l2':
        if unregularized_targets:
            err = np.stack(
                [np.square(np.abs(betas_train[:, frac_i, ] - betas_test[:, -1, :])).sum(axis=0) for frac_i in
                 range(nfracs)])
        else:
            err = np.stack(
                [np.square(np.abs(betas_train[:, frac_i, :] - betas_test[:, frac_i, :])).sum(axis=0) for frac_i in
                 range(nfracs)])
        performance = err * -1
    return performance.astype(np.single)


def list_stb_outputs_for_mcnc(
        sub: str = '01',
        bidsroot: str = '/LOCAL/ocontier/thingsmri/bids',
        betas_basedir: str = '/LOCAL/ocontier/thingsmri/bids/derivatives/betas',
        stack_betas: bool = True,
        is_return_condnames: bool = False,
) -> tuple:
    betas_dir = pjoin(betas_basedir, f'sub-{sub}')#creating betas basedir for subvject 
    betas = []#specifying empty list for betas
    if is_return_condnames:
        condnames = []#specifying empty list for condition names
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
        if is_return_condnames:condnames.append(conds[repcond_is])#extracting the repeated condition names for the session
        #vectorized alternative
        rep_betas = ses_betas[repcond_is]#extracting the beta weights for the repeated conditions
        betas.append(rep_betas)#adding the beta weights to the list[in the end list will contain all the beta weights for all the repeated conditions across all sessions]
    if stack_betas:
        betas = np.moveaxis((np.moveaxis(np.stack(betas), 0, -1)), 0, -1)  # shape (voxX, voxY, nRepetitions, nStimuli)
    if is_return_condnames:return betas, run_niis[0], condnames[0]
    else:return betas, run_niis[0]


def _run_job_sim_notnested(overallmn, signalsd, noisesd):
    simsig = normal(overallmn, signalsd)
    mes = simsig + normal(0, noisesd)
    return r2_ndarr(simsig, mes)


def match_lsq(reg, unreg, nnls: bool = False):
    lr = LinearRegression(fit_intercept=True, n_jobs=-1)#, positive=nnls)
    lr.fit(reg.reshape(-1, 1), unreg)
    pred = lr.predict(reg.reshape(-1, 1))
    return pred


def match_nnslope(reg, unreg):
    reg_c, unreg_c = reg - reg.mean(), unreg - unreg.mean()
    lr = LinearRegression(fit_intercept=False, n_jobs=-1, positive=True)
    lr.fit(reg_c.reshape(-1, 1), unreg_c)
    pred_c = lr.predict(reg_c.reshape(-1, 1))
    pred = pred_c + unreg.mean()
    return pred


def load_betas(
        sub: str,
        mask: str,
        bidsroot: str = '/LOCAL/ocontier/thingsmri/bids',
        betas_derivname: str = 'betas_loo/on_residuals/scalematched',
        smoothing=0.,
        dtype=np.single,
) -> np.ndarray:
    if not smoothing:
        smoothing = None
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname, f'sub-{sub}')
    betafiles = [
        pjoin(betasdir, f'ses-things{ses_i:02d}', f'sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_betas.nii.gz')
        for ses_i in range(1, 13) for run_i in range(1, 11)
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

def load_betas_diffpath(
        sub: str,
        mask: str,
        #bidsroot: str = 'uLOCAL/ocontier/thingsmri/bids',
        betas_path:str='/DATA/satwick22/Documents/fMRI/fMRI_processing/trash/',
        betas_derivname: str = '',
        smoothing=0.,
        dtype=np.single,
) -> np.ndarray:
    if not smoothing:
        smoothing = None
    betasdir = pjoin(betas_path,'derivatives',betas_derivname, f'sub-{sub}')
    betafiles = [
        pjoin(betasdir, f'ses-things{ses_i:02d}', f'sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_betas.nii.gz')
        for ses_i in range(1, 13) for run_i in range(1, 11)
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


def load_betas_img(sub, bidsroot, betas_derivname='betas_loo/on_residuals/scalematched', dtype=np.single):
    betafiles = [
        pjoin(bidsroot, 'derivatives', betas_derivname,
              f'ses-things{ses_i:02d}', f'sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_betas.nii.gz')
        for ses_i in range(1, 13) for run_i in range(1, 11)
    ]
    return load_img(betafiles)
def read_condnames():
    pass
def load_filenames(sub, bidsroot, betas_derivname):
    # load condition names
    betas_basedir = pjoin(bidsroot, 'derivatives', betas_derivname)
    tsv_files = [
        pjoin(betas_basedir, f'sub-{sub}', f'ses-things{ses_i:02d}',
              f'sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_conditions.tsv')
        for ses_i in range(1, 13) for run_i in range(1, 11)
    ]
    filenames = np.hstack([pd.read_csv(tsv, sep='\t').image_filename.to_numpy() for tsv in tsv_files]).astype(str)
    return filenames


def filter_catch_trials(betas, filenames):
    noncatch_is = np.array([False if 'catch' in f else True for f in filenames])
    betas_noncatch, filenames_noncatch = betas[noncatch_is], filenames[noncatch_is]
    return betas_noncatch, filenames_noncatch, noncatch_is


def average_betas_per_concept(betas, filenames):
    """
    catch trials should have been excluded. Does not distinguish between test and normal trials.
    """
    trial_concepts = np.array([
        fn.split('/')[1][:-8]
        for fn in filenames
    ])
    concepts, counts = np.unique(trial_concepts, return_counts=True)
    assert np.sum(counts % 12) == 0
    betas_concepts = np.zeros((len(concepts), betas.shape[1]))
    for i, c_ in tqdm(enumerate(concepts), desc='averaging betas per concept'):
        mask = trial_concepts == c_
        assert np.sum(mask) % 12 == 0
        betas_c_ = betas[mask].mean(axis=0)
        betas_concepts[i] = betas_c_
    return np.stack(betas_concepts), concepts


def posthoc_scaling(
        sub='01',
        bidsroot='/LOCAL/ocontier/thingsmri/bids',
        reg_derivname='betas/regularized',
        unreg_derivname='betas/unregularized',
        out_derivname='betas/scalematched',
        njobs=30,
        nconds=92,
        method: str = 'ols'
):
    """
    Use OLS to find (intercept) and scalar to match the regularized betas to the scale of the unregularized betas.
    """
    assert method in ['ols', 'nnls', 'nnslope']
    # set up output directory
    outdirbase='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_'
    outdir = pjoin(outdirbase, 'derivatives', out_derivname, f'sub-{sub}')
    # get union mask
    thingsglm = THINGSGLM(bidsroot, sub, out_deriv_name='tmp')
    _, _, _, masks = thingsglm.get_inputs()
    thingsglm.add_union_mask(masks)
    # load regularized betas
    betas_dir_reg=pjoin(outdirbase,reg_derivname)
    reg_betas = load_betas(sub, thingsglm.union_mask, outdirbase, reg_derivname)
    # load unregularized betas
    betas_dir_unreg=pjoin(outdirbase,unreg_derivname)
    unreg_betas = load_betas(sub, thingsglm.union_mask,outdirbase, unreg_derivname)
    # run regressions in parallel
    nvox = reg_betas.shape[1]
    print('\nrescaling\n')
    if method in ['ols', 'nnls']:
        #matched_l=[]
        #for vox_i in tqdm(range(nvox), 'voxels'):
        #    matched_l.append(match_lsq(reg=reg_betas[:, vox_i],unreg=unreg_betas[:, vox_i],nnls=False))#True if method == 'nnls' else False))
        with Parallel(n_jobs=njobs) as parallel:
            matched_l = parallel(
                delayed(match_lsq)(reg_betas[:, vox_i],
                    unreg=unreg_betas[:, vox_i],
                    nnls=True if method == 'nnls' else False
                )
                for vox_i in tqdm(range(nvox), 'voxels')
            )
    elif method == 'nnslope':
        with Parallel(n_jobs=njobs) as parallel:
            matched_l = parallel(
                delayed(match_nnslope)(
                    reg=reg_betas[:, vox_i],
                    unreg=unreg_betas[:, vox_i],
                )
                for vox_i in tqdm(range(nvox), 'voxels')
            )
    matched = np.stack(matched_l, axis=1)
    # save output
    for ses_i in tqdm(range(12), desc='saving output for each session'):#for each session
        sesdir = pjoin(outdir, f'ses-things{ses_i + 1:02d}')
        if not os.path.exists(sesdir):
            os.makedirs(sesdir)
        for run_i in tqdm(range(10), desc='runs'): #for each run
            flat_i = ses_i * 10 + run_i
            starti, stopi = flat_i * nconds, flat_i * nconds + nconds
            matched_img = unmask(matched[starti:stopi], thingsglm.union_mask)
            nii = pjoin(sesdir, f'sub-{sub}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz')#creating the file name
            matched_img.to_filename(nii)#saving the file
    # copy condition files
    orig_tsvs = glob.glob(pjoin(bidsroot, 'derivatives', reg_derivname, f'sub-{sub}', 'ses-*', '*.tsv'))
    matched_tsvs = [t.replace(reg_derivname, out_derivname) for t in orig_tsvs]
    for src, trgt in tqdm(zip(orig_tsvs, matched_tsvs), 'copying tsv files'):
        copyfile(src, trgt)
    return
def roi_reader(stb_obj,mask_dir,sub_id,reg_id,reg_ind):
    """Reads the mask files for the given subject and region id"""
    #assigning reg_id to a class variable 
    run_id='02'
    sub_dir_base = pjoin(mask_dir, f'sub-{sub_id}')
    roi_path_dict={'EBA':'body_parcels','FFA':'face_parcels','OFA':'face_parcels','LOC':'object_parcels','PPA':'scene_parcels','STS':'face_parcels','RSC':'scene_parcels','TOS':'scene_parcels'}
    sub_dir=pjoin(sub_dir_base,roi_path_dict[reg_id])
    mask_img_left = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_l{reg_id}.nii.gz'))
    mask_img_right = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_r{reg_id}.nii.gz'))
    ##convert the numpy array to a nifti image
    roi_mask_image_path=pjoin(sub_dir,f'sub-{sub_id}_{reg_id}.nii.gz')
    #check if path exists
    if os.path.exists(roi_mask_image_path):
        roi_mask_img=image.load_img(roi_mask_image_path)
    else:
        roi_mask_l_data=mask_img_left.get_fdata()
        #convert the mask image to a numpy array
        roi_mask_r_data=mask_img_right.get_fdata()
        #add the numpy arrays
        roi_mask_data=roi_mask_l_data+roi_mask_r_data
        #convert the numpy array to a nifti image
        roi_mask_img=image.new_img_like(mask_img_left,roi_mask_data)
    #find the best fitting hrf for each of the hemisphere
    #checking if this is the first run of the function
    stb_obj.roi_mask=roi_mask_img
    stb_obj.roi_hrf_fit=True
    #stb_obj.reg_id=reg_id
    for reg_hf in ['l','r','']:
        stb_obj.reg_id=reg_hf+reg_id
        curr_out_deriv_name=f'betas_combi_run{run_id}/unregularized_{reg_hf}{reg_id}'
        stb_obj.change_out_dir(curr_out_deriv_name)
        if reg_ind==0 and reg_hf=='l':
            #stb_obj.run_combi()
            gc.collect()
        if reg_hf=='l': 
            stb_obj.roi_mask_hf=mask_img_left
        elif reg_hf=='r':
            stb_obj.roi_mask_hf=mask_img_right
        else:
            stb_obj.roi_mask_hf=roi_mask_img
        stb_obj.mask_transform()
        stb_obj.hrf_overfit_by_session()
def roi_reader2(stb_obj,out_derivname_base,mask_dir,sub_id,reg_id,reg_ind):
    """Reads the mask an sets the outdir for the particular ROI"""
    #mask_dir=pjoin(mask_dir,f'{n_runs}_runs')#adding the number of runs to the mask dir
    sub_dir_base = pjoin(mask_dir, f'sub-{sub_id}')
    roi_path_dict={'EBA':'body_parcels','FFA':'face_parcels','OFA':'face_parcels','LOC':'object_parcels','PPA':'scene_parcels','STS':'face_parcels','RSC':'scene_parcels','TOS':'scene_parcels'}
    sub_dir=pjoin(sub_dir_base,roi_path_dict[reg_id])
    mask_img_left = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_l{reg_id}.nii.gz'))
    mask_img_right = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_r{reg_id}.nii.gz'))
    #stb_obj.reg_id=reg_id
    for reg_hf in ['l','r']:
        stb_obj.reg_id=reg_hf+reg_id
        curr_out_deriv_name=pjoin(out_derivname_base,f'{reg_hf}{reg_id}')
        stb_obj.change_out_dir(curr_out_deriv_name)
        if reg_ind==0 and reg_hf=='l':
            #stb_obj.run_combi()
            gc.collect()
        if reg_hf=='l': 
            stb_obj.roi_mask_hf=mask_img_left
        elif reg_hf=='r':
            stb_obj.roi_mask_hf=mask_img_right
        stb_obj.mask_transform2()
        stb_obj.overfit_parameter()
def roi_reader3(stb_obj,out_derivname_base,mask_dir,sub_id,reg_id,reg_ind,n_runs,combine_method):
    """Reads the mask an sets the outdir for the particular ROI"""
    stb_obj.flush_roi()
    mask_dir=pjoin(mask_dir,f'{n_runs}_runs')#adding the number of runs to the mask dir
    sub_dir = pjoin(mask_dir, f'sub-{sub_id}')
    mask_img = image.load_img(pjoin(sub_dir, f'sub_{sub_id}_{reg_id}_{combine_method}.nii.gz'))
    #stb_obj.reg_id=reg_id
    stb_obj.reg_id=reg_id
    curr_out_deriv_name=pjoin(out_derivname_base,f'{reg_id}')
    stb_obj.change_out_dir(curr_out_deriv_name)
    gc.collect()
    stb_obj.roi_mask_hf=mask_img#reusing mask_hf variable for storing the mask image of entire area
    stb_obj.mask_transform2()
    stb_obj.overfit_parameter()
def driver_roi_overfit(stb_obj,run_id):
    """Receives the object of the SingleTrialBetas after data loading and the outdir base"""
    #creating the outdir
    #first getting the best hrf indices based upon mode
    stb_obj.roi_hrf_fit=True
    for hrf_inds_by in ['mode','mean']:
        stb_obj.best_hrf_inds_by=hrf_inds_by
        out_derivname=pjoin(f'betas_combi_roi_run{run_id}',f'unregularized_by{self.best_hrf_inds_by}')
        stb_obj.change_out_dir(out_derivname)
        #calling the hrf alpha overfit function
        stb_obj.hrf_alpha_overfit_rois()




if __name__ == '__main__':
    import sys
    from fracridge import FracRidgeRegressor
    #sub, bidsroot = sys.argv[1], sys.argv[2]
    sub, bidsroot='01','/DATA1/satwick22/Documents/fMRI/thingsmri'
    mask_runid='final__'
    mask_dir=pjoin(bidsroot,'derivatives',f'roi_masks_run_{mask_runid}')#SET
    #mask_dir = pjoin('/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_roi','rois','category_localizer')
    #creating a list of all ROIs
    #roi_list=['EBA','OFA','PPA','RSC','TOS','FFA','LOC']#SET
    roi_list=['EBA','FFA']#set
    #iterate through the list of ROIs
    #creating a global variable for storing index of best fitting HRF
    #best_hrf_inds_by_reg={}
    #creating the stb_unreg object
    #sub, bidsroot='01','/DATA1/satwick22/Documents/fMRI/thingsmri'
    run_id='debug2'#SET
    betas_fit='regularized'#SET
    fixed_alpha=True#SET
    flag=False
    #create the object here
    stb_unreg=SingleTrialBetas(bidsroot=bidsroot, subject=sub, out_deriv_name='betas_dummy')
    best_frac_methods=['mean']#SET
    best_hrf_methods=['mode']#SET
    n_shuffle=2#SET
    n_shuffle_offset=1#SET
    #create the empty dictionary of 12 sessions for concatenating the betas

    #reading the bold data and removing the noise
    #loading data all at once
    stb_unreg.batched_data_loading=False
    stb_unreg.data_reader()
    for i_shuffle in range(n_shuffle_offset,n_shuffle+n_shuffle_offset):#SET
        print(f'Performing the estimation for Sample{i_shuffle+1}')
        if i_shuffle:#when shuffle is 0 estimation is performed for the original data
            stb_unreg.shuffle_data=True
        stb_unreg.convolve_creator()#creates the design matrices
        stb_unreg.flush_sample()#removing the result of the previoous sample
        for best_hrf_by in best_hrf_methods:
            print(f'Performing the estimation for HRF estimation by {best_hrf_by}')
            for nruns in range(6,2,-1):
                print(f'Performing the estimation for {nruns} runs')
                for combine_method in ['intersection']:
                    if nruns==6:
                        if combine_method=='union':
                            continue
            #for best_frac_by in best_frac_methods:
                    if betas_fit=='unregularized':
                        best_hrf_inds_by=best_hrf_by
                        #best_frac_inds_by='_'
                    elif betas_fit=='regularized':
                        best_hrf_inds_by=best_hrf_by
                        #best_frac_inds_by=best_frac_by 
                    out_derivname_base=pjoin(f'betas_roi_combi_run-{run_id}',f'{betas_fit}',f'hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}',f'sample{i_shuffle+1}') 
                    if betas_fit=='unregularized':
                        stb_unreg.cv_scheme = betas_fit
                    stb_unreg.best_hrf_inds_by=best_hrf_inds_by
                    if betas_fit=='regularized':
                        if fixed_alpha:#using 0.1 as the fixed alpha
                            stb_unreg.assume_alpha=stb_unreg.fracs[0]
                    #calling the roi reader version 2
                    stb_unreg.flush_specification()
                    for roi_ind, roi in enumerate(roi_list):
                        #initialize the dictionary for storing the sessionwise beta
                        roi_reader3(stb_unreg,out_derivname_base,mask_dir,sub,roi,roi_ind,nruns,combine_method)
                        #concatenating the betas from session dict to concat dict
                        for ses_i in range(stb_unreg.n_sessions):
                            if not np.any(stb_unreg.betas_concat[ses_i+1]):
                                stb_unreg.betas_concat[ses_i+1]=stb_unreg.betas_session[ses_i+1]
                            else:    
                                stb_unreg.betas_concat[ses_i+1]=np.concatenate((stb_unreg.betas_concat[ses_i+1],stb_unreg.betas_session[ses_i+1]),axis=0)

                    #save the dictionary of betas and condnames
                    #iterate through the sessions
                    for ses_i in range(stb_unreg.n_sessions):
                        #if ses_i not in [2,3,7]:continue
                        #extract the concatenated beta weights
                        betas_concat_ses_i=stb_unreg.betas_concat[ses_i+1]
                        #extract the concatenated condition names
                        condnames_concat_ses_i=stb_unreg.condnames_concat[ses_i+1]
                        #sort the condnames after removing the / and subsequent elements for a condition
                        condname_filtered=[condname.split('/')[0] for condname in condnames_concat_ses_i]
                        if not np.any(stb_unreg.condnames_sortind[ses_i+1]):
                            condnames_concat_ses_i_sorted_ind=np.argsort(np.array(condname_filtered))
                            stb_unreg.condnames_sortind[ses_i+1]=condnames_concat_ses_i_sorted_ind
                        #sort the beta weights after interchanging the rows and columns
                        betas_concat_ses_i_sorted=betas_concat_ses_i.reshape(-1,betas_concat_ses_i.shape[0])[condnames_concat_ses_i_sorted_ind]
                        #sort the condition
                        condnames_concat_ses_i_sorted=np.array(condname_filtered)[stb_unreg.condnames_sortind[ses_i+1]]
                        #save the ndarrays
                        betas_path=pjoin(bidsroot,'derivatives',out_derivname_base,f'sub-{sub}')
                        if not os.path.exists(betas_path):
                            os.makedirs(betas_path)
                        betas_file=f'ses-things{ses_i+1:02d}_betas'
                        condnames_path=pjoin(bidsroot,'derivatives',out_derivname_base,f'sub-{sub}')
                        if not os.path.exists(condnames_path):
                            os.makedirs(condnames_path)
                        condnames_file=f'ses-things{ses_i+1:02d}_condnames'
                        #save the ndarray 
                        save_ndarray(betas_concat_ses_i_sorted,betas_path,betas_file)
                        save_ndarray(condnames_concat_ses_i_sorted,condnames_path,condnames_file)

                    best_hrf_dir=pjoin(bidsroot,'derivatives',out_derivname_base)
                    if not os.path.exists(best_hrf_dir):
                        os.makedirs(best_hrf_dir)
                    best_hrf_file_path=pjoin(best_hrf_dir,f'sub-{sub}_best_hrf_by_{best_hrf_inds_by}.pickle')
                    with open(best_hrf_file_path, 'wb+') as f:
                        pickle.dump(stb_unreg.best_hrf_inds_by_roi,f)
                        #empty the dictionary
                        #stb_unreg.best_hrf_inds_by_roi={}
                    if betas_fit=='regularized' and not stb_unreg.assume_alpha:
                        best_frac_dir=pjoin(bidsroot,'derivatives',out_derivname_base)
                        if not os.path.exists(best_frac_dir):
                            os.makedirs(best_frac_dir)
                        best_frac_file_path=pjoin(best_frac_dir,f'sub-{sub}_best_hrf_by_{best_hrf_inds_by}_best_frac_by_{best_frac_inds_by}.pickle')
                        with open(best_frac_file_path, 'wb+') as f:
                            pickle.dump(stb_unreg.best_frac_inds_by_roi,f)
                            #empty the dictionary 
                            stb_unreg.best_frac_inds_by_roi={} 
    