"""RUNS STAT MODEL FOR BETA WEIGHT EXTRACTION"""
import gc
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
from nilearn.image import load_img,new_img_like
from nilearn.masking import unmask, apply_mask
from numpy.random import normal
from scipy.linalg import block_diag
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from glm import GODGLM, df_to_boxcar_design, get_nuisance_df
from utils import pearsonr_nd, get_hrflib, spearman_brown, match_scale, apply_mask_smoothed, regress_out, r2_ndarr



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

class SingleTrialBetas(GODGLM):
    """
    Calculate single trial response estimates for the GOD-fMRI dataset.
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
            on_residuals_design: bool = True,
            cv_scheme: str = 'loo',
            perf_metric: str = 'l2',
            assume_hrf: int or False = False,
            assume_alpha: int or False = 0.1,
            masking: bool = False,
            match_scale_runwise: bool = False,
            use_spearman_brown: bool = False,
            unregularized_targets: bool = True,
            hrflib_url: str = 'https://raw.githubusercontent.com/kendrickkay/GLMdenoise/master/utilities'
                              '/getcanonicalhrflibrary.tsv',
            rescale_hrflib_amplitude: bool = True,
            hrflib_resolution: float = .1,
            stim_duration: float = 9,
            overfit_hrf_model: str = 'onoff',
            fracs: np.ndarray = np.hstack([np.arange(.1, .91, .05), np.arange(.91, 1.01, .01)]),
            fmriprep_noiseregs: list = [],
            fmriprep_compcors: bool or int = 0,
            aroma_regressors: bool = False,
            manual_ica_regressors: bool = False,
            nuisance_regressors: bool = False,
            drift_model: str = 'polynomial',
            poly_order: int = 4,
            high_pass: float = None,
            rescale_runwise_data: str = 'z',
            batched_loading: bool = False,
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
        self.batched_loading=batched_loading
        self.manual_ica_regressors = manual_ica_regressors
        self.nuisance_regressors = nuisance_regressors
        self.usecache = usecache
        self.on_residuals_data = on_residuals_data
        self.on_residuals_design = on_residuals_design
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
        self.stim_duration = stim_duration
        self.unregularized_targets = unregularized_targets
        self.nfracs = len(self.fracs)
        assert rescale_runwise_data in ['z', 'psc', 'center']  # don't allow uncentered data
        self.rescale_runwise_data = rescale_runwise_data
        assert overfit_hrf_model in ['onoff', 'single-trial']
        self.overfit_hrf_model = overfit_hrf_model
        self.assume_hrf = assume_hrf  # picked 10 as canonical hrf
        self.assume_alpha= assume_alpha
        self.apply_mask=masking
        self.n_parallel_hrf = n_parallel_hrf
        self.hrflib_resolution = hrflib_resolution
        self.rescale_hrflib_amplitude = rescale_hrflib_amplitude
        self.microtime_factor = int(self.tr / self.hrflib_resolution)  # should be 30 in our case
        self.frame_times_microtime = np.arange(0, self.ntrs * self.tr, self.hrflib_resolution) + self.stc_reftime
        self.frame_times=np.arange(0, self.ntrs * self.tr, self.tr) + self.stc_reftime
        self.frf = FracRidgeRegressor(fracs=1., fit_intercept=False, normalize=False)
        # mcnc settings
        self.mcnc_n_sig, self.mcnc_n_mes = mcnc_nsig, mcnc_nmes
        self.mcnc_njobs = mcnc_njobs
        self.mcnc_ddof = mcnc_ddof
        # directories and files
        self.workdirbase='/DATA1/satwick22/Documents/fMRI/wdir/python_work_dir'
        self.workdir = pjoin(self.workdirbase, 'betas_py')
        self.outdirbase='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids'
        self.outdir = pjoin(self.outdirbase, 'derivatives', self.out_deriv_name, f'sub-{self.subject}')
        self.best_hrf_nii = pjoin(self.outdir, 'best_hrf_inds.nii.gz')
        self.best_frac_inds_nii = pjoin(self.outdir, 'best_frac_inds.nii.gz')
        self.max_performance_nii = pjoin(self.outdir, 'max_performance.nii.gz')
        self.best_fracs_nii = pjoin(self.outdir, 'best_fracs.nii.gz')
        self.out_dtype = out_dtype
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

    def _make_design_df(self, event_file):
        """
        Make data frame containing specifying each trial as a separate condition.
        Also returns the list of condition names, and condition names for the repeated trials.
        """
        design_df = pd.read_csv(event_file, sep='\t')[['duration', 'onset', 'image_file', 'event_type','response_time']]
        #extract the events where the event type is rest
        design_df.loc[design_df['event_type'] == 'rest', 'image_file'] = 'rest'
        #rename the one back trials with 'r' at the end
        #check if the current image file is the same as the previous image file
        one_back_index=design_df['image_file']==design_df['image_file'].shift(1)
        design_df.loc[one_back_index,'image_file'] = design_df.loc[one_back_index, 'image_file'].replace('.JPEG', 'r.JPEG',regex=True)
        rep_condnames = design_df.loc[(design_df['event_type'] == 'stimulus') & ~(one_back_index), 'image_file'].to_numpy()
        design_df = design_df.drop(columns=['event_type','response_time'])
        design_df = design_df.rename(columns={'image_file': 'trial_type'})
        design_df = design_df.sort_values(by='onset', ignore_index=True)
        return design_df, rep_condnames

    def _onoff_df_from_design_df(self, design_df):
        """Take a single trial design data frame and turn it into an onoff design data frame"""
        onoff_df = design_df.copy(deep=True)
        onoff_df['trial_type'] = 'onoff'
        return onoff_df

    def make_design_dfs(self, event_files):
        """
        Given our event files, give us the design data frames and condition names per run
        as well as the list of unique names (across all runs) of repeated conditions.
        """
        designs_zipped = [self._make_design_df(ef) for ef in tqdm(event_files, desc='reading event files')]
        design_dfs, rep_condnames_runs = list(zip(*designs_zipped))
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
        design_dfs, onoff_dfs, rep_condnames = self.make_design_dfs(event_files)#gets the design dfs and onoff dfs,where onoff dfs are just design_dfs with onoff on trial type column
        def debug_dfs():
            df_to_boxcar_design(design_dfs[0], self.frame_times_microtime)
        debug_dfs()
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
        mean_r2s=np.nanmean(np.stack(chunk_scores), axis=0)
        best_hrf_inds = np.argmax(mean_r2s, axis=0)
        print(f'HRF overfitting completed in {(time.time() - start) / 60:.1f} minutes')
        return best_hrf_inds, mean_r2s

    def get_repeated_betas(self, data, convolved_designs, noise_mats, condnames_runs, rep_condnames, hrf_inds):
        """
        Fit single trial GLM to each session and return the beta estimates of the repeatedly presented stimuli.
        """
        betas_per_session=[]
        for ses_i in tqdm(range(self.n_sessions), desc='Getting repeated stimuli runwise', leave=True):
            startrun, stoprun = sum([self.nruns_perses_[self.ds.target_sessions[ses_prev]] for ses_prev in range(ses_i)]), sum([self.nruns_perses_[self.ds.target_sessions[ses_ind]] for ses_ind in range(ses_i+1)])
            startsample, stopsample = startrun * self.ntrs, stoprun * self.ntrs
            data_ses = data[startsample:stopsample]
            designs_ses = convolved_designs[startrun:stoprun]
            noisemats_ses = noise_mats[startrun:stoprun] if noise_mats else []
            condnames_ses = np.hstack(condnames_runs[startrun:stoprun])
            repis_ses = np.hstack([np.where(condnames_ses == repcond) for repcond in rep_condnames]).squeeze()
            # _ = get_betas_run(0, 0, data_ses, designs_ses, noisemats_ses, hrf_inds, self.fracs, self.nvox_masked)
            with Parallel(n_jobs=-1) as parallel:
                betas_per_run = parallel(
                    delayed(get_betas_run)(  # TODO: could be parallelized more elegantly
                        ses_i, run_i, data_ses, designs_ses, noisemats_ses, hrf_inds, self.fracs, self.nvox_masked,
                    )
                    for run_i in range(self.nruns_perses_[self.ds.target_sessions[ses_i]])
                )
            # saving the betas per run into the list betas_per_session
            betas_per_session.append(np.concatenate(betas_per_run, axis=0)[repis_ses])
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
            sesdir = pjoin(self.outdir, f'ses-{self.target_session}{ses_i + 1:02d}')
            ses_run_buffer=sum([self.nruns_perses_[self.ds.target_sessions[ses_prev]] for ses_prev in range(ses_i)])#computed the session index buffer
            if not os.path.exists(sesdir):
                os.makedirs(sesdir)
            for run_i in tqdm(range(self.nruns_perses_[self.ds.target_sessions[ses_i]]), desc='runs'):
                # figure out indices
                flat_i = ses_run_buffer + run_i
                startsample, stopsample = flat_i * self.ntrs, flat_i * self.ntrs + self.ntrs             
                nconds = len(condnames_runs[flat_i])
                # iterate over voxel sets and populate our results array
                betas = np.zeros(shape=(self.nvox_masked, nconds))
                for param_i in tqdm(np.unique(best_param_inds), 'parameter combinations'):
                    hrf_i, frac_i = np.unravel_index(param_i, shape=(self.nhrfs, self.nfracs))
                    voxel_inds = np.where(best_param_inds == param_i)
                    data_sub = data[startsample:stopsample, voxel_inds[0]].squeeze()
                    design = convolved_designs[flat_i][hrf_i]
                    if noise_mats:
                        design = np.hstack([design, noise_mats[flat_i]])
                    if self.match_scale_runwise:
                        self.frf.fracs = [self.fracs[frac_i], 1.]
                        self.frf.fit(design, data_sub)
                        betas_thisparam = match_scale(self.frf.coef_[:nconds, 0], self.frf.coef_[:nconds, 1])
                    else:
                        self.frf.fracs = self.fracs[frac_i]#setting the alpha value
                        self.frf.fit(design, data_sub) #fitting the model
                        betas_thisparam = self.frf.coef_[:nconds]
                    betas[voxel_inds[0]] = betas_thisparam.T
                # save betas and condition names for this run to file
                betas_nii = pjoin(
                    sesdir, f'sub-{self.subject}_ses-{self.target_session}{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz'
                )
                conds_tsv = betas_nii.replace('_betas.nii.gz', '_conditions.tsv')
                betas_img = unmask(betas.T.astype(self.out_dtype), self.union_mask)
                betas_img.to_filename(betas_nii)
                pd.DataFrame(condnames_runs[flat_i]).to_csv(conds_tsv, sep='\t', header=['image_filename'])

    def read_data(self):
        
        print('\nLoading design and noise regressors\n')
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()#returns bold_files,.... in form of list 
        self.add_union_mask(masks)#get the masks based on intersection of all masks
        if not self.apply_mask:# if we do  not want masked data
            #extract the tuple for shape of the mask
            one_mask = np.ones(self.union_mask.shape,dtype=np.int8)
            self.union_mask = new_img_like(self.union_mask, one_mask)
        

    #@profile
    def run(self):
        print('\nLoading design and noise regressors\n')
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()#returns bold_files,.... in form of list 
        self.add_union_mask(masks)#get the masks based on intersection of all masks
        if not self.apply_mask:# if we do  not want masked data
            #extract the tuple for shape of the mask
            one_mask = np.ones(self.union_mask.shape,dtype=np.int8)
            self.union_mask = new_img_like(self.union_mask, one_mask)

        convolved_designs, convolved_onoff, condnames_runs, rep_condnames = self.make_designs(event_files)#get the convolved design matrix and the convolved on off matrix along with the condition names

        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [self.make_noise_mat(nuisance_tsv, ica_tsv)
                          for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)]#get the noise design matrix 
        elif self.nuisance_regressors:
            noise_mats = [self.make_noise_mat(nuisance_tsv) for nuisance_tsv in nuisance_tsvs]
        else:
            noise_mats = []
        #standardizing the design matrices
        if self.standardize_noiseregs:
            if np.any(noise_mats):
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
        if self.batched_loading:
            print('\nLoading data\n')
            #running the data creator from bold signal in batches
            batch_size=20
            data = np.vstack([self.vstack_data_masked(bold_files[batch_i-batch_size:batch_i], rescale_runwise=self.rescale_runwise_data)
                                         for batch_i in [i for i in range(batch_size,len(bold_files)+1,batch_size)]])
        else:
            data=self.vstack_data_masked(bold_files, rescale_runwise=self.rescale_runwise_data)
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
                    if self.assume_alpha:
                        best_frac_inds = np.full(fill_value=np.where(self.fracs == self.assume_alpha)[0][0],
                                                 shape=self.nvox_masked, dtype=int)
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
                        max_performance = np.max(performances, axis=0)
                        best_frac_inds = np.argmax(performances, axis=0)
                        unmask(max_performance, self.union_mask).to_filename(self.max_performance_nii)
        #default behaviour: only save if the best frac indices are estimated
        if not self.assume_alpha:
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

    ###function for creating rdm
    def rdm_creator(self):
        pass



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





def load_betas(
        sub: str,
        glm_obj : GODGLM,
        bidsroot: str = '/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
        betas_derivname: str = None,
        smoothing=0.,
        dtype=np.single,
) -> np.ndarray:
    if not smoothing:
        smoothing = None
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname, f'sub-{sub}')
    betafiles = [
        pjoin(betasdir, f'ses-{glm_obj.target_session}{ses_i+1:02d}', f'sub-{sub}_ses-{glm_obj.target_session}{ses_i+1:02d}_run-{run_i+1:02d}_betas.nii.gz')
        for ses_i in range(glm_obj.n_sessions) for run_i in range(glm_obj.nruns_perses_[glm_obj.ds.target_sessions[ses_i]])
    ]
    for b in betafiles:
        assert os.path.exists(b)
    with Parallel(n_jobs=-1) as parallel:
        betas_l = parallel(
            delayed(apply_mask_smoothed)(bf, glm_obj.union_mask, smoothing, dtype)
            for bf in tqdm(betafiles, desc='loading betas')
        )
    betas = np.vstack(betas_l)
    return betas  # shape (ntrials, nvoxel)


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

def posthoc_scaling(
        sub='01',
        bidsroot='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
        outdirbase='betas_run-test',
        out_derivname='scalematched',
        njobs=30,
        nconds=56,
        method: str = 'ols',
        masking: bool = False
):
    """
    Use OLS to find (intercept) and scalar to match the regularized betas to the scale of the unregularized betas.
    """
    assert method in ['ols', 'nnls', 'nnslope']
    # set up output directory
    outdir = pjoin(bidsroot,'derivatives', outdirbase,out_derivname, f'sub-{sub}')
    # get union mask
    godglm = GODGLM(bidsroot, sub, out_deriv_name='tmp')
    _, _, _, masks = godglm.get_inputs()
    godglm.add_union_mask(masks)
    if not masking:
        ones=np.ones(godglm.union_mask.shape,dtype=np.int8)
        godglm.union_mask=new_img_like(godglm.union_mask,ones)
    # load regularized betas
    betas_dir_reg=pjoin(outdirbase,'regularized')
    reg_betas = load_betas(sub, godglm, bidsroot=bidsroot,betas_derivname=betas_dir_reg)
    # load unregularized betas
    betas_dir_unreg=pjoin(outdirbase,'unregularized')
    unreg_betas = load_betas(sub, godglm,bidsroot=bidsroot,betas_derivname=betas_dir_unreg)
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
    for ses_i in tqdm(range(godglm.n_sessions), desc='saving output for each session'):#for each session
        sesdir = pjoin(outdir, f'ses-{godglm.target_session}{ses_i + 1:02d}')
        ses_run_buffer = sum([godglm.nruns_perses_[godglm.ds.target_sessions[ses_prev]] for ses_prev in range(ses_i)])#computing the session index buffer
        if not os.path.exists(sesdir):
            os.makedirs(sesdir)
        for run_i in tqdm(range(godglm.nruns_perses_[godglm.ds.target_sessions[ses_i]]), desc='runs'): #for each run
            flat_i = ses_run_buffer + run_i
            starti, stopi = flat_i * nconds, flat_i * nconds + nconds
            matched_img = unmask(matched[starti:stopi], godglm.union_mask)
            nii = pjoin(sesdir, f'sub-{sub}_ses-{godglm.target_session}{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz')#creating the file name
            matched_img.to_filename(nii)#saving the file
    # copy condition files
    orig_tsvs = glob.glob(pjoin(bidsroot, 'derivatives', betas_dir_reg, f'sub-{sub}', 'ses-*', '*.tsv'))
    matched_tsvs = [t.replace(betas_dir_reg, pjoin(outdirbase,out_derivname)) for t in orig_tsvs]
    for src, trgt in tqdm(zip(orig_tsvs, matched_tsvs), 'copying tsv files'):
        copyfile(src, trgt)
    return



if __name__=='__main__':
    import sys
    #sub, bidsroot = sys.argv[1], sys.argv[2]
    for sub_id in range(3,5):
        sub=f'0{sub_id}'
        bidsroot='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids'
        #get an unregularized estimate of responses
        stb_unreg = SingleTrialBetas(bidsroot=bidsroot, subject=sub, out_deriv_name='betas_run-02/unregularized',cv_scheme='unregularized',nuisance_regressors=True)
        stb_unreg.run()
    ## get a regularized estimate of responses
    #stb_reg = SingleTrialBetas(bidsroot=bidsroot, subject=sub, out_deriv_name='betas_run-02/regularized',nuisance_regressors=True)
    #stb_reg.run()
    #posthoc_scaling(sub=sub, bidsroot=bidsroot, outdirbase='betas_run-02',out_derivname='scalematched')