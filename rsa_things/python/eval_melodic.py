
"""
generates ICA feature maps
"""
#importing the necessary libararies
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



#defining the class for running eval melodic


class EvalMelodic:
    """
    Evaluate Melodic results in terms of ...
        - correlation with motion/design parameters
        - High frequency content
        - 'Edge Fraction'
    Estimated features are saved to a tsv file and visual reports are generated for manual classification.
    Since plotting and saving visual reports for all ~24,000 components would take very long (estimate ~18 hours),
    one may specify to generate reports at random with a certain probability.
    """

    def __init__(
            self,
            
            bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_',
            fmriprepdir='/DATA/satwick22/Documents/fMRI/fMRI_processing/bids/derivatives/fmriprep/',
            out_deriv_name: str = 'melodic_features',
            space: str = 'T1w',
            edgefrac_thickness: int = 2,
            report_threshold: float = .9,
            try_first_n: int = 0,  # 0 means run for all
            report_dpi: int = 200,
            random_report_prob: float = 0.,  # 1. means generate report always, .5 means only 50% of the time, etc.
            exclude_comp_ids: np.ndarray = None,
            # no new reports will be generated for these comp IDs (e.g. '2_things01_things_7_8)
            stc_reftime: float = 0.701625,
    ):
        self.fmriprepdir = fmriprepdir#specifying the fmriprep directory
        self.bidsroot = bidsroot#specifying the bidsroot directory
        self.space = space#specifying the space of the data
        self.out_deriv_name = out_deriv_name#specifying the oiutput directory
        self.out_basedir = pjoin(bidsroot, 'derivatives', out_deriv_name)#complete path of the output directory
        if not os.path.exists(self.out_basedir):#create the output dircetory if it doesn't exist
            os.makedirs(self.out_basedir)
        spaces_naming = {'T1w': '_space-T1w', 'func_preproc': ''}#ceating the space dictionary for extracting the space keyword depending on the required space 
        self.space_str = spaces_naming[space]#extracting the space keyword
        self.edgefrac_thickness = edgefrac_thickness#
        self.melodic_basedir = pjoin(self.bidsroot, 'derivatives', 'melodic_run_n1', 'runwise', f'space-{self.space}')#specifying and creating the path to the melodic output
        #self.physioregs_basedir = pjoin(self.bidsroot, 'derivatives', 'physio_regressors')#specifying and creating the path to the physio regressors
        self.ds = ThingsMRIdataset(self.bidsroot)#creating the dataser object with the bidsroot provided
        self.bidsobs = self.ds.layout.get(task='things',suffix='bold', extension='nii.gz')  # all runs as pybids objects i.e bids.layout.models.BIDSImageFile
        # shuffle_list(self.bidsobs)#shuffling the list of pybids files
        self.try_first_n = try_first_n#assigbing the value
        if self.try_first_n:#checking if the output needs to be limited to a certain number of files
            self.bidsobs = self.bidsobs[:self.try_first_n]
        self.render3 = get_render3_cmap()
        self.lr = LinearRegression(fit_intercept=True, normalize=True, n_jobs=20)
        self.dpi = report_dpi
        self.random_report_prob = random_report_prob#probability of geenrating report for a given run
        self.report_threshold = report_threshold#threshold for the report
        self.exclude_comp_ids = exclude_comp_ids
        self.stc_reftime = stc_reftime

        warnings.filterwarnings("ignore", category=UserWarning)




#get_designmatrix function

    def get_designmat(self, bo, runinfo):
        events_tsv = bo.path.replace('_bold.nii.gz', '_events.tsv')
        events_df = pd.read_csv(events_tsv, sep='\t')[['trial_type', 'onset', 'duration']]
        events_df['trial_type'] = 'all'
        designmat = make_first_level_design_matrix(
            frame_times=np.arange(0, runinfo['ntrs'] * runinfo['tr'], runinfo['tr']) + self.stc_reftime,
            events=events_df, hrf_model='spm', drift_model=None, high_pass=None, drift_order=None
        )['all']
        tmp_mean = np.mean(designmat[25:270])
        tmp_sd = np.std(designmat[25:270])
        designmat_rescaled = (designmat.to_numpy() - tmp_mean) / tmp_sd
        return designmat, designmat_rescaled




    def get_motion(self, runinfo):
        confounds_tsv = pjoin(
            self.fmriprepdir, f"sub-{runinfo['subject']}", f"ses-{runinfo['session']}", 'func',
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']:02d}_desc-confounds_timeseries.tsv")
        #print(confounds_tsv)
        confounds_df = pd.read_csv(confounds_tsv, sep='\t')
        return confounds_df[[c for c in confounds_df.columns if 'trans' in c or 'rot' in c]]






    def get_comps(self, runinfo):
        melodic_outdir_run =pjoin(
            self.melodic_basedir, f"sub-{runinfo['subject']}", f"ses-{runinfo['session']}",
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']:02d}_melodic"
        )
        #print(melodic_outdir_run)
        mixmat = np.loadtxt(pjoin(melodic_outdir_run, 'melodic_mix'))
        ica_nii_f = pjoin(melodic_outdir_run, 'melodic_IC.nii.gz')
        comps_arr = load_img(ica_nii_f).get_fdata()
        return mixmat, comps_arr






    def get_edge_mask(self, runinfo):
        brainmask_f =pjoin(
            self.fmriprepdir, f"sub-{runinfo['subject']}", f"ses-{runinfo['session']}", 'func',
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']:02d}{self.space_str}_desc-brain_mask.nii.gz"
        )
        csf_anat_f = pjoin(
            self.fmriprepdir, f"sub-{runinfo['subject']}", 'anat',
            f"sub-{runinfo['subject']}_acq-prescannormalized_rec-pydeface_label-CSF_probseg.nii.gz"
        )
        #print(csf_anat_f)
        #print(brainmask_f)
        csf_func = threshold_img(
            resample_to_img(csf_anat_f, brainmask_f, interpolation='linear'),
            threshold=1.
        )
        brainmask = load_img(brainmask_f).get_fdata()
        mask_img = math_img('img1 - img2', img1=brainmask_f, img2=csf_func)
        mask_arr = mask_img.get_fdata()
        # worked okayish with erosion iterations=2
        ero_mask = binary_erosion(mask_arr, iterations=self.edgefrac_thickness).astype(int)
        edgemask = mask_arr - ero_mask
        return edgemask.astype(bool), brainmask.astype(bool)


    def calc_fits(self,comp_motion_df, maxmo_ind, comp_ts):
        """Fit motion regressors to component timeseries. Returns predicted time series."""
        #maxphy_ts = np.nan_to_num(comp_physio_df.to_numpy()[:, maxphy_ind + 1].reshape(-1, 1))
        #self.lr.fit(maxphy_ts, comp_ts)
        #phy_fit = self.lr.predict(maxphy_ts)
        maxmo_ts = np.nan_to_num(comp_motion_df.to_numpy()[:, maxmo_ind + 1].reshape(-1, 1))
        self.lr.fit(maxmo_ts, comp_ts)
        mo_fit = self.lr.predict(maxmo_ts)
        return mo_fit




    def calc_edgefrac(self,comp_arr, edgemask, brainmask):
        """Calculate the edge fraction, i.e. the tendency of the IC to occur at brain edges"""
        return np.absolute(comp_arr[edgemask]).sum() / np.absolute(comp_arr[brainmask]).sum()


    def generate_report(self,runinfo, results_dict, bo, comp_ts, comp_arr, designmat_rescaled, mo_fit,
                        report_filename):
        """
        Create a Plot that summarizes the characteristics of a given IC:
        spatial map, edge fraction, frequency spectrum, high frequency content, fit to design/motion
        """
        clim_ = 7
        fd = {'fontsize': 11}
        freqs, power = periodogram(comp_ts, fs=1. / runinfo['tr'])
        seconds = np.arange(0, runinfo['tr'] * runinfo['ntrs'], runinfo['tr'])
        comp_arr = np.rot90(np.copy(comp_arr), axes=(0, 2))
        comp_arr[np.logical_and(comp_arr < self.report_threshold, comp_arr > - self.report_threshold)] = np.nan
        func_f = pjoin(
            self.fmriprepdir, f'sub-{runinfo["subject"]}', f"ses-{runinfo['session']}",
            'func',
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']:02d}{self.space_str}_desc-preproc_bold.nii.gz"
        )
        meanbold = np.rot90(load_img(func_f).get_fdata().mean(axis=-1), axes=(0, 2))
        hor_is = np.linspace(5, comp_arr.shape[-1] - 15, num=10, dtype=int, endpoint=False)
        hor_img = np.flip(np.concatenate([comp_arr[slice_i, :, :] for slice_i in hor_is], axis=-1))
        hor_bg = np.flip(np.concatenate([meanbold[slice_i, :, :] for slice_i in hor_is], axis=-1))
        cor_is = np.linspace(5, comp_arr.shape[1], num=5, dtype=int, endpoint=False)
        cor_img = np.concatenate([comp_arr[:, slice_i, :] for slice_i in cor_is], axis=-1)
        cor_bg = np.concatenate([meanbold[:, slice_i, :] for slice_i in cor_is], axis=-1)
        sag_is = np.linspace(10, comp_arr.shape[2], num=5, dtype=int, endpoint=False)
        sag_img = np.concatenate([comp_arr[:, :, slice_i] for slice_i in sag_is], axis=-1)
        sag_bg = np.concatenate([meanbold[:, :, slice_i] for slice_i in sag_is], axis=-1)
        fig = plt.figure(figsize=(16.54, 9.45), facecolor='white')  # constrained_layout=True
        gs = GridSpec(4, 2, figure=fig, hspace=0.2, wspace=0.05)
        # component maps from different views
        hor_ax = fig.add_subplot(gs[0, :])
        hor_ax.imshow(hor_bg, cmap='gray')
        hor_ax.imshow(hor_img, cmap=self.render3, clim=(-clim_, clim_))
        hor_ax.set_title(f"Edge Fraction = {results_dict['edgefrac']:.2f}", x=.07, y=.84, fontdict=fd, color='white')
        cor_ax = fig.add_subplot(gs[1, 0])
        cor_ax.imshow(cor_bg, cmap='gray')
        cor_ax.imshow(cor_img, cmap=self.render3, clim=(-clim_, clim_))
        sag_ax = fig.add_subplot(gs[1, 1])
        sag_ax.imshow(sag_bg, cmap='gray')
        sag_ax.imshow(sag_img, cmap=self.render3, clim=(-clim_, clim_))
        # frequency spectrum
        freq_ax = fig.add_subplot(gs[2, 0])
        freq_ax.plot(freqs, power, color='black', alpha=.7)
        freq_ax.set_title(f'Frequency (HFC = {results_dict["hfc"]:.3f})', y=.85, fontdict=fd)
        # design and motion fit
        design_ax = fig.add_subplot(gs[2, 1])
        design_ax.plot(seconds, comp_ts, color='black', alpha=.7)
        design_ax.plot(seconds, designmat_rescaled, alpha=.7, color=plt.cm.tab10.colors[2])
        design_ax.set_ylim([-3, None])
        design_ax.set_title(f'Design (r = {results_dict["designcorr"]:.3f})', color=plt.cm.tab10.colors[2], y=.85,
                            fontdict=fd)
        #phy_ax = fig.add_subplot(gs[3, 0])
        #phy_ax.plot(seconds, zscore(comp_ts), color='black', alpha=.8)
        #phy_ax.plot(seconds, zscore(phy_fit), alpha=.8, color=plt.cm.tab10.colors[0])
        #phy_ax.set_title(f'Physio (r = {results_dict["maxphysiocorr"]:.3f})', color=plt.cm.tab10.colors[0], y=.85,
        #                 fontdict=fd)
        mo_ax = fig.add_subplot(gs[3, 1])
        mo_ax.plot(seconds, zscore(comp_ts), color='black', alpha=.8)
        mo_ax.plot(seconds, zscore(mo_fit), alpha=.8, color=plt.cm.tab10.colors[1])
        mo_ax.set_title(f'Motion (r = {results_dict["maxmotioncorr"]:.3f})', color=plt.cm.tab10.colors[1], y=.85,
                        fontdict=fd)
        for imax in [hor_ax, cor_ax, sag_ax]:
            imax.set(xticks=[], yticks=[])
        plt.suptitle(f"{bo.filename} (Component #{results_dict['comp_i']})".replace('_bold.nii.gz', ''), y=.92)
        # plt.show()
        fig.savefig(report_filename, dpi=self.dpi)
        plt.close(fig=fig)






    def runall(self):
        results_dicts = []
        report_dicts = []  # make an extra csv file containing only ICs for which we've generated reports
        for bo in tqdm(self.bidsobs, desc='iterating over runs'):
            runinfo = bo.get_entities()
            # TODO: missing event files for prf task and some memory runs (skipped for now)
            if (runinfo['task'] == 'pRF') or (runinfo['task'] == 'memory' and runinfo['run'] > 10):
                continue
            runinfo['tr'] = bo.get_metadata()['RepetitionTime']
            run_img = load_img(bo.path)
            runinfo['ntrs'] = run_img.shape[-1]
            designmat, designmat_rescaled = self.get_designmat(bo, runinfo)
            #physio_df = self.get_physio(bo, runinfo)
            motion_df = self.get_motion(runinfo)
            mixmat, comps_arr = self.get_comps(runinfo)
            edgemask, brainmask = self.get_edge_mask(runinfo)
            outdir = pjoin(self.out_basedir, f'sub-{runinfo["subject"]}', f'ses-{runinfo["session"]}')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for comp_i in range(mixmat.shape[-1]):
                results_dict = {'subject': runinfo['subject'], 'session': runinfo['session'], 'task': runinfo['task'],
                                'run': runinfo['run'], "comp_i": comp_i}
                comp_ts = mixmat[:, comp_i]
                comp_arr = comps_arr[:, :, :, comp_i]
                compdf = pd.DataFrame({'comp_ts': comp_ts})
                # design correlation
                results_dict['designcorr'] = np.absolute(np.corrcoef(comp_ts, designmat)[0, -1])
                # motion correlation
                comp_motion_df = pd.concat([compdf, motion_df], axis=1)
                motioncorrs = comp_motion_df.corr().to_numpy()[1:, 0]
                maxmo_ind = np.argmax(np.absolute(motioncorrs))
                results_dict['maxmotioncorr'] = motioncorrs[maxmo_ind]
                # physio correlation
                #comp_physio_df = pd.concat([compdf, physio_df], axis=1)
                #physiocorrs = comp_physio_df.corr().to_numpy()[1:, 0]
                #maxphy_ind = np.argmax(np.absolute(physiocorrs))
                #results_dict['maxphysiocorr'] = physiocorrs[maxphy_ind]
                # edge fraction
                results_dict['edgefrac'] = self.calc_edgefrac(comp_arr, edgemask, brainmask)
                # high frequency content
                results_dict['hfc'] = calc_hfc(comp_ts)
                results_dicts.append(results_dict)
                if self.exclude_comp_ids:
                    # skip if in self.exclude_comp_ids
                    comp_id = f"{int(runinfo['subject'])}_{runinfo['session']}_{runinfo['task']}_{int(runinfo['run'])}_{comp_i}"
                    if comp_id in self.exclude_comp_ids:
                        continue
                # Decide whether to generate report, given probability
                if not rndchoice(2, p=[1.0 - self.random_report_prob, self.random_report_prob]):
                    continue
                report_filename = pjoin(outdir, f"{bo.filename}_comp-{comp_i}.png".replace('_bold.nii.gz', ''))
                mo_fit = self.calc_fits(comp_motion_df, maxmo_ind, comp_ts)
                self.generate_report(runinfo, results_dict, bo, comp_ts, comp_arr, designmat_rescaled, mo_fit,
                                     report_filename)
                report_dicts.append({
                    'subject': runinfo['subject'], 'session': runinfo['session'], 'task': runinfo['task'],
                    'run': runinfo['run'], 'comp_i': comp_i, 'ManualRating': ''
                })
        results_df = pd.DataFrame(results_dicts)
        results_df.to_csv(pjoin(self.out_basedir, f'melodic_correlations_space-{self.space}.tsv'), sep='\t')
        reports_df = pd.DataFrame(report_dicts)
        reports_df.to_csv(pjoin(self.out_basedir, f'melodic_correlations_space-{self.space}_reports.tsv'), sep='\t')


if __name__ == '__main__':

    import os
    #setting fsl environment variables
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    os.environ['FSLDIR'] = '/home/satwick22/fsl'
    #make fsl executable available in the path
    os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']
    subject = '01'
    bidsroot = '/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_'
    fmriprepdir = pjoin('/DATA/satwick22/Documents/fMRI/fMRI_processing/bids', 'derivatives', 'fmriprep')
    #provide the bidsroot if required else it wont be tuned to work for a particular subject
    eval_melodic = EvalMelodic(bidsroot, fmriprepdir,out_deriv_name='melodic_features_run_n1', space='T1w', random_report_prob=.0)
    eval_melodic.runall()