"""
Contains dataset class source
"""
#! /usr/env/python

import warnings
from os.path import exists as pexists
from os.path import join as pjoin

from bids import BIDSLayout


class ThingsMRIdataset:
    """
    Data loader for the THINGS-fMRI dataset.
    """

    def __init__(self, root_path: str, validate: bool = True):
        # path attributes
        assert isinstance(root_path, str)
        self.root_path = root_path
        self.rawdata_path = pjoin(self.root_path, 'rawdata') # i.e rawdata_path = root_path/rawdata
        self.sourcedata_path = pjoin(self.root_path, 'sourcedata')# i.e sourcedata_path = root_path/sourcedata
        self.derivs_path = pjoin(self.root_path, 'derivatives')# i.e derivs_path = root_path/derivatives
        # pybids layout
        #creating BIDSLayout object
        #BIDSLayout is a class that represents a BIDS project, and provides methods for querying and manipulating the project.
        self.layout = BIDSLayout(self.rawdata_path, validate=validate)#get the layout of the rawdata
        #returns a list of ids for all the subjects in the rawdata directory
        self.subjects = self.layout.get(return_type='id', target='subject')#get the subjects under the rawdata directory
        self.sessions = self.layout.get(return_type='id', target='session')#get all the sessions in the raw directory
        self.things_sessions = [ses for ses in self.layout.get(return_type='id', target='session') if 'things' in ses]# get all the sessions where the functional image was recorded
        runids = self.layout.get(return_type='id', target='run')#get the number of runs of all the sessions
        self.maxnruns = int(runids[-1])  # maximum number of runs per session

    def update_layout(self, validate: bool = True):
        self.layout = BIDSLayout(self.rawdata_path, validate=validate)
        self.layout.add_derivatives(self.derivs_path) # add derivatives to layout
        return None

    def include_derivs(self):
        """Note that this only captures folders in the derivs_path which have a dataset_description.json."""
        if pexists(self.derivs_path):
            self.layout.add_derivatives(self.derivs_path)
        else:
            warnings.warn("Could not find derivatives directory\n{}".format(self.derivs_path))
        return None

    def get_reconall_anat(self, subject: str) -> dict:
        """Collect paths to relevant outputs of reconall"""
        return dict(
            wmseg=pjoin(self.derivs_path, 'reconall', f'sub-{subject}', 'mri', 'wm.seg.mgz'),
            t1=pjoin(self.derivs_path, 'reconall', f'sub-{subject}', 'mri', 'nu.mgz'),
            t1_brain=pjoin(self.derivs_path, 'reconall', f'sub-{subject}', 'mri', 'norm.mgz'),
        )

    def get_fieldmap_files(self, subject: str) -> dict:
        phasediff_files = self.layout.get(subject=subject, return_type='file', extension='.nii.gz', suffix='phasediff')
        mag1_files = self.layout.get(subject=subject, return_type='file', extension='.nii.gz', suffix='magnitude1')
        mag2_files = self.layout.get(subject=subject, return_type='file', extension='.nii.gz', suffix='magnitude2')
        return dict(
            phasediff_files=phasediff_files,
            mag1_files=mag1_files,
            mag2_files=mag2_files,
        )#creating a dictionary with required filetypes as key and the paths as values

    def get_fmriprep_t1w(self, subject):
        return pjoin(self.root_path, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat',
                     f'sub-{subject}_acq-prescannormalized_rec-pydeface_desc-preproc_T1w.nii.gz')
