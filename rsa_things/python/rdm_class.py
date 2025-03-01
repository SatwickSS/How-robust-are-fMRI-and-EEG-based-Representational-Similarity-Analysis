#create class for estiamting teh RDMs

#importing the necessary libraries and modules
"""
Creates RDM from nifti files"""

import os
import numpy as np
from os.path import join as pjoin
from rdm_helper_rois import mask_driver2
from rdm_driver import beta_linearizer_roi
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr


class RDM:
    def __init__(self,sub_id:str,bidsroot:str,betas_base:str,betas_fit:str,pipeline_spec:tuple=None):
        #assigning the class variables
        self.sub_id=sub_id
        self.bidsroot=bidsroot
        if pipeline_spec:pipeline_spec=pjoin(pipeline_spec)
        else:pipeline_spec=''
        self.betas_dir=pjoin(betas_base,betas_fit,pipeline_spec)#as of now it has no additional usecase , so think if it's necessary
        self.rdms:list=[]#list for storing the RDMs

    def beta_linearizer(self,bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_',transform:bool=True):
        """Reads the beta_weights and linearizes and saves in a dictionary
        Transform:bool,if True then union_mask is resampled to mask_template"""
        #creates the empty dictionary to store the beta weights for each session
        temp_dict={}
        betas_list,condnames_list,sessions=beta_linearizer_roi(self.sub_id,self.betas_dir,bidsroot=bidsroot,transform=transform)
        #creating the dictionary 
        for session,betas,condnames in zip(sessions,betas_list,condnames_list):
            temp_dict[session]={}
            temp_dict[session]['betas']=betas
            if session==sessions[0]:
                temp_dict['condnames']=condnames
                temp_dict['reg_ids']=[]
        self.betas=temp_dict
    def roi_indexer(self,reg_id:str,mask_dir:str=None,transform:bool=True):
        """Extracts the roi mask indexes for indexing the beta weights later"""
        if reg_id not in self.betas['reg_ids']:
            if mask_dir:_,roi_indices=mask_driver2(self.sub_id,reg_id,self.bidsroot,mask_dir=mask_dir,transform=transform)
            else:_,roi_indices=mask_driver2(self.sub_id,reg_id,transform=transform)
            #take an intersection and remap the values in the roi indices from overall space to brain masked voxel space
            current_roi_indices=self.betas.get('roi_indices',np.array([],dtype=int))
            #concatenate the left and right roi indices
            roi_indices=np.unique(np.concatenate(roi_indices,axis=0))
            #update the roi indices
            #concatenate the current roi indices with the new roi indices and the sort the indices
            if np.any(current_roi_indices):self.betas['roi_indices']=np.sort(roi_indices)
            else:self.betas['roi_indices']=np.sort(np.concatenate((current_roi_indices,roi_indices)))
            #update the reg_ids
            self.betas['reg_ids'].append(reg_id)
    def rdm_evaluator(self,calc_method='correlation',roi_indexer=True):
        """Evaluates the rdm based on the method speciifed"""
        current_rdm={}
        betas=np.array([])
        #iterate through the sessions
        for session in self.betas.keys():
            if session in ['condnames','reg_ids','roi_indices']:continue
            #iterate through the reg_ids
            #for reg_id in self.betas['reg_ids']:
                #get the roi indices
            if roi_indexer:
                #extarct the roi indices
                roi_indices=self.betas['roi_indices']
                #get the beta weights
                betas=self.betas[session]['betas'][:,roi_indices]
            else:
                betas=self.betas[session]['betas']
                #get the rdm dataset object
            rdm_dataset=rsd.Dataset(measurements=betas,descriptors={'session':session,'subj':self.sub_id},obs_descriptors={'conds':self.betas['condnames']})
            #calculate the rdm
            rdm=rsr.calc_rdm(rdm_dataset,method=calc_method,descriptor='conds')
            #save the rdm in the dictionary
            current_rdm[session]=rdm
        current_rdm['reg_id']=self.betas['reg_ids']
        #appending the rdm dicitonary to the list
        self.rdms.append(current_rdm)


if __name__=='__main__':
    #specify the subject id
    sub_id='01'
    #specify the bidsroot
    bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/bids'
    #specify the betas_base
    betas_base='betas_vanilla'
    #specify the betas_fit
    betas_fit='betas_vol'
    #create class object
    rdm=RDM(sub_id,bidsroot,betas_base,betas_fit)
    #linearize the betas
    rdm.beta_linearizer(bidsroot=bidsroot,transform=False)
    #add roi 
    #estimate the rdm
    rdm.rdm_evaluator(roi_indexer=False)




