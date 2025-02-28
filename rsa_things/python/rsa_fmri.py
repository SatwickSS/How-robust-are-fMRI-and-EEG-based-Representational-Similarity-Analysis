"""Pipeline for RSA from beta and region maps"""
 # relevant imports
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import rsatoolbox.rdm as rsr
import json,re,pickle
from rdm_comparison import data_grabber_rdm,rdm_creator
from rdm_helper_rois import mask_driver
from os.path import join as pjoin
import os



#defining the function for reading the beta weight and condition names
def data_grabber_rdm(betasdir:str, session:str):
    """ Function to grab the beta weights and condition names from the disc
    Args:
        betasdir (str): path to the betas directory
        session (str): session name
    Returns:
        beta_values (ndarray): beta values
        condnames (list): condition names """
    #create the file name for the prticular sessions beta
    beta_file_name=f'sub-{sub}_{session}_betas.npy'
    cond_file_name=f'sub-{sub}_{session}_condnames.json'
    #creating the path for basedir
    input_dir=pjoin(betasdir,session)
    #reading the beta values
    beta_values=np.load(pjoin(input_dir,beta_file_name))
    #reading the condition names
    with open(pjoin(input_dir,cond_file_name)) as f:
        condnames=json.load(f)
    return beta_values,condnames
def rdm_creator(rdm_data,method:str='correlation',descriptor:str='conds',subset_data:dict=None):
    """ Function to create the RDMs from the beta values
    Args:
        rdm_data (rsd.rdm data object): list of dataset objects
        method (str): method to calculate the RDMs
        descriptor (str): descriptor to calculate the RDMs
    Returns:
        rdm_data (dataset): dataset object containing the RDMs """
    if subset_data:
        subset_param=next(iter(subset_data))
        subset_conds=subset_data[subset_param]
        new_data=rdm_data.subset_channel(by=subset_param,value=subset_conds)
        rdms=rsr.calc_rdm(new_data,method=method,descriptor=descriptor)
    #creating the RDMs
    else:
        rdms=rsr.calc_rdm(rdm_data,method=method,descriptor=descriptor)
    return rdms
def rdm_driver(sub:str,betas_derivname:str,sessions_target_ind:tuple,voxel_nums=np.array([]),method='correlation',descriptor='conds',subset_data:dict=None):
    """ Function to drive the RDM creation process
    Args:
        sub (str): subject number
        betas_derivname (str): name of the betas directory
        sessions_target (list): list of sessions to retrieve the betas from
        method (str): method to calculate the RDMs
        descriptor (str): descriptor to calculate the RDMs
        subset_data (dict): dictionary containing the subset parameters
    Returns:
        res_rdms (dataset): dataset object containing the RDMs"""
    #specifying the sessions
    sessions_orig = [f'ses-things{i:02d}' for i in range(1, 13)]
    sessions=sessions_orig+['ses-thingsavg','ses-thingsall']
    bidsroot='/DATA1/satwick22/Documents/fMRI/thingsmri'
    ##specifying the subjects
    #sub='01'
    ##specifying the betas derivname
    #betas_derivname='betas_run01/regularized'
    #specifying the betas directory
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname,'betas_linearized')#CRITICAL
    #specifying the session to retrieve the betas from
    sessions_target=sessions[sessions_target_ind[0]:sessions_target_ind[1]]#use slicing to retrieve the sessions
    #iterating through the sessions
    for session in sessions_target:
        #grabbing the betas
        beta_values,condnames=data_grabber_rdm(betasdir,session)
        des = {'session': session, 'subj': int(sub[1:])}
        measurements = beta_values
        nVox=measurements.shape[1]
        obs_des = {'conds': condnames}
        #obs_des = {'conds': np.array(['cond_' + str(x) for x in np.arange(0,nCond)])} # indices␣˓ →from 1
        if voxel_nums.any():
            chn_des = {'voxels': voxel_nums}
        else:
            chn_des = {'voxels': np.array(['voxel' + str(x) for x in np.arange(0,nVox)])} # indices␣˓ →from 0
        rdm_data=rsd.Dataset(measurements=measurements,
        descriptors=des,
        obs_descriptors=obs_des,
        channel_descriptors=chn_des)
    #creating the RDMs
    res_rdms=rdm_creator(rdm_data,method,descriptor,subset_data)
    return res_rdms
def vanilla_rdm_creator(sub:str,betas_derivname:str,sessions_target_ind:tuple,method='correlation',descriptor='conds',subset_data:dict=None):
    vanilla_rdms={}
    for i in range(sessions_target_ind[0]+1,sessions_target_ind[1]+1):
        sessions_target_ind=(i-1,i)
        vanilla_rdms[f'ses{i:02d}']=rdm_driver(sub,betas_derivname,sessions_target_ind,union_mask_method,descriptor,subset_data)
    return vanilla_rdms
    #rdms=rdm_driver(sub,betas_derivname,sessions_target_ind,method,descriptor,subset_data)
    return rdms#creating and returning the vanilla RDMs
def session_number_calc(session_target_ind:tuple):
    total_len=14
    start,end=session_target_ind
    if end==0:
        val=0
    if start>=0 and end>0:
        if start<end:
            val=end-start
    elif start<0 and end<0:
        if start>end:
            raise ValueError('start should be less than end')
        else:
            val=abs(start-end)
    elif start>=0 and end<0:
        val=(total_len+end+1)-start
    elif start<0 and end>0:
        new_start=total_len+start
        val=end-new_start
    if val<0:
        val=0
    return val
def combi_driver(sub,betas_derivname_base,param_combi,param_list,vanilla_rdms,outdir):
    sessions_target_ind,method,descriptor,subset_data=(0,2),'correlation','conds',None
    #create a data structure for comapring the RDMs
    #create a empty numpy array of size nparam X sessions
    rdm_corr_matrix=np.empty((len(param_list),2))
    def derivname_generator():
        for param in param_list:
            betas_derivname=pjoin(betas_derivname_base,f'{param_combi}_{param}')
            yield betas_derivname
    for ind,betas_derivname in enumerate(derivname_generator()):
        for i in range(sessions_target_ind[0]+1,sessions_target_ind[1]+1):
            sessions_target_ind=(i-1,i)
            combi_rdm=rdm_driver(sub,betas_derivname,sessions_target_ind,method,descriptor,subset_data)
            #comparing the two RDMs
            rdm_corr_matrix[ind,i-1]=rsr.compare(combi_rdm,vanilla_rdms[f'ses{i:02d}'],method='corr')

    #saving the numpy array rdm_corr_matrix
    np.save(pjoin(outdir,f'rdm_corr_matrix_{param_combi}_{param_list[0]}-{param_list[-1]}.npy'),rdm_corr_matrix)
def pickle_reader(sub_id,best_hrf_pickle_path:str='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/best_hrf'):
    #create the path for the pickle file
    pickle_file_path=pjoin(best_hrf_pickle_path,f'sub-{sub_id}_best_hrf.pkl')
    #read the pickle file
    with open(pickle_file_path,'rb') as f:
        best_hrf=pickle.load(f)
    return best_hrf
def rdm_roi_driver(sub_id:str,sessions_target:tuple,mask_dir:str=None):
    #create the code for reading the beta hrfs for each ROI 
    #iterate through the ROIs and create the RDMs
    #roi_list=['EBA','OFA','PPA','RSC','TOS','FFA','LOC']
    roi_list=['FFA']#SET
    rdm_store={}#setting the container for storing the rdms of all rois
    for reg_id in roi_list:
        #initializing the list object for storing 
        rdm_store_hf={}
        #getting the mask indices
        union_mask_indices,mask_rois_indices=mask_driver(sub_id,reg_id)
        #create the directory for the estimated linearized betaweights
        for hf in ['l','r','']:
            #create the path for the particular hf
            betas_derivname=f'betas_combi_run02/unregularized_{hf}{reg_id}'#SET
            #get the mask indexes
            #create the data subsetter dictionary
            #extacting the roi indices for the combined mask 
            subset_data={'voxels': mask_rois_indices[-1]}
            #create the rdms using the rdm creator function
            for ses_ind in range(sessions_target[0],sessions_target[1]):
                sessions_target_ind=(ses_ind,ses_ind+1)
                rdm_=rdm_driver(sub_id,betas_derivname,sessions_target_ind,union_mask_indices,method='correlation',descriptor='conds',subset_data=subset_data)
                rdm_store_hf[hf]=rdm_
            #add the list to the dictionary
        rdm_store[reg_id]=rdm_store_hf
    #estimating the correlations across sessions for the overfit
    #iterating through the reg ids
    rdm_comp={'reg_id':['l vs r',"l vs 'combined'","r vs 'combined'"]}
    for reg_id in roi_list:
        #extractint the rdms for the particular reg_ID
        rdm_store_hf=rdm_store[reg_id]
        #estimating the correlations across the hemispheres for a single session
        rdm_comp_hf=[]
        for ind1,hf1 in enumerate(['l','r','']):
            for ind2,hf2 in enumerate(['l','r','']):
                if ind1>=ind2:
                    continue
                rdm_comp_hf.append(rsr.compare(rdm_store_hf[hf1],rdm_store_hf[hf2],method='corr'))
        rdm_comp[reg_id]=rdm_comp_hf
    print(rdm_comp)
def rdm_roi_driver2(sub_id:str,sessions_target:tuple=None,betas_dir=None):
    """Creates RDM for all conditions of best frac indices and hrf indices estimation"""
    run_id='00'#SET
    betas_fit='regularized'#SET
    #create a dictionary for storing the RDMs
    rdm_dict={}
    nsessions=12#SET
    #iterate through the four conditions
    best_frac_methods=['mean','mode']#SET
    best_hrf_methods=['mean','mode']#SET
    for best_hrf_inds_by in best_hrf_methods:
        for best_frac_inds_by in best_frac_methods:
            if best_hrf_inds_by=='mean' and best_frac_inds_by=='mode':
                continue
            betas_dir_base=pjoin(f'betas_roi_combi_run{run_id}',f'{betas_fit}',f'{best_hrf_inds_by}_{best_frac_inds_by}')#SET
            rdm_dict[f'hrf_by_{best_hrf_inds_by}_frac_by_{best_frac_inds_by}']={}
            #iterate through all sessions
            for ses_ind in range(nsessions):
                #get the rdm for session ses_ind
                sessions_target_ind=(ses_ind,ses_ind+1)
                #call the rdm driver function
                rdm=rdm_driver(sub_id,betas_dir_base,sessions_target_ind,method='correlation',descriptor='conds',subset_data=None)
                #save the rdm
                rdm_dict[f'hrf_by_{best_hrf_inds_by}_frac_by_{best_frac_inds_by}'][f'ses_things{ses_ind+1:02}']=rdm

    #find the correlations between the 4 conditions RDMs
    #create a dictionary for storing the correlations
    rdm_comp_dict={}
    #iterate through the four conditions
    keys=list(rdm_dict.keys())
    #iterate through the keys and get the rdm correlation
    for ses in range(nsessions):
        rdm_comp_dict[f'ses_things{ses+1:02}']={}
        for ind1 in range(len(keys)):
            for ind2 in range(ind1+1,len(keys)):
                rdm_comp_dict[f'ses_things{ses+1:02}'][f'{keys[ind1]}x{keys[ind2]}']=rsr.compare(rdm_dict[keys[ind1]][f'ses_things{ses+1:02}'],rdm_dict[keys[ind2]][f'ses_things{ses+1:02}'],method='tau-a')
    return rdm_dict,rdm_comp_dict
    








if __name__=='__main__':
    sub='01'
    #calling the function
    rdm_roi_driver2(sub)


