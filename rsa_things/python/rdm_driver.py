""" Creates RDMs across regions from region masks and beta weights"""
#importing the required libraries
import os
import numpy as np,re
from rdm_builder_helper import get_beta_weights_subject,custom_concat,save_list,save_ndarray_dict,save_ndarray
from os.path import join as pjoin
from rdm_helper_rois import mask_driver2
# specifying the driver function
def beta_linearizer(sub,betas_derivname,bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_'):
    #sub,betaspath,outpath= '01','/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_','/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/betas_run01/regularized/','/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/betas_run01/regularized/sub-01/'                 
    #betas_derivname='betas_run01/regularized'
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname, f'sub-{sub}')
    #calling the function for extracting the beta weights for one subject over sessions
    betas,condnames=get_beta_weights_subject(sub=sub,bidsroot=bidsroot,betas_dir=betasdir,betas_derivname=betas_derivname)
    #print(len(betas),betas[0].shape,condnames[0].shape)
    #creating the all session beta weight and the conditions
    all_ses_betas=custom_concat(list(betas.values()))
    all_ses_condnames=custom_concat(condnames)
    #converting condnames to list
    all_ses_condnames=all_ses_condnames.tolist()
    #converting all the ndarrays in condnames to list
    condnames=[condnames[i].tolist() for i in range(len(condnames))]
    #creating the average session betas and the conditions
    avg_ses_betas=np.mean(np.dstack(list(betas.values())),axis=-1)
    #repalcing the condition names with the average session condition names
    avg_ses_condnames=[re.sub(r'ses_[0-9]+','avg',i) for i in condnames[0]]
    #adding the two beta arrays to the betas list
    betas.update({'ses_avg':avg_ses_betas,'ses_all':all_ses_betas})
    #adding the two condition names to the condnames list
    condnames+=[avg_ses_condnames,all_ses_condnames]
    print(len(list(betas.values())),len(condnames))
    #creating the outdir for storing the varios sessionwise betas
    #creating the list of session indexes
    sessions_orig = [f'ses-things{i:02d}' for i in range(1, 13)]
    sessions=sessions_orig+['ses-thingsavg','ses-thingsall']
    sub='01'
    betas_list=list(betas.values())
    outdir_base=pjoin(betasdir,'betas_linearized')
    if not os.path.exists(outdir_base):
        os.makedirs(outdir_base)
    for ind,session in enumerate(sessions):
        outdir=pjoin(outdir_base,session)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        beta_file_name=f'sub-{sub}_{session}_betas'
        cond_file_name=f'sub-{sub}_{session}_condnames'
        save_ndarray(betas_list[ind],outdir,beta_file_name)
        save_list(condnames[ind],outdir,cond_file_name)
def beta_linearizer_roi(sub,betas_derivname,bidsroot='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_',transform=False):
    #sub,betaspath,outpath= '01','/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_','/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/betas_run01/regularized/','/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/betas_run01/regularized/sub-01/'                 
    #betas_derivname='betas_run01/regularized'
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname, f'sub-{sub}')
    betasdir = pjoin(bidsroot, 'beta_derivatives', betas_derivname, f'sub-{sub}')#DELETE
    #calling the function for extracting the beta weights for one subject over sessions
    betas,condnames=get_beta_weights_subject(sub=sub,bidsroot=bidsroot,betas_dir=betasdir,betas_derivname=betas_derivname,transform=transform)
    #converting all the ndarrays in condnames to list
    condnames=[condnames[i].tolist() for i in range(len(condnames))]
    #creating the average session betas and the conditions
    avg_ses_betas=np.mean(np.dstack(list(betas.values())),axis=-1)
    #repalcing the condition names with the average session condition names
    avg_ses_condnames=[re.sub(r'ses_[0-9]+','avg',i) for i in condnames[0]]
    #adding the average beta array to the betas list
    betas.update({'ses_avg':avg_ses_betas})
    #adding the condition name of average condition to the condnames list
    condnames+=[avg_ses_condnames]
    #creating the outdir for storing the varios sessionwise betas
    #creating the list of session indexes
    sessions_orig = [f'ses-things{i:02d}' for i in range(1, 13)]
    sessions=sessions_orig+['ses-thingsavg']
    #sub='01'
    betas_list=list(betas.values())
    return betas_list,condnames,sessions
def save_betas(betas_outdir,betas_list,condnames,sub_id,sessions,bidsroot='/DATA1/satwick22/Documents/fMRI/thingsmri'):#function for saving the betaweights
    run_id='01'
    outdir_base=pjoin(betas_outdir,f'betas_linearized')
    if not os.path.exists(outdir_base):
        os.makedirs(outdir_base)
    for ind,session in enumerate(sessions):
        outdir=pjoin(bidsroot,'derivatives',outdir_base,session)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        beta_file_name=f'sub-{sub_id}_{session}_betas'
        cond_file_name=f'sub-{sub_id}_{session}_condnames'
        save_ndarray(betas_list[ind],outdir,beta_file_name)
        save_list(condnames[ind],outdir,cond_file_name)
def combi_driver(sub,betas_derivname_base,param_combi,param_list):
    for param in param_list:
        betas_derivname=pjoin(betas_derivname_base,f'{param_combi}_{param}')
        yield betas_derivname
def roi_combi_driver(sub_id,roi_beta_deriv_base,reg_id,run_id):
    for roi in reg_id:#iterating through the rois
        for hf in ['l','r','']:
            betas_derivname=pjoin(f'{roi_beta_deriv_base}_run{run_id}',f'unregularized_{hf}{roi}')
            beta_linearizer(sub_id,betas_derivname)
def roi_concat_driver(reg_ids,run_id,best_hrf_inds_by,best_frac_inds_by,betas_fit='regularized'):
    sub_id='01'#SET
    nsessions=13#SET
    betas_dir_base=pjoin(f'betas_roi_combi_run{run_id}',f'{betas_fit}',f'{best_hrf_inds_by}_{best_frac_inds_by}')#SET
    betas_roi_concat=[np.array([]) for i in range(nsessions)]  # array for storing sessionwise concatenated betaweights
    #iterate through the rois
    for roi in reg_ids:
        print('Concatenating region:',roi)
        #get the mask indices for the roi
        _,roi_indices=mask_driver2(sub_id,roi)
        #iterate through the hemispheres
        for hf_ind,hf in enumerate(['l','r']):
            #create the directory for the roi for the particular hf
            roi_dir=pjoin(betas_dir_base,f'{hf}{roi}')        
            #extract the betas for the sessions
            betas_list,condnames,sessions=beta_linearizer_roi(sub_id,roi_dir)
            #extract the betas for the roi for each session
            betas_list_roi=[betas_list[i][:,roi_indices[hf_ind]] for i in range(len(betas_list))]
            #iterate through the sessions and add the roi betas
            for ind,betas in enumerate(betas_list_roi):
                if not np.any(betas_roi_concat[ind]):
                    betas_roi_concat[ind]=betas
                else:
                    betas_roi_concat[ind]=np.concatenate([betas_roi_concat[ind],betas],axis=1)
    #save the betas
    save_betas(betas_dir_base,betas_roi_concat,condnames,sub_id,sessions)#will save the roi based betas 

def driver():
    #creating the list of reg_ids
    reg_ids=['FFA','PPA','EBA','LOC','TOS','OFA','RSC']
    best_frac_inds_by=['mean','mode']#SET
    best_hrf_inds_by=['mean','mode']#SET
    run_id='00'#SET
    for best_hrf_ind_by in best_hrf_inds_by:
        for best_frac_ind_by in best_frac_inds_by:
            roi_concat_driver(reg_ids,run_id,best_hrf_ind_by,best_frac_ind_by)
    
def driver_test():
    """General Purpose Function for extracting the betas for any type of betaweight estimated"""
    #extrating,linearizing, and saving the betas for the rois
    reg_ids=['FFA','PPA','EBA','LOC','TOS','OFA','RSC']
    sub_id='01'#SET
    nsessions=13#SET
    betas_dir_base=pjoin('betas')
    betas_roi_concat=[np.array([]) for i in range(nsessions)]  # array for storing sessionwise concatenated betaweights
    #iterate through the rois
    for roi in reg_ids:
        print('Concatenating region:',roi)
        #get the mask indices for the roi
        _,roi_indices=mask_driver2(sub_id,roi,transform=False)
        #take union of roi_indices
        roi_indices=np.unique(np.concatenate(roi_indices))
        #crete the directory path for extracting the betas
        betas_base='betas'
        #specify the type of betaweight estimated
        betas_fit='scalematched'
        #specify the specification of the pipeline
        pipeline_spec=''
        #combine the above three to create the betas directory
        betas_dir=pjoin(betas_base,betas_fit,pipeline_spec)
        betas_list,condnames,sessions=beta_linearizer_roi(sub_id,betas_dir)
        betas_list_roi=[betas_list[i][:,roi_indices] for i in range(len(betas_list))]
        for ind,betas in enumerate(betas_list_roi):
            if not np.any(betas_roi_concat[ind]):
                betas_roi_concat[ind]=betas
            else:
                betas_roi_concat[ind]=np.concatenate([betas_roi_concat[ind],betas],axis=1)
    save_betas(betas_dir_base,betas_roi_concat,condnames,sub_id,sessions)#will save the roi based betas 


        
if __name__=='__main__':
    beta_linearizer('01','betas_run01/scalematched')
