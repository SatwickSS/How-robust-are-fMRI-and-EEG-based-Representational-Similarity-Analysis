"""creating rdm from the estimated beta weights
"""

import os
import numpy as np
import pandas as pd
import glob
from glm_class import GODGLM
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr

from os.path import join as pjoin
from joblib import Parallel, delayed
from utils import apply_mask_smoothed
from nilearn.image import new_img_like,load_img
from tqdm import tqdm

### creating the RDM class

class RDM:
    
    ##constructor
    def __init__(self,bidsroot,sub_id,derivname,prepdir='fmriprep_run-01',masking=False):
        self.bidsroot=bidsroot
        self.sub_id=sub_id
        self.derivname=derivname
        #create a object of godglm 
        godglm = GODGLM(bidsroot, sub_id, out_deriv_name='tmp',prepdir=prepdir)
        _, _, _, masks = godglm.get_inputs()
        godglm.add_union_mask(masks)
        if not masking: 
            one_mask = np.ones(godglm.union_mask.shape,dtype=np.int8)
            godglm.union_mask = new_img_like(godglm.union_mask, one_mask)
        self.glmobj=godglm


    def beta_reader(self,betas_type,template=''):
        #assign self.template
        self.template=template

        #read the betas for each session
        def load_betas(
                sub: str,
                glm_obj : GODGLM,
                bidsroot: str = '/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
                betas_type: str = 'regularized' ,
                smoothing=0.,
                dtype=np.single,
                template=''
        ) -> np.ndarray:
            if not smoothing:
                smoothing = None
            self.betas_type=betas_type
            betasdir = pjoin(bidsroot, 'derivatives', self.derivname,self.betas_type, f'sub-{sub}')
            betafiles = [
                pjoin(betasdir, f'ses-{glm_obj.target_session}{ses_i+1:02d}', f'sub-{sub}_ses-{glm_obj.target_session}{ses_i+1:02d}_run-{run_i+1:02d}_betas{template}.nii.gz')
                for ses_i in range(glm_obj.n_sessions) for run_i in range(glm_obj.nruns_perses_[glm_obj.ds.target_sessions[ses_i]])
            ]
            conditionfiles=[betafile.replace(f'_betas{template}.nii.gz','_conditions.tsv') for betafile in betafiles]
            # check if all files exist
            for c in conditionfiles:
                assert os.path.exists(c)
            for b in betafiles:
                assert os.path.exists(b)
            #if the template is not the orginal T1w 
            #reshape the mask into the template space
            if template:
                beta_ref=load_img(betafiles[0])
                #extract the shape of the beta reference
                beta_shape=beta_ref.shape
                #create a new mask in the template space
                mask_data=np.ones(beta_shape[:-1],dtype=np.int8)
                glm_obj.union_mask=new_img_like(beta_ref,mask_data)

            with Parallel(n_jobs=-1) as parallel:
                betas_l = parallel(
                    delayed(apply_mask_smoothed)(bf, glm_obj.union_mask, smoothing, dtype)
                    for bf in tqdm(betafiles, desc='loading betas')
                )
            #betas = np.vstack(betas_l)
            #read the condition files
            conditions_l=[pd.read_csv(c,sep='\t') for c in conditionfiles]
            conditions_dict={}
            betas_dict={}
            for run_i in range(glm_obj.nruns_total):
                #extract the beta file for the session
                betas=betas_l[run_i]
                #extract the condition file for the session
                conditions=conditions_l[run_i]
                #create the boolea mask for rest condition
                rest_mask=conditions['image_filename']=='rest'
                #create the boolean mask for imagefile names ending with r.JPEG
                r_mask=conditions['image_filename'].str.endswith('r.JPEG')
                #combine the masks
                mask=rest_mask | r_mask
                #extract the rows from the beta matrix
                betas=betas[~mask]
                #extract the rows from the condition files
                conditions=conditions.loc[~mask,'image_filename'].to_numpy()
                #sort the condition indexes
                conditions_sorted_index=np.argsort(conditions)
                #sort the conditions
                conditions_sorted=conditions[conditions_sorted_index]
                #sort the betas
                betas_sorted=betas[conditions_sorted_index]
                conditions_dict[f'{run_i+1:02d}']=conditions_sorted
                betas_dict[f'{run_i+1:02d}']=betas_sorted 


            #conditions=pd.concat(conditionfiles)
            ##drop the rows for rest condition
            ##create the boolea mask for rest condition
            #rest_mask=conditions['image_filename']=='rest'
            ##create the boolean mask for imagefile names ending with r.JPEG
            #r_mask=conditions['image_filename'].str.endswith('r.JPEG')
            ##combine the masks
            #mask=rest_mask | r_mask
            ##extract the rows from the beta matrix
            #betas=betas[~mask]
            ##extract the rows from the condition files
            #conditions=conditions.loc[~mask,'image_filename']

            ##for each session segregate the condiitions and the betas into a dictionary
            ##extract the number of unique conditions
            #unique_conditions=conditions.unique()
            #for ses_i in range(glm_obj.n_sessions):
            #    start_index,end_index=unique_conditions.shape[0]*ses_i,unique_conditions.shape[0]*(ses_i+1)
            #    #extract the conditions
            #    conditions_ses=conditions.iloc[start_index:end_index].to_numpy()
            #    #extract the sorted condition index
            #    conditions_sorted_index=np.argsort(conditions_ses)
            #    #sort the conditioms
            #    conditions_sorted=conditions_ses[conditions_sorted_index]
            #    #sort the betas
            #    betas_sorted=betas[start_index:end_index][conditions_sorted_index]
            #    conditions_dict[f'{self.glmobj.target_session}{ses_i+1:02d}']=conditions_sorted
            #    #sort
            #    betas_dict[f'{self.glmobj.target_session}{ses_i+1:02d}']=betas_sorted
            return betas_dict,conditions_dict  # shape (ntrials, nvoxel)
        #load the betas
        self.betas,self.conditions=load_betas(self.sub_id,self.glmobj,betas_type=betas_type,template=self.template)


    def create_RDM(self,dist_method='correlation'):
        self.rdms={}
        beta_avg=np.mean(np.stack(list(self.betas.values()),axis=0),axis=0)
        #itearte through each session
        for run in self.betas.keys():
            #create the dataset object
            rdm_dataset=rsd.Dataset(measurements=self.betas[run],descriptors={'run':run,'subj':self.sub_id},obs_descriptors={'conds':self.conditions[run]})
            #calculate the rdm
            rdm=rsr.calc_rdm(rdm_dataset,method=dist_method,descriptor='conds')
            #save the rdm in the dictionary
            self.rdms[run]=rdm
        #create the average rdm
        rdm_dataset=rsd.Dataset(measurements=beta_avg,descriptors={'run':f'avg','subj':self.sub_id},obs_descriptors={'conds':self.conditions[run]})
        #calculate the rdm
        rdm=rsr.calc_rdm(rdm_dataset,method=dist_method,descriptor='conds')
        self.rdms[f'avg']=rdm
    
    def create_RDM_from_og(self,betas_dir,sub,dist_method='correlation'):
        #extract the beta files
        beta_files=glob.glob(pjoin(betas_dir,f'sub-{sub}','ses-perceptionTest-avg','*.nii'))
        #extract the conditions from the filenames
        conditions=np.array([os.path.basename(bf).split('-')[-1].replace('.nii','') for bf in beta_files])
        #sort the condition names
        condition_names_sorted_index=np.argsort(conditions)
        #sort the conditions
        conditions_sorted=conditions[condition_names_sorted_index]
        #read the beta files
        beta_data=[load_img(bf).get_fdata().flatten() for bf in beta_files]
        #create the betas matrix
        beta_data=np.stack(beta_data,axis=0)
        #sort the beta data
        beta_data_sorted=beta_data[condition_names_sorted_index]
        #create the dataset object
        rdm_dataset=rsd.Dataset(measurements=beta_data_sorted,descriptors={'session':'perceptionTest-avg','subj':sub},obs_descriptors={'conds':conditions_sorted})
        #calculate the rdm
        rdm=rsr.calc_rdm(rdm_dataset,method=dist_method,descriptor='conds')
        return rdm

        
        
        #return
        ##extract the condition files
        #condition_files=glob.glob(pjoin(betas_dir,f'sub-{sub}','ses-perceptionTest-avg','*.tsv'))
        ##extract the beta files
        #beta_data=[load_img(bf).get_fdata() for bf in beta_files]
        ##extract the condition files
        #conditions=[pd.read_csv(cf,sep='\t')['image_filename'].to_numpy() for cf in condition_files]
        ##extract the condition names
        #condition_names=np.array([os.path.basename(cf).replace('_conditions.tsv','') for cf in condition_files])
        ##sort the condition names
        #condition_names_sorted=np.sort(condition_names)
        ##sort the condition indexes
        #condition_names_sort_index=np.argsort(condition_names)
        ##sort the conditions
        #conditions_sorted=[conditions[ind] for ind in condition_names_sort_index]
        ##sort the betas
        #beta_data_sorted=[beta_data[ind] for ind in condition_names_sort_index]
        ##create the dataset object
        #rdm_dataset=rsd.Dataset(measurements=beta_data_sorted,descriptors={'session':'perceptionTest-avg','subj':sub},obs_descriptors={'conds':conditions_sorted})
        ##calculate the rdm
        #rdm=rsr.calc_rdm(rdm_dataset,method='correlation',descriptor='conds')
        #self.rdms['perceptionTest-avg']=rdm
        #return rdm


    def save_rdm(self,rdm_dir=None,og=False,rdm=None):
        if og:
            rdm_dir=pjoin(self.bidsroot, 'derivatives', self.derivname,'original', f'sub-{self.sub_id}',f'rdm{self.template}')
            if not os.path.exists(rdm_dir):os.makedirs(rdm_dir)
            np.save(pjoin(rdm_dir,f'sub-{self.sub_id}_avgrdm'),rdm.dissimilarities)
            return
        #save under the beta directory
        else:rdm_dir=pjoin(self.bidsroot, 'derivatives', self.derivname,self.betas_type, f'sub-{self.sub_id}','rdm')
        #create the directory if it does not exist
        if not os.path.exists(rdm_dir):os.makedirs(rdm_dir)
        #creating the rdm directory
        #if not rdm_dir:
        #    rdm_dir=pjoin(self.bidsroot, 'derivatives', self.derivname,self.betas_type)
        #save the rdms
        for session in self.rdms.keys():
            np.save(pjoin(rdm_dir,f'sub-{self.sub_id}_ses-{session}_rdm'),self.rdms[session].dissimilarities)





if __name__ == '__main__':
    #rdms=[]
    #for sub_id in range(1,6):
    #    sub=f'{sub_id:02d}'
    #    if sub in ['03','01']:
    #        preprdir='fmriprep_run-02'
    #    elif sub =='05':
    #        preprdir='fmriprep_run-03'
    #    else:
    #        preprdir='fmriprep_run-01'
    #    rdm = RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',sub,'betas_run-02',prepdir=preprdir)
    #    rdm.beta_reader('scalematched',template='')
    #    rdm.create_RDM()
    #    rdms.append(rdm)
    #for sub_id in range(1,6):
    #    sub=f'{sub_id:02d}'
    #for sub_id in range(1,6):
    #     sub=f'{sub_id:02d}'
    ##    rdm1 = RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',sub,'betas_run-02',prepdir='fmriprep-final')
    ##    rdm1.beta_reader('glm_single',template='')
    ##    rdm1.create_RDM(dist_method='euclidean')
    ##    rdm1.save_rdm()
    #sub_og_rdm=rdm1.create_RDM_from_og('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids/derivatives/betas_run-02/original',sub,dist_method='euclidean')
    #    rdm1.save_rdm(og=True,rdm=sub_og_rdm)
    for sub in range(1,5):
        sub_id=f'{sub:02d}'
        prepdir='fmriprep-final'
        derivname='scalematched'
        rdm=RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',sub_id,'betas_run-03',prepdir=prepdir)
        rdm.beta_reader(derivname,template='MNI305')
        rdm.create_RDM(dist_method='euclidean')
        rdm.save_rdm()
    #sub_og_rdm=rdm1.create_RDM_from_og('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids/derivatives/betas_run-03/original','05',dist_method='euclidean')
    #rdm1.save_rdm()
    #rdm2 = RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids','05','betas_run-03',prepdir=prepdir)
    #rdm2.beta_reader('glm_single',template='')
    #rdm2.create_RDM(dist_method='euclidean')
    #rdm2.save_rdm()
    #rdm3= RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids','03','betas_run-02',prepdir=prepdir)
    #rdm3.beta_reader(derivname,template='')
    #rdm3.create_RDM(dist_method='euclidean')
    #rdm3.save_rdm()
    #rdm4=RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids','04','betas_run-02',prepdir=prepdir)
    #rdm4.beta_reader(derivname,template='')
    #rdm4.create_RDM(dist_method='euclidean')
    #rdm4.save_rdm()
    #rdm5= RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids','05','betas_run-02',prepdir=prepdir)
    #rdm5.beta_reader(derivname,template='')
    #rdm5.create_RDM(dist_method='euclidean')
    #rdm5.save_rdm()
    pass
##
#
#