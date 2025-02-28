"""Creating RDMs from estimated beta weights"""

import os
import gc
import numpy as np
import pandas as pd
from glm_class import GODGLM
import nibabel as nib

from os.path import join as pjoin
from joblib import Parallel, delayed
from utils import apply_mask_smoothed
from nilearn.image import new_img_like,load_img
from tqdm import tqdm

### creating the rdm class
get_data=lambda filename:nib.load(filename).get_fdata()
class beta_avgizer:
    
    ##constructor
    def __init__(self,bidsroot,sub_id,derivname):
        self.bidsroot=bidsroot
        self.sub_id=sub_id
        self.derivname=derivname
        #create a object of godglm 
        godglm = GODGLM(bidsroot, sub_id, out_deriv_name='tmp')
        self.glmobj=godglm


    def beta_reader(self,betas_type,template=''):
        if template:self.template=template
        else: self.template='t1w'

        #read the betas for each session
        def load_betas(
                sub: str,
                glm_obj : GODGLM,
                bidsroot: str = '/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
                betas_type: str = 'regularized' ,
                dtype=np.single,
                template:str=''
        ) -> np.ndarray:
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
            #if the template is not the orginal t1w 
            #reshape the mask into the template space
            #read the betafiles
            #betas_l=[get_data(bf) for bf in betafiles[:2]]
            batch_size=5
            #extract the beta files in batches
            betas_l=[]
            for batch_start in range(0,len(betafiles),batch_size):
                batch_end=min(batch_start+batch_size,len(betafiles))
                with Parallel(n_jobs=batch_size) as parallel:
                    betas_batch = parallel(
                        delayed(get_data)(bf)
                        for bf in tqdm(betafiles[batch_start:batch_end], desc='loading betas')
                    )
                betas_l+=betas_batch
                gc.collect()
                  
            #read the condition files
            conditions_l=[pd.read_csv(c,sep='\t') for c in conditionfiles]
            for run_i in tqdm(range(glm_obj.nruns_total), desc='processing runs'):
                #extract the beta file for the session
                betas=betas_l[run_i]
                #extract the condition file for the session
                conditions=conditions_l[run_i]
                #create the boolea mask for rest condition
                rest_mask=conditions['image_filename']=='rest'
                #create the boolean mask for imagefile names ending with r.jpeg
                r_mask=conditions['image_filename'].str.endswith('R.JPEG')
                #combine the masks
                mask=rest_mask | r_mask
                #extract the rows from the beta matrix
                betas=betas[:,:,:,~mask]
                #extract the rows from the condition files
                conditions=conditions.loc[~mask,'image_filename'].to_numpy()
                #sort the condition indexes
                conditions_sorted_index=np.argsort(conditions)
                #sort the conditions
                conditions_sorted=conditions[conditions_sorted_index]
                #sort the betas
                betas_sorted=betas[:,:,:,conditions_sorted_index]
                betas_l[run_i]=betas_sorted
                if run_i==len(betas_l)-1:#debugger condition
                    break
            #extract the mean of the betas
            #create a placeholder of 0s
            betas_init=np.zeros(betas_l[0].shape)
            for batch_start in range(0,len(betas_l),batch_size):
                batch_end=min(batch_start+batch_size,len(betas_l))
                #extract the betas and sum them
                betas_sum=np.sum(np.stack(betas_l[batch_start:batch_end]),axis=0)
                betas_init=betas_init+betas_sum
                gc.collect()
            #calculate the mean of the betas
            betas=betas_init/len(betas_l)
            #betas=np.mean(np.stack(betas_l),axis=0)
            #make the betas into a image
            #extract each conditions beta values and create a new image
            beta_conditions=[betas[:,:,:,cond_ind] for cond_ind in range(betas.shape[-1])]
            ref_image=load_img(betafiles[0])
            betas_images=[new_img_like(ref_image,beta_conditions[cond_ind]) for cond_ind in range(len(beta_conditions))]
            return betas_images,conditions_sorted


        #load the betas
        self.betas,self.conditions=load_betas(self.sub_id,self.glmobj,betas_type=betas_type,template=template)

    def save_files(self,out_derivname:str='perceptionTest-avg'):
        #save the betas
        betas_dir=pjoin(self.bidsroot,'derivatives',self.derivname,self.betas_type,f'sub-{self.sub_id}',f'ses-{out_derivname}')
        if not os.path.exists(betas_dir):os.makedirs(betas_dir)
        for ind,beta_condition in enumerate(self.betas):
            beta_condition_file=pjoin(betas_dir,f'sub-{self.sub_id}_ses-{self.glmobj.target_session}-{self.conditions[ind]}_{self.template}.nii.gz')
            beta_condition.to_filename(beta_condition_file)
        conditions=pjoin(betas_dir,f'sub-{self.sub_id}_ses-{self.glmobj.target_session}avg_conditions.npy')
        np.save(conditions,self.conditions)

if __name__ == '__main__':
    for sub_ind in range(1,6):
        sub_id=f'0{sub_ind}' if sub_ind<10 else str(sub_ind)
        obj = beta_avgizer('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',sub_id,'betas_run-01')
        #obj.betas_type='scalematched'
        obj.beta_reader('regularized',template='MNI305')
        obj.save_files()
