"""Extracting the image names for a particular session
"""
## extracting the image names for a particular session

#import the necessary libraries
import numpy as np
import os
import pandas as pd
from os.path import join as pjoin
from glm_class import GODGLM


#read a condiitons file 
def extract_conditions(
    sub: str,
    glm_obj : GODGLM,
    bidsroot: str = '/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
    betas_outdir: str = 'betas_run-01/regularized',
) -> pd.DataFrame:
    conditions_dir=pjoin(bidsroot,'derivatives',betas_outdir,f'sub-{sub}')
    condition_files=[
                pjoin(conditions_dir, f'ses-{glm_obj.target_session}{ses_i+1:02d}', f'sub-{sub}_ses-{glm_obj.target_session}{ses_i+1:02d}_run-{run_i+1:02d}_conditions.tsv')
                for ses_i in range(glm_obj.n_sessions) for run_i in range(glm_obj.nruns_perses_[glm_obj.ds.target_sessions[ses_i]])
            ]
    #assert if the conditions exist
    for c in condition_files:
        assert os.path.exists(c)
    #read one of the condition files
    conditions=pd.read_csv(condition_files[0],sep='\t')
    #convert into a nd_array
    image_filenames=conditions['image_filename'].values
    #filter the image filenames
    #create the boolea mask for rest condition
    rest_mask=conditions['image_filename']=='rest'
    #create the boolean mask for imagefile names ending with r.JPEG
    r_mask=conditions['image_filename'].str.endswith('r.JPEG')
    mask=rest_mask|r_mask
    #extract the image filenames
    image_filenames=image_filenames[~mask]
    #sort the array
    image_filenames=np.sort(image_filenames)
    #save the image filenames
    np.save(pjoin('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/images',f'{godglm.target_session}-image_filenames.npy'),image_filenames)
    return
    #save the image filenames in the location of the iamge folder 


if __name__=='__main__':
        #create a GOGLM object
        bidsroot='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids'
        sub_id='01'
        #for extracting a different sessions images specify the target session in the GODGLM object
        godglm = GODGLM(bidsroot, sub_id, out_deriv_name='tmp')
        extract_conditions(sub_id,godglm)