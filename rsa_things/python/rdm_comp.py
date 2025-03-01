"""Simple RDM creation funciton from numpy arrays"""
import numpy as np
import pandas as pd
import glob

from os.path import join as pjoin
from joblib import Parallel, delayed
from tqdm import tqdm


#creating the function for reading RDMs
def read_RDM(bidsroot,derivname,sub_id):
    #create the directory 
    rdm_dir=pjoin(bidsroot,'derivatives',derivname,f'sub-{sub_id}','rdm')
    #extract all the files ending with npy in the directory
    rdm_files=glob.glob(pjoin(rdm_dir,'*.npy'))
    #create an empty dictionary to store the RDMs
    rdm_dict={}
    #iterate over the files
    for rdm_file in rdm_files:
        #extract the name of the file
        rdm_name=rdm_file.split('/')[-1].split('_')[-2][4:]
        #load the RDM
        rdm=np.load(rdm_file,allow_pickle=True).squeeze()
        #store the RDM in the dictionary
        rdm_dict[rdm_name]=rdm
    og_rdm=np.load(pjoin(bidsroot,'derivatives','betas_run-03','original',f'sub-{sub_id}','rdm',f'sub-{sub_id}_avgrdm.npy'))
    rdm_dict['og']=og_rdm.squeeze()
    return rdm_dict

if __name__=='__main__':
    #extract the scalematched rdms for the subjects
    scalematched_rdms=[]
    derivname='betas_run-03/scalematched'
    for sub in range(1,6):
        sub_id=f'{sub:02d}'
        scalematched_rdms.append(read_RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',derivname,sub_id))
    #extract the glm single rdms for the subjects
    glmsingle_rdms=[]
    derivname='betas_run-03/glm_single'
    for sub in range(1,6):
        sub_id=f'{sub:02d}'
        glmsingle_rdms.append(read_RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',derivname,sub_id))

    pass