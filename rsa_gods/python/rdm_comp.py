"""
For comparing two RDMs 
"""
import numpy as np
import pandas as pd
import glob
#from glm_class import GODGLM
#import rsatoolbox.data as rsd
#import rsatoolbox.rdm as rsr

from os.path import join as pjoin
from joblib import Parallel, delayed
#from utils import apply_mask_smoothed
#from nilearn.image import new_img_like,load_img
from tqdm import tqdm


#creating the function for reading RDMs
def read_RDM(bidsroot,derivname,sub_id,template=''):
    #create the directory 
    rdm_dir=pjoin(bidsroot,'derivatives',derivname,f'sub-{sub_id}',f'rdm{template}')
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
    scalematched_rdms_=[]
    derivname='betas_run-03/scalematched'
    for sub in range(1,6):
        sub_id=f'{sub:02d}'
        scalematched_rdms_.append(read_RDM('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',derivname,sub_id,template='MNI305'))

    pass