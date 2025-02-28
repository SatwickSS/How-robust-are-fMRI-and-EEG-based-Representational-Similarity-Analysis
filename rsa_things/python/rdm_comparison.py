"""Creates RDMs from npz"""
#importing the necessary libraries

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset objects
import rsatoolbox.rdm as rsr # abbreviation to deal with RDMs
import json,re
from os.path import join as pjoin
#loading the dataset i.e the beta weights for various sessions and the mean beta weights across sessions

#function for extracting the beta weights and creating the rsatoolbox dataset object
def data_grabber_rdm(
        indir_base:str='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/betas/scalematched/sub-01/',
):
    "returns the rsatoolbox dataset object given the path of the dictionary of measurements and the path of the condition names"
    #indir_base='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/betas/scalematched/sub-01/'
    betas_path=pjoin(indir_base,'betas_dict.npz')
    betas=np.load(betas_path)
    #loading the condition names
    condnames_path=pjoin(indir_base,'condnames_list.json')
    with open(condnames_path, "r") as fp:
        condnames=json.load(fp)
    #print(len(betas),len(condnames))
    #creating the empty list to store the data
    data=[]
    for ind,dict_item in enumerate(list(betas.items())): # iterating over the betas dictionary
        key,value=dict_item
        des = {'session': re.search(r'_.+',key)[0][1:], 'subj': 1}
        measurements = value
        obs_des = {'conds': condnames[ind]}
        #obs_des = {'conds': np.array(['cond_' + str(x) for x in np.arange(2,nCond+1)])} # indices␣˓ →from 1
        #chn_des = {'conds': np.array(['voxel' + str(x) for x in np.arpjoin(ange(1,nVox+1)])} # indices␣˓ →from 1
        data.append(rsd.Dataset(measurements=measurements,
        descriptors=des,
        obs_descriptors=obs_des,
        ))
    return data
def rdm_creator(outdir:str='/DATA/satwick22/Documents/fMRI/fMRI_processing/trash/plots/'):
    "creates the rdm and saves the plot in the directory specified in the function argument"
    #calculating the RDMs
    data=data_grabber_rdm()
    rdms=rsr.calc_rdm(data[:3],method='correlation',descriptor='conds')
    #plot the rdm
    fig,ax,retval=rsatoolbox.vis.show_rdm(rdms[0],descriptor='conds',show_colorbar='panel')
    #save the figure object 
    fig.savefig(pjoin(outdir,'rdm.png'))
    return 
if __name__=='main':
    rdm_creator()