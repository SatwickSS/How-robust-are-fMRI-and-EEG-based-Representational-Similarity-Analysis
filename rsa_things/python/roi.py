"""Plots ROIs"""
##importing the necessary libraries

from os.path import join as pjoin
import numpy as np
import pandas as pd
import re
from nilearn import image, plotting
from nilearn.masking import intersect_masks



#create the directory path for the top level directory of the masks
mask_dir = pjoin('/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_roi','rois','category_localizer')
#setting the background image for visualization
bg_img='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/fmriprep/sub-01/anat/sub-01_acq-prescannormalized_rec-pydeface_desc-preproc_T1w.nii.gz'




#create the loop for extracting the roi masks across the subjects
#create sub_id list
sub_ids=['01','02','03']
#loop through the sub_ids
#create an empty list to store the roi images
roi_imgs=[]
mask_img_left_list=[]
mask_img_right_list=[]
for sub_id in sub_ids:
    sub_dir_base = pjoin(mask_dir, f'sub-{sub_id}')
    reg_id='EBA'
    #creating a dictionary for paths based on reg_id
    roi_path_dict={'EBA':'body_parcels','FFA':'face_parcels','OFA':'face_parcels','LOC':'object_parcels','PPA':'scene_parcels','STS':'face_parcels','RSC':'scene_parcels','TOS':'scene_parcels'}
    #creating the path for the mask images
    sub_dir=pjoin(sub_dir_base,roi_path_dict[reg_id])
    #loading the mask images
    #loading the left and right body parcels
    mask_img_left = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_l{reg_id}.nii.gz'))
    mask_img_right = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_r{reg_id}.nii.gz'))
    #create the path for the mask images
    rois_path=[pjoin(sub_dir, f'sub-{sub_id}_{h_sphere}{reg_id}.nii.gz') for h_sphere in ['l','r']]
    #combine rois from left and right hemisphere
    roi_mask=intersect_masks([mask_img_left,mask_img_right],threshold=0)
    #convert the mask image to a numpy array
    roi_mask_l_data=mask_img_left.get_fdata()
    #convert the mask image to a numpy array
    roi_mask_r_data=mask_img_right.get_fdata()
    #add the numpy arrays
    roi_mask_data=roi_mask_l_data+roi_mask_r_data
    #convert the numpy array to a nifti image
    roi_mask_img=image.new_img_like(mask_img_left,roi_mask_data)
    #store the roi images in a list
    roi_imgs+=[roi_mask_img]
    mask_img_left_list+=[mask_img_left]
    mask_img_right_list+=[mask_img_right]
    #plot the roi images
    plotting.plot_roi(roi_mask_img,bg_img=bg_img,title=f'sub-{sub_id}_{reg_id}',display_mode='ortho',colorbar=True,cmap='Paired')

roi_mean_img=image.mean_img(roi_imgs)
#plotting the roi image
plotting.plot_stat_map(roi_mean_img,bg_img=bg_img,title=f'mean_{reg_id}',display_mode='ortho',colorbar=True)
