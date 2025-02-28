"""For transforming mask across templates"""
from os.path import join as pjoin
import numpy as np
import pandas as pd
import re
from nilearn import image, plotting
from nilearn.masking import intersect_masks,unmask
from nilearn import plotting, image
import numpy as np
import os
#creating the path for the roi_mask

#create the directory path for the top level directory of the masks
mask_dir='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_roi/rois/category_localizer'
out_dir='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_rois_transformed/'
#creating the directory if it does not exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
#create the loop for extracting the roi masks across the subjects
#create sub_id list
sub_id='01'
sub_dir_base = pjoin(mask_dir, f'sub-{sub_id}') 
reg_ids=['FFA','OFA','LOC','PPA','STS','RSC','TOS']#specifying the region ID
#creating a dictionary for paths based on reg_id
for reg_id in reg_ids:
    roi_path_dict={'EBA':'body_parcels','FFA':'face_parcels','OFA':'face_parcels','LOC':'object_parcels','PPA':'scene_parcels','STS':'face_parcels','RSC':'scene_parcels','TOS':'scene_parcels'}
    #creating the path for the mask images
    sub_dir=pjoin(sub_dir_base,roi_path_dict[reg_id])
    #loading the mask images
    #loading the left and right body parcels
    mask_img_left = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_l{reg_id}.nii.gz'))
    mask_img_right = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_r{reg_id}.nii.gz'))
    roi_mask_image_path=pjoin(sub_dir,f'sub-{sub_id}_{reg_id}.nii.gz')
    #check if path exists
    if os.path.exists(roi_mask_image_path):
        roi_mask_img=image.load_img(roi_mask_image_path)
    else:
        roi_mask_l_data=mask_img_left.get_fdata()
        #convert the mask image to a numpy array
        roi_mask_r_data=mask_img_right.get_fdata()
        #add the numpy arrays
        roi_mask_data=roi_mask_l_data+roi_mask_r_data
        #convert the numpy array to a nifti image
        roi_mask_img=image.new_img_like(mask_img_left,roi_mask_data)
    target_mask=image.load_img('/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/fmriprep/sub-01/ses-things01/func/sub-01_ses-things01_task-things_run-10_space-T1w_desc-brain_mask.nii.gz')
    #transform to numpy array
    #transform mask from one shape to another
    roi_mask_transformed=image.resample_to_img(roi_mask_img,target_mask,interpolation='nearest')
    #saving the transformed mask
    roi_mask_transformed.to_filename(pjoin(out_dir,f'sub-{sub_id}_{reg_id}_transformed.nii.gz'))

