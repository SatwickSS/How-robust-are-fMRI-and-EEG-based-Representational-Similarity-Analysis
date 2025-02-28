"""Contains helper function for RDM creation  from RDMs"""
#importing the necessary nilearn masking modules

from os.path import join as pjoin
import numpy as np
import pandas as pd
import re
from nilearn import image, plotting
from nilearn.masking import intersect_masks,unmask
from nilearn import plotting, image
import numpy as np
import os
from rdm_ppln_prep import get_masks
union_mask=None
#designing the function for indexing the masks
def mask_indexer(mask_img:object):
    """ Function to index the mask image """
    #transform the mask image to a numpy array
    mask_img_data=mask_img.get_fdata().flatten()
    #get the indices of the mask image
    mask_indices=np.where(mask_img_data==1)[0]
    return mask_indices
#designing the function for findig intersect of the mask indexes
def mask_index_intersect(source_mask,target_mask:list):
    source_mask_index=mask_indexer(source_mask)
    intersected_indexes=[]
    for mask in target_mask:
        mask_index=mask_indexer(mask)
        intersected_mask_index=np.intersect1d(source_mask_index,mask_index)
        intersected_indexes.append(intersected_mask_index)
    return intersected_indexes
def mask_index_intersect2(source_mask,target_mask:list):
    """Gets the intersection based on source masks index"""
    source_mask_index=mask_indexer(source_mask)
    intersected_indexes=[]
    for mask in target_mask:
        mask_index=mask_indexer(mask)
        intersected_mask_index=np.where(source_mask_index[:,np.newaxis]==mask_index[np.newaxis,:])[0]
        intersected_indexes.append(intersected_mask_index)
    return intersected_indexes        
#designing the decorator function for transforming the masks
#designing the decorator function for transforming the masks
def mask_transformer(func):
    """ Decorator function for transforming the masks """
    def wrapper(target_mask,mask_dir:str,bidsroot:str,reg_id:str,sub_id:str,transform:bool=True):
        masks=func(mask_dir,bidsroot,reg_id,sub_id)
        #masks are a list of nifti images
        if transform:
            transformed_masks=[]
            for mask in masks:
                transformed_masks.append(image.resample_to_img(mask,target_mask,interpolation='nearest'))
            return transformed_masks
        else:
            return masks
    return wrapper

#function for reading the roi masks depending on the roi specified and the root directory mentioned
@mask_transformer
def roi_reader(mask_dir:str,bidsroot:str,reg_id:str,sub_id:str):
    """ Function to read the roi masks and create the combined masks"""
    roi_path_dict={'EBA':'body_parcels','FFA':'face_parcels','OFA':'face_parcels','LOC':'object_parcels','PPA':'scene_parcels','STS':'face_parcels','RSC':'scene_parcels','TOS':'scene_parcels'}
    sub_dir_base = pjoin(mask_dir, f'sub-{sub_id}')
    #creating the path for the mask images
    sub_dir=pjoin(sub_dir_base,roi_path_dict[reg_id])
    #loading the mask images
    #loading the left and right body parcels
    mask_img_left = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_l{reg_id}.nii.gz'))
    mask_img_right = image.load_img(pjoin(sub_dir, f'sub-{sub_id}_r{reg_id}.nii.gz'))
    roi_mask_image_path=pjoin(sub_dir,f'sub-{sub_id}_{reg_id}.nii.gz')
    #check if path exists
    if os.path.exists(roi_mask_image_path):
        mask_img_combi=image.load_img(roi_mask_image_path)
    else:
        roi_mask_l_data=mask_img_left.get_fdata()
        #convert the mask image to a numpy array
        roi_mask_r_data=mask_img_right.get_fdata()
        #add the numpy arrays
        roi_mask_data=roi_mask_l_data+roi_mask_r_data
        #convert the numpy array to a nifti image
        mask_img_combi=image.new_img_like(mask_img_left,roi_mask_data)
    return [mask_img_left,mask_img_right,mask_img_combi]
#testing the roi_reader
#function for driving the funcitons and returning the indexes of the rois 
#and the union mask
def mask_driver(sub_id:str,reg_id:str,bidsroot:str = '/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_',mask_dir:str = '/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_roi/rois/category_localizer',transform=True):
    """ Function to drive the mask functions """
    """Transform=True transforms the roi masks to the union mask
    Transform=False transforms the union mask to the roi masks and leaves the roi masks intact"""

    #creating the session names
    sessions = [f'things{i:02d}' for i in range(1, 13)]
    #creating the subject prepdir [fmri dir for the subject]
    subj_prepdir=pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{sub_id}')
    #creating the union mask
    union_mask=get_masks(sub_id,subj_prepdir,sessions)
    #reading the roi masks
    if transform:
        mask_img_left,mask_img_right,mask_img_combi=roi_reader(union_mask,mask_dir,bidsroot,reg_id,sub_id,transform=transform)
    else:
        mask_img_left,mask_img_right,mask_img_combi=roi_reader(union_mask,mask_dir,bidsroot,reg_id,sub_id,transform=transform)
        union_mask=image.resample_to_img(union_mask,mask_img_left,interpolation='nearest')
    mask_roi_indexes=mask_index_intersect(union_mask,[mask_img_left,mask_img_right,mask_img_combi])
    union_mask_indexer=mask_indexer(union_mask)
    return union_mask_indexer,mask_roi_indexes

def mask_driver2(sub_id:str,reg_id:str,bidsroot:str = '/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_',mask_dir:str = '/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_roi/rois/category_localizer',transform=True,return_masks=False):
    """ Function to drive the mask functions 
    V2 uses mask_index_intersect2 and ignores the combined mask"""
    #creating the session names
    sessions = [f'things{i:02d}' for i in range(1, 13)]
    #creating the subject prepdir [fmri dir for the subject]
    subj_prepdir=pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{sub_id}')
    #creating the union mask
    union_mask=get_masks(sub_id,subj_prepdir,sessions)
    #reading the roi masks
    if transform:
        mask_img_left,mask_img_right,_=roi_reader(union_mask,mask_dir,bidsroot,reg_id,sub_id,transform=transform)
    else:
        mask_img_left,mask_img_right,_=roi_reader(union_mask,mask_dir,bidsroot,reg_id,sub_id,transform=transform)
        union_mask=image.resample_to_img(union_mask,mask_img_left,interpolation='nearest')
    ##reading the roi masks
    mask_roi_indexes=mask_index_intersect2(union_mask,[mask_img_left,mask_img_right])
    union_mask_indexer=mask_indexer(union_mask)
    if return_masks:
        return union_mask_indexer,mask_roi_indexes,mask_img_left,mask_img_right
    else:return union_mask_indexer,mask_roi_indexes
if __name__=='__main__':
    #creating the path for the roi_mask
    mask_dir='/DATA/satwick22/Documents/fMRI/fMRI_processing/thingsmri_/derivatives/masks_roi/rois/category_localizer'
    #creating the directory if it does not exist
    #create the loop for extracting the roi masks across the subjects
    #create sub_id list
    sub_id='01'
    reg_id='FFA'
    union_mask_indexer,mask_roi_indexes,_,_=mask_driver2(sub_id,reg_id,transform=False,return_masks=True)
    pass