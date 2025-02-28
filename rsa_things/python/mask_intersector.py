
"""Creates the intersection masks for the localizer data"""




##importing the necessary libraries
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import re
from nilearn import image, plotting
from nilearn.masking import intersect_masks
from nilearn.image import threshold_img,math_img,resample_img
import matplotlib.pyplot as plt
import math
from rdm_ppln_prep import get_masks


def wrapper(func):
    def inner(n,k):
        combi_lists=func(n,k)
        #convert teh list to strings
        list_str=['_'.join(str(run_id+1) for run_id in combi_list) for combi_list in combi_lists ]
        return list_str
    return inner

@wrapper
def combinations(n,k):
    def inner_combinations(n,k):
        if k == 0:
            return [[]]
        else:
            return [pre + [i] for i in range(n) for pre in inner_combinations(i, k-1)]
    return inner_combinations(n,k)
def mask_converter_from_img(img,threshold=0):
    #convert the image into a mask
    thresh_img=threshold_img(img,threshold=threshold)
    mask=math_img('img>0',img=thresh_img)
    return mask
def extract_parcel_mask(parcel_type:str,subject:str,bidsroot:str='/DATA1/satwick22/Documents/fMRI/thingsmri'):
    #setting the base dir
    parcel_basedir='/DATA1/satwick22/Documents/fMRI/parcel_regs/'
    #create the mask for resampling the parcel image
    brain_mask=get_masks(subject)

    #create a threshold dictionary
    thresh_dict={'face':float(1.0000001),'body':float(1.0000001),'object':float(1.0000001),'scene':float(1.0000001)}
    #extracting the parcel nifti image
    if parcel_type=='all':
        #combine all the parcels
        #union the masks using math_img
        parcel_nii4d=image.load_img(pjoin(parcel_basedir,'cvs_face_parcels','fROIs-fwhm_5-0.0001.nii'))
        #resample the image to the same size as the brain mask
        parcel_nii=resample_img(parcel_nii,target_affine=brain_mask.affine,target_shape=brain_mask.shape)
        #convert the image into a mask 
        parcel_mask=mask_converter_from_img(parcel_nii)

        for parcel in ['body','object','scene']:
            parcel_nii_=image.load_img(pjoin(parcel_basedir,f'cvs_{parcel}_parcels','fROIs-fwhm_5-0.0001.nii'))
            #resample the image
            parcel_nii_=resample_img(parcel_nii_,target_affine=brain_mask.affine,target_shape=brain_mask.shape,interpolation='nearest')
            #convert the image into a mask
            parcel_mask_=mask_converter_from_img(parcel_nii_,threshold=thresh_dict[parcel_type])
             
            parcel_mask=math_img('img1 | img2',img1=parcel_nii,img2=parcel_mask_)

    else:#read->load->resample->convert
        parcel_nii_path=pjoin(parcel_basedir,f'cvs_{parcel_type}_parcels','fROIs-fwhm_5-0.0001.nii')
        #load the nifti image
        parcel_nii_4d=image.load_img(parcel_nii_path)
        #extract the first image
        parcel_nii=image.index_img(parcel_nii_4d,0)
        #resample the image to the same size as the brain mask
        parcel_nii=resample_img(parcel_nii,target_affine=brain_mask.affine,target_shape=brain_mask.shape,interpolation='nearest')
        #convert the image into a mask
        parcel_mask=mask_converter_from_img(parcel_nii,threshold=thresh_dict[parcel_type])
    #threshold nad convert the image into a mask

    return parcel_mask   
def mask_generator(roi:str,bidsroot:str,subject:str,nruns:int,combine_method:str):
    #setting up the function run id 
    func_run_id='final__'#SET
    #create the ROI dictionary for mapping ROI to parcels
    roi_category={'FFA':'face','PPA':'scene','EBA':'body','LOC':'object'}
    #extract the parcel mask
    parcel_mask=extract_parcel_mask(roi_category[roi],subject,bidsroot)
    total_runs=6#SET
    save_mask=True#SET
    bg_img=pjoin(bidsroot,'derivatives','fmriprep',f'sub-{subject}','anat',f'sub-{subject}_acq-prescannormalized_rec-pydeface_desc-preproc_T1w.nii.gz')
    #create the directory for the intersection masks
    #get the indexes of the combinations
    #for each value of k we will extract a new mask 
    #debug and prepare the code for one value of k first
    parent_tag=str(nruns)+'runs'
    mask_list=[]
    localizer_dir_base=pjoin(bidsroot,'derivatives',f'localizer-{parent_tag}')
    #create the output directory for the masks
    outdir_base=pjoin(bidsroot,'derivatives',f'roi_masks_run_{func_run_id}',f'{nruns}_runs')
    outdir=pjoin(outdir_base,f'sub-{subject}')
    if save_mask:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    for i,run_id_tag in enumerate(combinations(total_runs,nruns)):
        #create the directory for the intersection masks
        #since LOC does not have any contrast-LOC_all directory
        if roi=='LOC': localizer_dir=pjoin(localizer_dir_base,f'runs_{run_id_tag}',f'sub-{subject}',f'contrast-{roi}')
        else: localizer_dir=pjoin(localizer_dir_base,f'runs_{run_id_tag}',f'sub-{subject}',f'contrast-{roi}_all')
        #extracting the threshold map
        threshold_map=pjoin(localizer_dir,'zstat1_threshold.nii.gz')
        #extracting the image
        threshold_contrast_img=image.load_img(threshold_map)
        #convert the image into a mask
        threshold_contrast_mask=mask_converter_from_img(threshold_contrast_img)

        mask_list.append(threshold_contrast_mask)
        
        
        #read the threshold image
    #create the intersection mask
    #for the case when the
    if len(mask_list)==1:#when the mask is already a mean over all sessions
        #intersect the mask with the parcel mask
        intersect_mask=math_img('img1 & img2',img1=mask_list[0],img2=parcel_mask)
        #save the mask
        if save_mask:
            intersect_mask.to_filename(pjoin(outdir,f'sub_{subject}_{roi}_{combine_method}.nii.gz'))
        return
    else:
        if combine_method=='intersection':
            intersection_mask = mask_list[0]
            for mask in mask_list[1:]:
                intersection_mask = math_img('img1 & img2', img1=intersection_mask, img2=mask)
        elif combine_method=='union':
            intersection_mask = mask_list[0]
            for mask in mask_list[1:]:
                intersection_mask = math_img('img1 | img2', img1=intersection_mask, img2=mask)
        #intersect the resulting mask with the parcel mask
        intersection_mask=math_img('img1 & img2',img1=intersection_mask,img2=parcel_mask)
        ##plot the parcel mask
    if save_mask:
        #save the intersection mask
        #create the directory for the storing the intersection masks
        intersection_mask.to_filename(pjoin(outdir,f'sub_{subject}_{roi}_{combine_method}.nii.gz'))
    
    return

#driving the function
if __name__=='__main__':
    roi_list=['EBA','PPA','FFA','LOC']
    bidsroot='/DATA1/satwick22/Documents/fMRI/thingsmri'
    for roi in roi_list:
        for nruns in range(6,2,-1):
            for combine_method in ['intersection','union']:
                if nruns==6:
                    mask_generator(roi,bidsroot,'01',nruns,combine_method)
                    break
                else:
                    mask_generator(roi,bidsroot,'01',nruns,combine_method)
    


