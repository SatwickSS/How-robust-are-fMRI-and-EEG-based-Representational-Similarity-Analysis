"""
Get RDM scores for specifications
"""
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import rsatoolbox.rdm as rsr
import json,re,pickle
from rdm_comparison import data_grabber_rdm,rdm_creator
from rdm_helper_rois import mask_driver
from os.path import join as pjoin
import os,sys
from tqdm import tqdm


def rdm_roi_driver3(sub_id:str,run_id:str,n_shuffle:int,n_shuffle_offset:int=0):
    """Creates RDM for all conditions of ROI estimation and HRF indices estimation"""
    #assigning the variables 
    #run_id='test'#SET
    #SET A FEW THINGS EXTRA
    #n_shuffle=5#SET
    betas_fit='regularized'#SET
    bidsroot='/scratch/satwick22/DATA1/satwick22/Documents/fMRI/thingsmri'
    results={}
    #for the original sample get the RDM for the original specification
    #best_hrf_inds_by,nruns,combine_method='mean',6,'intersection'
    def vanilla_rdm_generator(sub_id:str,run_id:str,sample_ind:int,
                              n_sessions:int=12,
                              data_centering_method:str='psc',
                              data_rescaling:str='re-center',
                              zscore_residuals:bool=True,
                              betas_fit:str='regularized',
                              best_hrf_inds_by:str='mean',
                              nruns:int=6,
                              combine_method:str='intersection'):#change function definition
        for session in tqdm(range(n_sessions),desc=f'Session for Sample-{sample_ind+1}'):
            sample_ind_beta=0
            
            betas_dir_base=pjoin(f'betas_roi_combi_run-{run_id}',f'{betas_fit}',f'centered_by_{data_centering_method}',f'recentered_by_{data_rescaling}',f'zscored_{int(zscore_residuals)}',f'hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}') 
            #betas_dir_base=pjoin(bidsroot,'derivatives',f'betas_roi_combi_run-{run_id}',f'{betas_fit}',f'hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}')
            #for now adding the sample_index, will have to be removed later
            betas_dir=pjoin(bidsroot,'derivatives',betas_dir_base,f'sub-{sub_id}')
            #betas_dir=pjoin(betas_dir_base,f'sub-{sub_id}')
            #read the betas numpy array
            betas_concat=np.load(pjoin(betas_dir,f'ses-things{session+1:02}_betas.npy'))
            #create the condnames for the shuffle
            condnames_concat=np.load(pjoin(bidsroot,'derivatives',f'betas_roi_combi_run-{run_id}','shuffle_indexes',f'sample-{sample_ind+1}',f'ses-things{session+1:02}_condnames.npy'))
            #argsort the condnames so the orders are the same across session
            condnames_concat_index=np.argsort(condnames_concat)
            #sort the betas and the condnames based on the sorted condnames index
            betas_concat=betas_concat[condnames_concat_index]
            condnames_concat=condnames_concat[condnames_concat_index]
            #condnames_concat=np.load(pjoin(betas_dir,f'ses-things{session+1:02}_condnames.npy'))
            #creae the rdm_dataset_object
            betas_data=rsd.Dataset(measurements=betas_concat,descriptors={'session':f'ses-things{session+1:02}','subj':sub_id},obs_descriptors={'conds':condnames_concat},channel_descriptors={'voxels':np.array(['voxel'+str(i) for i in range(betas_concat.shape[1])])})
            #create the data structure for the betas
            #create the RDMs
            print('Estmating the RDMs for the original specification : for')
            print(f'sample_{sample_ind+1}:ses_things{session+1:02}:',f'centered_by_{data_centering_method}',f'recentered_by_{data_rescaling}',f'zscored_{int(zscore_residuals)}',f'hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}')
            if session:#that is for any session > 0
                rdm_vanilla_sess_list.append(rsr.calc_rdm(betas_data,method='correlation',descriptor='conds'))
            else:
                rdm_vanilla_sess_list=rsr.calc_rdm(betas_data,method='correlation',descriptor='conds')
        avg_rdm_vanilla=rdm_vanilla_sess_list.mean()
        return avg_rdm_vanilla
    #rdm_vanilla_avg=vanilla_rdm_generator(sub_id=sub_id,run_id=run_id)
    #set the parameters inside the function
    def parallel_driver(sample_ind,
                        sub_id,
                        run_id,
                        data_centering_methods=['psc','z','center'],
                        #data_rescaling_methods=['off'],
                        data_rescaling_methods=['psc','z','re-center','off'],#SET 
                        best_hrf_methods=['mean','mode'],#SET
                        n_sessions=12,#SET
                        betas_fit='regularized'#SET
                        ):
        #driver function for running the program parallely for each shuffled sample
        rdm_vanilla_avg=vanilla_rdm_generator(sub_id=sub_id,run_id=run_id,sample_ind=sample_ind)
        for data_centering_method in data_centering_methods:
            for data_rescaling in data_rescaling_methods:
                for zscore_residuals in [True,False]:#SET
                    for best_hrf_inds_by in best_hrf_methods:
                        for nruns in range(6,2,-1):#SET
                            for combine_method in ['intersection','union']:
                                #skipping the intersection condition for nruns=6
                                if nruns==6 and combine_method=='union':continue
                                for session in range(n_sessions):
                                    #read then condnames for the session
                                    condnames=np.load(pjoin(bidsroot,'derivatives',f'betas_roi_combi_run-{run_id}','shuffle_indexes',f'sample-{sample_ind+1}',f'ses-things{session+1:02}_condnames.npy')) 
                                    #condnames=np.load(pjoin(bidsroot,'derivatives',f'betas_roi_combi_run-{run_id}','regularized','shuffle_indexes',f'sample-{sample_ind+1}',f'ses-things{session+1:02}_condnames.npy'))#change
                                    #argsort the condnames
                                    condnames_index=np.argsort(condnames)
                                    #comp_rdm=None
                                    #create the alternate specification dictionary key
                                    specification_tup=(f'centered_by_{data_centering_method}',f'recentered_by_{data_rescaling}',f'zscored_{int(zscore_residuals)}',f'hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}')
                                    #combine the specification tuple to create the dictionary key
                                    dict_key=str.join('_',specification_tup)
                                    alt_spec_dict_key=dict_key#change
                                    #if the specification key does not exist create a dictionary agaisnt it
                                    #if not results.get(alt_spec_dict_key):
                                    #    results[alt_spec_dict_key]={}

                                    #creating the sample dictionary if it does not exist
                                    # if not results[alt_spec_dict_key].get(f'sample_{sample_ind+1}'):
                                    #     results[alt_spec_dict_key][f'sample_{sample_ind+1}']={}
                                    #if not results[alt_spec_dict_key][f'sample_{sample_ind+1}'].get(f'ses_things{session+1:02}'):
                                     #   results[alt_spec_dict_key][f'sample_{sample_ind+1}'][f'ses_things{session+1:02}']={}

                                    #create the betas directory
                                    betas_dir_base=pjoin(f'betas_roi_combi_run-{run_id}',f'{betas_fit}',f'centered_by_{data_centering_method}',f'recentered_by_{data_rescaling}',f'zscored_{int(zscore_residuals)}',f'hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}') 
                                    #betas_dir=
                                    #betas_dir_base=pjoin(bidsroot,'derivatives',f'betas_roi_combi_run-{run_id}',f'{betas_fit}',f'hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}',f'sample{1}')#change
                                    #extract the betas from first sample because the shuffling is based on condanmes 
                                    betas_dir=pjoin(bidsroot,'derivatives',betas_dir_base,f'sub-{sub_id}')
                                    #read the betas numpy array
                                    betas_concat=np.load(pjoin(betas_dir,f'ses-things{session+1:02}_betas.npy'))
                                    #sort the betas_concat and condnames_concat
                                    betas_concat=betas_concat[condnames_index]
                                    condnames=condnames[condnames_index]
                                    #condnames_concat=np.load(pjoin(betas_dir,f'ses-things{session+1:02}_condnames.npy'))
                                    #creae the rdm_dataset_object
                                    betas_data=rsd.Dataset(measurements=betas_concat,descriptors={'session':f'ses-things{session+1:02}','subj':sub_id},obs_descriptors={'conds':condnames},channel_descriptors={'voxels':np.array(['voxel'+str(i) for i in range(betas_concat.shape[1])])})

                                    #chekcing if this specification is the original specification
                                    #if best_hrf_inds_by=='mean' and nruns==6 and combine_method=='intersection':
                                        #create the data structure for the betas
                                        #create the RDMs
                                        #print('Estmating the RDMs for the original specification : for')
                                        #print(f'sample_{sample_ind+1}:ses_things{session+1:02}:hrf_by_{best_hrf_inds_by}_{nruns}_runs_{combine_method}')
                                        #comp_rdm=rsr.calc_rdm(betas_data,method='correlation',descriptor='conds')                            #save the RDMs
                                    rdm=rsr.calc_rdm(betas_data,method='correlation',descriptor='conds')
                                    if session:#that is for any session > 0
                                        rdm_sess_list.append(rdm)
                                    else:
                                        rdm_sess_list=rdm

                                try:
                                    rdm_avg=rdm_sess_list.mean()
                                    comp_score=rsr.compare(rdm_avg,rdm_vanilla_avg,method='tau-a')[0][0]
                                except:
                                    pass
                                #store the result
                                results[alt_spec_dict_key]=comp_score
        return results
    #parallelize the loop on shuffles using joblib
    #debugging code block
    #_=parallel_driver(sample_ind=0,sub_id=sub_id,run_id=run_id)
    from joblib import Parallel, delayed
    results=Parallel(n_jobs=20)(delayed(parallel_driver)(sample_ind,sub_id,run_id) for sample_ind in range(n_shuffle_offset,n_shuffle_offset+n_shuffle))
    out_dir_base=pjoin(bidsroot,'derivatives',f'betas_roi_combi_run-{run_id}',f'{betas_fit}')
    #creating the subject directory
    out_dir=pjoin(out_dir_base,f'sub-{sub_id}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #saving the results in a pickle file
    estimator_run_id='02'
    with open(pjoin(out_dir,f'sub-{sub_id}_avg_rdm_roi_comparison_nshuffles-from{n_shuffle_offset}-to-{n_shuffle}-run-{estimator_run_id}.pkl'),'wb') as f:
        pickle.dump(results,f)
    #with open(pjoin(out_dir,f'sub-{sub_id}_rdm_roi_comparison.json'),'w') as f:
    #    json.dump(results,f)
    return results


if __name__=='__main__':
    sub_id='01'
    #speicfy the run id
    run_id='06'#SET
    #specify the number of shuffles
    import sys
    n_shuffle=100#SET
    n_shuffle_offset=0#SET
    
#    n_shuffle_offset=0
    #run the fiunction and extract the results
    results=rdm_roi_driver3(sub_id,run_id=run_id,n_shuffle=n_shuffle,n_shuffle_offset=n_shuffle_offset)
#defining the rdm comparison function for a subject