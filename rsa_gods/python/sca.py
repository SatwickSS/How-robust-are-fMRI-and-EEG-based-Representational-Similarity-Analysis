
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')

import scipy
from scipy.spatial import distance
from scipy.stats import spearmanr,pearsonr,ttest_ind
from scipy.spatial.distance import squareform
from scipy import stats
from tqdm.auto import tqdm
import gc


import re
import os
from os.path import join as opj

import pickle
import nibabel as nib
from glob import glob
from loguru import logger

from general_utils import *
from kamitani_utils import *
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from itertools import product



cm1 , cnum1 = plt.get_cmap('Blues') , 6.
cm2 , cnum2 = plt.get_cmap('Reds')  , 8.
cm3 , cnum3 = plt.get_cmap('Greens'), 3.

model_plot_dict = {
    'clip'          : ('CLIP',                1.5, cm3(1/cnum1)),
    'virtex'      : ('VirTex',           1.5, cm3(3/cnum1)),
    #'icmlm-attfc'   : ('ICMLM-attfc',         1.5, cm3(4/cnum1)), 
    #'icmlm-tfm'     : ('ICMLM-tfm',           1.5, cm3(5/cnum1)), 
    #'tsmresnet50'   : ('TSMResNet50-visual',  1.5, cm3(6/cnum1)),


    'BiT-M'              : ('BiT-M',       3.0, cm2(1/cnum2)),
    'resnet'           : ('ResNet50',    3.0, cm2(2/cnum2)),
    'madryli8'            : ('AR-Linf8',    3.0, cm2(3/cnum2)),
    'madryli4'         : ('AR-Linf4',    3.0, cm2(4/cnum2)),
    'madryl23'         : ('AR-L2',       3.0, cm2(5/cnum2)),
    'geirhos_sin'          : ('SIN',         3.0, cm2(6/cnum2)),
    'geirhos_sinin'    : ('SIN-IN',      3.0, cm2(7/cnum2)),
    'geirhos_sininfin' : ('SIN-IN+FIN',  3.0, cm2(8/cnum2)),


    'BERT'   : ('BERT',     4.5, cm1(1/cnum3)),
    'GPT2'   : ('GPT2',     4.5, cm1(2/cnum3)),
    'CLIP-L' : ('CLIP-L',   4.5, cm1(3/cnum3)),


    'M'     : ('M', 6.0, cm3(0.25*cnum3)),
    'V'     : ('V', 6.0, cm2(0.1*cnum2)),
    'L'     : ('L', 6.0, cm1(0.3*cnum1)),

}


def _get_mask_(region_idx,atlas):

    # get the atlas
    if atlas == 'destrieux':
        aparc_file = nib.load(DESKIAN_ATLAS)
    else:
        aparc_file = nib.load(DESKIAN_ATLAS)
    aparc_data = aparc_file.get_fdata()

    # get the mask
    if region_idx is not None:

        if isinstance(region_idx,int):
            region_mask = aparc_data == region_idx
        elif isinstance(region_idx,tuple):
            region_mask = np.isin(aparc_data,region_idx)

    else:
        region_mask = aparc_data != np.nan    #basically whole brain

    return region_mask
def get_region_vectors_(sub_idx,test_beta_dir,region_idx,atlas='default',use_train=True,use_test=True):

    '''
    Function to get beta values in a specific regions of the brain.

    Args : 
        sub_idx     : index of the subject (from [1,2,3,4,5])
        region_idx  : (int/tuple/None) index of the region as per the atlas used. None would give the values in the whole brain.
        atlas       : atlas used to define the region_idx. Can be 'default' or 'destrieux'
        use_train   : include train betas
        use_test    : include test betas

        NOTE : if both use_train and use_test are given to be True, the first 150 will we train and the next 50 will be test. This order has to be preserved for all the subsequent analysis.
    '''

    region_mask = _get_mask_(region_idx,atlas)

    #train_beta_dir = ""
    test_beta_dir.insert(1,f'sub-{sub_idx:02d}')
    test_beta_dir=opj(*test_beta_dir)

    # get all names for beta files
    beta_fnames = []
    #if use_train==True:
    #    beta_fnames = sorted([f for f in os.listdir(train_beta_dir) if f[:4]=='beta'])
    #    beta_fnames = [opj(train_beta_dir,f) for f in beta_fnames[2:152]]

    if use_test==True:
        test_beta_fnames = sorted(glob(opj(test_beta_dir,'*MNI305.nii.gz'),recursive=True))
        assert len(test_beta_fnames)==50

        for x in test_beta_fnames:
            beta_fnames.append(x)


    # get the betas
    beta_data = []
    for fname in beta_fnames:
        beta_file = nib.load(fname) 
        orig_data = beta_file.get_fdata()
        #assert beta_file.shape == region_mask.shape

        region_data = np.nan*np.ones(orig_data.shape,dtype=orig_data.dtype)
        region_data[region_mask] = orig_data[region_mask]

        beta_data.append(region_data)

    #if use_train == True and use_test == True  : assert len(beta_data) == 200
    #if use_train == True and use_test == False : assert len(beta_data) == 150 
    #if use_train == False and use_test == True : assert len(beta_data) == 50 

    # reshape and remove nans
    betas = np.stack(beta_data,axis=0)
    n,h,w,c = betas.shape
    sub_data = betas.reshape(n,h*w*c) 

    nans = np.isnan(sub_data[0])
    sub_data = sub_data[:,np.logical_not(nans)]
    logger.debug(f'Shape of data for region {region_idx} is {sub_data.shape}')

    return sub_data if sub_data.shape[1] != 0 else None

def _get_sort_idx(mat):

    # we want to find voxels that respond maximally to the images
    # hence, let's first find voxels that have highest betas
    var_list=[mat[:,vox].max() for vox in range(mat.shape[1])]  
    sort_idx = np.argsort(var_list) 

    return sort_idx

def voxel_selection(data_dict,number_based=False,threshold_based=False,**kwargs):

    '''
        Voxel selection based on 2 criteria -- either number of voxels (number_based) or threshold on the beta values (threshold_based)
    '''

    if number_based:

        def select_topk_voxels(sampledata,num_voxels,sort_idx=None):
            '''
            Select the top n voxels based on the sort idx provided
            if sort_idx is None, then the max values will be calculated using max values 

            Remember that we are technically changing the order of the voxels. There will be effects because of this. 

            # TODO : what if there are high negative beta values? (Or do we not care about those? -- we dont since it means that the voxel shows anti-expt behaviour?)
            '''


            if num_voxels == None:
                return sampledata
            
            sort_idx = _get_sort_idx(sampledata) if sort_idx is None else sort_idx
            
            subsample = []
            for c in sort_idx[::-1]:        
                if len(subsample) == num_voxels:
                    # print (c,'--',sampledata[:,c].max())
                    break

                if np.all(sampledata[:,c] != 0.):
                    subsample.append(sampledata[:,c])

            subsample = np.stack(subsample,axis=1)
            return subsample




        num_voxel_dict = kwargs.pop('num_voxel_dict')
        sort_idx_dict = {_region:{sub_idx:_get_sort_idx(data_dict[_region][sub_idx]) for sub_idx in range(1,args.NUM_SUB+1)} for _region in data_dict.keys()}
        smaller_data_dict = {region : {sub: select_topk_voxels(data_dict[region][sub],num_voxels=num_voxel_dict[region],sort_idx=sort_idx_dict[region][sub]) for sub in range(1,args.NUM_SUB+1) } for region in data_dict.keys()}

        return smaller_data_dict




    if threshold_based:
        num_voxels_of_hippocampus = kwargs.pop('num_voxels_of_hippocampus')


        def get_threshold_dict(num_voxels_of_hippocampus):
            '''
            Get thresholds that you need to use as per the num_voxels_of_hippocampus. The thresholds are given per subject in a form of a dictionary.
            The idea is that we want to threshold the values of betas and use only those that are quite high. 
            '''

            threshold_dict = {s:0. for s in range(1,args.NUM_SUB+1)}
            
            for s in range(1,args.NUM_SUB+1):
                var_list = [data_dict[(17,53)][s][:,vox].max() for vox in range(data_dict[(17,53)][s].shape[1]) ]
                threshold_dict[s] = sorted(var_list)[-(num_voxels_of_hippocampus+1)]
            
            return threshold_dict


        def select_thresholded_voxels(sampledata,threshold=0.):
            '''
            Select the top voxels based on the threshold. 
            '''

            var_list = [sampledata[:,vox].max() for vox in range(sampledata.shape[1])]

            subsample = []
            for c,max_val in enumerate(var_list):
                if max_val > threshold:
                    subsample.append(sampledata[:,c])

            subsample = np.stack(subsample,axis=1)

            print ('Using threshold found : ',subsample.shape)

            return subsample


        threshold_dict = get_threshold_dict(num_voxels_of_hippocampus)
        smaller_data_dict = {region : {sub : select_thresholded_voxels(smaller_data_dict[region][sub],threshold=threshold_dict[sub]) for sub in range(1,args.NUM_SUB+1)} for region in data_dict.keys() }

        return smaller_data_dict

def select_voxels(data_dict):

    if args.MAIN_TASK == 'normal_correlations':
        return data_dict #don't change anything

    if args.MAIN_TASK == 'select_top30_voxels':
        # print ("HERE")
        num_voxel_dict = {r:30 for r in data_dict.keys()}
        smaller_data_dict = voxel_selection(data_dict,number_based=True,num_voxel_dict=num_voxel_dict)

        return smaller_data_dict

    if args.MAIN_TASK == 'select_voxels_based_on_hippocampus':
        num_voxels_of_hippocampus = 30
        smaller_data_dict = voxel_selection(data_dict,threshold_based=True,num_voxels_of_hippocampus=num_voxels_of_hippocampus)
        return smaller_data_dict


    if args.MAIN_TASK == 'max_of_each_region':

        num_voxel_dict = {
            (1007)                   : 235,
            (2007)                   : 355,
            (1007, 2007)             : 629,
            (1011, 1021)             : 888,
            (2011, 2021)             : 343,
            (1011, 2011, 1021, 2021) : 1367,
            (17)                     : 8,
            (53)                     : 17,
            (17, 53)                 : 29,
            (1016)                   : 56,
            (2016)                   : 108,
            (1016, 2016)             : 89,
        }

        smaller_data_dict = voxel_selection(data_dict,number_based=True,num_voxel_dict=num_voxel_dict)
        return smaller_data_dict


def get_upper_ceilings(smaller_data_dict):
    '''
        Calculate the upper_ceilings
    '''
    upper_ceiling_dict = {region: calculate_noise_ceilings([ get_rdm(smaller_data_dict[region][sub],distance=args.DISTANCE) for sub in range(1,args.NUM_SUB+1)  ],corr_func=CORR_FUNC) for region in smaller_data_dict.keys() }
    
    return upper_ceiling_dict

def get_model_rdms():
    tstart = time.time()
    model_data_dict = {model_1[0]:get_rdm(get_model_vectors(model_1[0],layer_to_use=model_1[1],use_test=args.USE_TEST_BETAS),distance=args.DISTANCE) for model_1 in args.MODELS}
    tend = time.time()
    print ("Time taken to get all model RDMS : ",tend-tstart)
    return model_data_dict

def get_correlations(smaller_data_dict):

    '''
        Main function that takes data dict and provides correlations with model rdms.
        smaller_data_dict : data dict of selected voxels (based on number or thresholds or No selection) 
    '''
    
    model_rdms_dict = get_model_rdms()


    corr_data_dict = {}
    for model_1 in args.MODELS:

        corr_data_dict[model_1[0]] = {k:{sub:None for sub in range(1,args.NUM_SUB+1)} for k in smaller_data_dict.keys()}
        for _region in smaller_data_dict.keys():
            for sub_idx in range(1,args.NUM_SUB+1):
            
                model_rdm = model_rdms_dict[model_1[0]]
                brain_vectors = smaller_data_dict[_region][sub_idx]
                corr_data_dict[model_1[0]][_region][sub_idx] = CORR_FUNC(model_rdm,get_rdm(brain_vectors,distance=args.DISTANCE)) if brain_vectors is not None else np.nan
    
    return corr_data_dict


def driver(betas_dir:str,
           bidsroot:str,
           betas_type:str,
           ses:str='ses-perceptionTest-avg'):
    
    def plot_fancyfig(save_name=None,data_type=''):


        fp = {'fontsize':16}
        plt.figure(figsize=(20,30))

        title_init = f"{args.MAIN_TASK} :"

        if args.MAIN_TASK == 'select_from_train_for_test':
            plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC} \n selecting top voxels from test data on train data",fontsize=10)
        if args.MAIN_TASK == 'select_voxels_based_on_hippocampus':
            plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC} \n thresholded using max value of the 30th hippocampus voxel",fontsize=10)
        if args.MAIN_TASK == 'normal_correlations' or args.MAIN_TASK=='select_top30_voxels':
            plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC}",fontsize=10)


        plt.axhline(1.,linestyle='dashed')
        plt.ylabel('Normalized Correlation',fontsize=14)

        xticks_for_models = []
        labels_for_models = []

        heights = [0.4,0.1,0.3,0.2,0.1,0.2]

        beh = 0
        for ind,region in enumerate(smaller_data_dict.keys()):

            upper_ceilings = upper_ceiling_dict[region]
            labels_for_models.append(interesting_regions_mapping[region])
            xticks_for_models.append(5*ind+1.5)

            plt.xticks([])
            plt.yticks(**fp)


            factors = [1.0/x for x in upper_ceilings]

            combined_values = {'M':[],'V':[],"L":[]}

            for mind,model in enumerate(model_plot_dict.keys()):

                if model in ['M','V','L']:
                    continue


                val = [corr_data_dict[model][region][s][0]*factors[s-1] for s in range(1,args.NUM_SUB+1)]
                datum = np.array(val)
                mean = datum.mean()
                # sem = scipy.stats.sem(datum)

                if model in ['clip','virtex','virtexv2','icmlm_attfc','icmlm_tfm','tsmresnet50','audioclip']:
                    combined_values['M'].append(mean)
                elif model in ['BERT','GPT2','CLIP-L']:
                    combined_values['L'].append(mean)
                else:
                    combined_values['V'].append(mean)
            p_values = {'MxV':[],'MxL':[],'VxL':[]}
            _,p1 = ttest_ind(combined_values['M'],combined_values['L'],equal_var=False)
            p_values['MxL'].append(p1)
            if p1 < 0.05:
                print ('ML \t \t',region)
            _,p1 = ttest_ind(combined_values['M'],combined_values['V'],equal_var=False)
            p_values['MxV'].append(p1)
            if p1 < 0.05:
                print ('MV \t \t',region,'\t \t',p1)
            _,p1 = ttest_ind(combined_values['V'],combined_values['L'],equal_var=False)
            p_values['VxL'].append(p1)
            if p1 < 0.05:
                print ('VL \t \t',region)

            print ("COMBINED_VALUES")
            print (combined_values)
            for xind,x in enumerate(['M','V','L']):
                label,addendum,color=model_plot_dict[x]

                plt.bar(((5*ind)+xind+addendum-5.5),np.array(combined_values[x]).mean(),width=1.,color=color,edgecolor='black',alpha=0.7,linestyle='--')
                _,c,_ = plt.errorbar(1.*(5*ind+xind+addendum-5.5),np.array(combined_values[x]).mean(),yerr=scipy.stats.sem(np.array(combined_values[x])),lolims=True,color='black',capsize=0)
                for capline in c:
                    capline.set_marker('_')

                _ ,p = ttest_ind(combined_values[x],combined_values['V'],equal_var=False)

                if p < 0.05:
                    label,addendum,color=model_plot_dict[x]
                    print (region,x)

                    if xind < 1:
                        x1,x2 = ((5*ind)+xind+addendum-5.5) , ((5*ind)+xind+addendum-5.5) + 1.
                    else:
                        x1,x2 = ((5*ind)+xind+addendum-5.5) -1. , ((5*ind)+xind+addendum-5.5)


                    print (beh)
                    h = heights[beh]
                    y,col = np.array(combined_values[x]).mean()+scipy.stats.sem(np.array(combined_values[x])),'k'
                    plt.text((x1+x2)*.5,y+h,"*",ha='center',va='bottom',color=col)
                    plt.annotate('',xy=(x1,y+h-0.04),xytext=(x2,y+h-0.04),arrowprops={'connectionstyle':'bar','arrowstyle':'-','shrinkA':20,'shrinkB':20})
                    beh += 1

        plt.xticks(xticks_for_models,labels_for_models,rotation=15,fontsize=12)



        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.legend(bbox_to_anchor=(1.,1.),fontsize=12,ncol=2)

        if save_name is not None:
            plt.savefig(f"{save_name}/{data_type}_fancyfig.png",bbox_inches='tight')
        #return p_values


    def estimate_effect(type:str):
        for ind,region in enumerate(smaller_data_dict.keys()):
            upper_ceilings = upper_ceiling_dict[region]
            factors = [1.0/x for x in upper_ceilings]
            combined_values = {'M':[],'V':[],"L":[]}
            for mind,model in enumerate(model_plot_dict.keys()):
                if model in ['M','V','L']:
                    continue
                val = [corr_data_dict[model][region][s][0]*factors[s-1] for s in range(1,args.NUM_SUB+1)]
                datum = np.array(val)
                mean = datum.mean()
                # sem = scipy.stats.sem(datum)

                if model in ['clip','virtex','virtexv2','icmlm_attfc','icmlm_tfm','tsmresnet50','audioclip']:
                    combined_values['M'].append(mean)
                elif model in ['BERT','GPT2','CLIP-L']:
                    combined_values['L'].append(mean)
                else:
                    combined_values['V'].append(mean)

            #create dictionary for storing the p-values
            p_values = {'MxV':[],'MxL':[],'VxL':[]}

            _,p1 = ttest_ind(combined_values['M'],combined_values['L'],equal_var=False)
            p_values['MxL'].append(p1)
            if p1 < 0.05:
                print ('ML \t \t',region)
            _,p1 = ttest_ind(combined_values['M'],combined_values['V'],equal_var=False)
            p_values['MxV'].append(p1)
            if p1 < 0.05:
                print ('MV \t \t',region,'\t \t',p1)
            _,p1 = ttest_ind(combined_values['V'],combined_values['L'],equal_var=False)
            p_values['VxL'].append(p1)
            if p1 < 0.05:
                print ('VL \t \t',region)
            #region_pvals=p_values
            #estimate the effect of the modalities
            corr_diff={'MxV':[],'MxL':[],'VxL':[]}
            for pair in corr_diff.keys():
                diff = np.mean(combined_values[pair[0]]) - np.mean(combined_values[pair[-1]])
                corr_diff[pair].append(diff)
            #create the region key
            region=f'{interesting_regions_mapping[region]}'
            #save the region effect 
            if type=='vanilla':
                if region_effect.get(region) is None:
                    region_effect[region]={}
                region_effect[region]['vanilla']=corr_diff
                if region_pvals.get(region) is None:
                    region_pvals[region]={}
                region_pvals[region]['vanilla']=p_values
            else:
                if region_effect[region].get('shuffle') is None:
                    region_effect[region]['shuffle']=[]
                if region_pvals[region].get('shuffle') is None:
                    region_pvals[region]['shuffle']=[]
                region_effect[region]['shuffle'].append(corr_diff)
                region_pvals[region]['shuffle'].append(p_values)



    #create the betas dir
    betas_test=opj(bidsroot,'derivatives',betas_dir,betas_type)
    #extract the effect for the shuffles
    #for the vanilla data
    #betas_test=opj(betas_test,ses,'vanilla_data')
    #get the region vectors
    data_dict = {_region : {sub_idx : get_region_vectors_(sub_idx,[betas_test,ses,'vanilla_data'],_region,atlas=args.USE_ATLAS,use_train=args.USE_TRAIN_BETAS,use_test=args.USE_TEST_BETAS)  for sub_idx in range(1,args.NUM_SUB+1)} for _region in interesting_regions_mapping.keys()}
    global CORR_FUNC
    for masking_method in ['normal_correlations','select_top30_voxels']:
        args.MAIN_TASK=masking_method
        for distance_ in ['euclidean','correlation'][:]:
            args.DISTANCE=distance_
            for corr_method in ['kendalltau','pearsonr','spearmanr'][:-1]:
                CORR_FUNC=getattr(stats,corr_method)
                smaller_data_dict = select_voxels(data_dict)
                upper_ceiling_dict = get_upper_ceilings(smaller_data_dict)
                corr_data_dict = get_correlations(smaller_data_dict)

                #save the results 
                save_dir = opj(betas_test,f"masking_method-{masking_method}/dist-{distance_}/corr-{corr_method}/sca_results_run-03")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                args.SAVE_DIR = save_dir
                args.SAVE_IMAGE= opj(args.SAVE_DIR,'plot')
                if not os.path.exists(args.SAVE_IMAGE):
                    os.makedirs(args.SAVE_IMAGE)
                #crete the dictionary for storing the p-values and effect sizes
                region_pvals={}
                region_effect={}
                #plot_fancyfig(f"{args.SAVE_IMAGE}",'vanilla')
                #return
                estimate_effect(type='vanilla')
                # #save the results
                with open(f"{args.SAVE_DIR}/p_values.pkl",'wb') as f:
                    pickle.dump(region_pvals,f)
                with open(f"{args.SAVE_DIR}/region_effect.pkl",'wb') as f:
                    pickle.dump(region_effect,f)

        
    gc.collect()

def process_combination(rescale_runwise, rescale_global, hrf_method, bidsroot, betas_type):
    # Construct the betas directory path
    if betas_type=="glm_single":betas_dir=f"betas_sca_v2"
    else:betas_dir = f'betas_sca_v2/rescale_runwise-{rescale_runwise}/rescale_global-{rescale_global}/hrf-{hrf_method}'
    driver(betas_dir,
           bidsroot,
           betas_type,
           ses='ses-perceptionTest-avg')

def parallelizer():
    # Define all possible values
    rescale_runwise_options = ['psc']#['z', 'psc', 'center']#, 'off']
    rescale_global_options = ["off"]#["psc"]#['z', 'psc','re-center', 'off']
    hrf_method_options = ["mean"]#['mode','mean']

    # Generate all combinations
    combinations = list(product(rescale_runwise_options, rescale_global_options, hrf_method_options))
    combinations.append((None,None,None))
    process_combination(*combinations[-1],bidsroot,betas_type)
    Parallel(n_jobs=8)(
        delayed(process_combination)(
            rescale_runwise, 
            rescale_global, 
            hrf_method, 
            bidsroot,  # You'll need to pass this variable
            betas_type  # You'll need to pass this variable
        )
        for rescale_runwise, rescale_global, hrf_method in tqdm(combinations, 
                                                           desc="Processing combinations",
                                                           total=len(combinations))
    )



if __name__=='__main__':

    args = OmegaConf.load('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/code/config.yaml')
    assert args.USE_TRAIN_BETAS == False  #we stopped doing that analysis now


    if args.CORR_FUNC == 'pearsonr':
        CORR_FUNC = pearsonr
    if args.CORR_FUNC == 'spearmanr':
        CORR_FUNC = spearmanr    


    #if os.path.exists(args.SAVE_DIR) == False:
    #    os.mkdir(args.SAVE_DIR)

    interesting_regions_mapping = get_interesting_regions(args.USE_ATLAS)
    bidsroot='/path/to/data/directory'
    betas_type='scalematched'
    betas_type="glm_single"
    parallelizer()
