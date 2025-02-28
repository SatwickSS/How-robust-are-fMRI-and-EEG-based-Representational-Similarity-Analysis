
import os
from os.path import join as pjoin
#iterate through the directories
bidsroot='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids'
betas_type='scalematched'
for sub in range(1,6):
    sub_id=f'{sub:02d}'
    for rescale_runwise in ['z','psc','center','off']:
            for rescale_global in ['z', 'psc', 're-center', 'off']:
                for hrf_method in ['mean','mode']:
                    #for shuffle_index in range(10):  
                        betas_dir=f'betas_sca/rescale_runwise-{rescale_runwise}/rescale_global-{rescale_global}/hrf-{hrf_method}'
                        #join the paths
                        betas_dir=pjoin(bidsroot,'derivatives',betas_dir,betas_type,f'sub-{sub_id}')
                        if not os.path.exists(betas_dir):
                            #print(betas_dir)
                            os.makedirs(betas_dir)

