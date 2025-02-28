import os
from os.path import join as pjoin
import re
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataSink
from nipype import Node, Workflow, MapNode
from nipype.interfaces.freesurfer import RobustRegister, ApplyVolTransform, Tkregister2
from nipype.pipeline.plugins import MultiProcPlugin
from tqdm import tqdm
from glob import glob

def grab_betas(
        sub:str,
        bidsroot: str = '/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
        betas_derivname:str = 'betas_run-01/scalematched'):
    import os
    from os.path import join as pjoin
    from glm import GODGLM
    glm_obj = GODGLM(bidsroot, sub, out_deriv_name='tmp')
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname, f'sub-{sub}')
    betafiles = [
        pjoin(betasdir, f'ses-{glm_obj.target_session}{ses_i+1:02d}', f'sub-{sub}_ses-{glm_obj.target_session}{ses_i+1:02d}_run-{run_i+1:02d}_betas.nii.gz')
        for ses_i in range(glm_obj.n_sessions) for run_i in range(glm_obj.nruns_perses_[glm_obj.ds.target_sessions[ses_i]])
    ]
    for b in betafiles:
        assert os.path.exists(b)
    return betafiles

def grab_betasv2(
        sub:str,
        bidsroot: str = '/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
        betas_derivname:str = 'betas_run-01/scalematched',
        ses_dir='ses-perceptionTest-avg'):
    import os
    from os.path import join as pjoin
    from glm import GODGLM
    from glob import glob
    glm_obj = GODGLM(bidsroot, sub, out_deriv_name='tmp')
    betasdir = pjoin(bidsroot, 'derivatives', betas_derivname, f'sub-{sub}',ses_dir)
    #extract the files using the glob function
    betafiles = glob(pjoin(betasdir,'*','*.nii.gz'))
    for b in betafiles:
        assert os.path.exists(b)
    return betafiles
def create_target_file_name(
        infiles:list,
        template: str='MNI305'):
    import re
    #infiles=grab_betas(sub=sub,bidsroot=bidsroot,betas_derivname=betas_derivname)
    outfiles=[re.sub(r'(.+)\.nii\.gz', fr'\1{template}.nii.gz', filename) for filename in infiles]
    return outfiles

                            

def transform(
        sub:str,
        bidsroot:str = '/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids',
        reconall_dir:str ='fmriprep_run-04/sourcedata/freesurfer/',
        betas_derivname:str = 'betas_run-01/scalematched',
        template='MNI305'):


    #define the os environment paths
    freesurfer_home = os.environ['FREESURFER_HOME']
    fsl_home = os.environ['FSLDIR']

    #create the path for the reconall directory
    reconall_dir = pjoin(bidsroot, 'derivatives', reconall_dir, f'sub-{sub}','mri')

    #create the node for the datagrabber
    datagrabber=Node(Function(function=grab_betasv2,input_names=['sub','betas_derivname'],output_names=['betafiles']),name='datagrabber')
    datagrabber.inputs.sub=sub
    datagrabber.inputs.betas_derivname=betas_derivname

    #create the node for the target file name
    target_file_name=Node(Function(function=create_target_file_name,input_names=['infiles'],output_names=['outfiles']),name='transformed_file_name')
    target_file_name.inputs.template=template

    #extract the brain image
    brain_image = pjoin(reconall_dir,'brain.mgz')
    
    #create node for registation from subject T1w to mni305
    #extract the mni305 image
    if template=='MNI305':
        target_image = pjoin(freesurfer_home, 'subjects', 'fsaverage', 'mri', 'brain.mgz')
    elif template=='MNI152':
        target_image = pjoin(fsl_home, 'data','standard', 'MNI152_T1_2mm.nii.gz')
    #create the node for the robust register
    robust_register = Node(RobustRegister(),name='robust_register')
    robust_register.inputs.target_file = target_image
    robust_register.inputs.source_file = brain_image
    robust_register.inputs.auto_sens = True

    #create the node for converting from lta file to fsl matrix file
    tkregister=Node(Tkregister2(),name='tkregister')
    tkregister.inputs.moving_image = brain_image
    tkregister.inputs.target_image = target_image



    #create the node for volume transformation
    apply_vol_transform = MapNode(ApplyVolTransform(),
                                  iterfield=['source_file','transformed_file'],
                                  name='apply_vol_transform') 
    apply_vol_transform.inputs.target_file = target_image
    #connect the nodes
    wf = Workflow(name='wf')
    #connect all the nodes to applyvoltransform
    wf.connect(robust_register,'out_reg_file',tkregister,'lta_in')
    wf.connect(datagrabber,'betafiles',target_file_name,'infiles')
    wf.connect(datagrabber,'betafiles',apply_vol_transform,'source_file')
    wf.connect(target_file_name,'outfiles',apply_vol_transform,'transformed_file')
    wf.connect(tkregister,'reg_file',apply_vol_transform,'reg_file')

    ##create the datasink node
    #datasink = Node(DataSink(),name='datasink')
    #datasink.inputs.base_directory = pjoin(bidsroot,'derivatives',betas_derivname,f'{transformed_space}',f'sub-{sub}')
    #wf.connect(apply_vol_transform,'transformed_file',datasink,'transformed_files')
    
    
    
    #visualize the workflow
    wf.base_dir='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/code/python/workflows'
    #wf.write_graph(graph2use='flat')
    wf.run()

        
        
if __name__=='__main__':
    #specify the betas type
    betas_type='scalematched'
    for sub in range(1,3):
        sub_id=f'{sub:02d}'
        for rescale_runwise in ['z','psc','center','off']:
                for rescale_global in ['z', 'psc', 're-center', 'off']:
                    for hrf_method in ['mean','mode']:
                        #for shuffle_index in range(10):  
                            betas_dir=f'betas_sca/rescale_runwise-{rescale_runwise}/rescale_global-{rescale_global}/hrf-{hrf_method}/{betas_type}'
                            transform(sub=sub_id,template='MNI305',betas_derivname=betas_dir)