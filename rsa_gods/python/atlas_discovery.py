
import os,re
from os.path import join as pjoin
from nipype import Node, Workflow
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function

def extract_roi_indices(txt_file,outdir=None):
    indexes=[]
    region_names=[]
    hemisphere_dict={'lh':'Left','rh':'right'}
    hemisphere=[]
    with open(txt_file,'r') as f:
        #iterate through each line
        for line in f:
            #extract lines having desired string
            if re.match(r'^\d+.*', line):
                #extract the region and the index 
                match = re.match(r'^(\d+)\s+([\w\.-]+)', line)
                if match:
                    index = match.group(1)
                    region_name = match.group(2)
                    #check if region name contains lh or rh
                    hemisphere_name1=re.search(r'(lh|rh)([-,_])', region_name)
                    if hemisphere_name1:
                        hemisphere_name=hemisphere_dict[hemisphere_name1.groups()[0]]
                        region_name = re.sub(r'(lh|rh)[-,_]','',region_name)
                    #check if region name contains left or right
                    hemisphere_name2=re.search(r'(Left|Right)', region_name)
                    if hemisphere_name2:
                        hemisphere_name=hemisphere_name2.group(1)
                        region_name = re.sub(r'(Left|Right)[-,_]','',region_name)
                        region_name = re.sub(r'[-,_](Left|Right)$','',region_name)

                    if (not hemisphere_name1) and (not hemisphere_name2):hemisphere_name=None
                    #hemisphere_name=re.search(r'(lh|rh)', region_name)
                    #region_name = re.sub(r'^\w+-(lh|rh)-','',region_name)
                    indexes.append(index)
                    region_names.append(region_name)
                    hemisphere.append(hemisphere_name)

    #create a pandas dataframe and save it as a csv file
    import pandas as pd
    df=pd.DataFrame({'region_index':indexes,'region_name':region_names,'hemisphere':hemisphere})
    if outdir:
        df.to_csv(f'{outdir}desikan_atlas.csv',index=False,sep='\t')



    

def mgz_to_nii(mgz_file, nii_file,out_dir):
    # Create a node to execute mri_convert
    mri_convert = Node(MRIConvert(), name='mri_convert')
    mri_convert.inputs.in_file = mgz_file
    mri_convert.inputs.out_file = nii_file
    mri_convert.inputs.out_type = 'niigz'


    #creating the datasink node
    datasink=Node(DataSink(),name='datasink')
    datasink.inputs.base_directory=out_dir


    # Create a workflow
    wf = Workflow(name='mri_convert_workflow')
    wf.base_dir='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/code/python/workflows'

    # Add the mri_convert node to the workflow
    wf.connect(mri_convert, 'out_file', datasink, '@out_file')

    # Run the workflow
    #wf.write_graph(graph2use='flat')
    wf.run()

if __name__=='__main__':
    #specify the outdir
    out_dir='/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids/derivatives/atlas/'
    #call the function

    extract_roi_indices('/DATA1/satwick22/Documents/fMRI/multimodal_concepts/generic_object_decoding_bids/derivatives/atlas/FreeSurferColorLUT.txt',outdir=out_dir)