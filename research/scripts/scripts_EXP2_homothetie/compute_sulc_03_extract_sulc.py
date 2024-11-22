
import numpy as np
import nibabel as nib
import os

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
subjects=['CC00735XX18_222201']
scales=['32500', '65000','130000', '260000', '520000', '1040000', '1755000', '4160000', '8125000']


for sub in subjects:
    print(sub)
    for scale in scales:
        print(scale)
        path_in_gifti = os.path.join(wd, 'data/scaled_meshes/sulc', sub + '_' + scale + '_sulcal_depth.gii')
        path_out_gifti = os.path.join(wd, 'data/scaled_meshes/sulc', sub + '_' + scale + '_sulc.gii')
        input_gifti = nib.load(path_in_gifti)
        output_gifti = nib.GiftiImage()
        # create a new gifti file containing only the sulc values
        output_gifti.add_gifti_data_array(input_gifti.darrays[-1])
        # inverse the values to have positive ones
        output_gifti.darrays[-1].data = -output_gifti.darrays[-1].data
        # save the sulc texture file into gifti (equivalent gifti shape format)
        nib.save(output_gifti, path_out_gifti)
