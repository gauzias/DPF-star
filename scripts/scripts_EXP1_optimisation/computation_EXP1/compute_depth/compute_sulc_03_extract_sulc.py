
import numpy as np
import nibabel as nib
import os

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
subjects=['KKI2009_113_MR1',
          'KKI2009_505_MR1',
          'KKI2009_142_MR1',]


for sub in subjects:
    print(sub)
    path_in_gifti = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub,  'sulc', sub + '_sulcal_depth.gii')
    path_out_gifti = os.path.join(wd, 'data_EXP1/result_EXP1/depth', sub,  'sulc', sub + '_sulc.gii')
    input_gifti = nib.load(path_in_gifti)
    output_gifti = nib.GiftiImage()
    # create a new gifti file containing only the sulc values
    output_gifti.add_gifti_data_array(input_gifti.darrays[-1])
    # inverse the values to have positive ones
    output_gifti.darrays[-1].data = -output_gifti.darrays[-1].data
    # save the sulc texture file into gifti (equivalent gifti shape format)
    nib.save(output_gifti, path_out_gifti)
