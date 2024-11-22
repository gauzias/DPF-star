import numpy as np
import nibabel as nib
import os
import nibabel.freesurfer as nibf
import slam.io as sio
import slam.texture as stex

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
subjects = ['KKI2009_800', 'KKI2009_239',
       'KKI2009_505', 'KKI2009_679', 'KKI2009_934', 'KKI2009_113',
       'KKI2009_422', 'KKI2009_815', 'KKI2009_906', 'KKI2009_127',
       'KKI2009_742', 'KKI2009_849', 'KKI2009_913', 'KKI2009_346',
       'KKI2009_502', 'KKI2009_814', 'KKI2009_916', 'KKI2009_959',
       'KKI2009_142', 'KKI2009_656']


for idx, sub in enumerate(subjects):
    # load mesh
    mesh_name = 'lh.white.gii'
    mesh_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                             sub + '_' + 'MR1', 'surf', mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # load label
    label_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                              sub + '_MR1','label', 'lh.cortex.label')

    label, scalar = nibf.io.read_label(label_path,read_scalars = True)

    # create texture
    interH = np.ones(len(mesh.vertices))
    interH[label] = 0
    sio.write_texture(stex.TextureND(darray=interH), os.path.join(wd,'/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                              sub + '_MR1','label', 'mask_interH.gii'))




