import slam.io as sio
import slam.texture as stex
import os
import scipy.stats as ss

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'
folder_KKI = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0'
folder_KKI_sulc = '/media/maxime/Expansion/outputs_mirtkdeformmesh_KKIFS6'
# dpfstar and voronoi
folder_voronoi = os.path.join(wd, 'data/EXP_3_bassins/voronoi_textures')
folder_dpf100 = os.path.join(wd, 'data/EXP_3_bassins/dpfstar100')
# info dataset
KKI_info_path = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0/KKI_info.csv'
dHCP_rel3_info_path = '/media/maxime/Expansion/rel3_dHCP/all_seesion_info_order.csv'

#sub = 'CC00154XX06'
#ses = '50700'

sub = 'KKI2009_113'
ses = 'MR1'

#sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
#sulc_path = os.path.join(folder_dHCP , 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)

sulc_name = sub + '_' + ses + '_l_sulc.gii'
sulc_path = os.path.join(folder_KKI_sulc, sulc_name)


sulc = sio.load_texture(sulc_path).darray[0]
sulc_Z = ss.zscore(sulc)

#sio.write_texture(stex.TextureND(darray=sulc_Z),
#                  os.path.join(folder_dHCP , 'sub-' + sub, 'ses-' + ses, 'anat', sub + '_' + ses + '_sulc_Z.gii'))


sio.write_texture(stex.TextureND(darray=sulc_Z),
                  os.path.join(folder_KKI, sub + '_' + ses, 'surf' , sub + '_' + ses + '_l_sulc_Z.gii'))

