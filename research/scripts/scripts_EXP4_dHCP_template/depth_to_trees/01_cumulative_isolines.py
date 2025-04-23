import numpy as np
from research.tools import rw
import os
from app.functions import texture as stex
from app.functions import topology as slt


# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'


sub = 'CC00576XX16'
ses = '163200'

# load mesh
mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat')
mesh_path = os.path.join(mesh_folder, mesh_name)
mesh = rw.load_mesh(mesh_path)

# load mask cortex
cortex = rw.read_gii_file(os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat',
                                               'sub-' + sub + '_ses-' + ses + '_hemi-left_desc-drawem_dseg.label.gii'))
cortex = cortex.astype(bool)

# load depth map
depth_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data_EXP3/result_EXP3/dpfstar500',
                           sub + '_' + ses + '_dpfstar500.gii')

depth = rw.read_gii_file(depth_path)
#depth = depth[cortex]

# discrtesise depth map
min_depth = np.min(depth[cortex])
max_depth = np.max(depth[cortex])
depth[np.invert(cortex)] = max_depth
depth_bin = np.linspace(min_depth, max_depth, 50)
isolines = np.digitize(depth, bins=depth_bin)


# create different levels
isolines[np.invert(cortex)] = 50+10
cumulative_sulci = list()
for level in np.linspace(1,50):
    print(level)
    cml_iso = np.zeros(len(depth))
    cml_iso[np.where(isolines<=level)[0]] = 1
    cumulative_sulci.append(cml_iso)

isolines[np.invert(cortex)] = 0-10
cumulative_gyri = list()
for level in np.linspace(1,50):
    cml_iso = np.ones(len(depth))
    cml_iso[np.where(isolines<=level)[0]] = 0
    cumulative_gyri.append(cml_iso)


save_folder = os.path.join(wd, 'data_EXP4/result_EXP4',sub + '_' + ses )
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
rw.write_texture(stex.TextureND(darray=cumulative_sulci), os.path.join(save_folder,
                                                                        sub + '_' + ses + '_cumulative_sulci.gii'))
rw.write_texture(stex.TextureND(darray=cumulative_gyri), os.path.join(save_folder,
                                                                        sub + '_' + ses + '_cumulative_gyri.gii'))



