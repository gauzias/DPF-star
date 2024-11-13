
import numpy as np
import os
import slam.io as sio
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import slam.texture as stex


def objective(x, a, b):
    return a * x + b

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
subjects=['CC00777XX19_239102', 'KKI2009-113_MR1']

scales=[65000, 130000, 260000, 520000, 1040000, 1755000, 4160000, 8125000]

sub= subjects[0]

#su = 'CC00777XX19'
#se = '239102'
#su = 'KKI2009-113'
#se = 'MR1'

sub = 'CC00735XX18'
ses = '222201'

# load sulc ref
sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_sulc.shape.gii'
sulc_path = os.path.join(wd, 'data/subject_analysis', sub + '_' + ses,'surface_processing/sulc', sulc_name)
sulc_ref = sio.load_texture(sulc_path).darray[0]

# load mesh
mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
mesh_path = os.path.join(wd, 'data/meshes',   mesh_name)
mesh = sio.load_mesh(mesh_path)
vol_ref = mesh.convex_hull.volume


#fig,ax = plt.subplots()
list_afit = list()
for scale in scales:
    sulc = sio.load_texture(os.path.join(wd, 'data/scaled_meshes/sulc',
                                         sub + '_' + ses + '_' + str(scale) + '_sulc.gii')).darray[0]
    coef = np.power(scale/vol_ref, 1/3)
    popt, _ = curve_fit(objective, sulc_ref, sulc)
    a, b = popt
    print(a)
    list_afit.append(a)
    sulc = (sulc-b)/a
    diff = sulc_ref - sulc
    err = np.square(diff)
    sio.write_texture(stex.TextureND(darray=diff), os.path.join(wd, 'data/scaled_meshes/perturbation',sub + '_' + ses,
                                                                sub +'_ref'+'_diff_' + str(scale) + '.gii'))


