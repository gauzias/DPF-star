
import numpy as np
import os
import slam.io as sio
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def objective(x, a, b):
    return a * x + b

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'


#scales=[65000, 130000, 260000, 520000, 1040000, 1755000, 4160000, 8125000]
volclass = [32500, 65000, 130000, 260000, 520000, 1040000, 1755000, 4160000, 8125000]
volclass = [ 65000, 520000, 1755000, 4160000, 8125000]

#su = 'CC00777XX19'
#se = '239102'
#su = 'KKI2009-113'
#se = 'MR1'

sub = 'CC00735XX18'
ses = '222201'

#sub = 'CC00777XX19'
#ses = '239102'

# load sulc ref
sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_sulc.shape.gii'
sulc_path = os.path.join(wd, 'data/subject_analysis', sub + '_' + ses,'surface_processing/sulc', sulc_name)

sulc_path = os.path.join(wd, 'data/scaled_meshes/dpfstar100', sub + '_' + ses + '_' + str(65000) + '_dpfstar100.gii')
sulc_ref = sio.load_texture(sulc_path).darray[0]

# load mesh
mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
mesh_path = os.path.join(wd, 'data/meshes',   mesh_name)
mesh = sio.load_mesh(mesh_path)
vol_ref = mesh.convex_hull.volume

EQM = list()
fig,ax = plt.subplots()
list_afit = list()
for vol in volclass:
    sulc = sio.load_texture(os.path.join(wd, 'data/scaled_meshes/sulc', sub + '_' + ses  + '_' + str(vol) + '_sulc.gii')).darray[0]
    sulc = sio.load_texture(
        os.path.join(wd, 'data/scaled_meshes/dpfstar100', sub + '_' + ses + '_' + str(vol) + '_dpfstar100.gii')).darray[0]
    coef = np.power(vol/vol_ref, 1/3)
    popt, _ = curve_fit(objective, sulc_ref, sulc)
    a, b = popt
    list_afit.append(a)
    print('y = %.5f * x + %.5f' % (a, b))
    x_line = np.arange(min(sulc_ref), max(sulc_ref), 1)
    y_line = objective(x_line, a, b)
    err = np.sum(np.square(a*sulc_ref+b - sulc))/len(sulc)
    EQM.append(err)
    ax.plot(x_line, y_line, label=str(np.round(coef)))
    ax.scatter(sulc_ref*10000,sulc*10000, s = 0.2)
ax.set_xlabel('SULC on initial mesh')
ax.set_ylabel('SULC on scaled meshes')
#plt.legend()
plt.grid()

coefs = [np.power(sc/vol_ref , 1/3) for sc in volclass]


fig2, ax2 = plt.subplots()
ax2.plot(coefs, list_afit)
ax2.plot(coefs, coefs)
ax2.scatter(coefs, list_afit)

fig3, ax3 = plt.subplots()
ax3.plot(np.power(volclass/vol_ref,1/3), EQM)


plt.show()

