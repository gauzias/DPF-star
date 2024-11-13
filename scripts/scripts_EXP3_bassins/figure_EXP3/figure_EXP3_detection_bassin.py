import os
import pandas as pd
import numpy as np
import slam.io as sio

import tools.voronoi as tv
import matplotlib.pyplot as plt
import slam.texture as stex
import slam.curvature as scurv
import tools.depth as depth
import os
import slam.texture as stex
import seaborn as sns

# load sulc

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
dir_sulcKKI = '/home/maxime/callisto/data/outputs_mirtkdeformmesh_KKIFS6'
dir_dHCP = '/media/maxime/DATA/rel3_dhcp_anat_pipeline'
dir_KKI = '/home/maxime/callisto/data/databases_copied_from_hpc/REPRO_database/FS_database_KKI_test_retest_FS6.0'

dir_voronoi = '/data/rel3/voronoi'
dir_dpf100 = '/data/rel3/dpf100'

sub_KKI = list()
ses_KKI = list()
sub_dHCP = list()
ses_dHCP = list()

for file in os.listdir(dir_voronoi):
    if file.endswith('_MR1_voronoi.gii'):
        subK = file.split('_')[0] + '_' + file.split('_')[1]
        sesK = 'MR1'
        sub_KKI.append(subK)
        ses_KKI.append(sesK)
    else:
        subH = file.split('_')[0]
        sesH = file.split('_')[1]
        sub_dHCP.append(subH)
        ses_dHCP.append(sesH)

subjects = np.hstack([sub_dHCP, sub_KKI])
sessions = np.hstack([ses_dHCP, ses_KKI])

dataset = np.hstack([np.repeat('dHCP', len(sub_dHCP)), np.repeat('KKI', len(sub_KKI))])




## define threshold
value_dpf100 = np.array([])
value_sulc = np.array([])

GI_list = np.array([])
surface_list = np.array([])

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    if dset == 'dHCP':
        # load mesh
        mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
        mesh_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   mesh_name)
        mesh = sio.load_mesh(mesh_path)
        surface = mesh.area
        volume = mesh.volume

        # compute GI
        GI = mesh.area/mesh.convex_hull.area
        GI_list = np.hstack([GI_list, GI])
        surface_list = np.hstack([surface_list, surface])

        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
        sulc = sio.load_texture(sulc_path).darray[0]

        sulc = sulc / np.power(surface, 1 / 2)

        value_sulc = np.hstack([value_sulc, sulc])
    else:
        # load mesh
        mesh_path = os.path.join(dir_KKI, sub + '_' + ses, 'surf/lh.white.gii')
        mesh = sio.load_mesh(mesh_path)
        surface = mesh.area
        volume = mesh.volume

        # compute GI
        GI =  mesh.area/mesh.convex_hull.area
        GI_list = np.hstack([GI_list, GI])
        surface_list = np.hstack([surface_list, surface])
        # load sulc KKI
        sulc_name = sub + '_' + ses + '_l_sulc.gii'
        sulc = sio.load_texture(os.path.join(dir_sulcKKI, sulc_name)).darray[0]

        sulc = sulc / np.power(surface, 1 / 2)

        value_sulc = np.hstack([value_sulc, sulc])

    # load dpf100
    dpf100_path = os.path.join(dir_dpf100, sub + '_' + ses + '_dpf100.gii')
    dpf100 = sio.load_texture(dpf100_path).darray[0]
    value_dpf100 = np.hstack([value_dpf100, dpf100])



nbin=9
Th1_list = [ np.percentile(value_dpf100,pct)  for pct in np.linspace(10,90, 9)]
Th2_list = [ np.percentile(value_sulc,pct)  for pct in np.linspace(10,90, 9)]



#Th1_list = np.linspace(np.min(value_dpf100), np.max(value_dpf100), nbin)
#Th2_list = np.linspace(np.min(value_sulc), np.max(value_dpf100), nbin )



df1 = pd.DataFrame(dict( sub=[], surface=[], surface_rounded=[], Th=[],percentil = [], value=[]))
df2 = pd.DataFrame(dict( sub=[], surface=[], surface_rounded=[],Th=[], percentil = [], value=[]))
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    if dset == 'dHCP':
        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
        sulc = sio.load_texture(sulc_path).darray[0]



    else :
        # load sulc KKI
        sulc_name = sub + '_' + ses + '_l_sulc.gii'
        sulc = sio.load_texture(os.path.join(dir_sulcKKI, sulc_name)).darray[0]



    # load dpf100
    dpf100_path = os.path.join(dir_dpf100, sub + '_' + ses + '_dpf100.gii')
    dpf100 = sio.load_texture(dpf100_path).darray[0]

    # load voronoi
    voronoi_path = os.path.join(dir_voronoi, sub + '_' + ses + '_voronoi.gii')
    voronoi = sio.load_texture(voronoi_path).darray[0]

    surface =  np.sum(voronoi)
    surface_rounded = round(surface / 5000) * 5000

    list_metric_sulc = list()
    list_metric_dpf100 = list()

    sulc = sulc / np.power(surface, 1 / 2)


    for pidx, pct in enumerate(np.linspace(10,90, nbin)):
        Th1 = Th1_list[pidx]
        Th2 = Th2_list[pidx]
        # bassin detection
        bassin_dpf100 = dpf100 <= Th1
        bassin_sulc = sulc <= Th2
        # metric
        metric_dpf100 = np.sum(voronoi[bassin_dpf100]) / np.sum(voronoi)
        metric_sulc = np.sum(voronoi[bassin_sulc]) / np.sum(voronoi)

        list_metric_dpf100.append(metric_dpf100)
        list_metric_sulc.append(metric_sulc)

    df1ij = pd.DataFrame(dict(sub=np.repeat(sub, len(Th1_list)), surface=np.repeat(surface, len(Th1_list)),
                              surface_rounded=np.repeat(surface_rounded, len(Th1_list)),
                              Th=Th1_list, percentil = np.linspace(10, 90, nbin),  value=list_metric_dpf100))

    df1 = pd.concat([df1, df1ij], ignore_index=True)

    df2ij = pd.DataFrame(dict(sub=np.repeat(sub, len(Th2_list)), surface=np.repeat(surface, len(Th2_list)),
                              surface_rounded=np.repeat(surface_rounded, len(Th2_list)),
                              Th=Th2_list,percentil = np.linspace(10, 90, nbin), value=list_metric_sulc))

    df2 = pd.concat([df2, df2ij], ignore_index=True)

    # ax1.scatter(list_surface, list_metric_dpf100, label='dpf100_' + str(Th1))
    # ax2.scatter(list_surface, list_metric_sulc, label='sulc_' + str(Th2))



## figure 1
fig1, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [5,1]})

ax1 = axs[0]
ax11  = axs[1]

f1 = sns.lineplot(data=df1, x="surface_rounded", y="value", hue='percentil', ax=ax1, legend = False)
f2 = sns.lineplot(data=df2, x="surface_rounded", y="value", hue='percentil', ax=ax1, legend = 'full',
                  linestyle='--')

g1 = sns.scatterplot(data=df1, x="surface", y="value", hue='percentil', ax=ax1, legend = False, )
g2 = sns.scatterplot(data=df2, x="surface", y="value", hue='percentil', ax=ax1, legend = False,
                  marker='X')

f1.set(xlabel="surface area mm2", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded DPF-star ')
f2.set(xlabel="surface area mm2", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded Depth ')


ax1.yaxis.grid()
ax1.legend()
leg = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg.set_title('percentil of \n Depth values')


ax2 = ax1.twinx()
styles = ['-', '--']
lab = ['DPF-star', 'SULC']
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss],
             label=lab[ss], c='black')
ax2.get_yaxis().set_visible(False)

ax2.legend()
leg2 = ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
leg2.set_title('method')

plt.tight_layout()



### fig 2
fig2, axx = plt.subplots()
h2 = sns.scatterplot(x = surface_list, y = GI_list, ax = ax11)


plt.tight_layout()
plt.show()

