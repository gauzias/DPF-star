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
import scipy.stats as ss
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


subjects_add = ["CC00735XX18","CC00672AN13", "CC00672BN13", "CC00621XX11", "CC00829XX21",
            "CC00617XX15", "CC00385XX15"]

sessions_add = ["222201", "197601" , "200000", "177900", "17610",  "176500" ,"118500" ]


subjects = np.hstack([sub_dHCP, sub_KKI, subjects_add])
sessions = np.hstack([ses_dHCP, ses_KKI, sessions_add])

dataset = np.hstack([np.repeat('dHCP', len(sub_dHCP)), np.repeat('KKI', len(sub_KKI)),
                     np.repeat('dHCP', len(subjects_add))])

addi = np.hstack([np.repeat(0, len(sub_dHCP)+len(sub_KKI)), np.repeat(1,len(subjects_add))])


## define threshold
value_dpf100 = np.array([])
value_sulc = np.array([])
value_sulc_Z = np.array([])
value_sulc_N = np.array([])

GI_list = np.array([])
surface_list = np.array([])
mesh_error = []
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    add = addi[idx]
    if (dset == 'dHCP') & (add == 0):
        print('bonjour')
        try :
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
            sulc_N = sulc / np.power(surface, 1 / 2)
            #sulc_N = sulc / np.power(volume, 1 / 3)
            sulc_Z = ss.zscore(sulc)
            value_sulc = np.hstack([value_sulc, sulc])
            value_sulc_N = np.hstack([value_sulc_N, sulc_N])
            value_sulc_Z = np.hstack([value_sulc_Z, sulc_Z])
        except:
            mesh_error.append(idx)

    if (dset== 'dHCP') & (add==1):
        mesh_path = os.path.join('/media/maxime/DATA/dHCP_anat_pipeline', 'sub-'+sub, 'ses-'+ses,
                                     'anat', 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii')
        mesh = sio.load_mesh(mesh_path)
        surface = mesh.area
        volume = mesh.volume
        # compute GI
        GI = mesh.area / mesh.convex_hull.area
        GI_list = np.hstack([GI_list, GI])
        surface_list = np.hstack([surface_list, surface])

        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_sulc.shape.gii'
        sulc_path = os.path.join('/media/maxime/DATA/dHCP_anat_pipeline', 'sub-' + sub, 'ses-' + ses, 'anat',
                                 sulc_name)
        sulc = sio.load_texture(sulc_path).darray[0]
        sulc_N = sulc / np.power(surface, 1 / 2)
        # sulc_N = sulc / np.power(volume, 1 / 3)
        sulc_Z = ss.zscore(sulc)
        value_sulc = np.hstack([value_sulc, sulc])
        value_sulc_N = np.hstack([value_sulc_N, sulc_N])
        value_sulc_Z = np.hstack([value_sulc_Z, sulc_Z])
    if (dset== 'KKI') :
        print('bizar')
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
        sulc_N = sulc / np.power(surface, 1 / 2)
        #sulc_N = sulc / np.power(volume, 1 / 3)
        sulc_Z = ss.zscore(sulc)
        value_sulc = np.hstack([value_sulc, sulc])
        value_sulc_N = np.hstack([value_sulc_N, sulc_N])
        value_sulc_Z = np.hstack([value_sulc_Z, sulc_Z])

    # load dpf100
    dpf100_path = os.path.join(dir_dpf100, sub + '_' + ses + '_dpf100.gii')
    dpf100 = sio.load_texture(dpf100_path).darray[0]
    value_dpf100 = np.hstack([value_dpf100, dpf100])



nbin=9
#linp = np.linspace(10,90, nbin)
linp = [50]
#Th1_list = [ np.percentile(value_dpf100,pct)  for pct in np.linspace(10,90, 9)]
#Th2_list = [ np.percentile(value_sulc,pct)  for pct in np.linspace(10,90, 9)]

#Th1_list = np.linspace(np.min(value_dpf100), np.max(value_dpf100), nbin)
#Th2_list = np.linspace(np.min(value_sulc), np.max(value_dpf100), nbin )

Th1_list = [np.median(value_dpf100)]
Th2_list = [np.median(value_sulc)]
ThN_list = [np.median(value_sulc_N)]
ThZ_list = [np.median(value_sulc_Z)]


Th1_list = [np.percentile(value_dpf100,45)]
Th2_list = [np.percentile(value_sulc,45)]
ThN_list = [np.percentile(value_sulc_N,45)]
ThZ_list = [np.percentile(value_sulc_Z,45)]


print(Th1_list)
print(Th2_list)
print(ThN_list)
print(ThZ_list)


df1 = pd.DataFrame(dict( sub=[], surface=[], surface_rounded=[], Th=[],percentil = [], value=[]))
df2 = pd.DataFrame(dict( sub=[], surface=[], surface_rounded=[],Th=[], percentil = [], value=[]))
dfN = pd.DataFrame(dict( sub=[], surface=[], surface_rounded=[],Th=[], percentil = [], value=[]))
dfZ = pd.DataFrame(dict( sub=[], surface=[], surface_rounded=[],Th=[], percentil = [], value=[]))


subjects2 = np.delete(subjects, mesh_error)
sessions2 = np.delete(sessions, mesh_error)
dataset2 = np.delete(dataset, mesh_error)
addi2 = np.delete(addi, mesh_error)

for idx, sub in enumerate(subjects2):
    print(sub)
    ses = sessions2[idx]
    dset = dataset2[idx]
    add = addi2[idx]
    if (dset == 'dHCP') & (add==0):
        print('dhcp')
        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
        sulc = sio.load_texture(sulc_path).darray[0]
        sulc_N = sulc / np.power(surface, 1/2)
        #sulc_N = sulc / np.power(volume, 1 / 3)
        sulc_Z = ss.zscore(sulc)
    if (dset == 'dHCP') & (add ==1):
        print('dhcp plus')
        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_sulc.shape.gii'
        sulc_path = os.path.join('/media/maxime/DATA/dHCP_anat_pipeline', 'sub-' + sub, 'ses-' + ses, 'anat',
                                 sulc_name)
        sulc = sio.load_texture(sulc_path).darray[0]
        sulc_N = sulc / np.power(surface, 1/2)
        #sulc_N = sulc / np.power(volume, 1 / 3)
        sulc_Z = ss.zscore(sulc)
    if (dset == 'KKI') :
        # load sulc KKI
        sulc_name = sub + '_' + ses + '_l_sulc.gii'
        sulc = sio.load_texture(os.path.join(dir_sulcKKI, sulc_name)).darray[0]
        sulc_N = sulc / np.power(surface, 1/2)
        #sulc_N = sulc / np.power(volume, 1/3)
        sulc_Z = ss.zscore(sulc)

    # load dpf100
    dpf100_path = os.path.join(dir_dpf100, sub + '_' + ses + '_dpf100.gii')
    dpf100 = sio.load_texture(dpf100_path).darray[0]

    # load voronoi
    voronoi_path = os.path.join(dir_voronoi, sub + '_' + ses + '_voronoi.gii')
    voronoi = sio.load_texture(voronoi_path).darray[0]

    surface = np.sum(voronoi)
    surface_rounded = round(surface / 8000) * 8000

    list_metric_sulc = list()
    list_metric_sulcN = list()
    list_metric_sulcZ = list()
    list_metric_dpf100 = list()


    for pidx, pct in enumerate(linp):
        Th1 = Th1_list[pidx]
        Th2 = Th2_list[pidx]
        ThN = ThN_list[pidx]
        ThZ = ThZ_list[pidx]
        # bassin detection
        bassin_dpf100 = dpf100 <= Th1
        bassin_sulc = sulc <= Th2
        bassin_sulcZ = sulc_Z <= ThZ
        bassin_sulcN = sulc_N <= ThN
        # metric
        metric_dpf100 = np.sum(voronoi[bassin_dpf100]) / np.sum(voronoi)
        metric_sulc = np.sum(voronoi[bassin_sulc]) / np.sum(voronoi)
        metric_sulcN = np.sum(voronoi[bassin_sulcN]) / np.sum(voronoi)
        metric_sulcZ = np.sum(voronoi[bassin_sulcZ]) / np.sum(voronoi)

        list_metric_dpf100.append(metric_dpf100)
        list_metric_sulc.append(metric_sulc)
        list_metric_sulcZ.append(metric_sulcZ)
        list_metric_sulcN.append(metric_sulcN)

    df1ij = pd.DataFrame(dict(sub=np.repeat(sub, len(Th1_list)), surface=np.repeat(surface, len(Th1_list)),
                              surface_rounded=np.repeat(surface_rounded, len(Th1_list)),
                              Th=Th1_list, percentil = linp,  value=list_metric_dpf100))

    df1 = pd.concat([df1, df1ij], ignore_index=True)

    dfZij = pd.DataFrame(dict(sub=np.repeat(sub, len(ThZ_list)), surface=np.repeat(surface, len(ThZ_list)),
                              surface_rounded=np.repeat(surface_rounded, len(ThZ_list)),
                              Th=ThZ_list,percentil = linp, value=list_metric_sulcZ))

    dfZ = pd.concat([dfZ, dfZij], ignore_index=True)

    dfNij = pd.DataFrame(dict(sub=np.repeat(sub, len(ThN_list)), surface=np.repeat(surface, len(ThN_list)),
                              surface_rounded=np.repeat(surface_rounded, len(ThN_list)),
                              Th=ThN_list,percentil = linp, value=list_metric_sulcN))

    dfN = pd.concat([dfN, dfNij], ignore_index=True)

    df2ij = pd.DataFrame(dict(sub=np.repeat(sub, len(Th2_list)), surface=np.repeat(surface, len(Th2_list)),
                              surface_rounded=np.repeat(surface_rounded, len(Th2_list)),
                              Th=Th2_list,percentil = linp, value=list_metric_sulc))

    df2 = pd.concat([df2, df2ij], ignore_index=True)

    # ax1.scatter(list_surface, list_metric_dpf100, label='dpf100_' + str(Th1))
    # ax2.scatter(list_surface, list_metric_sulc, label='sulc_' + str(Th2))



## figure 1
fig1, axs = plt.subplots()



f1 = sns.lineplot(data=df1, x="surface_rounded", y="value" ,label = 'DPF-star alpha 100')
f2 = sns.lineplot(data=df2, x="surface_rounded", y="value", label = 'SULC',
                  linestyle='--')
fZ = sns.lineplot(data=dfZ, x="surface_rounded", y="value",  label = 'SULC Zscore normalisation',
                  linestyle='--')
#fN = sns.lineplot(data=dfN, x="surface_rounded", y="value",  label = 'SULC size normalisation',
#                  linestyle='--')

g1 = sns.scatterplot(data=df1, x="surface", y="value",  legend = False, )
g2 = sns.scatterplot(data=df2, x="surface", y="value",  legend = False,
               marker='X')
g3 = sns.scatterplot(data=dfZ, x="surface", y="value",  legend = False,
               marker='X')

f1.set(xlabel="surface area mm2", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded DPF-star ')
f2.set(xlabel="surface area mm2", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded Depth ')

axs.legend()
plt.grid()
plt.tight_layout()



plt.show()

