import pandas as pd
import numpy as np
import slam.io as sio
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as ss
import pickle

subjects = ['sub-CC00530XX11', 'sub-CC00530XX11', 'sub-CC00618XX16',
       'sub-CC00629XX19', 'sub-CC00634AN16', 'sub-CC00666XX15',
       'sub-CC00672BN13', 'sub-CC00694XX19', 'sub-CC00718XX17',
       'sub-CC00796XX22', 'sub-CC00136AN13', 'sub-CC00227XX13',
       'sub-CC00389XX19', 'sub-CC00518XX15', 'sub-CC00526XX15',
       'sub-CC00576XX16', 'sub-CC00621XX11', 'sub-CC00657XX14',
       'sub-CC00661XX10', 'sub-CC00728AN19', 'sub-CC00070XX05',
       'sub-CC00072XX07', 'sub-CC00136BN13', 'sub-CC00140XX09',
       'sub-CC00152AN04', 'sub-CC00154XX06', 'sub-CC00169XX13',
       'sub-CC00177XX13', 'sub-CC00248XX18', 'sub-CC00281BN10',
       'sub-CC00053XX04', 'sub-CC00059XX10', 'sub-CC00060XX03',
       'sub-CC00063AN06', 'sub-CC00063BN06', 'sub-CC00064XX07',
       'sub-CC00067XX10', 'sub-CC00071XX06', 'sub-CC00074XX09',
       'sub-CC00075XX10', 'sub-CC00051XX02', 'sub-CC00052XX03',
       'sub-CC00054XX05', 'sub-CC00056XX07', 'sub-CC00057XX08',
       'sub-CC00062XX05', 'sub-CC00065XX08', 'sub-CC00066XX09',
       'sub-CC00068XX11', 'sub-CC00069XX12', 'sub-CC00055XX06',
       'sub-CC00080XX07', 'sub-CC00119XX12', 'sub-CC00120XX05',
       'sub-CC00130XX07', 'sub-CC00134XX11', 'sub-CC00135AN12',
       'sub-CC00135BN12', 'sub-CC00136BN13', 'sub-CC00137XX14',
       'sub-CC00058XX09', 'sub-CC00167XX11', 'sub-CC00168XX12',
       'sub-CC00200XX02', 'sub-CC00218AN12', 'sub-CC00284AN13',
       'sub-CC00286XX15', 'sub-CC00290XX11', 'sub-CC00316XX11',
       'sub-CC00335XX14', 'sub-CC00194XX14', 'sub-CC00883XX18',
       'sub-CC00886XX21', 'sub-CC00986BN22']
subjects = [su.split('sub-')[1] for su in subjects]
sessions =['152300', '153600', '177201', '182000', '184100', '198200',
       '200000', '201800', '210400', '245100', '45100', '76601', '119100',
       '145700', '150500', '163200', '177900', '193700', '195801',
       '214100', '26700', '27600', '45000', '46800', '49200', '50700',
       '55500', '58500', '83000', '90500', '8607', '11900', '12501',
       '15102', '15104', '18303', '20200', '27000', '28000', '28400',
       '7702', '8300', '8800', '10700', '11002', '13801', '18600',
       '19200', '20701', '26300', '9300', '30300', '39400', '41600',
       '44001', '44600', '54400', '54500', '64300', '45200', '11300',
       '55600', '55700', '67204', '85900', '111400', '91700', '92900',
       '101300', '106300', '65401', '14430', '18030', '41830']


subjects2 = ['KKI2009_800', 'KKI2009_239',
       'KKI2009_505', 'KKI2009_679', 'KKI2009_934', 'KKI2009_113',
       'KKI2009_422', 'KKI2009_815', 'KKI2009_906', 'KKI2009_127',
       'KKI2009_742', 'KKI2009_849', 'KKI2009_913', 'KKI2009_346',
       'KKI2009_502', 'KKI2009_814', 'KKI2009_916', 'KKI2009_959',
       'KKI2009_142', 'KKI2009_656']
sessions2 = ['MR1',
       'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1',
       'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1',
       'MR1']


dataset = np.repeat('dHCP', len(subjects))
dataset2 = np.repeat('KKI2009', len(subjects2))

subjects_both = np.hstack([subjects, subjects2])
sessions_both = np.hstack([sessions, sessions2])
dataset_both = np.hstack([dataset, dataset2])

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'
folder_KKI = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0'
folder_KKI_sulc = '/media/maxime/Expansion/outputs_mirtkdeformmesh_KKIFS6'
# dpfstar and voronoi
folder_voronoi = os.path.join(wd, 'data_EXP3/result_EXP3/voronoi_textures')
folder_dpf500 = os.path.join(wd, 'data_EXP3/resul_EXP3/dpfstar500')


# load threshold # computed with the script threshold2
with open(os.path.join(wd, 'data_EXP3/result_EXP3', 'value_dpfstar500.pkl'), 'rb') as file:
    value_dpf500 = pickle.load(file)

with open(os.path.join(wd, 'data_EXP3/result_EXP3', 'value_sulc.pkl'), 'rb') as file:
    value_sulc = pickle.load(file)

with open(os.path.join(wd, 'data_EXP3/result_EXP3', 'value_dpfstar500_abs.pkl'), 'rb') as file:
    value_dpfA = pickle.load(file)

# set threshold
pct = 50
Th_dpf500 = np.percentile(value_dpf500,pct)
Th_dpfA = np.percentile(value_dpfA,pct)
Th_sulc = np.percentile(value_sulc,pct)



# init
list_metric_dpf500 = list()
list_metric_sulc = list()
list_metric_dpfA = list()

list_age = list()
list_sub = list()
list_ses = list()
list_surface = list()
list_volume = list()

list_GI = list()

for idx, sub in enumerate(subjects_both):
    print(sub)
    ses = sessions_both[idx]
    dset = dataset_both[idx]

    if (dset == 'dHCP') :
        # load mask cortex
        cortex = sio.load_texture(os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat',
                                               'sub-' + sub + '_ses-' + ses + '_hemi-left_desc-drawem_dseg.label.gii'))
        cortex = cortex.darray[0].astype(bool)

        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(folder_dHCP , 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
        dpfstar500_name = sub + '_' + ses + '_dpfstar500.gii'
        voronoi_name = sub + '_' + ses + '_voronoi.gii'

        mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
        mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat' )
        mesh_path = os.path.join(mesh_folder, mesh_name)
        mesh = sio.load_mesh(mesh_path)
        volume = mesh.volume
        list_volume.append(volume)



    if (dset== 'KKI2009') :
        # load mask cortex
        cortex = sio.load_texture(os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                                 sub + '_' + ses, 'label', 'mask_interH.gii')).darray[0]
        cortex = np.invert(cortex.astype(bool))
        # load mesh
        # KKI
        mesh_name = 'lh.white.gii'
        mesh_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                                 sub + '_' + ses, 'surf', mesh_name)
        mesh = sio.load_mesh(mesh_path)
        volume = mesh.volume
        list_volume.append(volume)

        # load sulc KKI
        sulc_name =  sub + '_' + ses + '_l_sulc.gii'
        sulc_path = os.path.join(folder_KKI_sulc, sulc_name)
        dpfstar500_name = sub + '_' + ses + '_dpfstar500.gii'
        voronoi_name = sub + '_' + ses + '_voronoi.gii'

    # compute GI
    surface = mesh.area
    volume = mesh.volume
    GI = mesh.area / mesh.convex_hull.area
    list_GI.append(GI)
    # compute sulc
    sulc = sio.load_texture(sulc_path).darray[0]
    sulc = sulc[cortex]
    # load dpf500
    dpf500_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data_EXP3/result_EXP3/dpfstar500',
                               dpfstar500_name)

    dpf500 = sio.load_texture(dpf500_path).darray[0]
    dpf500 = dpf500[cortex]
    dpfA = dpf500 * np.power(volume, 1/3)
    # load voronoi
    voronoi_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data_EXP3/result_EXP3/voronoi_textures',
                            voronoi_name)
    voronoi = sio.load_texture(voronoi_path).darray[0]
    surface = np.sum(voronoi)
    voronoi = voronoi[cortex]

    # bassin detection
    bassin_dpf500 = dpf500 <= Th_dpf500
    bassin_sulc = sulc <= Th_sulc
    bassin_dpfA = dpfA <= Th_dpfA

    # metric
    metric_dpf500 = np.sum(voronoi[bassin_dpf500]) / np.sum(voronoi)
    metric_sulc = np.sum(voronoi[bassin_sulc]) / np.sum(voronoi)
    metric_dpfA = np.sum(voronoi[bassin_dpfA]) / np.sum(voronoi)


    # store for dataframe
    list_metric_dpf500.append(metric_dpf500)
    list_metric_sulc.append(metric_sulc)
    list_metric_dpfA.append(metric_dpfA)

    list_sub.append(sub)
    list_ses.append(ses)
    list_surface.append(np.sum(voronoi))


df_dpf500 = pd.DataFrame(dict(sub=list_sub, ses = list_ses, surface=list_surface, volume = list_volume,GI = list_GI,
                               value=list_metric_dpf500))


df_sulc = pd.DataFrame(dict(sub=list_sub, ses = list_ses, surface=list_surface, volume = list_volume,GI = list_GI,
                               value=list_metric_sulc))


df_dpfA = pd.DataFrame(dict(sub=list_sub, ses = list_ses, surface=list_surface, volume = list_volume,GI = list_GI,
                               value=list_metric_dpfA))


## figure 1
plt.rcParams["font.size"]= "30"
size = 100
fontsize = 30
fig1, axs = plt.subplots()

f1 = sns.scatterplot(data=df_dpf500, x="volume", y="value",  label = 'dpf500', s=size)
f2 = sns.scatterplot(data=df_sulc, x="volume", y="value",  label = 'sulc',
                s = size)
#f3 = sns.scatterplot(data=df_dpfA, x="volume", y="value",  label = 'dpf500A')


f1.set(xlabel="volume  mm3", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded DPF-star ')
#axs.set_xlabel("volume  mm3" , fontsize=fontsize)
#axs.set_ylabel("percentage surface bassin", fontsize=fontsize)

axs.legend()
plt.grid()
plt.tight_layout()

## figure 2
fig2, ax2s = plt.subplots()

g1 = sns.scatterplot(data=df_dpf500, x="GI", y="value",  label = 'dpf500', s = size)
g2 = sns.scatterplot(data=df_sulc, x="GI", y="value",  label = 'sulc',
                s =size)
#f3 = sns.scatterplot(data=df_dpfA, x="volume", y="value",  label = 'dpf500A')


g1.set(xlabel="Giration index", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded DPF-star ')
#ax2s.set_xlabel("Giration index",fontsize=fontsize)
#ax2s.set_ylabel("percentage surface bassin", fontsize=fontsize)
#sns.set_context('paper', font_scale=3)

ax2s.legend()
plt.grid()
plt.tight_layout()

plt.show()

