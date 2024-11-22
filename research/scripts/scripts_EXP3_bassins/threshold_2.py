import pandas as pd
import numpy as np
import slam.io as sio
import os
import scipy.stats as ss
import pickle
import matplotlib.pyplot as plt

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'
folder_KKI = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0'
folder_KKI_sulc = '/media/maxime/Expansion/outputs_mirtkdeformmesh_KKIFS6'

# dpfstar and voronoi
folder_voronoi = os.path.join(wd, 'data/result_EXP3/voronoi_textures')
folder_dpf500 = os.path.join(wd, 'data/result_EXP3/dpfstar500')

#load subject

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

## define threshold
value_dpf500 = np.array([])
value_dpfA = np.array([])
value_sulc = np.array([])

# compute threshold for Th sulc
for idx, sub in enumerate(subjects_both):
    ses = sessions_both[idx]
    dset = dataset_both[idx]
    if (dset == 'dHCP') :
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(folder_dHCP , 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
        # load dpf500
        dpf500_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data_EXP3/result_EXP3/dpfstar500',
                                       sub + '_' + ses + '_dpfstar500.gii')
        #load mesh
        mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
        mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat')
        mesh_path = os.path.join(mesh_folder, mesh_name)
        mesh = sio.load_mesh(mesh_path)
        vol = mesh.volume


    if (dset== 'KKI2009') :
        sulc_name =  sub + '_' + ses + '_l_sulc.gii'
        sulc_path = os.path.join(folder_KKI_sulc, sulc_name)
        # load dpf100
        dpf500_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data_EXP3/result_EXP3/dpfstar500',
                                        sub + '_' + ses + '_dpfstar500.gii')
        # load mesh
        mesh_name = 'lh.white.gii'
        mesh_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                                 sub + '_' + ses, 'surf', mesh_name)
        mesh = sio.load_mesh(mesh_path)
        vol = mesh.volume

    dpf500 = sio.load_texture(dpf500_path).darray[0]
    dpfA = dpf500 * np.power(vol, 1/3)
    sulc = sio.load_texture(sulc_path).darray[0]

    value_sulc = np.hstack([value_sulc, sulc])
    value_dpf500 = np.hstack([value_dpf500, dpf500])
    value_dpfA = np.hstack([value_dpfA, dpfA])


#save

with open(os.path.join(wd, 'data_EXP3/result_EXP3', 'value_dpfstar500.pkl'), 'wb') as file:
    pickle.dump(value_dpf500, file)

with open(os.path.join(wd, 'data_EXP3/result_EXP3', 'value_sulc.pkl'), 'wb') as file:
    pickle.dump(value_sulc, file)

with open(os.path.join(wd, 'data_EXP3/result_EXP3', 'value_dpfstar500_abs.pkl'), 'wb') as file:
    pickle.dump(value_dpfA, file)




