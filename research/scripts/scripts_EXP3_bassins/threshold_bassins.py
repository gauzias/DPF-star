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
folder_voronoi = os.path.join(wd, 'data/EXP_3_bassins/voronoi_textures')
folder_dpf100 = os.path.join(wd, 'data/EXP_3_bassins/dpfstar100')

info_dataset = '/home/maxime/callisto/repo/paper_sulcal_depth/data/info_database/info_database_EXP3.csv'
info_data = pd.read_csv(info_dataset)
info_data = info_data[(info_data['dpfstar100'] == True) &(info_data['curvature'] == True)&(info_data['voronoi'] == True)]

subjects = info_data['suj']
subjects = [su.split('sub-')[1] for su in subjects]
nb = len(subjects)
sessions = info_data['session_id']
age = info_data['scan_age_days']
dataset = info_data['dataset']

volume = info_data['volume']

hist_volume = plt.hist(volume, bins = 'auto')
bins = hist_volume[1]
count = hist_volume[0]

list_weight = list()

for idx, vol in enumerate(volume):
    test_volume = np.array([])
    test_volume = np.hstack([bins, vol])
    test_volume = np.unique(test_volume)
    test_volume = np.sort(test_volume)
    bin_vol = np.where(test_volume==vol)[0][0]
   #print(bin_vol)
    if bin_vol>=len(count):
        bin_vol = len(count)-1
    if bin_vol == 0:
        bin_vol = 1
    count_vol = count[bin_vol-1]
    #print(count_vol)
    weight = 1/(len(count) * count_vol)
    list_weight.append(weight)
    print(weight)


## define threshold
value_dpf100 = np.array([])
value_sulc = np.array([])
value_sulc_Z = np.array([])
#GI_list = list()
volume_list = list()
# compute threshold for Th sulc
for idx, sub in enumerate(subjects):
    ses = sessions[idx]
    dset = dataset[idx]
    if (dset == 'dHCP') :
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(folder_dHCP , 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
        # load dpf100
        dpf100_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data/EXP_3_bassins/dpfstar100',
                                       sub + '_' + ses + '_dpfstar100.gii')
        dpf100 = sio.load_texture(dpf100_path).darray[0]
    if (dset== 'KKI2009') :
        sulc_name = 'KKI2009_'+ sub + '_' + ses + '_l_sulc.gii'
        sulc_path = os.path.join(folder_KKI_sulc, sulc_name)
        # load dpf100
        dpf100_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data/EXP_3_bassins/dpfstar100',
                                       'KKI2009_'+ sub + '_' + ses + '_dpfstar100.gii')
        dpf100 = sio.load_texture(dpf100_path).darray[0]
    sulc = sio.load_texture(sulc_path).darray[0]
    sulc_Z = ss.zscore(sulc)
    value_sulc = np.hstack([value_sulc, sulc])
    value_sulc_Z = np.hstack([value_sulc_Z, sulc_Z])
    value_dpf100 = np.hstack([value_dpf100, dpf100])


"""

#save

with open(os.path.join(wd, 'data/EXP_3_bassins', 'value_dpfstar100.pkl'), 'wb') as file:
    pickle.dump(value_dpf100,file)

with open(os.path.join(wd, 'data/EXP_3_bassins', 'value_sulc.pkl'), 'wb') as file:
    pickle.dump(value_sulc,file)

with open(os.path.join(wd, 'data/EXP_3_bassins', 'value_sulcZ.pkl'), 'wb') as file:
    pickle.dump(value_sulc_Z,file)


with open(os.path.join(wd, 'data/EXP_3_bassins', 'GI.pkl'), 'wb') as file:
    pickle.dump(GI_list,file)

with open(os.path.join(wd, 'data/EXP_3_bassins', 'volume.pkl'), 'wb') as file:
    pickle.dump(volume_list,file)

"""
