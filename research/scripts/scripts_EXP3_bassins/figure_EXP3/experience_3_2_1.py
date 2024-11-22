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

value_dpf100 = np.array([])
value_sulc = np.array([])


Th1_list = np.linspace(-0.1, 0.1, 30)
Th2_list = np.linspace(-10,10, 30)



df1 = pd.DataFrame(dict(Th = [], sub = [],  surface = [], surface_rounded = [],  value = [] ))
df2 = pd.DataFrame(dict(Th = [], sub = [],  surface = [], surface_rounded = [],  value = [] ))


for tx, Th1 in enumerate(Th1_list):
    Th2 = Th2_list[tx]
    list_metric_sulc = list()
    list_metric_dpf100 = list()
    list_surface = list()
    list_surface_rounded = list()

    for idx, sub in enumerate(subjects):
        print(sub)
        ses = sessions[idx]
        dset = dataset[idx]

        if dset == 'dHCP':
            # load sulc dHCP
            sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
            sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
            sulc = sio.load_texture(sulc_path).darray[0]
        if dset == 'KKI':
            # load sulc KKI
            sulc_name = sub + '_' + ses + '_l_sulc.gii'
            sulc = sio.load_texture(os.path.join(dir_sulcKKI, sulc_name)).darray[0]

        # load dpf100
        dpf100_path = os.path.join(dir_dpf100, sub + '_' + ses + '_dpf100.gii')
        dpf100 = sio.load_texture(dpf100_path).darray[0]



        # load voronoi
        voronoi_path = os.path.join(dir_voronoi, sub + '_' + ses + '_voronoi.gii')
        voronoi = sio.load_texture(voronoi_path).darray[0]

        # bassin detection
        bassin_sulc = sulc <= Th2
        bassin_dpf100 = dpf100 <= Th1
        # metric
        metric_sulc = np.sum(voronoi[bassin_sulc]) / np.sum(voronoi)
        metric_dpf100 = np.sum(voronoi[bassin_dpf100]) / np.sum(voronoi)
        list_metric_sulc.append(metric_sulc)
        list_metric_dpf100.append(metric_dpf100)

        list_surface.append(np.sum(voronoi))
        list_surface_rounded.append(round(np.sum(voronoi)/5000)*5000)


    df1ij = pd.DataFrame(dict(  Th = np.repeat(Th1, len(subjects)) , sub = subjects, surface = list_surface,
                               surface_rounded= list_surface_rounded,
                               value = list_metric_dpf100))

    df1 = pd.concat([df1, df1ij], ignore_index=True)

    df2ij = pd.DataFrame(dict(Th=np.repeat(Th2, len(subjects)), sub=subjects, surface=list_surface,
                              surface_rounded=list_surface_rounded,
                              value=list_metric_sulc))

    df2 = pd.concat([df2, df2ij], ignore_index=True)

    #ax1.scatter(list_surface, list_metric_dpf100, label='dpf100_' + str(Th1))
    #ax2.scatter(list_surface, list_metric_sulc, label='sulc_' + str(Th2))



fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

f1 = sns.lineplot(data = df1, x = "surface_rounded", y = "value", hue = 'Th', ax= ax1)
f2 = sns.lineplot(data = df2, x = "surface_rounded", y = "value", hue = 'Th', ax= ax2)

f1.set(xlabel ="surface area mm2", ylabel = "percentage surface bassin", title ='sulcal bassin detection with thresholded DPF-star ')
f2.set(xlabel ="surface area mm2", ylabel = "percentage surface bassin", title ='sulcal bassin detection with thresholded SULC ')


#ax1.set_xlabel('surface area')
#ax1.set_ylabel('ratio surface bassin / total surface ')

#ax2.set_xlabel('surface area')
#ax2.set_ylabel('ratio surface bassin / total surface ')
#plt.legend()

plt.show()

