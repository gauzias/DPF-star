import pandas as pd
import numpy as np
import slam.io as sio
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as ss
import pickle
from scipy import interpolate
from scipy.interpolate import RBFInterpolator, InterpolatedUnivariateSpline

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'
folder_KKI = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0'
folder_KKI_sulc = '/media/maxime/Expansion/outputs_mirtkdeformmesh_KKIFS6'
# dpfstar and voronoi
folder_voronoi = os.path.join(wd, 'data/EXP_3_bassins/voronoi_textures')
folder_dpf100 = os.path.join(wd, 'data/EXP_3_bassins/dpfstar100')
# info dataset
#KKI_info_path = '/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0/KKI_info.csv'
#dHCP_rel3_info_path = '/media/maxime/Expansion/rel3_dHCP/info_rel3.csv'

info_data_path = os.path.join(wd, 'data/info_database/info_database_EXP3.csv')
info_data = pd.read_csv(info_data_path)

info_data = info_data[(info_data['curvature']==True)
                      & (info_data['voronoi']==True)
                      & (info_data['dpfstar100'] == True)]

# list of subjects
# load KKI subjects
#KKI_info = pd.read_csv(KKI_info_path)
#sub_KKI = KKI_info['Subject_ID'].values
#sub_KKI = ['KKI2009_' + str(xx) for xx in sub_KKI]
#nbk = len(sub_KKI)
#ses_KKI = np.repeat('MR1', nbk)
#scan_age_KKI = KKI_info['Age']
#convert scan age into days
#scan_age_KKI = np.array([ss*365 for ss in scan_age_KKI])
#tri_age_KKI = np.argsort(scan_age_KKI)
# sort by age
#scan_age_KKI = np.sort(scan_age_KKI)
#sub_KKI= np.array(sub_KKI)[tri_age_KKI]

# load dHCP subjects
#dHCP_rel3_info = pd.read_csv(dHCP_rel3_info_path)
#sub_dHCP = dHCP_rel3_info['suj'].values
#sub_dHCP = [xx.split('-')[1] for xx in sub_dHCP]
#nbh = len(sub_dHCP)
#session_dHCP = dHCP_rel3_info['session_id'].values
#session_dHCP = [str(xx) for xx in session_dHCP]
#scan_age = dHCP_rel3_info['scan_age'].values
# convert age into days
#scan_age = np.array([ss * 7 for ss in scan_age])
#tri_age = np.argsort(scan_age)
# sort by age
#scan_age = np.sort(scan_age)
#sub_dHCP= np.array(sub_dHCP)[tri_age]
#session_dHCP = np.array(session_dHCP)[tri_age]



# list subjects dHCP + KKI
#subjects = np.hstack([sub_dHCP, sub_KKI])
#sessions = np.hstack([session_dHCP, ses_KKI])
#ages = np.hstack([scan_age, scan_age_KKI])
#dataset = np.hstack([np.repeat('dHCP', nbh), np.repeat('KKI', nbk)])


# load GI
#with open(os.path.join(wd, 'data/EXP_3_bassins', 'GI.pkl'), 'rb') as file:
#    GI_list = pickle.load(file)
# load volume
#with open(os.path.join(wd, 'data/EXP_3_bassins', 'volume.pkl'), 'rb') as file:
#    volume_list = pickle.load(file)

# load threshold
with open(os.path.join(wd, 'data/EXP_3_bassins', 'value_dpfstar100.pkl'), 'rb') as file:
    value_dpf100 = pickle.load(file)

with open(os.path.join(wd, 'data/EXP_3_bassins', 'value_sulc.pkl'), 'rb') as file:
    value_sulc = pickle.load(file)

with open(os.path.join(wd, 'data/EXP_3_bassins', 'value_sulcZ.pkl'), 'rb') as file:
    value_sulc_Z = pickle.load(file)


# set threshold
pct = 45
Th_dpf100 = np.percentile(value_dpf100,pct)
Th_sulc = np.percentile(value_sulc,pct)
Th_sulcZ = np.percentile(value_sulc_Z,pct)

#Th_dpf100 = 0
#Th_sulc = 0
#Th_sulcZ = 0


# init
list_metric_dpf100 = list()
list_metric_sulc = list()
list_metric_sulcZ = list()
list_age = list()
list_sub = list()
list_ses = list()
list_surface = list()

buggy = list()

dataset =  info_data['dataset'].values
sessions = info_data['session_id'].values
sessions = [str(xx) for xx in sessions]
subjects = info_data['suj'].values
subjects = [xx.split('-')[1] for xx in subjects]
ages = info_data['scan_age_days'].values

GI_list   = list()
volume_list = list()
for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    dset = dataset[idx]
    age = ages[idx]

    if (dset == 'dHCP') :
        # load sulc dHCP
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join(folder_dHCP , 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
        dpfstar100_name = sub + '_' + ses + '_dpfstar100.gii'
        voronoi_name = sub + '_' + ses + '_voronoi.gii'
    if (dset== 'KKI2009') :
        # load sulc KKI
        sulc_name = 'KKI2009_' + sub + '_' + ses + '_l_sulc.gii'
        sulc_path = os.path.join(folder_KKI_sulc, sulc_name)
        dpfstar100_name = 'KKI2009_' + sub + '_' + ses + '_dpfstar100.gii'
        voronoi_name = 'KKI2009_' + sub + '_' + ses + '_voronoi.gii'

    sulc = sio.load_texture(sulc_path).darray[0]
    sulc_Z = ss.zscore(sulc)
    # load dpf100
    dpf100_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data/EXP_3_bassins/dpfstar100',
                               dpfstar100_name)

    dpf100 = sio.load_texture(dpf100_path).darray[0]

    # load voronoi
    voronoi_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data/EXP_3_bassins/voronoi_textures',
                            voronoi_name)

    voronoi = sio.load_texture(voronoi_path).darray[0]
    surface = np.sum(voronoi)

    # load GI
    index = info_data.index[ (info_data['suj']== 'sub-' + sub) & (info_data['session_id'] == ses)]
    GI = info_data.loc[index, 'GI'].tolist()[0]
    volume = info_data.loc[index, 'volume'].tolist()[0]
    print(GI)
    GI_list.append(GI)
    volume_list.append(volume)

    # bassin detection
    bassin_dpf100 = dpf100 <= Th_dpf100
    bassin_sulc = sulc <= Th_sulc
    bassin_sulcZ = sulc_Z <= Th_sulcZ

    # metric
    metric_dpf100 = np.sum(voronoi[bassin_dpf100]) / np.sum(voronoi)
    metric_sulc = np.sum(voronoi[bassin_sulc]) / np.sum(voronoi)
    metric_sulcZ = np.sum(voronoi[bassin_sulcZ]) / np.sum(voronoi)

    # store for dataframe
    list_metric_dpf100.append(metric_dpf100)
    list_metric_sulc.append(metric_sulc)
    list_metric_sulcZ.append(metric_sulcZ)
    list_sub.append(sub)
    list_ses.append(ses)
    list_age.append(age)
    list_surface.append(np.sum(voronoi))


df_dpf100 = pd.DataFrame(dict(sub=list_sub, ses = list_ses, surface=list_surface, volume = volume_list,
                              age = list_age, GI =GI_list, value=list_metric_dpf100))

df_sulc = pd.DataFrame(dict(sub=list_sub, ses = list_ses, surface=list_surface, volume = volume_list,
                              age = list_age,GI =GI_list,  value=list_metric_sulc))

df_sulcZ = pd.DataFrame(dict(sub=list_sub, ses = list_ses, surface=list_surface,volume = volume_list,
                              age = list_age, GI =GI_list, value=list_metric_sulcZ))


# interpolate


# dpf 100
x = np.sort(df_dpf100["GI"].values)
y = df_dpf100["value"].values[np.argsort(np.sort(df_dpf100["GI"].values))]
knot_numbers = 0
x_new = np.linspace(0, 1, knot_numbers+1)[1:-1]
q_knots = np.quantile(x, x_new)
t,c,k = interpolate.splrep(x, y, t=q_knots, s=1)
yfit = interpolate.BSpline(t,c,k)(x)

# sulc
x2 = np.sort(df_sulc["GI"].values)
y2 = df_sulc["value"].values[np.argsort(np.sort(df_sulc["GI"].values))]
knot_numbers = 1
x2_new = np.linspace(0, 1, knot_numbers+1)[1:-1]
q_knots = np.quantile(x2, x2_new)
t2,c2,k2 = interpolate.splrep(x2, y2, t=q_knots, s=1)
y2fit = interpolate.BSpline(t2,c2,k2)(x2)

# sulc Z
x3 = np.sort(df_sulcZ["GI"].values)
y3 = df_sulcZ["value"].values[np.argsort(np.sort(df_sulcZ["GI"].values))]
knot_numbers = 1
x3_new = np.linspace(0, 1, knot_numbers+1)[1:-1]
q_knots = np.quantile(x3, x3_new)
t3,c3,k3 = interpolate.splrep(x3, y3, t=q_knots, s=1)
y3fit = interpolate.BSpline(t3,c3,k3)(x2)


## figure 1
fig1, axs = plt.subplots()

f1 = sns.scatterplot(data=df_dpf100, x="volume", y="value",  legend = False, )
f2 = sns.scatterplot(data=df_sulc, x="volume", y="value",  legend = False,
               marker='X')
f3 = sns.scatterplot(data=df_sulcZ, x="volume", y="value",  legend = False,
               marker='X')

f1.set(xlabel="surface area mm2", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded DPF-star ')
f2.set(xlabel="surface area mm2", ylabel="percentage surface bassin",
       title='sulcal bassin detection with thresholded Depth ')

axs.legend()
plt.grid()
plt.tight_layout()

fig2, ax2 = plt.subplots()
g1 = sns.scatterplot(data=df_dpf100, y="value", x="GI",  label = 'dpfstar100', size="volume", legend=False)
g2 = sns.scatterplot(data=df_sulc, y="value", x="GI",  label= 'sulc' , size = "volume", legend=False)
#g3 = sns.scatterplot(data=df_sulcZ, y="value", x="GI",  label = 'sulcZ' )

ax2.set_xlabel('Girification Index')
ax2.set_ylabel('percentage surface detected as sulcal bassins')

ax2.plot(x,yfit)
ax2.plot(x2,y2fit)
ax2.plot(x3,y3fit)

plt.grid()
plt.legend()
plt.show()

