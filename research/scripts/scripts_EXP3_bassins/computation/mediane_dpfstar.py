import os
import numpy as np
import pandas as pd
import slam.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'

# load subjects

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

# init
volumes = list()
p5 = list()
p25 = list()
p50 = list()
p75 = list()
p95 = list()
mea = list()


Ap5 = list()
Ap25 = list()
Ap50 = list()
Ap75 = list()
Ap95 = list()
Amea = list()

Sp5 = list()
Sp25 = list()
Sp50 = list()
Sp75 = list()
Sp95 = list()
Smea = list()

# loop
list_dpfr = list()
list_dpfa = list()
list_sulc = list()
for idx, sub in enumerate(subjects_both):
    print(sub)
    ses = sessions_both[idx]
    dset = dataset_both[idx]
    if (dset == 'dHCP'):
        # load mesh
        mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
        mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat')
        mesh_path = os.path.join(mesh_folder, mesh_name)
        mesh = sio.load_mesh(mesh_path)
        vol = mesh.volume
        volumes.append(vol)
        # load mask cortex
        cortex = sio.load_texture(os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat',
                                               'sub-' + sub + '_ses-' + ses + '_hemi-left_desc-drawem_dseg.label.gii'))
        cortex = cortex.darray[0].astype(bool)

    if (dset == 'KKI2009'):
        # load mesh
        mesh_name = 'lh.white.gii'
        mesh_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                                 sub + '_' + ses, 'surf', mesh_name)
        mesh = sio.load_mesh(mesh_path)
        vol = mesh.volume
        volumes.append(vol)
        # load mask cortex
        cortex = sio.load_texture(os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                                 sub + '_' + ses, 'label', 'mask_interH.gii')).darray[0]
        cortex = np.invert(cortex.astype(bool))
    # load sulc
    if dset == 'dHCP':
        sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
        sulc_path = os.path.join('/media/maxime/Expansion/rel3_dHCP', 'sub-' + sub, 'ses-' + ses, 'anat', sulc_name)
    if dset == 'KKI2009':
        sulc_path = os.path.join('/media/maxime/Expansion/outputs_mirtkdeformmesh_KKIFS6',
        sub + '_' + ses + '_l_sulc.gii')
    sulc = sio.load_texture(sulc_path).darray[0]
    sulc = sulc[cortex]
    list_sulc.append(sulc)
    # load dpfstar 500
    dpfstar_name = sub + '_' + ses + '_dpfstar500.gii'
    dpfstar_folder = os.path.join(wd, 'data_EXP3/result_EXP3/dpfstar500')
    dpfstar_path = os.path.join(dpfstar_folder, dpfstar_name)
    dpfstar = sio.load_texture(dpfstar_path).darray[0]
    dpfstar = dpfstar[cortex]
    # add dpfstar to list
    list_dpfr.append(dpfstar)
    # compute dpfstar 500 absolut
    dpfA = dpfstar * np.power(vol, 1/3)
    list_dpfa.append(dpfA)

    # compute percentiles
    p5.append(np.percentile(dpfstar, 5))
    p25.append(np.percentile(dpfstar, 25))
    p50.append(np.percentile(dpfstar, 50))
    p75.append(np.percentile(dpfstar, 75))
    p95.append(np.percentile(dpfstar, 95))
    mea.append(np.mean(dpfstar))
    # compute percentiles
    Ap5.append(np.percentile(dpfA, 5))
    Ap25.append(np.percentile(dpfA, 25))
    Ap50.append(np.percentile(dpfA, 50))
    Ap75.append(np.percentile(dpfA, 75))
    Ap95.append(np.percentile(dpfA, 95))
    Amea.append(np.mean(dpfA))
    # compute percentiles
    Sp5.append(np.percentile(sulc, 5))
    Sp25.append(np.percentile(sulc, 25))
    Sp50.append(np.percentile(sulc, 50))
    Sp75.append(np.percentile(sulc, 75))
    Sp95.append(np.percentile(sulc, 95))
    Smea.append(np.mean(sulc))


## Threshod
Th_dpfr = np.mean(np.array([np.mean(np.array(d)) for d in list_dpfr]))
Th_dpfa = np.mean(np.array([np.mean(np.array(d)) for d in list_dpfa]))
Th_sulc = np.mean(np.array([np.mean(np.array(d)) for d in list_sulc]))

print('dpfr', Th_dpfr)
print('dpfa', Th_dpfa)
print('sulc', Th_sulc)

## construct dataframe
df_dpfr = pd.DataFrame(dict(subjects = subjects_both,
                       sessions = sessions_both,
                       volumes = volumes,
                       p5 = p5, p25 = p25, p50 = p50, p75 = p75, p95 = p95, mean = mea))

df_dpfa = pd.DataFrame(dict(subjects = subjects_both,
                       sessions = sessions_both,
                       volumes = volumes,
                       p5 = Ap5, p25 = Ap25, p50 = Ap50, p75 = Ap75, p95 = Ap95, mean = Amea))

df_sulc = pd.DataFrame(dict(subjects = subjects_both,
                       sessions = sessions_both,
                       volumes = volumes,
                       p5 = Sp5, p25 = Sp25, p50 = Sp50, p75 = Sp75, p95 = Sp95, mean = Smea))


### fit
def fit(df, p):
    x = np.sort(df["volumes"].values)
    y = df[p].values[np.argsort(np.sort(df["volumes"].values))]
    knot_numbers = 0
    x_new = np.linspace(0, 1, knot_numbers+1)[1:-1]
    q_knots = np.quantile(x, x_new)
    t,c,k = interpolate.splrep(x, y, t=q_knots, s=1)
    yfit = interpolate.BSpline(t,c,k)(x)
    return x, yfit


# diplay

def display(df, Th):
    sns.scatterplot(data = df, x = 'volumes', y = 'p5', size = 2, color = 'lightblue',legend=False)
    sns.scatterplot(data = df, x = 'volumes', y = 'p25', size = 2, color = 'blue',legend=False)
    sns.scatterplot(data = df, x = 'volumes', y = 'p50', size = 2, color = 'orange',legend=False)
    sns.scatterplot(data = df, x = 'volumes', y = 'mean', size = 2, color = 'red',legend=False)
    sns.scatterplot(data = df, x = 'volumes', y = 'p75', size = 2, color = 'blue',legend=False)
    sns.scatterplot(data = df, x = 'volumes', y = 'p95', size = 2, color = 'lightblue',legend=False)

    # line 5-95
    x5,y5 = fit(df, 'p5')
    x95,y95 = fit(df, 'p95')
    sns.lineplot(x5,y5, color = 'lightblue', label = 'p5')
    sns.lineplot(x95,y95, color = 'lightblue', label = 'p95')
    plt.fill_between(x5, y5, y95, color='blue', alpha=.2)

    # line 25 - 75
    x25,y25 = fit(df, 'p25')
    x75,y75 = fit(df, 'p75')
    sns.lineplot(x25,y25, color = 'blue', label = 'p25')
    sns.lineplot(x75,y75, color = 'blue', label = 'p75')
    plt.fill_between(x25, y25, y75, color='blue', alpha=.2)

    # line median mean
    x50,y50 = fit(df, 'p50')
    xm,ym = fit(df, 'mean')
    sns.lineplot(x50,y50, color = 'orange', label = 'median')
    sns.lineplot(xm,ym, color = 'red', label = 'mean')

    # line threshold
    plt.axhline(Th, label = 'threshold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()

fig1, ax1 = plt.subplots()
display(df_dpfr, Th_dpfr)
ax1.set(xlabel='Volume mm3', ylabel='Depth Percentils', title = 'DPF-star')
plt.tight_layout()

fig2, ax2 = plt.subplots()
display(df_dpfa, Th_dpfa)
ax2.set(xlabel='Volume mm3', ylabel='Depth Percentils', title = 'absolut DPF-star')
plt.tight_layout()

fig3, ax3 = plt.subplots()
display(df_sulc, Th_sulc)
ax3.set(xlabel='Volume mm3', ylabel='Depth Percentils', title = 'SULC')

plt.tight_layout()
plt.show()