import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pingouin as pg

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]

dataset_EXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))
# load metrics DEV
dev_curv = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/curv/curv_dev.csv'))

dev_sulc = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/sulc/sulc_dev.csv'))

dev_dpf003 = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf003/dpf003_dev.csv'))


dev_dpfstar = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpfstar/dpfstar_dev.csv'))
dev_dpfstar2 = dev_dpfstar.reset_index().pivot_table(index = 'subject', columns='alphas', values='angle_dpfstar')
dev_dpfstar2 = dev_dpfstar2.reset_index()

# load metrics STD
std_curv = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/curv/curv_std_crest.csv'))
std_sulc = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/sulc/sulc_std_crest.csv'))
std_dpf003 = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf003/dpf003_std_crest.csv'))

std_dpfstar = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpfstar/dpfstar_std_crest.csv'))
std_dpfstar2 = std_dpfstar.reset_index().pivot_table(index = 'subject', columns='alphas', values='std_crest_dpfstar')
std_dpfstar2 = std_dpfstar2.reset_index()

# load metric DIFF
diff_curv = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/curv/curv_diff_fundicrest.csv'))
diff_sulc = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/sulc/sulc_diff_fundicrest.csv'))
diff_dpf003 = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf003/dpf003_diff_fundicrest.csv'))

diff_dpfstar = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpfstar/dpfstar_diff_fundicrest.csv'))
diff_dpfstar2 = diff_dpfstar.reset_index().pivot_table(index = 'subject', columns='alphas', values='diff_fundicrest_dpfstar')
diff_dpfstar2 = diff_dpfstar2.reset_index()


# fuse
df_dev = dev_curv.merge(dev_sulc, how ='outer')
df_dev = df_dev.merge(dev_dpf003, how ='outer')
df_dev = df_dev.merge(dev_dpfstar2, how ='outer')
df_dev = df_dev.rename(columns={'angle_curv': 'curv',
                                'angle_sulc': 'sulc',
                                'angle_dpf003' : 'dpf003'})

df_std = std_curv.merge(std_sulc, how ='outer')
df_std = df_std.merge(std_dpf003, how ='outer')
df_std = df_std.merge(std_dpfstar2, how ='outer')
df_std = df_std.rename(columns={'std_crest_curv': 'curv',
                                'std_crest_sulc': 'sulc',
                                'std_crest_dpf003' : 'dpf003'})

df_diff = diff_curv.merge(diff_sulc, how ='outer')
df_diff = df_diff.merge(diff_dpf003, how ='outer')
df_diff = df_diff.merge(diff_dpfstar2, how ='outer')
df_diff = df_diff.rename(columns={'diff_fundicrest_curv': 'curv',
                                'diff_fundicrest_sulc': 'sulc',
                                'diff_fundicrest_dpf003' : 'dpf003'})

df_diff.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/stats/stats.csv'), index=False)
# T-test
#df = pg.read_dataset(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/stats/stats.csv'))
df_diff_stat = df_diff.drop(['subject', 'sessions'], axis=1)
df_dev_stat = df_dev.drop(['subject', 'sessions'], axis=1)
df_std_stat = df_std.drop(['subject', 'sessions'], axis=1)

stat_diff = pg.ptests(df_diff_stat, alternative = 'two-sided')
stat_dev = pg.ptests(df_dev_stat, alternative = 'two-sided')
stat_std = pg.ptests(df_std_stat, alternative = 'two-sided')

stat_diff.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/stats/stats_diff_2.csv'))
stat_dev.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/stats/stats_dev_2.csv'))
stat_std.to_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/stats/stats_std_2.csv'))


fig, ax = plt.subplots()

df_init = df_std_stat

df_mean = np.mean(df_init)[3:]
df_std = np.std(df_init)[3:]
df = pd.DataFrame({'mean': df_mean, 'std': df_std})
df = df.reset_index()

plt.scatter(df_mean , df_std)
for i, point in df.iterrows():
    ax.text(point['mean'], point['std'], str(point['index']))

ax.set_xlabel('mean metric std')
ax.set_ylabel('std metric std')
ax.set_title('optimal alpha for metric std')
plt.show()