import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pingouin as pg


wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
alphas = [0, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.5]


#



######  VIOLINPLOT

dataset_EXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))
# load metrics
dev_dpf = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf/dpf_dev.csv'))
std_dpf = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf/dpf_std_crest.csv'))
diff_dpf = pd.read_csv(os.path.join(wd, 'data_EXP1/result_EXP1/metrics/dpf/dpf_diff_fundicrest.csv'))

df = dev_dpf.merge(std_dpf, how='outer')
df = df.merge(diff_dpf, how = 'outer')
df['volume_hull'] = np.repeat(dataset_EXP1['volume_hull'].values, len(alphas), axis=0)


df2 = df[df['alphas'].isin([ 0.0005, 0.003,0.03, 0.3, ])]

fig, ax = plt.subplots()
sns.lineplot(data = df2, x = 'volume_hull', y = 'std_crest_dpf', hue = 'alphas')

fig1, ax1 = plt.subplots()
sns.lineplot(data = df2, x = 'volume_hull', y = 'diff_fundicres_dpf', hue = 'alphas')

fig2, ax2 = plt.subplots()
sns.lineplot(data = df2, x = 'volume_hull', y = 'angle_dpf', hue = 'alphas')

plt.show()