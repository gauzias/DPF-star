import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
""""
author : maxime.dieudonne@univ-amu.fr
date : 26/07/2023

This script creates the csv file that contains info relative to the dataset 2:
full dHCP rel3 + KKI repro database

"""


wd = '/home/maxime/callisto/repo/paper_sulcal_depth'

dataset_EXP1 = pd.read_csv(os.path.join(wd, 'datasets/dataset_EXP1.csv'))



length_area = np.power(dataset_EXP1['surface_area'].values, 1/2)
length_volume  = np.power(dataset_EXP1['volume'].values, 1/3)
length_volume_hull  = np.power(dataset_EXP1['volume_hull'].values, 1/3)

fig, ax = plt.subplots()

f1 = sns.scatterplot(data = dataset_EXP1, x = 'scan_age_days', y = length_volume, label = 'volume')
f2 = sns.scatterplot(data = dataset_EXP1, x = 'scan_age_days', y = length_area, label = 'surface')
f2 = sns.scatterplot(data = dataset_EXP1, x = 'scan_age_days', y = length_volume_hull, label = 'volume_hull')

ax.legend()

fig2, ax2 = plt.subplots(2,1)
g1 = sns.histplot(data = dataset_EXP1, x= 'volume', ax = ax2[0])
g2 = sns.histplot(data = dataset_EXP1, x= 'surface_area', ax = ax2[1])
ax2[0].set_title('dataset EXP1')
plt.show()

fig3, ax3 = plt.subplots()
h1 = sns.scatterplot(data = dataset_EXP1, x= 'participant_id', y = 'volume')
ax3.grid()
plt.yticks(np.arange(25000, 300000, 25000))
plt.xticks(rotation=45)
plt.show()




