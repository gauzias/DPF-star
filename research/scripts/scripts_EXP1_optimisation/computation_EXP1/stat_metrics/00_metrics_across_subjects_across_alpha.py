from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# Define working directory
wd = Path('D:/Callisto/repo/paper_sulcal_depth')
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]

# Load datasets
dataset_EXP1 = pd.read_csv(wd / 'datasets' / 'dataset_EXP1.csv')

# Load DEV metrics
dev_curv = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'curv' / 'curv_dev.csv')
dev_sulc = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'sulc' / 'sulc_dev.csv')
dev_dpf003 = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'dpf003' / 'dpf003_dev.csv')

dev_dpfstar = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'dpfstar' / 'dpfstar_dev.csv')
dev_dpfstar2 = dev_dpfstar.pivot(index='subject', columns='alphas', values='angle_dpfstar').reset_index()

# Load STD metrics
std_curv = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'curv' / 'curv_std_crest.csv')
std_sulc = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'sulc' / 'sulc_std_crest.csv')
std_dpf003 = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'dpf003' / 'dpf003_std_crest.csv')

std_dpfstar = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'dpfstar' / 'dpfstar_std_crest.csv')
std_dpfstar2 = std_dpfstar.pivot(index='subject', columns='alphas', values='std_crest_dpfstar').reset_index()

# Load DIFF metrics
diff_curv = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'curv' / 'curv_diff_fundicrest.csv')
diff_sulc = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'sulc' / 'sulc_diff_fundicrest.csv')
diff_dpf003 = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'dpf003' / 'dpf003_diff_fundicrest.csv')

diff_dpfstar = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'dpfstar' / 'dpfstar_diff_fundicrest.csv')
diff_dpfstar2 = diff_dpfstar.pivot(index='subject', columns='alphas', values='diff_fundicrest_dpfstar').reset_index()

# Merge and rename columns
def prepare_df(curv, sulc, dpf003, dpfstar, rename_map):
    df = curv.merge(sulc, on=['subject', 'sessions'], how='outer')
    df = df.merge(dpf003, on=['subject', 'sessions'], how='outer')
    df = df.merge(dpfstar, on='subject', how='outer')
    return df.rename(columns=rename_map)

rename_dev = {'angle_curv': 'curv', 'angle_sulc': 'sulc', 'angle_dpf003': 'dpf003'}
rename_std = {'std_crest_curv': 'curv', 'std_crest_sulc': 'sulc', 'std_crest_dpf003': 'dpf003'}
rename_diff = {'diff_fundicrest_curv': 'curv', 'diff_fundicrest_sulc': 'sulc', 'diff_fundicrest_dpf003': 'dpf003'}

df_dev = prepare_df(dev_curv, dev_sulc, dev_dpf003, dev_dpfstar2, rename_dev)
df_std = prepare_df(std_curv, std_sulc, std_dpf003, std_dpfstar2, rename_std)
df_diff = prepare_df(diff_curv, diff_sulc, diff_dpf003, diff_dpfstar2, rename_diff)


# Ensure subdirectories exist
(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon').mkdir(parents=True, exist_ok=True)


# Save combined DIFF data
df_dev.to_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_dev.csv', index=False)
df_std.to_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_std.csv', index=False)
df_diff.to_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_diff.csv', index=False)

# Wilcoxon tests
#def perform_wilcoxon(df):
#    return pg.wilcoxon(df.drop(columns=['subject', 'sessions']), alternative='two-sided')

#stat_diff = perform_wilcoxon(df_diff)
#stat_dev = perform_wilcoxon(df_dev)
#stat_std = perform_wilcoxon(df_std)

# Save stats
#(stat_diff.to_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_diff_2.csv'))
#(stat_dev.to_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_dev_2.csv'))
#(stat_std.to_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_std_2.csv'))