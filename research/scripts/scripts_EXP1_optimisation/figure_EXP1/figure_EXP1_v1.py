from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

# Working directory and constants
wd = Path('D:/Callisto/repo/paper_sulcal_depth')
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]

# Paths
path_dataset = wd / 'datasets' / 'dataset_EXP1.csv'
path_metrics = wd / 'data_EXP1' / 'result_EXP1' / 'metrics'
path_output_stats = path_metrics / 'stats'

# Load datasets
load_metric = lambda *parts: pd.read_csv(path_metrics.joinpath(*parts))

dataset_EXP1 = pd.read_csv(path_dataset)

# Load DEV metrics
dev_curv = load_metric('curv', 'curv_dev.csv')
dev_sulc = load_metric('sulc', 'sulc_dev.csv')
dev_dpf003 = load_metric('dpf003', 'dpf003_dev.csv')
dev_dpfstar = load_metric('dpfstar', 'dpfstar_dev.csv')
dev_dpfstar2 = dev_dpfstar.pivot_table(index='subject', columns='alphas', values='angle_dpfstar').reset_index()

# Load STD metrics
std_curv = load_metric('curv', 'curv_std_crest.csv')
std_sulc = load_metric('sulc', 'sulc_std_crest.csv')
std_dpf003 = load_metric('dpf003', 'dpf003_std_crest.csv')
std_dpfstar = load_metric('dpfstar', 'dpfstar_std_crest.csv')
std_dpfstar2 = std_dpfstar.pivot_table(index='subject', columns='alphas', values='std_crest_dpfstar').reset_index()

# Load DIFF metrics
diff_curv = load_metric('curv', 'curv_diff_fundicrest.csv')
diff_sulc = load_metric('sulc', 'sulc_diff_fundicrest.csv')
diff_dpf003 = load_metric('dpf003', 'dpf003_diff_fundicrest.csv')
diff_dpfstar = load_metric('dpfstar', 'dpfstar_diff_fundicrest.csv')
diff_dpfstar2 = diff_dpfstar.pivot_table(index='subject', columns='alphas', values='diff_fundicrest_dpfstar').reset_index()

# Merge and rename function
def merge_and_rename(df1, df2, df3, df4, rename_dict):
    df = df1.merge(df2, how='outer').merge(df3, how='outer').merge(df4, how='outer')
    return df.rename(columns=rename_dict)

# Combine metrics
df_dev = merge_and_rename(dev_curv, dev_sulc, dev_dpf003, dev_dpfstar2,
                          {'angle_curv': 'curv', 'angle_sulc': 'sulc', 'angle_dpf003': 'dpf003'})
df_std = merge_and_rename(std_curv, std_sulc, std_dpf003, std_dpfstar2,
                          {'std_crest_curv': 'curv', 'std_crest_sulc': 'sulc', 'std_crest_dpf003': 'dpf003'})
df_diff = merge_and_rename(diff_curv, diff_sulc, diff_dpf003, diff_dpfstar2,
                           {'diff_fundicrest_curv': 'curv', 'diff_fundicrest_sulc': 'sulc', 'diff_fundicrest_dpf003': 'dpf003'})

# Save merged data
df_diff.to_csv(path_output_stats / 'stats.csv', index=False)

# Run stats
df_diff_stat = df_diff.drop(columns=['subject', 'sessions'])
df_dev_stat = df_dev.drop(columns=['subject', 'sessions'])
df_std_stat = df_std.drop(columns=['subject', 'sessions'])

stat_diff = pg.wilcoxon(df_diff_stat, alternative='two-sided')
stat_dev = pg.wilcoxon(df_dev_stat, alternative='two-sided')
stat_std = pg.wilcoxon(df_std_stat, alternative='two-sided')

# Prepare violinplot data
def melt_df(df, value_name):
    df_melted = df.set_index(['subject', 'sessions']).stack().reset_index()
    df_melted.columns = ['subject', 'sessions', 'depth', value_name]
    df_melted['volume_hull'] = np.repeat(dataset_EXP1['volume_hull'].values, 21)
    return df_melted

df1 = melt_df(df_dev, 'Dev')
df2 = melt_df(df_std, 'StdCrest')
df3 = melt_df(df_diff, 'Sep')

# Subjects to select
subs = [
    'CC00735XX18', 'CC00672AN13', 'CC00672BN13', 'CC00621XX11', 'CC00829XX21',
    'CC00617XX15', 'CC00712XX11', 'CC00385XX15', 'CC00063AN06', 'CC00492AN15',
    'CC00100XX01', 'CC00777XX19', 'CC00839XX23', 'KKI2009_113', 'KKI2009_142', 'KKI2009_505']

# Plotting
fig = plt.figure(figsize=(7, 14))
w1, w2 = 6, 1
wf = 3 * (w1 + w2)

def create_ax(pos, row, label, ylabel):
    ax = plt.subplot2grid((wf, 1), (pos, 0), rowspan=row)
    ax.axvspan(-1, 2.5, facecolor='darkgray', alpha=0.3)
    ax.set_ylabel(ylabel)
    if 'top' in label:
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=40, ha='left', rotation_mode='anchor')
    else:
        ax.set(xticklabels=[])
    return ax

xlabels = ['curv', 'sulc', 'dpf'] + [str(a) for a in alphas]

ax1 = create_ax(0, w1, 'top', 'angular deviation (°)')
ax2 = create_ax(w1, w2, '', '')
ax2.set_ylim([0.38, 0.42])
ax2.set_yticklabels([])
ax2.tick_params(left=False, labelleft=False)

ax3 = create_ax(w1 + w2, w1, '', 'normalised std on crest')
ax4 = create_ax(2 * (w1 + w2) - w2, w2, '', '')
ax4.set_ylim([0.38, 0.42])
ax4.set_yticklabels([])
ax4.tick_params(left=False, labelleft=False)

ax5 = create_ax(2 * (w1 + w2), w1, '', 'normalised median diff crest/fundi')
ax6 = create_ax(3 * (w1 + w2) - w2, w2, '', '')
ax6.set_ylim([0.38, 0.42])
ax6.set_yticklabels([])
ax6.tick_params(left=False, labelleft=False, axis='x', rotation=40)

# Violin plots
sns.violinplot(data=df1[df1['subject'].isin(subs)], y='Dev', x='depth', ax=ax1).grid()
sns.violinplot(data=df2[df2['subject'].isin(subs)], y='StdCrest', x='depth', ax=ax3).grid()
sns.violinplot(data=df3[df3['subject'].isin(subs)], y='Sep', x='depth', ax=ax5).grid()

# Stat markers

def pv2color(pv):
    return ['white' if p in ['*', '**', '***'] else 'cornflowerblue' for p in pv]

def disstat(stat_df, ax, opt, split):
    x0 = stat_df.loc[opt].reset_index()['index'].astype(str).values
    y0 = np.repeat(0.4, len(x0))
    v0 = np.concatenate([
        stat_df.loc[opt].reset_index()[opt].values[:split],
        stat_dev.loc[opt].reset_index()[opt].values[split:]
    ])
    pv0 = pv2color(v0)
    ax.scatter(x0, y0, c=pv0, marker='_', linewidths=4, s=350)
    ax.text(-0.9, 0.4, 'optimal band')

disstat(stat_dev, ax2, 50, 7)
disstat(stat_std, ax4, 400, 13)
disstat(stat_diff, ax6, 2000, 20)

plt.subplots_adjust(hspace=0)
plt.savefig(wd / 'figEXP1.png')
plt.show()
