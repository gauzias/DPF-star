import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pingouin as pg

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]


################## STATS

################ STATS
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



################# VIOLINPLOT

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


## select sub
subs = [
'CC00735XX18',
'CC00672AN13',
'CC00672BN13',
'CC00621XX11',
'CC00829XX21',
'CC00617XX15',
'CC00712XX11',
'CC00385XX15',
'CC00063AN06',
'CC00492AN15',
'CC00100XX01',
'CC00777XX19',
'CC00839XX23',
'KKI2009_113',
'KKI2009_142',
'KKI2009_505'
]

## seaborn
df1 = df_dev.set_index(['subject','sessions']).stack()
df1 = df1.reset_index()
df1 = df1.rename(columns={'level_2': 'depth', 0: 'Dev'})
df1['volume_hull'] = np.repeat(dataset_EXP1['volume_hull'].values, 21, axis=0)

df2 = df_std.set_index(['subject','sessions']).stack()
df2 = df2.reset_index()
df2 = df2.rename(columns={'level_2': 'depth', 0: 'StdCrest'})
df2['volume_hull'] = np.repeat(dataset_EXP1['volume_hull'].values, 21, axis=0)

df3 = df_diff.set_index(['subject','sessions']).stack()
df3 = df3.reset_index()
df3 = df3.rename(columns={'level_2': 'depth', 0: 'Sep'})
df3['volume_hull'] = np.repeat(dataset_EXP1['volume_hull'].values, 21, axis=0)


fig = plt.subplots(figsize = (7,14), sharex = True)
w1 = 6
w2 = 1
wf = 3*(w1+w2)

xlabels = ['curv', 'sulc', 'dpf', '0', '1', '5', '10', '50', '100', '150', '200', '250',
           '300', '400', '500', '600', '700', '800', '900', '1000', '2000']
# ax1
ax1 = plt.subplot2grid((wf, 1), (0, 0), rowspan=w1, colspan=1)
ax1.set_ylabel('angular deviation (°)')
ax1.axvspan(xmin=-1, xmax=2.5, facecolor='darkgray', alpha=0.3)
ax1.xaxis.tick_top()
#ax1.tick_params(axis='x', rotation=45, size=3)
ax1.set_xticklabels(xlabels, rotation=40, ha='left',rotation_mode='anchor')

#ax2
ax2 = plt.subplot2grid((wf, 1), (w1, 0), rowspan=w2, colspan=1, sharex=ax1)
ax2.set_ylim([0.38,0.42])
ax2.set(yticklabels=[])
ax2.tick_params(left = False, labelleft = False )
#ax2.set_ylabel('stat')

# ax3
ax3 = plt.subplot2grid((wf, 1), (w1 + w2, 0), rowspan=w1, colspan=1)
ax3.set_ylabel('normalised std on crest')
ax3.axvspan(xmin=-1, xmax=2.5, facecolor='darkgray', alpha=0.3)
ax3.set(xticklabels=[])

#ax4
ax4 = plt.subplot2grid((wf, 1), (2*(w1 + w2)-w2, 0), rowspan=w2, colspan=1, sharex=ax3)
ax4.set_ylim([0.38,0.42])
ax4.set(yticklabels=[])
ax4.tick_params(left = False, labelleft = False )
#ax4.set_ylabel('stat')

# ax5
ax5 = plt.subplot2grid((wf, 1), (2*(w1 + w2), 0), rowspan=w1, colspan=1)
ax5.set_ylabel('normalised median diff crest/fundi')
ax5.axvspan(xmin=-1, xmax=2.5, facecolor='darkgray', alpha=0.3)
ax5.set(xticklabels=[])

#ax6
ax6 = plt.subplot2grid((wf, 1), (3*(w1 + w2)-w2, 0), rowspan=w2, colspan=1, sharex=ax5)
ax6.set_ylim([0.38,0.42])
ax6.set(yticklabels=[])
ax6.tick_params(left = False, labelleft = False )
#ax6.set_ylabel('stat')

ax6.tick_params(axis='x', rotation=40, )

#S = ['curv', 'sulc', 'dpf']
#S2 = set([200])
#for tick in ax1.get_xticklabels():
#    print(tick)
#    if tick.get_text() in S:
#        print(tick.get_text())
##        tick.set_color('grey')
#    if tick.get_text() in S2:
#        print(tick.get_text())
#        tick.set_color('r')

#violin
f1= sns.violinplot(data=df1[df1['subject'].isin(subs)], y='Dev', x = 'depth', ax = ax1)
#ax1.text(6.5, 30, '*')
f1.set(xticklabels=[])
f1.set(xlabel=None)
f1.grid()

f2= sns.violinplot(data=df2[df2['subject'].isin(subs)], y='StdCrest', x = 'depth', ax = ax3)
#ax3.text(12.5, 0.16, 'opt')
f2.set(xticklabels=[])
f2.set(xlabel=None)  # remove the axis label
f2.grid()

f3= sns.violinplot(data=df3[df3['subject'].isin(subs)], y='Sep', x = 'depth', ax = ax5)
#ax5.text(19.5, 1.03, 'opt')
f3.set(xticklabels=[])
f3.set(xlabel=None)  # remove the axis label
f3.grid()

#stat

def pv2color(pv):
    #pv[pv == '***']='royalblue'
    #pv[pv == '**']='cornflowerblue'
    #pv[pv == '*']='lightsteelblue'
    #for idx, p in enumerate(pv) :
    #    if p not in ['royalblue', 'cornflowerblue','lightsteelblue']:
    #        pv[idx] = 'white'

    #pv[0:3]='white'

    pv[pv == '***']='white'
    pv[pv == '**']='white'
    pv[pv == '*']='white'
    for idx, p in enumerate(pv) :
        if p not in ['white']:
            pv[idx] = 'cornflowerblue'

    pv[0:3]='white'
    return  pv



def disstat(stat_df, ax, opt, split):
    x0 = [str(xx) for xx in stat_df.loc[opt].reset_index()['index'].values]
    y0 = np.repeat(0.4, len(alphas) + 3)
    v0 = np.concatenate([stat_df[opt].reset_index()[opt].values[0:split], stat_dev.loc[opt].reset_index()[opt].values[split:]])
    pv0 = pv2color(v0)
    s0 = sns.scatterplot(x0, y0, c=pv0, ax=ax, marker="_", linewidth = 4, s=350)
    s0.text(-0.9, 0.4, 'optimal band')
    x1 = [str(xx) for xx in stat_df.loc['curv'].reset_index()['index'].values]
    y1 =  np.repeat(0.3, len(alphas)+3)
    pv1 = pv2color(stat_df.loc['curv'].reset_index()['curv'].values)
    x2 = [str(xx) for xx in stat_df.loc['sulc'].reset_index()['index'].values]
    y2 =  np.repeat(0.2, len(alphas)+3)
    pv2 = pv2color(stat_df.loc['sulc'].reset_index()['sulc'].values)
    x3 = [str(xx) for xx in stat_df.loc['dpf003'].reset_index()['index'].values]
    y3 =  np.repeat(0.1, len(alphas)+3)
    pv3 = pv2color(stat_df.loc['dpf003'].reset_index()['dpf003'].values)
    #s1 = sns.scatterplot(x1, y1,  c= pv1, ax= ax)
    #s1.text(-0.9, 0.28, 'ttest / curv')
    #s2 = sns.scatterplot(x2, y2,  c= pv2, ax= ax)
    #s2.text(-0.9, 0.18, 'ttest / sulc')
    #s3 = sns.scatterplot(x3, y3,  c= pv3, ax= ax)
    #s3.text(-0.9, 0.08, 'ttest / dpf')

disstat(stat_dev, ax2, 50, 7)
disstat(stat_std, ax4, 400, 13)
disstat(stat_diff, ax6, 2000, 20)

plt.subplots_adjust( hspace=0)
#plt.tight_layout()
plt.show()



plt.savefig('/home/maxime/callisto/repo/paper_sulcal_depth/figEXP1.png')
