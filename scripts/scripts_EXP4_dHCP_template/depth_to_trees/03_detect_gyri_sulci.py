import slam.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'

# init sub
#sub = 'CC00672BN13'
#ses = '200000'

sub = 'CC00576XX16'
ses = '163200'


# load voronoi

voronoi = sio.load_texture(os.path.join(wd, 'data_EXP3/result_EXP3/voronoi_textures',sub + '_' + ses + '_voronoi.gii'))
voronoi = voronoi.darray[0]

# load curvature

K1 = sio.load_texture(os.path.join(wd, 'data_EXP3/result_EXP3/curvature', sub + '_' + ses + '_K1.gii')).darray[0]
K2 = sio.load_texture(os.path.join(wd, 'data_EXP3/result_EXP3/curvature', sub + '_' + ses + '_K2.gii')).darray[0]
curv = 0.5 * (K1 + K2)


# import levels
save_folder = os.path.join(wd, 'data_EXP4/result_EXP4',sub + '_' + ses )
level_s = sio.load_texture(os.path.join( os.path.join(save_folder,
                                                      sub + '_' + ses + '_sulci_extraction.gii')))
level_s = level_s.darray

level_g = sio.load_texture(os.path.join( os.path.join(save_folder,
                                                      sub + '_' + ses + '_gyri_extraction.gii')))
level_g = level_g.darray

mean_mean_sulci = list()
mean_mean_gyri = list()
mean_vor_sulci = list()
mean_vor_gyri = list()

for idx, lvls in enumerate(level_s) :
    lvlg = level_g[idx]

    labels = np.unique(lvls)
    labels = labels[labels != 0]

    labelg = np.unique(lvlg)
    labelg = labelg[labelg != 0]

    mean_sulci = list()
    vor_sulci = list()
    for lab in labels :
        mean_sulci.append(np.mean(curv[lvls == lab]))
        vor_sulci.append(np.sum(voronoi[lvls == lab]))

    mean_gyri = list()
    vor_gyri = list()
    for lab in labelg :
        mean_gyri.append(np.mean(curv[lvlg == lab]))
        vor_gyri.append(np.sum(voronoi[lvlg == lab]))

    mean_mean_sulci.append(np.mean(mean_sulci))
    mean_mean_gyri.append(np.mean(mean_gyri))
    mean_vor_sulci.append(np.mean(vor_sulci))
    mean_vor_gyri.append(np.mean(vor_gyri))

diff = [np.abs(gy - sul) for (gy, sul) in zip(mean_mean_gyri, mean_mean_sulci)]

### number of gyri  / sulci
nb_sulci = list()
nb_gyri = list()
for idx, lvls in enumerate(level_s):
    lvlg = level_g[idx]

    labels = np.unique(lvls)
    labels = labels[labels != 0]

    labelg = np.unique(lvlg)
    labelg = labelg[labelg != 0]

    nb_sulci.append(len(labels))
    nb_gyri.append(len(labelg))



### figure
fig, [ax,ax2,ax4] = plt.subplots(3,1, sharex=True)

ax.plot(np.arange(len(level_s)), mean_mean_sulci, label = 'sulci')
ax.plot(np.arange(len(level_g)), mean_mean_gyri, label = 'gyri')
ax.plot(np.arange(len(level_g)), diff, label = 'diff')
ax.set_xticks(np.arange(len(level_s)))
ax.grid()
ax.legend()
ax.set_ylabel('mean mean curvature')
ax.set_xlabel('min <- depth level -> max')
ax.set_title('mean mean-curvature sulci and gyri per depth level')
plt.tight_layout()


sumgyrifundi = np.array(nb_gyri) + np.array(nb_sulci)
#fig2, ax2 = plt.subplots()
ax2.plot(np.arange(len(level_s)), nb_sulci, label = 'sulci')
ax2.plot(np.arange(len(level_g)), nb_gyri, label = 'gyri')
ax2.plot(np.arange(len(level_g)), np.array(nb_gyri) + np.array(nb_sulci), label = 'gyri+sulci')
ax2.set_xticks(np.arange(len(level_s)))
ax2.grid()
ax2.legend()
ax2.set_ylabel('number')
ax2.set_xlabel('min <- depth level -> max')
ax2.set_title('number of sulci and gyri per depth level')
plt.tight_layout()


#fig4, ax4 = plt.subplots()
ax4.plot(np.arange(len(level_s)), mean_vor_sulci, label = 'sulci')
ax4.plot(np.arange(len(level_g)), mean_vor_gyri, label = 'gyri')
ax4.plot(np.arange(len(level_g)), np.abs(np.array(mean_vor_gyri) - np.array(mean_vor_sulci)), label = 'diff')
ax4.set_xticks(np.arange(len(level_s)))
ax4.grid()
ax4.legend()
ax4.set_ylabel('mean area')
ax4.set_xlabel('min <- depth level -> max')
ax4.set_title('mean area sulci and gyri per depth level')

plt.tight_layout()
plt.show()






