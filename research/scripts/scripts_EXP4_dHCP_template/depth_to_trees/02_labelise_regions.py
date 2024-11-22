import numpy as np
import slam.io as sio
import os
import slam.texture as stex
import slam.topology as slt

# paths
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
folder_dHCP = '/media/maxime/Expansion/rel3_dHCP'

# init sub
#sub = 'CC00672BN13'
#ses = '200000'

sub = 'CC00576XX16'
ses = '163200'

# load mesh
mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat')
mesh_path = os.path.join(mesh_folder, mesh_name)
mesh = sio.load_mesh(mesh_path)

# load depth map
dpf500_path = os.path.join('/home/maxime/callisto/repo/paper_sulcal_depth/data_EXP3/result_EXP3/dpfstar500',
                           sub + '_' + ses + '_dpfstar500.gii')

depth = sio.load_texture(dpf500_path).darray[0]

# discrtesise depth map
min_depth = np.min(depth)
max_depth = np.max(depth)
depth_bin = np.linspace(min_depth, max_depth, 50)
isolines = np.digitize(depth, bins=depth_bin)

# load cimmulative sulci

isolines_path = os.path.join(wd, 'data_EXP4/result_EXP4', sub + '_' + ses + '_cumulative_sulci.gii')
cumulative_iso = sio.load_texture(isolines_path).darray

###
labels = list()
offlab = 0
for i in np.arange(len(np.unique(isolines))):
    print(i)
    texture = cumulative_iso[i]

    label = texture
    lab = 2
    vert_neg = [idx for idx, val in enumerate(texture) if val == 1]
    texture = texture.astype(bool)

    nb_vertex = len(texture)
    adj_matrix = slt.adjacency_matrix(mesh)

    for vertex in vert_neg:
        # print('current vertex : ',vertex)

        neigh = adj_matrix.indices[
                adj_matrix.indptr[vertex]:adj_matrix.indptr[vertex + 1]]
        neigh = neigh.astype(int)
        neigh_tex = [texture[i] for i in neigh]  # tell if neigh are negativ (true) or positiv(false)
        neigh_neg = neigh[neigh_tex]  # keep only the neig with curv negativ
        neigh_lab = [label[i] for i in neigh_neg]
        neigh_lab = [nl for nl in neigh_lab if nl != 0]
        neigh_lab = [nl for nl in neigh_lab if nl != 1]
        if len(neigh_lab) == 0:
            label[vertex] = lab
            lab = lab + 1
        if len(neigh_lab) != 0:
            neig_lab_unique = np.unique(neigh_lab)
            # print('not nul', neig_lab_unique)
            label[vertex] = np.min(neig_lab_unique)
            label = np.where(np.isin(label, neig_lab_unique), np.min(neig_lab_unique), label)

    #label_unique = np.sort(np.unique(label))
    #for i, l in enumerate(label_unique):
    #    label = np.where(label == l, i, label)
    label = np.digitize(label, bins = np.unique(label))-1
    # add offset
    label = np.digitize(label, bins = np.unique(label))-1 + offlab
    label[label == offlab] = 0
    print(np.unique(label))
    offlab = np.max(label)
    labels.append(label)

save_folder = os.path.join(wd, 'data_EXP4/result_EXP4',sub + '_' + ses )
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
fname = sub + '_' + ses + '_sulci_extraction.gii'
sio.write_texture(stex.TextureND(darray=labels), os.path.join( os.path.join(save_folder
                                                                           , fname)))



