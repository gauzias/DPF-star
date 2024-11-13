import os
import slam.io as sio
import numpy as np
import slam.texture as stex

# load mesh 1, 2, 3

#sub = 735 #1.3     # 65 000
#sub = 712 # 1.5    # 130 000
#sub = 777 # 1.75   # 260 000
#sub = KKI 113 # 2.0 #  520 000


def rescale_mesh (mesh, scale, saving_path, fname) :
    mesh.vertices = scale * mesh.vertices
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    sio.write_mesh(mesh, os.path.join(saving_path,fname))

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
subjects = ['CC00735XX18', 'CC00712XX11', 'CC00777XX19', 'KKI2009-113']
sessions = ['222201', '221400', '239102', 'MR1']

# import mesh

volclass = [32500, 65000, 130000, 260000, 520000, 1040000, 1755000, 4160000, 8125000]


#offset = [-5, 10, 60, 110, 180, 300, 400, 550, 720]


#  generate mesh scaled

for idx, sub in enumerate(subjects) :
    print(sub)
    ses = sessions[idx]
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(wd, 'data','meshes', mesh_name)
    for jdx, vc in enumerate(volclass):
        print(vc)
        #off = offset[jdx]
        saving_path = os.path.join(wd, 'data','scaled_meshes', sub + '_' + ses)
        fname = sub + '_' + ses + '_' + str(vc) + '.gii'

        mesh = sio.load_mesh(mesh_path)
        vol = mesh.convex_hull.volume
        coef = np.power(vc / vol, 1 / 3)
        mesh.vertices = coef * mesh.vertices
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        sio.write_mesh(mesh, os.path.join(saving_path, fname))



# compute sulc

