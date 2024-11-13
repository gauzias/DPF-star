import os
import numpy as np
import slam.differential_geometry as sdiff
import slam.curvature as scurv
import slam.io as sio
import slam.texture as stex
import tools.depth as depth

def rescale_mesh(mesh, scale):
    mesh.vertices = np.sqrt(scale) * mesh.vertices
    return mesh


wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
subjects = ['CC00735XX18', 'CC00712XX11', 'CC00777XX19', 'KKI2009-113']
sessions = ['222201', '221400', '239102', 'MR1']

volclass = [32500, 65000, 130000, 260000, 520000, 1040000, 1755000, 4160000, 8125000]

alphas = [10000]

for idx, sub in enumerate(subjects):
    print(sub)
    ses = sessions[idx]
    for vol in volclass:
        # import mesh
        mesh_name =  sub + '_' + ses + '_' + str(vol) + '.gii'
        mesh_path = os.path.join(wd, 'data','scaled_meshes', sub + '_' + ses ,mesh_name)
        mesh = sio.load_mesh(mesh_path)
        # compute curv
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        # compute dpf star 100
        dpfstar100 = depth.dpfstar(mesh, mean_curv, alphas)
        # save
        sio.write_texture(stex.TextureND(darray = mean_curv), os.path.join(wd,'data','scaled_meshes', sub + '_' + ses + '_' + str(vol) + '_curv.gii' ))
        sio.write_texture(stex.TextureND(darray = dpfstar100), os.path.join(wd,'data','scaled_meshes' , sub + '_' + ses + '_' + str(vol) + '_dpfstar100.gii'))


