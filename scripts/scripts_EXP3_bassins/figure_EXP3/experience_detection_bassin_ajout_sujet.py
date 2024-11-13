import os
import slam.io as sio
import slam.texture as stex
import tools.voronoi as tv
import tools.depth as depth
wd = '/home/maxime/callisto/repo/paper_sulcal_depth'


subjects = ["CC00735XX18","CC00672AN13", "CC00672BN13", "CC00621XX11", "CC00829XX21",
            "CC00617XX15", "CC00712XX11", "CC00385XX15","CC00063AN06"]

sessions = ["222201", "197601" , "200000", "177900", "17610",  "176500", "221400" ,"118500" ,"15102"]

nbs = len(subjects)
"""

for idx, sub in enumerate(subjects) :
    print(sub, idx ,'/', nbs)
    ses = sessions[idx]
    # load mesh
    mesh_name  = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_folder = os.path.join(wd, 'data/meshes')
    mesh_path = os.path.join(mesh_folder, mesh_name)

    mesh = sio.load_mesh(mesh_path)
    voronoi = tv.voronoi_de_papa(mesh)
    sio.write_texture(stex.TextureND(darray=voronoi),
                          os.path.join(wd, 'data/rel3/voronoi', sub + '_' + ses + '_voronoi.gii'))
"""

for idx, sub in enumerate(subjects) :
    print(sub, idx ,'/', nbs)
    ses = sessions[idx]
    # load mesh
    mesh_name  = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_folder = os.path.join(wd, '../../data/meshes')
    mesh_path = os.path.join(mesh_folder, mesh_name)
    mesh = sio.load_mesh(mesh_path)
    # load curvature
    K1_path = os.path.join(wd, '../../data/subject_analysis', sub + '_' + ses, 'surface_processing', 'curvature',
                           sub + '_' + ses + '_K1.gii')
    K2_path = os.path.join(wd, '../../data/subject_analysis', sub + '_' + ses, 'surface_processing', 'curvature',
                           sub + '_' + ses + '_K2.gii')
    K1 = sio.load_texture(K1_path).darray[0]
    K2 = sio.load_texture(K2_path).darray[0]
    curv = 0.5 * (K1 + K2)

    # curv = np.abs(curv)

    dpfstar = depth.dpfstar(mesh, curv, [100])
    sio.write_texture(stex.TextureND(darray=dpfstar), os.path.join(wd, '../../data/rel3/dpf100', sub + '_' + ses + '_dpf100.gii'))

