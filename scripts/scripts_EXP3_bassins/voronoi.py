
import slam.io as sio
import os
import tools.voronoi as tv
import slam.texture as stex


wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
alpha = 500
alpha_str = '500'

subjects = ['sub-CC00530XX11', 'sub-CC00530XX11', 'sub-CC00618XX16',
       'sub-CC00629XX19', 'sub-CC00634AN16', 'sub-CC00666XX15',
       'sub-CC00672BN13', 'sub-CC00694XX19', 'sub-CC00718XX17',
       'sub-CC00796XX22', 'sub-CC00136AN13', 'sub-CC00227XX13',
       'sub-CC00389XX19', 'sub-CC00518XX15', 'sub-CC00526XX15',
       'sub-CC00576XX16', 'sub-CC00621XX11', 'sub-CC00657XX14',
       'sub-CC00661XX10', 'sub-CC00728AN19', 'sub-CC00070XX05',
       'sub-CC00072XX07', 'sub-CC00136BN13', 'sub-CC00140XX09',
       'sub-CC00152AN04', 'sub-CC00154XX06', 'sub-CC00169XX13',
       'sub-CC00177XX13', 'sub-CC00248XX18', 'sub-CC00281BN10',
       'sub-CC00053XX04', 'sub-CC00059XX10', 'sub-CC00060XX03',
       'sub-CC00063AN06', 'sub-CC00063BN06', 'sub-CC00064XX07',
       'sub-CC00067XX10', 'sub-CC00071XX06', 'sub-CC00074XX09',
       'sub-CC00075XX10', 'sub-CC00051XX02', 'sub-CC00052XX03',
       'sub-CC00054XX05', 'sub-CC00056XX07', 'sub-CC00057XX08',
       'sub-CC00062XX05', 'sub-CC00065XX08', 'sub-CC00066XX09',
       'sub-CC00068XX11', 'sub-CC00069XX12', 'sub-CC00055XX06',
       'sub-CC00080XX07', 'sub-CC00119XX12', 'sub-CC00120XX05',
       'sub-CC00130XX07', 'sub-CC00134XX11', 'sub-CC00135AN12',
       'sub-CC00135BN12', 'sub-CC00136BN13', 'sub-CC00137XX14',
       'sub-CC00058XX09', 'sub-CC00167XX11', 'sub-CC00168XX12',
       'sub-CC00200XX02', 'sub-CC00218AN12', 'sub-CC00284AN13',
       'sub-CC00286XX15', 'sub-CC00290XX11', 'sub-CC00316XX11',
       'sub-CC00335XX14', 'sub-CC00194XX14', 'sub-CC00883XX18',
       'sub-CC00886XX21', 'sub-CC00986BN22']
subjects = [su.split('sub-')[1] for su in subjects]

sessions =['152300', '153600', '177201', '182000', '184100', '198200',
       '200000', '201800', '210400', '245100', '45100', '76601', '119100',
       '145700', '150500', '163200', '177900', '193700', '195801',
       '214100', '26700', '27600', '45000', '46800', '49200', '50700',
       '55500', '58500', '83000', '90500', '8607', '11900', '12501',
       '15102', '15104', '18303', '20200', '27000', '28000', '28400',
       '7702', '8300', '8800', '10700', '11002', '13801', '18600',
       '19200', '20701', '26300', '9300', '30300', '39400', '41600',
       '44001', '44600', '54400', '54500', '64300', '45200', '11300',
       '55600', '55700', '67204', '85900', '111400', '91700', '92900',
       '101300', '106300', '65401', '14430', '18030', '41830']


nbs = len(subjects)
# loop over dHCP subject
for idx, sub in enumerate(subjects) :
    print(sub, idx ,'/', nbs)
    ses = sessions[idx]
    # load mesh

    mesh_name  = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
    mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses,
                               'anat', )
    mesh_path = os.path.join(mesh_folder, mesh_name)

    mesh = sio.load_mesh(mesh_path)
        #if not os.path.exists(os.path.join(wd, 'data_EXP3/result_EXP3/voronoi_textures', sub + '_' + ses + '_voronoi.gii')):
    voronoi = tv.voronoi_de_papa(mesh)
    sio.write_texture(stex.TextureND(darray=voronoi),
                          os.path.join(wd, 'data_EXP3/result_EXP3/voronoi_textures_test', sub + '_' + ses + '_voronoi.gii'))