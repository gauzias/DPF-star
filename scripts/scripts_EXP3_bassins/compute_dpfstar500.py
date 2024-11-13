
import os
import slam.io as sio
import tools.depth as depth
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

subjects2 = ['KKI2009_800', 'KKI2009_239',
       'KKI2009_505', 'KKI2009_679', 'KKI2009_934', 'KKI2009_113',
       'KKI2009_422', 'KKI2009_815', 'KKI2009_906', 'KKI2009_127',
       'KKI2009_742', 'KKI2009_849', 'KKI2009_913', 'KKI2009_346',
       'KKI2009_502', 'KKI2009_814', 'KKI2009_916', 'KKI2009_959',
       'KKI2009_142', 'KKI2009_656']

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

sessions2 = ['MR1',
       'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1',
       'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1', 'MR1',
       'MR1']


nbs = len(subjects)
nbs2 = len(subjects2)
# loop over dHCP
print(".....COMPUTE DPFSTAR 500.....")

for idx, sub in enumerate(subjects2):
    print(sub)
    print(idx,'/' ,nbs2)
    #load mesh
    ses = sessions2[idx]

    # dHCP
    #mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
    #mesh_folder = os.path.join('/media/maxime/Expansion/rel3_dHCP/', 'sub-' + sub, 'ses-' + ses, 'anat', )
    #mesh_path = os.path.join(mesh_folder, mesh_name)

    #KKI
    mesh_name = 'lh.white.gii'
    mesh_path = os.path.join('/media/maxime/Expansion/FS_database_KKI_test_retest_FS6.0',
                             sub + '_' + ses , 'surf', mesh_name)


    mesh = sio.load_mesh(mesh_path)
    # check if already exist and compute
    fname_K1 = sub + '_' + ses + '_K1.gii'
    fname_K2 = sub + '_' + ses + '_K2.gii'
    folder_curv = os.path.join(wd, 'data/result_EXP3/curvature')

    fname_dpfstar = sub + '_' + ses + '_dpfstar'+ alpha_str + '.gii'
    folder_dpfstar = os.path.join(wd, 'data/result_EXP3/dpfstar' + alpha_str)

    if not os.path.exists(folder_dpfstar):
           os.makedirs(folder_dpfstar)

    # load curv
    try :
        K1 = sio.load_texture(os.path.join(folder_curv, fname_K1)).darray[0]
        K2 = sio.load_texture(os.path.join(folder_curv, fname_K2)).darray[0]
    except:
        K1, K2 = depth.curv(mesh)
        sio.write_texture(stex.TextureND(darray=K1), os.path.join(folder_curv, fname_K1))
        sio.write_texture(stex.TextureND(darray=K2), os.path.join(folder_curv, fname_K2))
    curv = 0.5 * (K1 + K2)
    # set alphas
    alphas_dpfstar = [alpha]
    # compute dpfstar
    dpfstar = depth.dpfstar(mesh, curv, alphas_dpfstar)
    # save dpfstar
    sio.write_texture(stex.TextureND(darray=dpfstar), os.path.join(folder_dpfstar, fname_dpfstar))


