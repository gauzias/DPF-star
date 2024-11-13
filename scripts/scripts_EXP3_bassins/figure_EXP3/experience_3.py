import os
import pandas as pd
import numpy as np
import slam.io as sio
import settings.path_manager as pm
import tools.voronoi as tv
import matplotlib.pyplot as plt
import slam.texture as stex

# load sulc

wd = '/home/maxime/callisto/repo/paper_sulcal_depth'
dir_sulcKKI = '/home/maxime/callisto/data/outputs_mirtkdeformmesh_KKIFS6'
dir_KKI = '/home/maxime/callisto/data/databases_copied_from_hpc/REPRO_database/FS_database_KKI_test_retest_FS6.0'
dir_dHCP = '/media/maxime/DATA/rel3_dhcp_anat_pipeline'


#init dataframe

#df = pd.DataFrame(dict())



# load list sub and session dHCP
sub_info_dhcp = pd.read_csv(os.path.join(wd, '../../data/info_database', 'dhcp_rel3_neurotypical.csv'))
sub_dhcp = sub_info_dhcp['participant_id'].values
ses_dhcp = sub_info_dhcp['session_id'].values
ses_dhcp = [str(se) for se in ses_dhcp]

list_sub_error = ['CC00284BN13', 'CC00286XX15', 'CC00287BN16', 'CC00287AN16', 'CC00287BN16', 'CC00289XX18',
                  'CC00290XX11', 'CC00292XX13', 'CC00293AN14', 'CC00293BN14', 'CC00295XX16', 'CC00298XX19',
                  'CC00299XX20', 'CC00300XX03', 'CC00301XX04', 'CC00301XX04', 'CC00302XX05', 'CC00303XX06', 'CC00304XX07',
                  'CC00305XX08', 'CC00305XX08', 'CC00306XX09', 'CC00307XX10', 'CC00308XX11', 'CC00309BN12', 'CC00313XX08',
                  'CC00314XX09', 'CC00316XX11', 'CC00319XX14', 'CC00320XX07', 'CC00322XX09', 'CC00324XX11', 'CC00325XX12',
                  'CC00326XX13', 'CC00326XX13', 'CC00328XX15', 'CC00329XX16', 'CC00330XX09', 'CC00332XX11', 'CC00334XX13',
                  'CC00335XX14', 'CC00336XX15', 'CC00337XX16', 'CC00338AN17', 'CC00338BN17', 'CC00339XX18', 'CC00340XX11',
                  'CC00341XX12', 'CC00343XX14', 'CC00344XX15', 'CC00345XX16', 'CC00346XX17', 'CC00347XX18', 'CC00348XX19',
                  'CC00349XX20', 'CC00351XX05', 'CC00352XX06', 'CC00354XX08', 'CC00355XX09', 'CC00356XX10', 'CC00357XX11',
                  'CC00358XX12', 'CC00361XX07', 'CC00361XX07', 'CC00362XX08', 'CC00363XX09', 'CC00364XX10', 'CC00367XX13',
                  'CC00368XX14', 'CC00370XX08', 'CC00371XX09', 'CC00375XX13', 'CC00376XX14', 'CC00377XX15', 'CC00378XX16',
                  'CC00380XX10', 'CC00381XX11', 'CC00382XX12', 'CC00383XX13', 'CC00385XX15', 'CC00385XX15', 'CC00388XX18',
                  'CC00389XX19', 'CC00389XX19', 'CC00400XX04', 'CC00401XX05', 'CC00402XX06', 'CC00403XX07', 'CC00404XX08',
                  'CC00406XX10', 'CC00407BN11', 'CC00408XX12', 'CC00409XX13', 'CC00410XX06', 'CC00412XX08', 'CC00413XX09',
                  'CC00415XX11', 'CC00421AN09', 'CC00421BN09', 'CC00424XX12', 'CC00425XX13', 'CC00426XX14', 'CC00427XX15',
                  'CC00428XX16', 'CC00429XX17', 'CC00430XX10', 'CC00431XX11', 'CC00433XX13', 'CC00438XX18', 'CC00439XX19',
                  'CC00440XX12', 'CC00441XX13', 'CC00442XX14', 'CC00443XX15', 'CC00444XX16', 'CC00445XX17', 'CC00446XX18',
                  'CC00447XX19', 'CC00448XX20', 'CC00450XX05', 'CC00451XX06', 'CC00453XX08', 'CC00455XX10', 'CC00457XX12',
                  'CC00458XX13', 'CC00461XX08', 'CC00465XX12', 'CC00466AN13', 'CC00466BN13', 'CC00467XX14', 'CC00468XX15',
                  'CC00469XX16', 'CC00470XX09', 'CC00472XX11', 'CC00473XX12', 'CC00474XX13', 'CC00475XX14', 'CC00476XX15',
                  'CC00477XX16', 'CC00478XX17', 'CC00479XX18', 'CC00480XX11', 'CC00481XX12', 'CC00482XX13', 'CC00483XX14',
                  'CC00484XX15', 'CC00485XX16', 'CC00486XX17', 'CC00492AN15', 'CC00492BN15', 'CC00497XX20', 'CC00498XX21',
                  'CC00499XX22', 'CC00500XX05', 'CC00501XX06', 'CC00502XX07', 'CC00504XX09', 'CC00505XX10', 'CC00506XX11',
                  'CC00507XX12', 'CC00508XX13', 'CC00509XX14', 'CC00512XX09', 'CC00514XX11', 'CC00515XX12', 'CC00516XX13',
                  'CC00520XX09', 'CC00525XX14', 'CC00525XX14', 'CC00528XX17', 'CC00529AN18', 'CC00529AN18', 'CC00529BN18',
                  'CC00529BN18', 'CC00532XX13', 'CC00534XX15', 'CC00535XX16', 'CC00536XX17', 'CC00537XX18', 'CC00538XX19',
                  'CC00541XX14', 'CC00542XX15', 'CC00543XX16', 'CC00544XX17', 'CC00548XX21', 'CC00549XX22', 'CC00550XX06',
                  'CC00551XX07', 'CC00552XX08', 'CC00553XX09', 'CC00554XX10', 'CC00557XX13', 'CC00560XX08', 'CC00561XX09',
                  'CC00563XX11', 'CC00564XX12', 'CC00566XX14', 'CC00568XX16', 'CC00569XX17', 'CC00569XX17', 'CC00570XX10',
                  'CC00572BN12', 'CC00576XX16', 'CC00577XX17', 'CC00578AN18', 'CC00578BN18', 'CC00580XX12', 'CC00581XX13',
                  'CC00583XX15', 'CC00584XX16', 'CC00586XX18', 'CC00588XX20', 'CC00589XX21', 'CC00590XX14', 'CC00591XX15',
                  'CC00592XX16', 'CC00593XX17', 'CC00594XX18', 'CC00596XX20', 'CC00597XX21', 'CC00598XX22', 'CC00607XX13',
                  'CC00616XX14', 'CC00617XX15', 'CC00617XX15', 'CC00618XX16', 'CC00620XX10', 'CC00621XX11', 'CC00622XX12',
                  'CC00628XX18', 'CC00628XX18', 'CC00629XX19', 'CC00630XX12', 'CC00632XX14', 'CC00632XX14', 'CC00637XX19',
                  'CC00639XX21', 'CC00642XX16', 'CC00647XX21', 'CC00648XX22', 'CC00648XX22', 'CC00649XX23', 'CC00650XX07',
                  'CC00652XX09', 'CC00653XX10', 'CC00654XX11', 'CC00655XX12', 'CC00656XX13', 'CC00657XX14', 'CC00663XX12',
                  'CC00664XX13', 'CC00667XX16', 'CC00668XX17', 'CC00669XX18', 'CC00670XX11', 'CC00671XX12', 'CC00672AN13',
                  'CC00672AN13', 'CC00672BN13', 'CC00685XX18', 'CC00686XX19', 'CC00687XX20', 'CC00692XX17', 'CC00693XX18',
                  'CC00698XX23', 'CC00705XX12', 'CC00712XX11', 'CC00712XX11', 'CC00713XX12', 'CC00714XX13', 'CC00716XX15',
                  'CC00718XX17', 'CC00719XX18', 'CC00720XX11', 'CC00731XX14', 'CC00734XX17', 'CC00735XX18', 'CC00736XX19',
                  'CC00737XX20', 'CC00740XX15', 'CC00741XX16', 'CC00747XX22', 'CC00749XX24', 'CC00753XX11', 'CC00754AN12',
                  'CC00754BN12', 'CC00757XX15', 'CC00760XX10', 'CC00764AN14', 'CC00764BN14', 'CC00765XX15', 'CC00768XX18',
                  'CC00769XX19', 'CC00770XX12', 'CC00770XX12', 'CC00771XX13', 'CC00777XX19', 'CC00782XX16', 'CC00783XX17',
                  'CC00788XX22', 'CC00789XX23', 'CC00791XX17', 'CC00792XX18', 'CC00792XX18', 'CC00798XX24', 'CC00799XX25',
                  'CC00801XX09', 'CC00802XX10', 'CC00802XX10', 'CC00810XX10', 'CC00811XX11', 'CC00815XX15', 'CC00816XX16',
                  'CC00818XX18', 'CC00822XX14', 'CC00823XX15', 'CC00823XX15', 'CC00824XX16', 'CC00829XX21', 'CC00830XX14',
                  'CC00830XX14', 'CC00832XX16', 'CC00833XX17', 'CC00839XX23', 'CC00840XX16', 'CC00841XX17', 'CC00843XX19',
                  'CC00845AN21', 'CC00845AN21', 'CC00845BN21', 'CC00845BN21', 'CC00846XX22', 'CC00847XX23', 'CC00850XX09',
                  'CC00851XX10', 'CC00852XX11', 'CC00854XX13', 'CC00855XX14', 'CC00855XX14', 'CC00856XX15', 'CC00858XX17',
                  'CC00860XX11', 'CC00861XX12', 'CC00863XX14', 'CC00867XX18', 'CC00867XX18', 'CC00868XX19', 'CC00870XX13',
                  'CC00871XX14', 'CC00874XX17', 'CC00875XX18', 'CC00879XX22', 'CC00879XX22', 'CC00880XX15', 'CC00881XX16',
                  'CC00882XX17', 'CC00883XX18', 'CC00884XX19', 'CC00889AN24', 'CC00889AN24', 'CC00889BN24', 'CC00889BN24',
                  'CC00890XX17', 'CC00891XX18', 'CC00897XX24', 'CC00898XX25', 'CC00904XX13', 'CC00907XX16', 'CC00907XX16',
                  'CC00911XX12', 'CC00914XX15', 'CC00915XX16', 'CC00917XX18', 'CC00919XX20', 'CC00923XX16', 'CC00924XX17',
                  'CC00925XX18', 'CC00926XX19', 'CC00928XX21', 'CC00929XX22', 'CC00930XX15', 'CC00938XX23', 'CC00939XX24',
                  'CC00940XX17', 'CC00945AN22', 'CC00947XX24', 'CC00948XX25', 'CC00949XX26', 'CC00955XX15', 'CC00956XX16',
                  'CC00958XX18', 'CC00961XX13', 'CC00962XX14', 'CC00964XX16', 'CC00966XX18', 'CC00967XX19', 'CC00971XX15',
                  'CC00973XX17', 'CC00974XX18', 'CC00976XX20', 'CC00979XX23', 'CC00980XX16', 'CC00982XX18', 'CC00987XX23',
                  'CC00990XX18', 'CC00992XX20', 'CC00997BN25', 'CC00998AN26', 'CC00998BN26', 'CC01004XX06', 'CC01005XX07',
                  'CC01005XX07', 'CC01007XX09', 'CC01011XX05', 'CC01011XX05', 'CC01011XX05', 'CC01013XX07', 'CC01014XX08',
                  'CC01015XX09', 'CC01018XX12', 'CC01019XX13', 'CC01021XX07', 'CC01022XX08', 'CC01023XX09', 'CC01027XX13',
                  'CC01029XX15', 'CC01032XX10', 'CC01034XX12', 'CC01037XX15', 'CC01038XX16', 'CC01041XX11', 'CC01042XX12',
                  'CC01044XX14', 'CC01045XX15', 'CC01047XX17', 'CC01050XX03', 'CC01051XX04', 'CC01055XX08', 'CC01057XX10',
                  'CC01059AN12', 'CC01059CN12', 'CC01069XX14', 'CC01070XX07', 'CC01074XX11', 'CC01077XX14', 'CC01077XX14',
                  'CC01082XX11', 'CC01084XX13', 'CC01086XX15', 'CC01087XX16', 'CC01089XX18', 'CC01093AN14', 'CC01093BN14',
                  'CC01096XX17', 'CC01104XX07', 'CC01105XX08', 'CC01108XX11', 'CC01111XX06', 'CC01116AN11', 'CC01117XX12',
                  'CC01145XX16', 'CC01153AN07', 'CC01176XX14', 'CC01181XX11', 'CC01190XX12', 'CC01191XX13', 'CC01192XX14',
                  'CC01194XX16', 'CC01195XX17', 'CC01198XX20', 'CC01199XX21', 'CC01200XX04', 'CC01201XX05', 'CC01206XX10',
                  'CC01207XX11', 'CC01208XX12', 'CC01208XX12', 'CC01209XX13', 'CC01211XX07', 'CC01212XX08', 'CC01215XX11',
                  'CC01218XX14', 'CC01218XX14', 'CC01220XX08', 'CC01223XX11', 'CC01232AN12', 'CC01234AN14', 'CC01234BN14',
                  'CC01236XX16', 'CC00908XX17']

nbs = len(sub_dhcp)
# load list sub and session KKI
# lits sub KKI
sub_KKI = list()
for file in os.listdir(dir_sulcKKI):
    if file.endswith('_MR1_l_sulc.gii'):
        sub = file.split('_')[0] + '_' + file.split('_')[1]
        sub_KKI.append(sub)


# compute voronoi KKI
# load mesh
for idx, sub in enumerate(sub_KKI):
    print(sub)
    ses = 'MR1'
    mesh_path = os.path.join(dir_KKI, sub + '_' + ses, 'surf/lh.white.gii')
    mesh = sio.load_mesh(mesh_path)
    voronoi = tv.voronoi_de_papa(mesh)
    sio.write_texture(stex.TextureND(darray=voronoi), os.path.join(wd, '../../data/rel3/voronoi', sub + '_' + ses + '_voronoi.gii'))



# compute voronoi dHCP
"""

list_metric = list()
list_surface = list()
list_error_mesh = list()
list_error_sulc = list()
for idx, sub in enumerate(sub_dhcp) :
    print(sub, idx ,'/', nbs)
    ses = ses_dhcp[idx]
    # load mesh
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_wm.surf.gii'
    mesh_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   mesh_name)
    try:
        mesh = sio.load_mesh(mesh_path)
        voronoi = tv.voronoi_de_papa(mesh)
        sio.write_texture(stex.TextureND(darray=voronoi),
                          os.path.join(wd, 'data/rel3/voronoi', sub + '_' + ses + '_voronoi.gii'))

    except:
        list_error_mesh.append(sub)
        # load sulc
    sulc_name = 'sub-' + sub + '_ses-' + ses + '_hemi-left_sulc.shape.gii'
    sulc_path = os.path.join(dir_dHCP, 'sub-' + sub, 'ses-' + ses, 'anat',   sulc_name)
    try:
        sulc = sio.load_texture(sulc_path).darray[0]
    except:
        list_error_sulc.append(sub)



print(list_error_mesh)
print(list_error_sulc)

"""

