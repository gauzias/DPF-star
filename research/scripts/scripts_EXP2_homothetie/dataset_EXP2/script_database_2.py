import slam.io as sio
import os
import pandas as pd
import settings.path_manager as pm

### 0. Upload Info subject tables

folder_info_database = pm.folder_info_dataset
folder_meshes = pm.folder_meshes
folder_subject_analysis = pm.folder_subject_analysis
folder_manual_labelisation = pm.folder_manual_labelisation

# function
def add_subject(folder_info_database,
                database = None, participant_id = None, gender= None, birth_age= None, birth_weight= None,
                singleton= None, session_id= None, preterm = None, scan_age= None, scan_age_unity = None,
                scan_head_circumference = None, scan_number = None,
                radiology_score= None, sedation= None, GI= None, surface_area= None, volume_hull= None, volume= None):
    """
    All the informations relativ to the database are stored in the csv file 'info_database.csv' saved in your
    folder_info_databe. This function allow you to add a new row specific to a new subject.
    :param folder_info_database: STRING. absolut path of your folder_info_database
    :param database : STRING. name of the source database (ex : 'dHCP' or 'KKI')
    :param participant_id: STRING.
    :param gender: STRING. 'Male' or 'Female'
    :param birth_age: FLOAT.
    :param birth_weight: FLOAT.
    :param singleton: STRING. 'Single' or 'Multiple'
    :param session_id: STRING.
    :param preterm : BOOL. True or False
    :param scan_age: FLOAT.
    :param scan_age_unity: STRING. 'years' or 'GW' (GW for gestational weeks)
    :param scan_head_circumference: FLOAT.
    :param scan_number: INT.
    :param radiology_score: INT.
    :param sedation: INT. 0 or 1
    :param GI: FlOAT. Girification index : (hull volume)/(inner Volume )
    :param surface_area: FLOAT.
    :param volume_hull: FLOAT. volume of the convex hull
    :param volume: FLOAT. inner volume of the mesh
    :return: no return, save directly the updated csv file in the specified folder_info_database
    """

    ## reading the csv file if csv file exist
    try:
        print('...reading csv file info database')
        info_database = pd.read_csv(os.path.join(folder_info_database, 'info_database.csv'))
        flag_info_exist = True
    except:
        print("CSV file info database not existing, we create a new one")
        flag_info_exist = False

    ## upload subject information
    new_sub = {'database' : database,
               'participant_id': participant_id,
               'gender': gender,
               'birth_age': birth_age,
               'birth_weight': birth_weight,
               'singelton': singleton,
               'session_id': session_id,
               'preterm' : preterm,
               'scan_age': scan_age,
               'scan_age_unity' : scan_age_unity,
               'scan_head_circumference': scan_head_circumference,
               'scan_number': scan_number,
               'radiology_score': radiology_score,
               'sedation': sedation,
               'GI': GI,
               'surface_area': surface_area,
               'volume_hull': volume_hull,
               'volume': volume}
    new_sub = pd.DataFrame([new_sub])

    ## check if the csv file already exist or not
    if flag_info_exist == True :
        info_database = pd.concat([info_database, new_sub], axis=0)
        print('...updating csv file')
    else:
        info_database = new_sub
        print('...updating csv file')

    ## Save the dataframe into csv in the specified folder
    try:
        info_database.to_csv(os.path.join(folder_info_database, 'info_database.csv'), index=False)
        print('...csv file updated.')
    except:
        print('Error, csv file not updated')



## script adding adult subject KKI2009_142_MR1
sub = 'KKI2009-142'
ses = 'MR1'
mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
mesh_path = os.path.join(folder_meshes, mesh_name)
mesh = sio.load_mesh(mesh_path)
hull = mesh.convex_hull
volume_hull = hull.volume
GI = volume_hull/mesh.volume
add_subject(folder_info_database=folder_info_database, database='KKI', participant_id=sub, session_id = ses,
            preterm=False, scan_age=25, scan_age_unity='years',surface_area=mesh.area, volume_hull=volume_hull,
            volume=mesh.volume, GI=GI)




## script upload volume and area information for all subjects
info_database = pd.read_csv(os.path.join(folder_info_database, 'info_database.csv'))
subjects = info_database['participant_id'].values
sessions = info_database['session_id'].values
for idx, sub in enumerate(subjects):
    ses = sessions[idx]
    print(sub)
    print(ses)
    mesh_name = 'sub-' + sub + '_ses-' + ses + '_hemi-L_space-T2w_wm.surf.gii'
    mesh_path = os.path.join(folder_meshes, mesh_name)
    mesh = sio.load_mesh(mesh_path)
    hull = mesh.convex_hull
    volume_hull = hull.volume
    GI = volume_hull / mesh.volume
    info_database.loc[idx , 'volume'] = mesh.volume
    info_database.loc[idx, 'volume_hull'] = volume_hull
    info_database.loc[idx, 'surface_area'] = mesh.area
    info_database.loc[idx, 'GI'] = GI

info_database.to_csv(os.path.join(folder_info_database, 'info_database.csv'), index=False)
