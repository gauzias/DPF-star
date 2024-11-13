import os

path = '/home/maxime/callisto/repo/paper_sulcal_depth/data/EXP_3_bassins/dpfstar100'

for filename in os.listdir(path):
    if filename.endswith('_dpf100.gii'):
        new_filename = filename.split('_')[0] + '_' + filename.split('_')[1] + '_dpfstar100.gii'
        os.rename(os.path.join(path,filename), os.path.join(path, new_filename))