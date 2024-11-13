import pandas as pd
import numpy as np
import os
import glob
import slam.io as sio
import matplotlib.pyplot as plt

# wd
OAS1_path = '/media/maxime/Expansion/FS_OASIS'

# get list of OAS1 subject
aa = glob.glob(os.path.join(OAS1_path, 'OAS1_0*'))
bb = [a for a in aa if os.path.isdir(a)]
subjects = [b.split(OAS1_path+'/')[1] for b in bb]

# loop
volumes = list()
buggy_mesh = list()
for idx, sub in enumerate(subjects):
    print(idx, '/', len(subjects))
    mesh_path = os.path.join(OAS1_path, sub, 'surf', 'lh.white.gii')
    try :
        mesh = sio.load_mesh(mesh_path)
        volumes.append(mesh.volume)
    except:
        print('error')
        buggy_mesh.append(sub)


# display hist volumes

fig, ax = plt.subplots()
ax.hist(volumes, bins = 'auto')

plt.show()