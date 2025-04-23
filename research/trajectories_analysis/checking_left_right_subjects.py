import os
import re
import shutil

# Dossiers contenant les fichiers .gii
hemi_left_dir = r"E:\\research_dpfstar\\results_rel3_dhcp\\dpfstar\\hemi_left"
hemi_right_dir = r"E:\\research_dpfstar\\results_rel3_dhcp\\dpfstar\\hemi_right"

# Expressions régulières pour extraire sujet et session
pattern = re.compile(r"sub-(.*?)_ses-(.*?)_hemi-(left|right)_wm\.surf")

def get_subject_session_set(directory):
    subject_session_set = set()
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            subject, session, _ = match.groups()
            subject_session_set.add((subject, session))
    return subject_session_set

# Obtenir les tuples (sujet, session) pour chaque hémisphère
left_set = get_subject_session_set(hemi_left_dir)
right_set = get_subject_session_set(hemi_right_dir)

# Trouver les tuples présents dans left mais pas dans right
missing_in_right =  right_set -left_set 
print(missing_in_right)
# Copier les dossiers correspondants
# Dossiers pour la copie
source_base_dir = r"E:\\rel3_dhcp_full"
destination_base_dir = r"E:\\rel3_dhcp_missing_right"
for subject, session in sorted(missing_in_right):
    print(f"sub-{subject}_ses-{session}")
    src_path = os.path.join(source_base_dir, f"sub-{subject}", f"ses-{session}")
    dest_path = os.path.join(destination_base_dir, f"sub-{subject}", f"ses-{session}")

    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
    else:
        print(f"[WARNING] Source path not found: {src_path}")
