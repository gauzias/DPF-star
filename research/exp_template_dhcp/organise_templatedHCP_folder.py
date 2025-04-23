import os
import shutil
import re

# Chemin de base
base_path = r"E:\dhcpSym_template"

# Lister tous les dossiers week-XX sauf "week-to-40-registration"
for item in os.listdir(base_path):
    item_path = os.path.join(base_path, item)
    
    if os.path.isdir(item_path) and item.startswith("week-") and item != "week-to-40-registration":
        print(f"Traitement de {item}...")
        
        # Créer les sous-dossiers
        for hemi in ["hemi_left", "hemi_right"]:
            for space in ["dhcpSym", "dhcpSym40"]:
                os.makedirs(os.path.join(item_path, hemi, space), exist_ok=True)

        # Parcourir les fichiers du dossier week-XX
        for file_name in os.listdir(item_path):
            if not file_name.endswith(".gii"):
                continue
            
            match = re.match(r"week-\d+_hemi-(left|right)_space-(dhcpSym40|dhcpSym)_.*\.gii", file_name)
            if not match:
                print(f"Nom de fichier non reconnu : {file_name}")
                continue

            hemi = f"hemi_{match.group(1)}"
            space = match.group(2)

            source_path = os.path.join(item_path, file_name)
            dest_path = os.path.join(item_path, hemi, space, file_name)

            shutil.move(source_path, dest_path)
            print(f"Déplacé : {file_name} -> {hemi}/{space}/")

print("Réorganisation terminée.")
