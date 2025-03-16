import os
import re
import pymeshlab

def find_pymeshlab_path():
    """ Trouve le chemin d'installation de PyMeshLab. """
    pymeshlab_path = os.path.dirname(pymeshlab.__file__)
    print(f"PyMeshLab est installé dans : {pymeshlab_path}")
    return pymeshlab_path

def search_function_in_pymeshlab(function_name, pymeshlab_path):
    """ Recherche une fonction spécifique dans les fichiers de PyMeshLab. """
    print(f"\nRecherche de '{function_name}' dans {pymeshlab_path}...\n")
    found_lines = []

    # Parcourir tous les fichiers dans le dossier PyMeshLab
    for root, _, files in os.walk(pymeshlab_path):
        for file in files:
            if file.endswith(".py"):  # Seulement les fichiers Python
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if function_name in line:  # Vérifie si la fonction est mentionnée
                                found_lines.append((file_path, i+1, line.strip()))
                except Exception as e:
                    print(f"Impossible de lire {file_path}: {e}")

    if found_lines:
        for file_path, line_num, line_content in found_lines:
            print(f"Trouvé dans {file_path} (ligne {line_num}): {line_content}")
    else:
        print("Aucune occurrence trouvée.")

def get_function_arguments(function_name, pymeshlab_path):
    """ Recherche les arguments de la fonction donnée. """
    print(f"\nRecherche des arguments de '{function_name}'...\n")
    for root, _, files in os.walk(pymeshlab_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Recherche une définition de fonction avec les arguments
                        match = re.search(rf"def\s+{function_name}\((.*?)\):", content, re.DOTALL)
                        if match:
                            print(f"Arguments trouvés dans {file_path}: ({match.group(1)})")
                            return match.group(1)
                except Exception as e:
                    print(f"Impossible de lire {file_path}: {e}")
    
    print("Impossible de trouver les arguments.")
    return None

# Exécution des fonctions
pymeshlab_path = find_pymeshlab_path()
search_function_in_pymeshlab("meshing_decimation_quadric_edge_collapse", pymeshlab_path)
get_function_arguments("meshing_decimation_quadric_edge_collapse", pymeshlab_path)
