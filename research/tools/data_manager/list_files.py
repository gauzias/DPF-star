import os

def list_files(directory):
    """
    Retourne une liste des chemins absolus de tous les fichiers dans un dossier donné.
    :param directory: Chemin du dossier à analyser
    :return: Liste des chemins absolus des fichiers
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Le chemin spécifié '{directory}' n'est pas un dossier valide.")
    
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.abspath(os.path.join(root, file)))
    
    return file_paths

# Exemple d'utilisation
#directory_path = ""
#try:
#    files = list_files(directory_path)
#    print("\n".join(files))
#except ValueError as e:
#    print(e)


def get_filename_without_extension(file_path, ext):
    """
    Extrait le nom du fichier sans son extension .gii
    :param file_path: Chemin du fichier
    :return: Nom du fichier sans extension
    """
    filename = os.path.basename(file_path)  # Récupère le nom du fichier
    if filename.endswith(ext): #exemple : if filename.endswith(".gii")
        return os.path.splitext(filename)[0]  # Retire l'extension .gii
    return filename
