
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib.pyplot as plt
import math
import re
from scipy.stats import gaussian_kde
import re



def extraire_id_session(nom_fichier):
    # Expression régulière adaptée à la nouvelle structure
    match = re.match(r"sub-[^_]+_([0-9]+)_ses-([^_]+)", nom_fichier)
    if match:
        identifiant = match.group(1)
        session = match.group(2)
        return identifiant, session
    else:
        raise ValueError("Le nom de fichier ne respecte pas le format attendu.")
    
def extraire_sub_ses(nom_fichier):
    # Utilisation d'une expression régulière pour extraire les champs
    match = re.match(r"sub-([a-zA-Z0-9]+)_ses-([0-9]+)", nom_fichier)
    if match:
        sub = match.group(1)
        ses = match.group(2)
        return sub, ses
    else:
        raise ValueError("Le nom de fichier ne respecte pas le format attendu.")

def plot_density_curves(dstar_list, res_list, subject_name, bins=40, save_dir="outputs/figures"):
    plt.figure(figsize=(10, 6))

    for dstar, res in zip(dstar_list, res_list):
        dstar = dstar[~np.isnan(dstar)]  # Nettoyage NaN
        kde = gaussian_kde(dstar)
        x_vals = np.linspace(np.min(dstar), np.max(dstar), 1000)
        y_vals = kde(x_vals)
        plt.plot(x_vals, y_vals, label=f"Résolution {res}", linewidth=2)

    plt.title(f"Courbes de densité DPF* – {subject_name}")
    plt.xlabel("Valeur DPF*")
    plt.ylabel("Densité")
    plt.legend(title="Résolutions")
    plt.grid(True)
    plt.tight_layout()

    # Création du dossier de sauvegarde s’il n’existe pas
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{subject_name}_density_plot.png")
    plt.savefig(save_path)
    plt.close()

def plot_histograms(*lists, bins):
    n = len(lists)
    cols = min(5, n)  # Maximum de 5 subplots par ligne
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]  # Assurer une itération correcte
    
    for i, data in enumerate(lists):
        axes[i].hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_title(f'Histogram {i+1}')
    
    # Cacher les subplots inutilisés
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()