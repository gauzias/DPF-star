import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chargement du fichier CSV
csv_path = "E:/research_dpfstar/results_rel3_dhcp/interpolation_errors_summary.csv"  # adapte si nécessaire
df = pd.read_csv(csv_path)

# Création du dossier de sauvegarde des figures
save_dir = os.path.join(os.path.dirname(csv_path), "figures")
os.makedirs(save_dir, exist_ok=True)

# Configuration graphique
sns.set(style="whitegrid")
metrics = ["MSE", "NRMSE", "MAE"]

# Génération et sauvegarde des figures
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Résolution cible", y=metric, palette="Set2")

    plt.title(f"Distribution de {metric} par résolution cible", fontsize=14)
    plt.xlabel("Résolution cible", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Sauvegarde
    fig_path = os.path.join(save_dir, f"boxplot_{metric}.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Figure sauvegardée : {fig_path}")

    plt.show()
