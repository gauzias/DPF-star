import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

# Chargement des données
csv_folder = "D:/Callisto/data/results_rel3_dhcp/cortical_metrics"
csv_files = glob(os.path.join(csv_folder, "sub-*_ses-*_cortical_metrics.csv"))
df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Nettoyage global
df_all["scan_age"] = pd.to_numeric(df_all["scan_age"], errors="coerce")
df_all = df_all.dropna(subset=["scan_age", "ROI"])

# Setup Seaborn
sns.set_theme(style="whitegrid")

# Colonnes à tracer
metrics = [
    "DPFStar_Mean",
    "DPFStar_Median",
    "DPFStar_Min",
    "DPFStar_NegCurv_Mean",
    "DPFStar_NegCurv_Median",
    "DPFStar_NegCurv_Min"
]

# ROIs
unique_rois = sorted(df_all["ROI"].unique())
n_rois = len(unique_rois)
cols = 4
rows = (n_rois + cols - 1) // cols

# Génération des figures
for metric in metrics:
    df_all[metric] = pd.to_numeric(df_all[metric], errors="coerce")
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, roi in enumerate(unique_rois):
        ax = axes[i]
        data = df_all[df_all["ROI"] == roi].dropna(subset=[metric])

        if not data.empty:
            sns.regplot(
                data=data,
                x="scan_age",
                y=metric,
                ax=ax,
                scatter_kws={'s': 20, 'alpha': 0.6},
                line_kws={'color': 'red'}
            )
            ax.set_title(roi, fontsize=10)
            ax.set_xlabel("Âge au scan (semaines)")
            ax.set_ylabel(metric.replace("_", " "))
            ax.tick_params(axis='x', labelrotation=45)

    # Supprimer les subplots vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Sauvegarde
    output_path = os.path.join(csv_folder, f"figure_{metric}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Figure sauvegardée : {output_path}")
