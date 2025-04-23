import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Paramètres ===
csv_path = "E:/research_dpfstar/results_rel3_dhcp/minima_count_sulc/sulc_summary.csv"
output_folder = os.path.dirname(csv_path)

# === Chargement des données ===
df = pd.read_csv(csv_path)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df = df.dropna(subset=["Age"])

# === Simplification des noms de ROI ===
def simplify_roi(roi_name):
    roi_lower = roi_name.lower()
    if "frontal" in roi_lower:
        return "Frontal_lobe"
    elif "parietal" in roi_lower:
        return "Parietal_lobe"
    elif "occipital" in roi_lower:
        return "Occipital_lobe"
    elif "temporal" in roi_lower:
        return "Temporal_lobe"
    else:
        return None

df["Simplified_ROI"] = df["ROI"].apply(simplify_roi)

# === Filtrage des ROIs d’intérêt ===
rois_to_keep = ["Frontal_lobe", "Parietal_lobe", "Occipital_lobe", "Temporal_lobe"]
df = df[df["Simplified_ROI"].isin(rois_to_keep)]

# === Agrégation des MinimaCount par combinaison unique ===
df_grouped = (
    df.groupby(["Subject", "Session", "Hemisphere", "Age", "Simplified_ROI"], as_index=False)
    .agg({"MinimaCount": "sum"})
)

# === Paramètres de tracé ===
sns.set_theme(style="whitegrid")
rois = sorted(df_grouped["Simplified_ROI"].unique())
cols = 2
rows = (len(rois) + cols - 1) // cols

# === Figure : Régression par ROI et hémisphère ===
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)
axes = axes.flatten()

for i, roi in enumerate(rois):
    ax = axes[i]
    data_roi = df_grouped[df_grouped["Simplified_ROI"] == roi]
    for hemi, color in zip(["left", "right"], ["blue", "red"]):
        data_hemi = data_roi[data_roi["Hemisphere"] == hemi]
        if not data_hemi.empty:
            sns.regplot(
                data=data_hemi,
                x="Age",
                y="MinimaCount",
                scatter_kws={'s': 20, 'alpha': 0.6},
                line_kws={'color': color},
                ax=ax,
                label=hemi.capitalize()
            )
    ax.set_title(roi.replace('_', ' '), fontsize=12)
    ax.set_xlabel("Scan Age")
    ax.set_ylabel("Minima Count")
    ax.legend()

# Supprimer les subplots inutilisés
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# === Sauvegarde de la figure ===
fig_path = os.path.join(output_folder, "figure_sulc_filtered_minima_regression_clean.png")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"[Figure] Régression sauvegardée : {fig_path}")
