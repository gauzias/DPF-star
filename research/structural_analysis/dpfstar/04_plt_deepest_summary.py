import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Paramètres ===
csv_path = "E:/research_dpfstar/results_rel3_dhcp/deepest_summary_from_textures.csv"
output_folder = os.path.join(os.path.dirname(csv_path), "figures_deepest_summary")
os.makedirs(output_folder, exist_ok=True)

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

# === Agrégation par combinaison unique ===
df_grouped = (
    df.groupby(["Subject", "Session", "Hemisphere", "Age", "Simplified_ROI"], as_index=False)
    .agg({
        "DeepestPoints": "sum",
        "MeanDepth": "mean"
    })
)

# === Paramètres de tracé ===
sns.set_theme(style="whitegrid")
rois = sorted(df_grouped["Simplified_ROI"].unique())
cols = 2
rows = (len(rois) + cols - 1) // cols

# === Figure 1 : Nombre de deepest points par ROI ===
fig1, axes1 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)
axes1 = axes1.flatten()

for i, roi in enumerate(rois):
    ax = axes1[i]
    data_roi = df_grouped[df_grouped["Simplified_ROI"] == roi]
    for hemi, color in zip(["left", "right"], ["blue", "red"]):
        data_hemi = data_roi[data_roi["Hemisphere"] == hemi]
        if not data_hemi.empty:
            sns.regplot(
                data=data_hemi,
                x="Age",
                y="DeepestPoints",
                scatter_kws={'s': 20, 'alpha': 0.6},
                line_kws={'color': color},
                ax=ax,
                label=hemi.capitalize()
            )
    ax.set_title(roi.replace('_', ' '), fontsize=12)
    ax.set_xlabel("Scan Age")
    ax.set_ylabel("Number of Deepest Points")
    ax.legend()

for j in range(i + 1, len(axes1)):
    fig1.delaxes(axes1[j])

fig1_path = os.path.join(output_folder, "figure_deepest_points_count.png")
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"[✓] Figure 1 sauvegardée : {fig1_path}")


# === Figure 2 : Profondeur moyenne des deepest points ===
fig2, axes2 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)
axes2 = axes2.flatten()

for i, roi in enumerate(rois):
    ax = axes2[i]
    data_roi = df_grouped[df_grouped["Simplified_ROI"] == roi]
    for hemi, color in zip(["left", "right"], ["blue", "red"]):
        data_hemi = data_roi[data_roi["Hemisphere"] == hemi]
        if not data_hemi.empty:
            sns.regplot(
                data=data_hemi,
                x="Age",
                y="MeanDepth",
                scatter_kws={'s': 20, 'alpha': 0.6},
                line_kws={'color': color},
                ax=ax,
                label=hemi.capitalize()
            )
    ax.set_title(roi.replace('_', ' '), fontsize=12)
    ax.set_xlabel("Scan Age")
    ax.set_ylabel("Mean Depth of Deepest Points")
    ax.legend()

for j in range(i + 1, len(axes2)):
    fig2.delaxes(axes2[j])

fig2_path = os.path.join(output_folder, "figure_mean_depth_deepest_points.png")
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"[✓] Figure 2 sauvegardée : {fig2_path}")
