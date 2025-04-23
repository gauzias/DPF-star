import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from matplotlib import colormaps
import seaborn as sns

# === Dictionnaire des labels et des couleurs ===
label_names = {
    2: "Sillon_Central", 3: "Sillon_Temporal_superieur", 4: "S.Pe.C.inf.", 5: "S.Pe.C.sup.", 6: "S.Po.C.", 7: "Fissure_Intra_Parietale",
    8: "Sillon_Frontal_superieur", 9: "Sillon_Frontal_inferieur", 10: "S.T.i.", 11: "F.C.M.", 12: "Fissure_Calcarine.", 13: "S.F.int.",
    14: "S.R.inf", 15: "F.Coll.", 16: "S.S.p.", 17: "S.Call.", 18: "Fissure_Parieto_Occipitale",
    19: "Sillon_Olfactif", 20: "S.Or.", 21: "S.O.T.Lat.", 22: "S.C.LPC.",
    23: "S.F.int.", 24: "S.F.marginal", 25: "S.F.Orbitaire", 26: "S.Rh.",
    27: "F.I.P.sup.", 28: "S.Pa.sup."
}

value_color_dict = {
    0: "floralwhite", 1: "floralwhite", 2: "blue", 3: "green", 4: "orange",
    5: "purple", 6: "teal", 7: "gold", 8: "pink", 9: "cyan", 10: "magenta",
    11: "lime", 12: "brown", 13: "navy", 14: "salmon", 15: "chocolate",
    16: "olive", 17: "turquoise", 18: "indigo", 19: "darkred", 20: "darkblue",
    21: "darkgreen", 22: "coral", 23: "deepskyblue", 24: "darkorange",
    25: "orchid", 26: "lightseagreen", 27: "steelblue", 28: "slategray"
}

def load_metric_data(root_folder, metric_prefix):
    all_data = []
    for week_folder in sorted(glob(os.path.join(root_folder, 'week-*'))):
        week_name = os.path.basename(week_folder)  # ex: week-30
        for hemi in ['left', 'right']:
            hemi_folder = os.path.join(week_folder, f'hemi_{hemi}')
            csv_files = glob(os.path.join(hemi_folder, '*_cortical_metrics.csv'))
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                df['Hemi'] = hemi
                df['Week'] = int(week_name.replace("week-", ""))
                # Filtrer les colonnes utiles
                cols = [c for c in df.columns if c.startswith(metric_prefix)] + ['Week', 'ROI', 'Hemi']
                df = df[[c for c in cols if c in df.columns]]
                all_data.append(df)
    if not all_data:
        print(f"Aucun fichier trouvé dans {root_folder} avec préfixe '{metric_prefix}'")
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def load_combined_data(dpf_folder, sulc_folder):
    df_dpf = load_metric_data(dpf_folder, "DPFstarAbs")
    df_sulc = load_metric_data(sulc_folder, "Sulc")
    df_merged = pd.merge(df_dpf, df_sulc, on=["Week", "ROI", "Hemi"], how="inner")
    return df_merged

def plot_all_rois(df, hemi, metric, title, output_path, label_names, value_color_dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    rois = sorted(df["ROI"].unique())

    # Associe chaque nom de ROI à sa couleur via label index
    roi_colors = {}
    inverse_label_names = {v: k for k, v in label_names.items()}  # nom -> index
    for roi in rois:
        index = inverse_label_names.get(roi)
        color = value_color_dict.get(index, "gray")  # default fallback
        roi_colors[roi] = color

    for roi in rois:
        df_roi = df[(df["ROI"] == roi) & (df["Hemi"] == hemi)]
        if df_roi.empty or metric not in df_roi.columns:
            continue
        sns.lineplot(
            data=df_roi,
            x="Week",
            y=metric,
            ax=ax,
            label=roi,
            color=roi_colors[roi],
            ci=None
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Âge (semaines)")
    ax.set_ylabel("Profondeur")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Figure enregistrée : {output_path}")


def main():
    sns.set_theme(style="whitegrid")

    dpf_folder = r"E:/research_dpfstar/result_dhcpSym/cortical_metrics_dpfstarAbs"
    sulc_folder = r"E:/research_dpfstar/result_dhcpSym/cortical_metrics_sulc"
    output_dir = os.path.join(dpf_folder, "figures_global_sulc_dpfstarAbs")
    os.makedirs(output_dir, exist_ok=True)

    df = load_combined_data(dpf_folder, sulc_folder)
    print("Données fusionnées :", df.shape)

    # 4 figures globales
    plot_all_rois(
        df=df,
        hemi="left",
        metric="DPFstarAbs_Mean",
        title="DPFstarAbs - Hémisphère Gauche (toutes ROIs)",
        output_path=os.path.join(output_dir, "Mean", "DPFstarAbs_left_allROIs.png"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

    plot_all_rois(
        df=df,
        hemi="left",
        metric="DPFstarAbs_Mean",
        title="DPFstarAbs - Hémisphère Gauche (toutes ROIs)",
        output_path=os.path.join(output_dir, "Mean", "DPFstarAbs_left_allROIs.png"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

    plot_all_rois(
        df=df,
        hemi="left",
        metric="DPFstarAbs_Mean",
        title="DPFstarAbs - Hémisphère Gauche (toutes ROIs)",
        output_path=os.path.join(output_dir,"Mean", "DPFstarAbs_left_allROIs.png"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

    plot_all_rois(
        df=df,
        hemi="left",
        metric="DPFstarAbs_Mean",
        title="DPFstarAbs - Hémisphère Gauche (toutes ROIs)",
        output_path=os.path.join(output_dir, "Mean",  "DPFstarAbs_left_allROIs.png"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

if __name__ == "__main__":
    main()
