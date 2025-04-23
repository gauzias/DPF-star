import os
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from glob import glob

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

# === Chargement des données ===
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

# === Plotly interactive ===
def plot_all_rois(df, hemi, metric, title, output_path, label_names, value_color_dict):
    fig = go.Figure()

    inverse_label_names = {v: k for k, v in label_names.items()}
    roi_colors = {
        roi: value_color_dict.get(inverse_label_names.get(roi, -1), "gray")
        for roi in sorted(df["ROI"].unique())
    }

    for roi in sorted(df["ROI"].unique()):
        df_roi = df[(df["ROI"] == roi) & (df["Hemi"] == hemi)]
        if df_roi.empty or metric not in df_roi.columns:
            continue

        fig.add_trace(go.Scatter(
            x=df_roi["Week"],
            y=df_roi[metric],
            mode="lines+markers",
            name=roi,
            line=dict(color=roi_colors[roi], width=2),
            marker=dict(size=6),
            text=[roi]*len(df_roi),
            hovertemplate="<b>%{text}</b><br>Semaine: %{x}<br>Valeur: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Âge (semaines)",
        yaxis_title="Profondeur",
        template="plotly_white",
        legend=dict(
            title="ROI",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,
            font=dict(size=10)
        ),
        width=1000,
        height=700
    )

    fig.write_html(output_path)
    print(f"📊 Figure interactive sauvegardée : {output_path}")

# === Main ===
def main():
    sns.set_theme(style="whitegrid")

    dpf_folder = r"E:/research_dpfstar/result_dhcpSym/cortical_metrics_dpfstarAbs"
    sulc_folder = r"E:/research_dpfstar/result_dhcpSym/cortical_metrics_sulc"
    #output_dir = os.path.join(dpf_folder, "figures_global_sulc_dpfstarAbs", "Mean")
    output_dir = os.path.join(dpf_folder, "figures_global_sulc_dpfstarAbs", "5p")
    os.makedirs(output_dir, exist_ok=True)

    df = load_combined_data(dpf_folder, sulc_folder)
    print("✅ Données fusionnées :", df.shape)

    # DPFstarAbs Gauche
    plot_all_rois(
        df=df,
        hemi="left",
        #metric="DPFstarAbs_Mean",
        metric="DPFstarAbs_5p",
        title="DPFstarAbs - Hémisphère Gauche (toutes ROIs)",
        output_path=os.path.join(output_dir, "DPFstarAbs_left_allROIs.html"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

    # DPFstarAbs Droit
    plot_all_rois(
        df=df,
        hemi="right",
        #metric="DPFstarAbs_Mean",
        metric="DPFstarAbs_5p",
        title="DPFstarAbs - Hémisphère Droit (toutes ROIs)",
        output_path=os.path.join(output_dir, "DPFstarAbs_right_allROIs.html"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

    # Sulc Gauche
    plot_all_rois(
        df=df,
        hemi="left",
        #metric="Sulc_Mean",
        metric="Sulc_5p",
        title="Sulc - Hémisphère Gauche (toutes ROIs)",
        output_path=os.path.join(output_dir, "Sulc_left_allROIs.html"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

    # Sulc Droit
    plot_all_rois(
        df=df,
        hemi="right",
        #metric="Sulc_Mean",
        metric = "Sulc_5p",
        title="Sulc - Hémisphère Droit (toutes ROIs)",
        output_path=os.path.join(output_dir, "Sulc_right_allROIs.html"),
        label_names=label_names,
        value_color_dict=value_color_dict
    )

if __name__ == "__main__":
    main()
