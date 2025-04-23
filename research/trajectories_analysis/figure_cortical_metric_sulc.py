import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

def main(plot_side='both'):
    sns.set_theme(style="whitegrid")

    # Dossiers
    folder_cortical_metric_left = "E:\\research_dpfstar\\results_rel3_dhcp\\cortical_metrics_sulc\\hemi_left"
    folder_cortical_metric_right = "E:\\research_dpfstar\\results_rel3_dhcp\\cortical_metrics_sulc\\hemi_right"

    # Chargement
    df_all_left = pd.DataFrame()
    df_all_right = pd.DataFrame()

    if plot_side in ['left', 'both']:
        cortical_metric_left = glob(os.path.join(folder_cortical_metric_left, "sub-*_ses-*_cortical_metrics.csv"))
        df_all_left = pd.concat([pd.read_csv(f) for f in cortical_metric_left], ignore_index=True)
        df_all_left["scan_age"] = pd.to_numeric(df_all_left["scan_age"], errors="coerce")
        df_all_left = df_all_left.dropna(subset=["scan_age", "ROI"])

    if plot_side in ['right', 'both']:
        cortical_metric_right = glob(os.path.join(folder_cortical_metric_right, "sub-*_ses-*_cortical_metrics.csv"))
        df_all_right = pd.concat([pd.read_csv(f) for f in cortical_metric_right], ignore_index=True)
        df_all_right["scan_age"] = pd.to_numeric(df_all_right["scan_age"], errors="coerce")
        df_all_right = df_all_right.dropna(subset=["scan_age", "ROI"])

    # Colonnes à tracer

    #metrics = [
    #    "Sulc_Mean_abs",
    #    "Sulc_Min",
    #    "Sulc_p5",
    #    "Sulc_p10",
    #    "Sulc_p15",
    #    "Sulc_p20",
    #    "Sulc_Q1",
    #    "Sulc_Mean",
    #    "Sulc_Median",
    #    "Sulc_Q3",
    #    "Sulc_Max",
    #    "Sulc_NegCurv_Min",
    #    "Sulc_NegCurv_Q1",
    #    "Sulc_NegCurv_Mean",
    #    "Sulc_NegCurv_Median",
    #    "Sulc_NegCurv_Q3",
    #    "Sulc_NegCurv_Max"
    #]


        metrics = [
        "Sulc_Mean_abs",
    ]

    # ROIs à utiliser (on prend celles de gauche, en supposant que ce sont les mêmes)
    if not df_all_left.empty:
        unique_rois = sorted(df_all_left["ROI"].unique())
    else:
        unique_rois = sorted(df_all_right["ROI"].unique())

    unique_rois = ["Parietal_lobe", "Temporal_lobe", "Frontal_lobe", "Occipital_lobe"]

    n_rois = len(unique_rois)
    cols = 4
    rows = (n_rois + cols - 1) // cols

    # Génération des figures
    for metric in metrics:
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
        axes = axes.flatten()

        for i, roi in enumerate(unique_rois):
            ax = axes[i]

            if plot_side in ['left', 'both']:
                data_left = df_all_left[df_all_left["ROI"] == roi].dropna(subset=[metric])
                if not data_left.empty:
                    sns.regplot(
                        data=data_left,
                        x="scan_age",
                        y=metric,
                        ax=ax,
                        scatter_kws={'s': 20, 'alpha': 0.6},
                        line_kws={'color': 'blue'},
                        label='Left'
                    )

            if plot_side in ['right', 'both']:
                data_right = df_all_right[df_all_right["ROI"] == roi].dropna(subset=[metric])
                if not data_right.empty:
                    sns.regplot(
                        data=data_right,
                        x="scan_age",
                        y=metric,
                        ax=ax,
                        scatter_kws={'s': 20, 'alpha': 0.6},
                        line_kws={'color': 'red'},
                        label='Right'
                    )

            ax.set_title(roi, fontsize=10)
            ax.set_xlabel("Âge au scan (semaines)")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_ylim(0, 5)  # <-- Limites Y fixées ici
            ax.tick_params(axis='x', labelrotation=45)
            ax.legend()

        # Supprimer les subplots vides
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Sauvegarde
        output_base = {
            'left': folder_cortical_metric_left,
            'right': folder_cortical_metric_right,
            'both': folder_cortical_metric_left  # on stocke dans le dossier gauche par défaut
        }
        output_path = os.path.join(output_base[plot_side], f"figure_{metric}_{plot_side}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Figure sauvegardée : {output_path}")

if __name__ == "__main__":
    # Choisis entre 'left', 'right' ou 'both'
    main(plot_side='both')
