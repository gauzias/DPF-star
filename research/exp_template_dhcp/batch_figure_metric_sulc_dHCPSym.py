import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from scipy.optimize import curve_fit
import numpy as np

# Modèle de Gompertz : f(t) = a * exp(-b * exp(-c * t))
def gompertz_model(t, a, b, c):
    return a * np.exp(-b * np.exp(-c * t))

#def gompertz_model(t, a, b, c):  # decroissant
#    return a - a * np.exp(-b * np.exp(-c * t))

def fit_and_plot_gompertz(ax, x, y, color, label):
    try:
        # Paramètres initiaux plus stables
        a0 = np.max(y)
        b0 = 1.0
        c0 = 0.05
        p0 = [a0, b0, c0]

        # Contraintes sur les paramètres
        bounds = (
            [0, 0, 0],          # a, b, c ≥ 0
            [np.inf, 100, 10]   # a, b, c ≤ seuils
        )

        # Fit
        params, _ = curve_fit(
            gompertz_model, x, y,
            p0=p0,
            bounds=bounds,
            maxfev=20000  # un peu plus de patience
        )

        # Affichage
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = gompertz_model(x_fit, *params)
        ax.plot(x_fit, y_fit, color=color, label=label)
    except Exception as e:
        print(f"Fit échoué ({label}) : {e}")
    finally:
        ax.scatter(x, y, s=20, alpha=0.6, color=color, label=f'{label} points')

def load_metric_data(base_folder, hemi):
    data = pd.DataFrame()
    week_folders = sorted(glob(os.path.join(base_folder, 'week-*')))
    for week_folder in week_folders:
        hemi_folder = os.path.join(week_folder, f"hemi_{hemi}")
        if not os.path.exists(hemi_folder):
            continue
        csv_files = glob(os.path.join(hemi_folder, '*_cortical_metrics.csv'))
        df_list = [pd.read_csv(f) for f in csv_files if os.path.exists(f)]
        if df_list:
            df_all = pd.concat(df_list, ignore_index=True)
            df_all['Week'] = pd.to_numeric(df_all['Week'], errors='coerce')
            data = pd.concat([data, df_all], ignore_index=True)
    return data

def main():
    sns.set_theme(style="whitegrid")

    # Dossier racine des résultats
    root_folder = r"E:/research_dpfstar/result_dhcpSym/cortical_metrics_sulc"

    # Chargement des données pour chaque hémisphère
    df_left = load_metric_data(root_folder, 'left')
    df_right = load_metric_data(root_folder, 'right')

    metrics = ['Sulc_Mean', 'Sulc_Median', 'Sulc_Min', 'Sulc_5p']

    # Vérifie si on a des données
    if df_left.empty and df_right.empty:
        print("Aucune donnée trouvée.")
        return

    unique_rois = sorted(set(df_left["ROI"].unique()).union(df_right["ROI"].unique()))
    n_rois = len(unique_rois)
    cols = 5
    rows = (n_rois + cols - 1) // cols
    rows = 6

    # Une figure par métrique
    for metric in metrics:
        # Calcul automatique du range Y pour chaque métrique (gauche + droite)
        all_values = pd.concat([
            df_left[[metric]].dropna(),
            df_right[[metric]].dropna()
        ])
        #ymin = all_values[metric].min()
        #ymax = all_values[metric].max()
        ymin = -all_values[metric].max()  # j'ai mis sulc negatif pour fitter gompertz (fit exponentiel croissant)
        ymax = -all_values[metric].min()  # pareil
        padding = 0.1 * (ymax - ymin) if ymax > ymin else 0.5
        ymin -= padding
        ymax += padding



        # Création de la figure
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= n_rois:
                fig.delaxes(ax)
                continue

            roi = unique_rois[i]
            print(roi)
            ax.set_title(roi, fontsize=10)

            # Données gauche
            data_l = df_left[df_left["ROI"] == roi].dropna(subset=[metric, "Week"])
            if not data_l.empty:
                fit_and_plot_gompertz(
                    ax,
                    data_l["Week"].values,
                    -data_l[metric].values,
                    color='blue',
                    label='Left'
                )

            # Données droite
            data_r = df_right[df_right["ROI"] == roi].dropna(subset=[metric, "Week"])
            if not data_r.empty:
                fit_and_plot_gompertz(
                    ax,
                    data_r["Week"].values,
                    -data_r[metric].values,
                    color='red',
                    label='Right'
                )

            ax.set_xlabel("Âge (semaines)")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_ylim(ymin, ymax)
            ax.legend()

        # Sauvegarde
        output_dir = os.path.join(root_folder, "figures_gompertz")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"figure_{metric}_gompertz.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Figure sauvegardée : {output_path}")

if __name__ == "__main__":
    main()
