import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

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

def main():
    sns.set_theme(style="whitegrid")

    # Dossiers
    dpf_folder = r"E:/research_dpfstar/result_dhcpSym/cortical_metrics_dpfstarAbs"
    sulc_folder = r"E:/research_dpfstar/result_dhcpSym/cortical_metrics_sulc"
    output_dir = os.path.join(dpf_folder, "figures_sulc_dpfstarAbs_combined")
    os.makedirs(output_dir, exist_ok=True)

    # Chargement et fusion
    df = load_combined_data(dpf_folder, sulc_folder)
    print("Données fusionnées :", df.shape)

    # Liste des ROI à tracer
    roi_list = sorted(df["ROI"].unique())

    for roi in roi_list:
        sub_df = df[df["ROI"] == roi]
        if sub_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"Comparaison Sulc / DPFstarAbs - ROI: {roi}", fontsize=12)

        # DPFstar
        sns.scatterplot(data=sub_df[sub_df["Hemi"] == "left"],
                        x="Week", y="DPFstarAbs_Mean", ax=ax,
                        color='navy', label="Left - DPFstarAbs", alpha=0.6)

        sns.scatterplot(data=sub_df[sub_df["Hemi"] == "right"],
                        x="Week", y="DPFstarAbs_Mean", ax=ax,
                        color='darkred', label="Right - DPFstarAbs", alpha=0.6)

        # Sulc
        sns.scatterplot(data=sub_df[sub_df["Hemi"] == "left"],
                        x="Week", y="Sulc_Mean", ax=ax,
                        color='cornflowerblue', label="Left - Sulc", alpha=0.6, marker='x')

        sns.scatterplot(data=sub_df[sub_df["Hemi"] == "right"],
                        x="Week", y="Sulc_Mean", ax=ax,
                        color='lightcoral', label="Right - Sulc", alpha=0.6, marker='x')

        ax.set_xlabel("Âge (semaines)")
        ax.set_ylabel("Profondeur moyenne")
        ax.legend()
        ax.grid(True)

        # Sauvegarde
        fname = roi.replace(" ", "_")
        fig.savefig(os.path.join(output_dir, f"{fname}_sulc_vs_dpfstarAbs.png"), dpi=300)
        plt.close()
        print(f"Figure enregistrée pour {roi}")

if __name__ == "__main__":
    main()
