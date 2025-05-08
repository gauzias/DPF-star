import pandas as pd

# Chargement des fichiers (corrige les noms si besoin)
dev_df = pd.read_csv("D:/Callisto/repo/paper_sulcal_depth/data_EXP1/result_EXP1/metrics/stats_wilcoxon/stats_dev.csv")
std_df = pd.read_csv("D:/Callisto/repo/paper_sulcal_depth/data_EXP1/result_EXP1/metrics/stats_wilcoxon/stats_std.csv")
diff_df = pd.read_csv("D:/Callisto/repo/paper_sulcal_depth/data_EXP1/result_EXP1/metrics/stats_wilcoxon/stats_diff.csv")

# Colonnes à traiter
target_columns = ['0', '1', '5', '10', '50', '100', '150', '200', '250', '300',
                  '400', '500', '600', '700', '800', '900', '1000', '2000']
target_columns = [str(c) for c in target_columns]

# Calcul des médianes
dev_medians = dev_df[target_columns].median()
std_medians = std_df[target_columns].median()
diff_medians = diff_df[target_columns].median()

# Extraction des meilleurs alphas
alpha_opt_dev = dev_medians.idxmin()
alpha_opt_std = std_medians.idxmin()
alpha_opt_diff = diff_medians.idxmax()

# Résultat sous forme d'une ligne
optimal_alphas = pd.DataFrame([{
    "alpha_opt_dev": alpha_opt_dev,
    "alpha_opt_std": alpha_opt_std,
    "alpha_opt_diff": alpha_opt_diff
}])

# Sauvegarde dans le fichier
output_path = "D:/Callisto/repo/paper_sulcal_depth/data_EXP1/result_EXP1/metrics/stats_wilcoxon/optimal_alphas.csv"
optimal_alphas.to_csv(output_path, index=False)
