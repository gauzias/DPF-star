from pathlib import Path
import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# Paths
base_path = Path("D:/Callisto/repo/paper_sulcal_depth")
stats_path = base_path / "data_EXP1" / "result_EXP1" / "metrics" / "stats_wilcoxon"
figures_path = base_path / "data_EXP1" / "result_EXP1" / "metrics" / "stats_wilcoxon" 

# Load data
df_dev = pd.read_csv(stats_path / "stats_dev.csv")
df_diff = pd.read_csv(stats_path / "stats_diff.csv")
df_std = pd.read_csv(stats_path / "stats_std.csv")
df_opt = pd.read_csv(figures_path / "optimal_alphas.csv")

# Prepare results container
results = {}

# Function to test each alpha against the optimal
def wilcoxon_vs_optimal(df, alpha_opt):
    alpha_columns = df.columns[3:]  # Skip metadata columns
    p_values = []
    labels = []

    for alpha in alpha_columns:
        print(alpha_opt, alpha)
        if alpha == str(alpha_opt):
            continue
        try:
            test = pg.wilcoxon(df[str(alpha)], df[str(alpha_opt)], alternative='two-sided')
            p_values.append(test['p-val'].values[0])
            labels.append(alpha)
        except Exception:
            p_values.append(1.0)
            labels.append(alpha)

    # Correction FDR
    #reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    pvals_corrected = [18*pv for pv in p_values] # beferoni correction
    reject = [pvc < 0.05 for pvc in pvals_corrected]
    # Build result
    results_df = pd.DataFrame({
        'alpha': labels,
        'p-raw': p_values,
        'p-corrected': pvals_corrected,
        'significant': ['yes' if r else 'no' for r in reject]
    })

    return results_df

# Run comparisons for each metric
res_dev = wilcoxon_vs_optimal(df_dev, df_opt['alpha_opt_dev'].iloc[0])
res_diff = wilcoxon_vs_optimal(df_diff, df_opt['alpha_opt_diff'].iloc[0])
res_std = wilcoxon_vs_optimal(df_std, df_opt['alpha_opt_std'].iloc[0])

# Save each result
res_dev.to_csv(figures_path / "wilcoxon_vs_optimal_dev.csv")
res_diff.to_csv(figures_path / "wilcoxon_vs_optimal_diff.csv")
res_std.to_csv(figures_path / "wilcoxon_vs_optimal_std.csv")

print("All Wilcoxon test results saved.")
