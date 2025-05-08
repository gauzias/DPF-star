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


alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
for alpha in alphas : 
    test = pg.wilcoxon(df_diff[str(alpha)], df_diff[str(2000)], alternative='two-sided')
    print(str(alpha), ':' , test['p-val'].values[0])
            


