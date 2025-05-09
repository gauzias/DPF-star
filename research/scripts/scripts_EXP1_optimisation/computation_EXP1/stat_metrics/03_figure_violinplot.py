import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Significance utilities ---
def pval_to_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

def pval_to_color(p):
    if p < 0.001: return '#08306b'
    elif p < 0.01: return '#2171b5'
    elif p < 0.05: return '#6baed6'
    else: return 'lightgray'

# --- Plot function ---
def plot_metric(file_data: Path, file_pval: Path, metric_name: str, out_dir: Path):
    # Load and format data
    df_data = pd.read_csv(file_data).iloc[:, 5:]  # Drop first 5 columns
    alpha_labels = df_data.columns.astype(str).tolist()
    df_long = df_data.melt(var_name='alpha', value_name='value')
    df_long['alpha'] = df_long['alpha'].astype(str)

    # Load p-values
    df_pval = pd.read_csv(file_pval, skiprows=[1, 2])
    pvals = dict(zip(df_pval['alpha'].astype(str), df_pval['p-corrected']))

    # Assign colors to each alpha
    color_map = {alpha: pval_to_color(pvals.get(alpha, 1.0)) for alpha in alpha_labels}

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=df_long, x='alpha', y='value', hue='alpha', order=alpha_labels,
                   palette=color_map, linewidth=1, saturation=1, ax=ax, legend=False)

    # Add significance stars
    y_max = df_long['value'].max()
    y_min = df_long['value'].min()
    spacing = (y_max - y_min) * 0.05
    for i, alpha in enumerate(alpha_labels):
        star = pval_to_star(pvals.get(alpha, 1.0))
        if star:
            ax.text(i, y_max + spacing, star, ha='center', va='bottom', fontsize=12, color='red')

    ax.set_title(f"{metric_name.upper()} distribution per alpha : colored by p-value vs optimal")
    ax.set_xlabel("Alpha")
    ax.set_ylabel(metric_name.upper())
    plt.tight_layout()

    output_file = out_dir / f"violin_{metric_name}.png"
    plt.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close()

# --- Paths ---
wd = Path('D:/Callisto/repo/paper_sulcal_depth/data_EXP1/result_EXP1/metrics/stats_wilcoxon')
out_dir = wd
metrics = ['dev', 'std', 'diff']

# --- Plot all metrics ---
for metric in metrics:
    plot_metric(
        file_data=wd / f"stats_{metric}.csv",
        file_pval=wd / f"wilcoxon_vs_optimal_{metric}.csv",
        metric_name=metric,
        out_dir=out_dir
    )
