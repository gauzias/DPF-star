from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Working directory
wd = Path('D:/Callisto/repo/paper_sulcal_depth')
out_dir = wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon'
out_dir.mkdir(exist_ok=True)

# Load datasets
df_std = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_std.csv')
df_dev = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_dev.csv')
df_diff = pd.read_csv(wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'stats_wilcoxon' / 'stats_diff.csv')

# List of datasets with plot labels and optimization goal
datasets = [
    ("std", df_std, "normalised std on crest", "min"),
    ("dev", df_dev, "angular deviation (°)", "min"),
    ("diff", df_diff, "median diff crest/fundi", "max"),
]

# Store optimal alphas
optimal_alphas = {}

for tag, df, ylabel, mode in datasets:
    df_metrics = df.drop(columns=['subject', 'sessions'], errors='ignore')

    df_summary = pd.DataFrame({
        'alpha': df_metrics.columns[3:],
        'mean': df_metrics.median()[3:].values,
        'std': df_metrics.std()[3:].values
    })

    # Normalize mean and std
    mean_min, mean_max = df_summary['mean'].min(), df_summary['mean'].max()
    std_min, std_max = df_summary['std'].min(), df_summary['std'].max()
    df_summary['mean_n'] = (df_summary['mean'] - mean_min) / (mean_max - mean_min)
    df_summary['std_n'] = (df_summary['std'] - std_min) / (std_max - std_min)

    # Compute score based on mode
    if mode == "min":
        df_summary['score'] = df_summary['mean_n'] + df_summary['std_n']
    elif mode == "max":
        df_summary['score'] = -(df_summary['mean_n'] - df_summary['std_n'])
    else:
        raise ValueError("mode must be 'min' or 'max'")

    optimal_row = df_summary.sort_values(by='score').iloc[0]
    optimal_alpha = optimal_row['alpha']
    optimal_alphas[f'alpha_opt_{tag}'] = [optimal_alpha]

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(df_summary['mean'], df_summary['std'])

    for _, row in df_summary.iterrows():
        ax.text(row['mean'], row['std'], str(row['alpha']))

    # Highlight optimal point
    ax.scatter(optimal_row['mean'], optimal_row['std'], color='red', label='optimal', zorder=5)
    ax.legend()

    ax.set_xlabel('mean')
    ax.set_ylabel('std')
    ax.set_title(f'Optimal alpha for metric: {ylabel}')
    plt.tight_layout()

    fig_path = out_dir / f'figure_alpha_opt_{tag}.png'
    plt.savefig(fig_path)
    print(f"Saved: {fig_path}")
    plt.show()

# Save optimal alphas to CSV
df_optimal = pd.DataFrame(optimal_alphas)
csv_path = out_dir / 'optimal_alphas.csv'
df_optimal.to_csv(csv_path, index=False)
print(f"Saved optimal alpha CSV: {csv_path}")
