from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# --- Paths ---
wd = Path('D:/Callisto/repo/paper_sulcal_depth')
folder_metrics = wd / 'data/group_analysis/dev'
df_subjects = pd.read_csv(wd / 'datasets' / 'dataset_EXP1.csv')
optimal_alphas = pd.read_csv(wd / "data_EXP1" / "result_EXP1" / "metrics" / "stats_wilcoxon" / 'optimal_alphas.csv')
out_dir = wd / "data_EXP1" / "result_EXP1" / "metrics" / "stats_wilcoxon"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Subjects ---
list_subs = df_subjects['participant_id'].values[:-3]
list_ses = df_subjects['session_id'].values[:-3]

# --- Alphas ---
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
alpha_labels = [str(a) for a in alphas]

# --- Load metrics ---
def load_metric(subs, ses, suffix):
    all_vals = []
    for i, sub in enumerate(subs):
        file = folder_metrics / f"{sub}_{ses[i]}_{suffix}.pkl"
        with open(file, 'rb') as f:
            all_vals.append(pickle.load(f))
    return np.transpose(np.array(all_vals))

def load_single_metric(subs, ses, suffix, transform=None, index=None):
    all_vals = []
    for i, sub in enumerate(subs):
        file = folder_metrics / f"{sub}_{ses[i]}_{suffix}.pkl"
        with open(file, 'rb') as f:
            data = pickle.load(f)
            if transform:
                data = transform(data)
            all_vals.append(data)
    all_vals = np.array(all_vals)
    if index is not None and all_vals.ndim > 1:
        return all_vals.T[index]
    else:
        return all_vals.flatten()

# --- Load data ---
dev_dpfs = load_metric(list_subs, list_ses, 'angle_dev_mean')
std_dpfs = np.sqrt(load_metric(list_subs, list_ses, 'var_crest_dpf_norm'))
diff_dpfs = load_metric(list_subs, list_ses, 'diff_dunficrest_dpf_norm')

dev_dpf003 = 180 - load_single_metric(list_subs, list_ses, 'angle_dev_mean_dpf003', index=8)
std_dpf003 = load_single_metric(list_subs, list_ses, 'var_crest_dpf003', transform=np.sqrt, index=8)
diff_dpf003 = load_single_metric(list_subs, list_ses, 'diff_dunficrest_dpf003', index=8)

dev_sulc = load_single_metric(list_subs, list_ses, 'angle_dev_mean_sulc')
std_sulc = load_single_metric(list_subs, list_ses, 'var_crest_sulc', transform=np.sqrt)
diff_sulc = load_single_metric(list_subs, list_ses, 'diff_dunficrest_sulc')

# --- Build long DataFrames ---
def make_long_df(matrix, name):
    df = pd.DataFrame()
    for i, alpha in enumerate(alpha_labels):
        df = pd.concat([df, pd.DataFrame({'sub': list_subs, 'alpha': alpha, name: matrix[i]})])
    return df.reset_index(drop=True)

def add_method(df, arr, label):
    return pd.concat([df, pd.DataFrame({'sub': list_subs, 'alpha': label, 'value': arr})], ignore_index=True)

df_dev = make_long_df(dev_dpfs, 'value')
df_dev = add_method(df_dev, dev_sulc, 'sulc')
df_dev = add_method(df_dev, dev_dpf003, 'dpf')

df_std = make_long_df(std_dpfs, 'value')
df_std = add_method(df_std, std_sulc, 'sulc')
df_std = add_method(df_std, std_dpf003, 'dpf')

df_diff = make_long_df(diff_dpfs, 'value')
df_diff = add_method(df_diff, diff_sulc, 'sulc')
df_diff = add_method(df_diff, diff_dpf003, 'dpf')

# --- Significance tools ---
def pval_to_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

def pval_to_color(p):
    if p < 0.001: return '#08306b'  # dark blue
    elif p < 0.01: return '#2171b5'
    elif p < 0.05: return '#6baed6'
    else: return 'lightgray'

# --- Load precomputed p-values ---
def load_pvals(metric):
    pval_path = out_dir / f"wilcoxon_vs_optimal_{metric}.csv"
    df_pval = pd.read_csv(pval_path)
    pval_dict = {row['alpha']: row['p-corrected'] for _, row in df_pval.iterrows()}
    return pval_dict

# --- Plot function ---
def plot_violin(df, metric, alpha_opt):
    order = df['alpha'].unique().tolist()
    data_per_alpha = {a: df[df['alpha'] == a]['value'].values for a in order}
    df_opt = data_per_alpha[alpha_opt]

    # Load precomputed p-values
    pvals = load_pvals(metric)

    colors = []
    for a in order:
        if a == alpha_opt:
            colors.append('red')
        else:
            p = pvals.get(a, 1.0)
            colors.append(pval_to_color(p))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=df, x='alpha', y='value', hue='alpha', order=order, palette=colors, ax=ax, linewidth=1, saturation=1, legend=False)

    y_max = df['value'].max()
    y_min = df['value'].min()
    spacing = (y_max - y_min) * 0.05

    for i, a in enumerate(order):
        if a == alpha_opt:
            continue
        p = pvals.get(a, 1.0)
        star = pval_to_star(p)
        if star:
            ax.text(i, y_max + spacing, star, ha='center', va='bottom', fontsize=12, color='red')

    ax.set_title(f"{metric.upper()} — violins colored by p-value vs α={alpha_opt}")
    plt.tight_layout()
    fig.savefig(out_dir / f"violin_{metric}.png")
    print(f"Saved: violin_{metric}.png")
    plt.close(fig)

# --- Execute for all metrics ---
plot_violin(df_dev, 'dev', str(optimal_alphas["alpha_opt_dev"].iloc[0]))
plot_violin(df_std, 'std', str(optimal_alphas["alpha_opt_std"].iloc[0]))
plot_violin(df_diff, 'diff', str(optimal_alphas["alpha_opt_diff"].iloc[0]))
