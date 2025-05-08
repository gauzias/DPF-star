from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import math
from app.functions import rw as sio

# --- Settings ---
wd = Path('D:/Callisto/repo/paper_sulcal_depth')
dataset_path = wd / 'datasets' / 'dataset_EXP1.csv'
depth_base_path = wd / 'data_EXP1' / 'result_EXP1' / 'depth'
output_path = wd / 'data_EXP1' / 'result_EXP1' / 'metrics'
mesh_path = wd / 'data_EXP1' / 'meshes'
label_path = wd / 'data_EXP1' / 'manual_labelisation' / 'MD_full'
sulcalwall_path = wd / 'data_EXP1' / 'sulcalWall_lines'

# Alpha values
alphas_dpfstar = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
alphas_dpf = [0, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.5]

# Load subject info
df_info = pd.read_csv(dataset_path)
subjects = df_info['participant_id'].values
sessions = df_info['session_id'].values
datasets = df_info['dataset'].values

# --- Helpers ---
def unit_vector(v): return v / np.linalg.norm(v)
def angle_between(v1, v2): return math.degrees(np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)))

# --- Compute all metrics ---
df_curv_std, df_sulc_std, df_dpf003_std = [], [], []
df_curv_diff, df_sulc_diff, df_dpf003_diff = [], [], []
df_dpfstar_std, df_dpfstar_diff, df_dpf_dev = [], [], []
df_sulc_dev, df_dpf003_dev, df_curv_dev, df_dpfstar_dev = [], [], [], []

for idx, sub in enumerate(subjects):
    ses = sessions[idx]
    dset = datasets[idx]
    print(f"Processing {sub}")

    # Paths
    curv_dir = depth_base_path / f'{sub}_{ses}' / 'curvature'
    sulc_dir = depth_base_path / f'{sub}_{ses}' / 'sulc'
    dpf003_dir = depth_base_path / f'{sub}_{ses}' / 'dpf003'
    dpfstar_path = depth_base_path / f'{sub}_{ses}' / 'dpfstar' / f'{sub}_{ses}_dpfstar.gii'
    mesh_file = mesh_path / f'sub-{sub}_ses-{ses}_hemi-L_space-T2w_wm.surf.gii'
    line_file = label_path / f'{sub}_{ses}_lines.gii'
    sulc_file = sulc_dir / (f"sub-{sub}_ses-{ses}_hemi-L_space-T2w_sulc.shape.gii" if dset == 'dHCP' else f"{sub}_{ses}_sulc.gii")

    # Load data
    K1 = sio.load_texture(curv_dir / f'{sub}_{ses}_K1.gii').darray[0]
    K2 = sio.load_texture(curv_dir / f'{sub}_{ses}_K2.gii').darray[0]
    curv = 0.5 * (K1 + K2)
    sulc = sio.load_texture(sulc_file).darray[0]
    dpf003 = sio.load_texture(dpf003_dir / f'{sub}_{ses}_dpf003.gii').darray[0]
    dpfstar_all = sio.load_texture(dpfstar_path).darray
    lines = np.round(sio.load_texture(line_file).darray[0]).astype(int)
    crest = np.where(lines == 100)[0]
    fundi = np.where(lines == 50)[0]

    def norm_std(x): return np.std(x) / np.abs(np.percentile(x, 95) - np.percentile(x, 5))
    def norm_diff(x): return np.abs(np.median(x[crest]) - np.median(x[fundi])) / np.abs(np.percentile(x, 95) - np.percentile(x, 5))

    df_curv_std.append({'subject': sub, 'sessions': ses, 'std_crest_curv': norm_std(curv[crest])})
    df_sulc_std.append({'subject': sub, 'sessions': ses, 'std_crest_sulc': norm_std(sulc[crest])})
    df_dpf003_std.append({'subject': sub, 'sessions': ses, 'std_crest_dpf003': norm_std(dpf003[crest])})

    df_curv_diff.append({'subject': sub, 'sessions': ses, 'diff_fundicrest_curv': norm_diff(curv)})
    df_sulc_diff.append({'subject': sub, 'sessions': ses, 'diff_fundicrest_sulc': norm_diff(sulc)})
    df_dpf003_diff.append({'subject': sub, 'sessions': ses, 'diff_fundicrest_dpf003': norm_diff(dpf003)})

    for j, alpha in enumerate(alphas_dpfstar):
        df_dpfstar_std.append({'subject': sub, 'sessions': ses, 'alphas': alpha, 'std_crest_dpfstar': norm_std(dpfstar_all[j][crest])})
        df_dpfstar_diff.append({'subject': sub, 'sessions': ses, 'alphas': alpha, 'diff_fundicrest_dpfstar': norm_diff(dpfstar_all[j])})

    mesh = sio.load_mesh(str(mesh_file))
    grad_surface = pd.read_csv(depth_base_path / f'{sub}_{ses}' / 'dmap_crest' / 'derivative' / f'{sub}_{ses}_grad_dmap_crest.csv')
    grad_dpf = pd.read_csv(depth_base_path / f'{sub}_{ses}' / 'dpf' / 'derivative' / f'{sub}_{ses}_grad_dpf.csv')
    grad_dpf003 = pd.read_csv(depth_base_path / f'{sub}_{ses}' / 'dpf003' / 'derivative' / f'{sub}_{ses}_grad_dpf003.csv')
    grad_curv = pd.read_csv(depth_base_path / f'{sub}_{ses}' / 'curvature' / 'derivative' / f'{sub}_{ses}_grad_curv.csv')
    grad_sulc = pd.read_csv(depth_base_path / f'{sub}_{ses}' / 'sulc' / 'derivative' / f'{sub}_{ses}_grad_sulc.csv')
    grad_dpfstar = pd.read_csv(depth_base_path / f'{sub}_{ses}' / 'dpfstar' / 'derivative' / f'{sub}_{ses}_grad_dpfstar.csv')

    with open(sulcalwall_path / f'{sub}_{ses}_sulcalwall.pkl', 'rb') as f:
        sw_lines = pickle.load(f)

    for alpha in alphas_dpf:
        g_dpf = grad_dpf[grad_dpf['alpha'] == alpha][['x', 'y', 'z']].values
        angles = [angle_between(-grad_surface.iloc[vtx], g_dpf[vtx])
                  for trace_group in sw_lines for trace in trace_group
                  if len(trace) > 5 for vtx in trace[2:-2]]
        df_dpf_dev.append({'subject': sub, 'sessions': ses, 'alphas': alpha, 'dev_dpf': np.mean(angles)})

    def compute_dev(metric, name):
        angles = [angle_between(-grad_surface.iloc[vtx], metric[vtx])
                  for trace_group in sw_lines for trace in trace_group
                  if len(trace) > 5 for vtx in trace[2:-2]]
        return {'subject': sub, 'sessions': ses, f'dev_{name}': np.mean(angles)}

    df_sulc_dev.append(compute_dev(grad_sulc[['x', 'y', 'z']].values, 'sulc'))
    df_curv_dev.append(compute_dev(grad_curv[['x', 'y', 'z']].values, 'curv'))
    df_dpf003_dev.append(compute_dev(grad_dpf003[['x', 'y', 'z']].values, 'dpf003'))
    df_dpfstar_dev.append(compute_dev(grad_dpfstar[['x', 'y', 'z']].values, 'dpfstar'))

# Ensure subdirectories exist
for subdir in ['curv', 'sulc', 'dpf003', 'dpfstar', 'dpf']:
    (output_path / subdir).mkdir(parents=True, exist_ok=True)

# Save all
pd.DataFrame(df_curv_std).to_csv(output_path / 'curv' / 'curv_std_crest.csv', index=False)
pd.DataFrame(df_sulc_std).to_csv(output_path / 'sulc' / 'sulc_std_crest.csv', index=False)
pd.DataFrame(df_dpf003_std).to_csv(output_path / 'dpf003' / 'dpf003_std_crest.csv', index=False)
pd.DataFrame(df_dpfstar_std).to_csv(output_path / 'dpfstar' / 'dpfstar_std_crest.csv', index=False)

pd.DataFrame(df_curv_diff).to_csv(output_path / 'curv' / 'curv_diff_fundicrest.csv', index=False)
pd.DataFrame(df_sulc_diff).to_csv(output_path / 'sulc' / 'sulc_diff_fundicrest.csv', index=False)
pd.DataFrame(df_dpf003_diff).to_csv(output_path / 'dpf003' / 'dpf003_diff_fundicrest.csv', index=False)
pd.DataFrame(df_dpfstar_diff).to_csv(output_path / 'dpfstar' / 'dpfstar_diff_fundicrest.csv', index=False)

pd.DataFrame(df_dpf_dev).to_csv(output_path / 'dpf' / 'dpfs_dev.csv', index=False)
pd.DataFrame(df_sulc_dev).to_csv(output_path / 'sulc' / 'sulc_dev.csv', index=False)
pd.DataFrame(df_curv_dev).to_csv(output_path / 'curv' / 'curv_dev.csv', index=False)
pd.DataFrame(df_dpf003_dev).to_csv(output_path / 'dpf003' / 'dpf003_dev.csv', index=False)
pd.DataFrame(df_dpfstar_dev).to_csv(output_path / 'dpfstar' / 'dpfstar_dev.csv', index=False)

print("All metrics computed and saved successfully.")
