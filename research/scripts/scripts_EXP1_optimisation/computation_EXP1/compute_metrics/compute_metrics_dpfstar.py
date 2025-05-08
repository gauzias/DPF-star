from pathlib import Path
import pandas as pd
import numpy as np
import math
import pickle
from app.functions import rw as sio

# --- Paths ---
wd = Path('D:/Callisto/repo/paper_sulcal_depth')
dataset_path = wd / 'datasets' / 'dataset_EXP1.csv'
depth_base_path = wd / 'data_EXP1' / 'result_EXP1' / 'depth'
output_path = wd / 'data_EXP1' / 'result_EXP1' / 'metrics' / 'dpfstar'
dpfstar_suffix = '_dpfstar.gii'
label_suffix = '_lines.gii'

output_path.mkdir(parents=True, exist_ok=True)

# --- Params ---
alphas = [0, 1, 5, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
df_info = pd.read_csv(dataset_path)
subjects = df_info['participant_id'].values
sessions = df_info['session_id'].values

# --- Helpers ---
def unit_vector(v): return v / np.linalg.norm(v)
def angle_between(v1, v2): return math.degrees(np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)))

# --- Data containers ---
list_dev, list_std, list_diff = [], [], []

# --- Loop over subjects ---
for i, sub in enumerate(subjects):
    ses = sessions[i]
    print(f"Processing {sub}_{ses}")

    # Load textures
    dpfstar_path = depth_base_path / f"{sub}_{ses}" / 'dpfstar' / f"{sub}_{ses}{dpfstar_suffix}"
    dpfstars = sio.load_texture(dpfstar_path).darray

    line_path = wd / 'data_EXP1' / 'manual_labelisation' / 'MD_full' / f"{sub}_{ses}{label_suffix}"
    lines = sio.load_texture(line_path).darray[0]
    lines = np.round(lines).astype(int)
    crest = np.where(lines == 100)[0]
    fundi = np.where(lines == 50)[0]

    for j, alpha in enumerate(alphas):
        values = dpfstars[j]
        
        # std crest
        std = np.std(values[crest]) / np.abs(np.percentile(values, 95) - np.percentile(values, 5))
        list_std.append({'subject': sub, 'sessions': ses, 'alphas': alpha, 'std_crest_dpfstar': std})

        # diff crest/fundi
        diff = np.abs(np.median(values[crest]) - np.median(values[fundi])) / np.abs(np.percentile(values, 95) - np.percentile(values, 5))
        list_diff.append({'subject': sub, 'sessions': ses, 'alphas': alpha, 'diff_fundicrest_dpfstar': diff})

        # dev (mean angle)
        grad_surf = pd.read_csv(depth_base_path / f"{sub}_{ses}" / 'dmap_crest' / 'derivative' / f"{sub}_{ses}_grad_dmap_crest.csv")
        grad_dpf = pd.read_csv(depth_base_path / f"{sub}_{ses}" / 'dpfstar' / 'derivative' / f"{sub}_{ses}_grad_dpfstar.csv")

        g_dpf = grad_dpf[grad_dpf['alpha'] == alpha][['x', 'y', 'z']].values

        sulcalwall_path = wd / 'data_EXP1' / 'sulcalWall_lines' / f"{sub}_{ses}_sulcalwall.pkl"
        with open(sulcalwall_path, 'rb') as f:
            sw_lines = pickle.load(f)

        angles = []
        for trace_group in sw_lines:
            for trace in trace_group:
                if len(trace) > 5:
                    trace = trace[2:-2]
                for vtx in trace:
                    angles.append(angle_between(-grad_surf.iloc[vtx].values, g_dpf[vtx]))
        list_dev.append({'subject': sub, 'sessions': ses, 'alphas': alpha, 'angle_dpfstar': np.mean(angles)})

# --- Save CSVs ---
pd.DataFrame(list_std).to_csv(output_path / 'dpfstar_std_crest.csv', index=False)
pd.DataFrame(list_diff).to_csv(output_path / 'dpfstar_diff_fundicrest.csv', index=False)
pd.DataFrame(list_dev).to_csv(output_path / 'dpfstar_dev.csv', index=False)

print("All dpfstar metrics saved.")
