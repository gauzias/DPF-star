import os
import pandas as pd
import numpy as np
from glob import glob
from research.tools import rw

# Configuration
mesh_root = r"E:\dhcpSym_template"
tsv_file = r"E:\dhcpSym_sulci_manual\dHCP_atlas_sulci_normenclature.tsv"
dir_result = r"E:/research_dpfstar/result_dhcpSym"
output_csv_root = os.path.join(dir_result, "cortical_metrics_sulc")

# Hémisphères à traiter
hemispheres = ["left","right"]

def extract_roi_data(tsv_path, gii_file):
    roi_df = pd.read_csv(tsv_path, sep='\t')
    roi_data = dict(zip(roi_df['index'], roi_df['name']))
    gii_data = rw.read_gii_file(gii_file)
    gii_data=  np.round(gii_data).astype(int)
    return gii_data, roi_data

def process_surface(mesh_path, hemi):
    subject_name = os.path.basename(mesh_path).replace('.gii', '')  # week-28_hemi-left_...
    week_id = subject_name.split("_")[0]  # week-28
    save_csv_folder = os.path.join(output_csv_root, week_id, f"hemi_{hemi}")
    os.makedirs(save_csv_folder, exist_ok=True)

    # Chemin label GII (même nom que le mesh mais avec `_label`)
    label_file = os.path.join("E:\dhcpSym_sulci_manual", f"week-44_hemi-{hemi}_space-dhcpSym_dens-32k_wm.sulci_manual.shape.gii")
    
    if not os.path.exists(label_file):
        print(f"Label manquant pour {subject_name} — ignoré.")
        return

    # Charger DPFstar
    depth_path = os.path.join(
        dir_result, subject_name, "dpfstar.gii"
    )
    depth_path = os.path.join("E:\dhcpSym_template", f"{week_id}", f"hemi_{hemi}", "dhcpSym",
                                 f"{week_id}_hemi-{hemi}_space-dhcpSym_dens-32k_sulc.shape.gii")
    print(depth_path)
    if not os.path.exists(depth_path):
        print(f"DPFstar manquant pour {subject_name}")
        return

    depth_data = rw.read_gii_file(depth_path)
    roi_mask, roi_labels = extract_roi_data(tsv_file, label_file)
    all_data = []

    for label, roi_name in roi_labels.items():
        #print(label, roi_name)
        print(np.unique(roi_mask))
        roi_mask_indices = np.where(roi_mask == label)
        if len(roi_mask_indices[0]) == 0:
            continue

        roi_depth = depth_data[roi_mask_indices]


        all_data.append({
            'Week': week_id.replace('week-', ''),
            'Hemi': hemi,
            'ROI': roi_name,
            'Label': label,
            'Sulc_Mean_abs': np.mean(np.abs(roi_depth)),
            'Sulc_Mean': np.mean(roi_depth),
            'Sulc_Median': np.median(roi_depth),
            'Sulc_Min': np.min(roi_depth),
            'Sulc_5p' : np.percentile(roi_depth, 5)
        })

    # Enregistrer CSV
    df = pd.DataFrame(all_data)
    if not df.empty:
        output_file = os.path.join(save_csv_folder, f"{subject_name}_cortical_metrics.csv")
        df.to_csv(output_file, index=False)
        print(f"Fichier enregistré : {output_file}")
    else:
        print(f"Aucune donnée ROI pour {subject_name}")

def main():
    for week_dir in sorted(glob(os.path.join(mesh_root, 'week-*'))):
        week_id = os.path.basename(week_dir)
        for hemi in hemispheres:
            mesh_pattern = os.path.join(week_dir, f"hemi_{hemi}", "dhcpSym", f"{week_id}_hemi-{hemi}_space-dhcpSym_dens-32k_wm.surf.gii")
            if os.path.exists(mesh_pattern):
                process_surface(mesh_pattern, hemi)
            else:
                print(f"Fichier mesh manquant : {mesh_pattern}")

if __name__ == "__main__":
    main()
