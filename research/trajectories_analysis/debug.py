import os
import pandas as pd
import numpy as np
import nibabel as nib
from glob import glob
from app import compute_dpfstar
from research.tools import rw

def extract_roi_data(tsv_file, gii_file):
    """Extrait les labels des ROI depuis le fichier TSV et applique les masques."""
    roi_df = pd.read_csv(tsv_file, sep='\t')
    roi_data = dict(zip(roi_df['index'], roi_df['name']))
    #nii_data = nib.load(nii_file).get_fdata()
    gii_data = rw.read_gii_file(gii_file)
    return gii_data, roi_data


# Extraction des ROI
tsv_file = os.path.join("D:/Callisto/data/rel3_dhcp", 'desc-drawem32_dseg.tsv')
gii_file = os.path.join("D:/Callisto/data/rel3_dhcp",'sub-CC00060XX03', 'ses-12501', 'anat', 'sub-CC00060XX03_ses-12501_hemi-left_desc-drawem_dseg.label.gii')
roi_mask, roi_labels = extract_roi_data(tsv_file, gii_file)

print(roi_mask)
print(roi_labels)
