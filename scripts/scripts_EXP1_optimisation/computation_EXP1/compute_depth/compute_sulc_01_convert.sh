module load all
module load freesurfer

DB=/envau/work/meca/users/dieudonne.m/EXP1_optimisation
cd $DB
subjects="KKI2009_113 KKI2009_505 KKI2009_142"
for subject in $subjects
do
  echo ${subject}
  in_mesh=${DB}/meshes/sub-${subject}_ses-MR1_hemi-L_space-T2w_wm.surf.gii
  out_mesh=${DB}/mris_meshes/sub-${subject}_ses-MR1_hemi-L_space-T2w_wm.surf.gii
  mris_convert ${in_mesh} ${out_mesh}
done