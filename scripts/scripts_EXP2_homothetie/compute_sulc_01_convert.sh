module load all
module load freesurfer

DB=/envau/work/meca/users/dieudonne.m/EXP2_homotety
cd $DB
subjects="CC00735XX18_222201"
scales="32500 65000 130000 260000 520000 1040000 1755000 4160000 8125000"
for subject in $subjects
do
  echo ${subject}
  for scale in $scales
  do
    echo ${subject}
    in_mesh=${DB}/MESH/${subject}/${subject}_${scale}.gii
    out_mesh=${DB}/MRIS_MESH/${subject}/${subject}_${scale}_mris.gii
    mris_convert in_mesh out_mesh
  done
done