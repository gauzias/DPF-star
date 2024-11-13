DB=/home/maxime/callisto/repo/paper_sulcal_depth
cd $DB
subjects="CC00735XX18_222201"
scales="32500 65000 130000 260000 520000 1040000 1755000 4160000 8125000"
for subject in $subjects
do
  echo ${subject}
  for scale in $scales
  do
   echo ${scale}
   mris_mesh=${subject}_${scale}_mris.gii
   out_mesh=${DB}/data/scaled_meshes/mris_convert/${subject}/${subject}_${scale}_sulcal_depth.gii
   cd ${DB}/data/scaled_meshes/mris_convert/${subject}
   mirtk deform-mesh ${mris_mesh} ${out_mesh} -inflate-brain -track SulcalDepth
  done
done