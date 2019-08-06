echo "start running python script"

mc_samples=(
ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8
)
for sample in ${mc_samples[@]}; do
  echo "submitting sample $sample"
  sbatch gpu_job.sh $sample
done

echo "done"
