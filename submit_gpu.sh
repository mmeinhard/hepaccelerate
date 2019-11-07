echo "start running python script"

mc_samples=(
#ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8
#ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8
#TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8
#TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8
#TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8
TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8
TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8
TTToHadronic_TuneCP5_13TeV-powheg-pythia8
)
for sample in ${mc_samples[@]}; do
  echo "submitting sample $sample"
  sbatch gpu_job.sh $sample
done

echo "done"
