echo "start running python script"

mc_samples=(
ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8
ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8
TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8
TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8
TTToHadronic_TuneCP5_13TeV-powheg-pythia8
)
for sample in ${mc_samples[@]}; do
  echo "submitting sample $sample"
  sbatch gpu_job.sh $sample
  #PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist datasets/RunIIFall17NanoAODv4/$sample.txt --sample $sample  --outdir results_studyBoostedStatistics/mc --from-cache --boosted --use-cuda #--DNN save-arrays --cache-location /scratch/druini/cache/ 
  #python run_analysis.py --filelist datasets/RunIIFall17NanoAODv4/$sample.txt --sample $sample  --cache-location /scratch/druini/cache/ --outdir /scratch/druini/results_studyBoostedStatistics/mc --from-cache --boosted #--DNN save-arrays
done

echo "done"
