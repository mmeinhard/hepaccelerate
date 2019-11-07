#!/bin/bash

#SBATCH --job-name=gpu_analysis                 

#SBATCH --account=gpu_gres  # to access gpu resources

#SBATCH --partition=gpu

#SBATCH --nodes=1       # request to run job on single node                                       

#SBATCH --ntasks=5     # request 10 CPU's (t3gpu01: balance between CPU and GPU : 5CPU/1GPU)      

#SBATCH --gres=gpu:1    # request 1 GPU's on machine                                         

echo "submitting test"

time PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist datasets/test.txt --sample ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8  --outdir ./bla/  --from-cache --use-cuda --DNN mass_fit --path-to-model /work/creissel/MODEL/Sep19/cms_ttbb/model.hdf5

