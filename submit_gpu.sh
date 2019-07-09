#!/bin/bash

#SBATCH --job-name=gpu_analysis                 

#SBATCH --account=gpu_gres  # to access gpu resources

#SBATCH --nodes=1       # request to run job on single node                                       

#SBATCH --ntasks=5     # request 10 CPU's (t3gpu01: balance between CPU and GPU : 5CPU/1GPU)      

#SBATCH --gres=gpu:1    # request 1 GPU's on machine                                         

echo "start running python script"

PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8.txt --sample ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8 --from-cache --evaluate-DNN --use-cuda

echo "done"
