#!/bin/bash

#SBATCH --job-name=gpu_analysis                 

#SBATCH --account=gpu_gres  # to access gpu resources

#SBATCH --nodes=1       # request to run job on single node                                       

#SBATCH --ntasks=5     # request 10 CPU's (t3gpu01: balance between CPU and GPU : 5CPU/1GPU)      

#SBATCH --gres=gpu:1    # request 1 GPU's on machine                                         

echo "submitting sample $1"

PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist datasets/$1.txt --sample $1  --outdir test/ --use-cuda --from-cache #--DNN save-arrays --cache-location /scratch/druini/cache/ 
