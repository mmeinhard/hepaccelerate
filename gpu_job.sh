#!/bin/bash

#SBATCH --job-name=gpu_analysis                 

#SBATCH --account=gpu_gres  # to access gpu resources

#SBATCH --partition=gpu

#SBATCH --nodes=1       # request to run job on single node                                       

#SBATCH --ntasks=5     # request 10 CPU's (t3gpu01: balance between CPU and GPU : 5CPU/1GPU)      

#SBATCH --gres=gpu:1    # request 1 GPU's on machine                                         

echo "submitting sample $1"

time PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist datasets_druini/$1.txt --sample $1  --outdir /work/creissel/GPUanalysis/hepaccelerate/massfit_druini/  --from-cache --cache-location /work/druini/hepaccelerate/cache/  --use-cuda --DNN mass_fit --path-to-model /work/creissel/MODEL/Oct16_ttHbb/ --categories sl_j4_t3 sl_j4_tge4 sl_j5_t3 sl_j5_tge4 sl_jge6_t3 sl_jge6_tge4 sl_jge4_tge2 sl_jge4_tge3
