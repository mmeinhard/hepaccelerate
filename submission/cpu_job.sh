#!/bin/bash

#SBATCH --job-name=cpu_analysis                 

#SBATCH -p wn

#SBATCH -w t3wn60

echo "submitting sample $1"

PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist datasets/$1.txt --sample $1  --outdir ./tests/ --from-cache #--DNN save-arrays --cache-location /scratch/druini/cache/ 
