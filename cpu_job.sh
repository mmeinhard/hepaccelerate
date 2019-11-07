#!/bin/bash

#SBATCH --job-name=cpu_analysis                 

#SBATCH -p wn

#SBATCH -w t3wn60

echo "submitting sample $1"

time PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist datasets/$1.txt --sample $1  --outdir ./t4/ --from-cache --nthreads 4 # --DNN cmb_binary --path-to-model /work/creissel/MODEL/Mar25/binaryclassifier/model.hdf5  
