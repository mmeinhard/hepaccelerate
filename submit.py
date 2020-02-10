#!/usr/bin/env python

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Runs a simple array-based analysis')
parser.add_argument('--datasets', help='directory with list of inputs', type=str, default=None, required=True)
parser.add_argument('--samples', nargs='+', help='List of samples to process', type=str, default=None, required=True)
parser.add_argument('--files-per-job', action='store', help='Number of files to process per job', type=int, default=1, required=False)
parser.add_argument('--batchSystem', help='batch system to submit to', type=str, default='slurm_cpu', required=False)

parser.add_argument('--from-cache', action='store_true', help='Load from cache (otherwise create it)')
parser.add_argument('--files-per-batch', action='store', help='Number of files to process per batch', type=int, default=1, required=False)
parser.add_argument('--nthreads', action='store', help='Number of CPU threads to use', type=int, default=4, required=False)
parser.add_argument('--cache-location', action='store', help='Path prefix for the cache, must be writable', type=str, default=os.path.join(os.getcwd(), 'cache'))
parser.add_argument('--cache-only', action='store_true', help='Produce only cached files')
parser.add_argument('--jets-met-corrected', action='store_true', help='defines usage of pt_nom vs pt for jets and MET', default=False)
parser.add_argument('--outdir', action='store', help='directory to store outputs', type=str, default=os.getcwd())
parser.add_argument('--DNN', action='store', choices=['save-arrays','cmb_binary', 'cmb_multiclass', 'ffwd_binary', 'ffwd_multiclass',False, 'mass_fit'], help='options for DNN evaluation / preparation', default=False)
parser.add_argument('--categories', nargs='+', help='categories to be processed (default: sl_jge4_tge2)', default="sl_jge4_tge2")
parser.add_argument('--path-to-model', action='store', help='path to DNN model', type=str, default=None, required=False)
parser.add_argument('--year', action='store', choices=['2016', '2017', '2018'], help='Year of data/MC samples', default='2017')
args = parser.parse_args()

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def partitionFileList(filelist, chunkSize=1):
    sampleFileList = np.loadtxt(filelist, dtype=str)
    return [sampleFileList[i:i+chunkSize] for i in range(0, len(sampleFileList), chunkSize)]   

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
job_directory = "{0}/logs_{1}".format(os.getcwd(), timestr) 

# Make top level directories
mkdir_p(job_directory)

samples=args.samples
if not type(args.categories)==list:
    categories = [args.categories]
else:
    categories = args.categories

for s in samples:

    sample_directory = "{0}/{1}".format(job_directory, s)
    mkdir_p(sample_directory)
    par_files = partitionFileList(args.datasets+'/%s.txt'%s, args.files_per_job * args.files_per_batch)

    for njob,f in enumerate(par_files): 

        job_file = os.path.join(sample_directory,"%s.job" %njob)

        if args.batchSystem=="slurm_cpu":

            with open(job_file, "w") as fh:
                fh.write("#!/bin/bash\n")
                fh.write("#SBATCH --job-name={0}_{1}.job\n".format(s,njob))
                fh.write("#SBATCH -p wn\n")

                fh.write("mkdir /scratch/c/\n")
                fh.write("PYTHONPATH=hepaccelerate:coffea:. python3 {0}/run_analysis.py ".format(os.getcwd()))
                fh.write("--categories ")
                fh.write(' '.join(map(str, categories)))
                fh.write(" --sample {0} --files-per-batch {1} --nthread {2} --cache-location {3} --outdir {4} --path-to-model {5} --year {6} --outtag _{7} ".format(s, args.files_per_batch, args.nthreads, "/scratch/c/", args.outdir, args.path_to_model, args.year, njob))
                if args.DNN:
                    fh.write("--DNN {0} ".format(args.DNN))
                if args.from_cache:
                    fh.write("--from-cache ")
                if args.cache_only:
                    fh.write("--cache-only ")
                if args.jets_met_corrected:
                    fh.write("--jets-met-corrected ")
                fh.write(' '.join(map(str, f)))
                fh.write('\n')
                fh.write('rm -r /scratch/c/')

            os.system("sbatch -o {0}/slurm-{1}.out {2}".format(sample_directory,njob,job_file))

        elif args.batchSystem=="slurm_gpu":

            with open(job_file, "w") as fh:
                fh.write("#!/bin/bash\n")
                fh.write("#SBATCH --job-name={0}_{1}.job\n".format(s,njob))
                fh.write("#SBATCH --account=gpu_gres  # to access gpu resources\n")
                fh.write("#SBATCH --nodes=1       # request to run job on single node\n")
                fh.write("#SBATCH --ntasks=5     # request 10 CPU's (t3gpu01: balance between CPU and GPU : 5CPU/1GPU)\n")
                fh.write("#SBATCH --gres=gpu:1    # request 1 GPU's on machine\n")

                fh.write("PYTHONPATH=hepaccelerate:coffea:. python3 {0}/run_analysis.py ".format(os.getcwd()))
                fh.write("--categories ")
                fh.write(' '.join(map(str, categories)))
                fh.write(" --sample {0} --files-per-batch {1} --nthread {2} --cache-location {3} --outdir {4} --path-to-model {5} --year {6} ".format(s, args.files_per_batch, args.nthreads, args.cache_location, args.outdir, args.path_to_model, args.year))
                if args.DNN:
                    fh.write("--DNN {0} ".format(args.DNN))
                if args.from_cache:
                    fh.write("--from-cache ")
                if args.cache_only:
                    fh.write("--cache-only ")
                fh.write(' '.join(map(str, f)))

            os.system("sbatch -o {0}/slurm-{1}.out {2}".format(sample_directory,njob,job_file))

        else:
            print("Unknown batch system.")  
