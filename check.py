import os, os.path

import argparse
parser = argparse.ArgumentParser(description='Checks output of batch jobs')
parser.add_argument('--logs', help='directory with submission scripts / logs', type=str, default=None, required=True)
parser.add_argument('--outdir', help='directory with output files', type=str, default=None, required=True)
parser.add_argument('--resubmit', action='store_true', help='resubmit missing files')
args = parser.parse_args()


# simple version for working with CWD
njobs = len([f for f in os.listdir(args.logs) if ".job" in f])
sample = args.logs.split('/')[-2]
print(sample)

for nCount in range(0,njobs):

    if not os.path.exists(args.outdir + "out_" + sample + "_" + str(nCount) + ".json"):
        print(args.outdir + "out_" + sample + "_" + str(nCount) + ".json")
        if args.resubmit:
            os.system("sbatch -o {0}/slurm-{1}_resubmit.out {0}/{1}.job".format(args.logs,nCount))
