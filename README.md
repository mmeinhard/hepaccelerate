# Costumized hepaccelerate framework for ttH(bb) analysis 

Framework for accelerated ttH(bb) array analysis on flat centrally produced nanoAOD samples applying object and event selection as well as pileup, lepton and btagging reweighting directly on a GPU! Or, if you don't have a GPU, no big deal, the same code works on the CPU too, just somewhat slower! No batch jobs or servers required!

Requirements:
 - python 3
 - uproot
 - awkward-array
 - numba
 - coffea tools (fermilab analysis tools)

Optional for CUDA acceleration:
 - cupy
 - cudatoolkit


## Getting started

~~~
git clone git@github.com:chreissel/hepaccelerate.git
cd hepaccelerate
~~~

The framework needs a list of files to process. These files can be either local files or files accessible via xrootd. Note reading files via xrootd will be much slower than caching local files, so if you are running the analysis you might consider copying the files to your own cluster using phedex.
~~~
#prepare a list of files to read
#replace /nvmedata with your local location of ROOT files
find /nvmedata/store/mc/RunIIFall17NanoAOD/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8*/NANOAODSIM -name "*.root > filelist.txt
~~~

In order to process the analysis, simply run:
~~~
#Run the test analysis
PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist filelist.txt --sample ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8

#output will be stored in this json
cat out_ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8.json
~~~
This script loads the ROOT files, prepares local caches from the branches you read and processes the data. The output file contains the weighted histograms.
~~~
#second time around, you can load the data from the cache, which is much faster
PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist filelist.txt --sample ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8 --from-cache

#use CUDA for array processing on a GPU!
PYTHONPATH=hepaccelerate:coffea:. python3 run_analysis.py --filelist filelist.txt --sample ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8 --from-cache --use-cuda
~~~

Object definitions, event selection cuts and files needed for scale factor calculations can be found in `definitions_analysis.py`. 

~~~
python counts.py --filelist filelist.txt
~~~
Running this script gives the total weighted sum of generated events for all files defined in `filelist.txt`. After changing the filelist, update the weighted sum of generated events in `definitions_analysis.py`.

Looking for the functions used for analysing the datasets? All important, analysis specific functions are defined in `lib_analysis.py`.

## Submitting slurm jobs to the PSI tier3
The scripts `submit_cpu.sh` and `submit_gpu.sh` in their current form submit one slurm job per sample. To use them, log in to the correct node (`ssh t3ui07` for cpu, `ssh t3login` for gpu), activate the correct environment:
~~~
export PATH=/work/creissel/anaconda3/bin:$PATH
source activate hepaccelerate_cpu
OR
source activate hepaccelerate_gpu
~~~
and simply source the `submit` script. The arguments passed to `run_analysis.py` are defined in `c(g)pu_job.sh`.

## Running the jupyter notebook from tier3
On your local machine run 
~~~
ssh -f YOURUSERNAME@t3ui02.psi.ch -L 8008:localhost:8008 -N
~~~
Then log in to tier3 and run
~~~
jupyter notebook --no-browser --port=8008
~~~
You will get a URL with a token to copy-paste into your browser. Done.
