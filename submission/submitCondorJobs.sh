#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'First argument, name of sample, is needed. Have a good day :)'
else

    sample=$1

    if [ ! -d condorlogs/ ]; then
        mkdir condorlogs/
    fi

    outputDir=/eos/home-a/algomez/tmpFiles/hepaccelerate/
    if [ ! -d "$outputDir" ]; then
        mkdir -p $outputDir
    fi
    if [ ! -d "${outputDir}/${sample}" ]; then
        mkdir -p $outputDir/$sample
    fi
    workingDir=${PWD}

    condorFile=${sample}_condorJob
    echo '''universe    =  vanilla
arguments   =  '${sample}' $(myfile) _$(ProcId)
executable  =  '${PWD}'/condorlogs/'${condorFile}'.sh
log         =  '${PWD}'/condorlogs/log_'${condorFile}'_$(ClusterId).log
error       =  '${PWD}'/condorlogs/log_'${condorFile}'_$(ClusterId)-$(ProcId).err
output      =  '${PWD}'/condorlogs/log_'${condorFile}'_$(ClusterId)-$(ProcId).out
initialdir  = '$outputDir'/'${sample}'/
getenv      =  True
requirements = (OpSysAndVer =?= "SLCern6")
+JobFlavour = "tomorrow"
queue
    ''' > condorlogs/${condorFile}.sub

    echo '''#!/bin/bash
export X509_USER_PROXY=/afs/cern.ch/user/a/algomez/x509up_u15148
export PATH=/afs/cern.ch/work/a/algomez/miniconda3/bin:$PATH
source activate hepaccelerate_cpu
cd '${workingDir}'
echo ${PWD}
echo "PYTHONPATH=hepaccelerate:coffea:. python3 '${workingDir}'/run_analysis.py --filelist '${workingDir}'/samples/2016_ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8.txt  --sample ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8 --cache-location='${workingDir}' --year 2016 --boosted True"
PYTHONPATH='${workingDir}'/coffea:. python3 '${workingDir}'/run_analysis.py --filelist '${workingDir}'/samples/2016_ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8.txt  --sample ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8 --cache-location='${outputDir}' --year 2016 --boosted True
    ''' > condorlogs/${condorFile}.sh

    condor_submit condorlogs/${condorFile}.sub

fi
