#!/bin/bash

# Run all DATA
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleElectron_RunB.txt --sample Run2017B_SingleElectron --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleElectron_RunC.txt --sample Run2017C_SingleElectron --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleElectron_RunD.txt --sample Run2017D_SingleElectron --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleElectron_RunE.txt --sample Run2017E_SingleElectron --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleElectron_RunF.txt --sample Run2017F_SingleElectron --from-cache

#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleMuon_RunB.txt --sample Run2017B_SingleMuon --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleMuon_RunC.txt --sample Run2017C_SingleMuon --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleMuon_RunD.txt --sample Run2017D_SingleMuon --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleMuon_RunE.txt --sample Run2017E_SingleMuon --from-cache
#PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/SingleMuon_RunF.txt --sample Run2017F_SingleMuon --from-cache

# Run all MC

PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8.txt --sample ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8 --from-cache --evaluate-DNN 
PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8.txt --sample ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8 --from-cache /scratch/creissel/cache/ --evaluate-DNN 


PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8.txt --sample TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8 --from-cache --cache-location /scratch/creissel/cache/ --evaluate-DNN
PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8.txt --sample TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8 --from-cache --cache-location /scratch/creissel/cache/ --evaluate-DNN
PYTHONPATH=.:$PYTHONPATH python3 simple_ttH.py --filelist datasets/TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8.txt --sample TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8 --from-cache --cache-location /scratch/creissel/cache/ --evaluate-DNN
