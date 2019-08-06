#!/usr/bin/python
import sys,os,time
import argparse, shutil
from dbs.apis.dbsClient import DbsApi
dbsPhys03 = DbsApi('https://cmsweb.cern.ch/dbs/prod/phys03/DBSReader')
dbsglobal = DbsApi('https://cmsweb.cern.ch/dbs/prod/global/DBSReader')

usage = 'usage: %prog [options]'

parser = argparse.ArgumentParser()
parser.add_argument( '-d', '--dataset', action='store', dest='dataset', default='ttHTobb', help='Type of sample' )
try: args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

##### Samples
allSamples = {}
allSamples[ '2016_ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8' ] = [ '/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer16NanoAODv4-PUMoriond17_Nano14Dec2018_102X_mcRun2_asymptotic_v6-v1/NANOAODSIM' ]


allSamples[ '2017_ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8' ] = [ '/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8/algomez-NANOAOD_v02-3ea2ff745e1084ea23260bd2ac726434/USER' ]
allSamples[ '2017_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8' ] = [ '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/algomez-NANOAOD_v02-3ea2ff745e1084ea23260bd2ac726434/USER' ]

allSamples['2018_SingleMuon_Run2018A'] = ['/SingleMuon/Run2018A-Nano14Dec2018-v1/NANOAOD' ]
allSamples['2018_SingleMuon_Run2018B'] = ['/SingleMuon/Run2018B-Nano14Dec2018-v1/NANOAOD' ]
allSamples['2018_SingleMuon_Run2018C'] = ['/SingleMuon/Run2018C-Nano14Dec2018-v1/NANOAOD' ]
allSamples['2018_SingleMuon_Run2018D'] = ['/SingleMuon/Run2018D-Nano14Dec2018_ver2-v1/NANOAOD' ]
allSamples[ '2018_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8' ] = [ '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv4-Nano14Dec2018_102X_upgrade2018_realistic_v16-v1/NANOAODSIM' ]


## trick to run only in specific samples
dictSamples = {}
for sam in allSamples:
    if sam.startswith( args.dataset ): dictSamples[ sam ] = allSamples[ sam ]
    #else: dictSamples = allSamples

for sample, jsample in dictSamples.items():

    ##### Create a list from the dataset
    lfnList = []
    for j in jsample:
        fileDictList = ( dbsPhys03 if j.endswith('USER') else dbsglobal).listFiles(dataset=j,validFileOnly=1)
        print "dataset %s has %d files" % (j, len(fileDictList))
        # DBS client returns a list of dictionaries, but we want a list of Logical File Names
        lfnList = lfnList + [ 'root://cms-xrd-global.cern.ch/'+dic['logical_file_name'] for dic in fileDictList ]

    textFile = open( sample+'.txt', 'w')
    textFile.write("\n".join(lfnList))

