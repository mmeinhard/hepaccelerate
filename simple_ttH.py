import os, glob
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import argparse

import uproot
import hepaccelerate
from hepaccelerate.utils import Results, NanoAODDataset, Histogram, choose_backend

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# function for lepton selection
def lepton_selection(leps, cuts):

    passes_eta = NUMPY_LIB.abs(leps.eta) < cuts["eta"]
    passes_subleading_pt = leps.pt > cuts["subleading_pt"]
    passes_leading_pt = leps.pt > cuts["leading_pt"]

    if cuts["type"] == "el":
        sca = NUMPY_LIB.abs(leps.deltaEtaSC + leps.eta)
        passes_id = ( (leps.cutBased >= 4) & NUMPY_LIB.invert( (sca>=1.4442) & (sca<1.5669)) & ( ((leps.dz < 0.10) & (sca <= 1.479)) | ((leps.dz < 0.20) & (sca > 1.479)) ) & ( ((leps.dxy < 0.05) & (sca <= 1.478)) | ((leps.dxy < 0.1) & (sca > 1.479)) ))

        #select electrons
        good_leps = passes_eta & passes_leading_pt & passes_id
        veto_leps = passes_eta & passes_subleading_pt & passes_id & NUMPY_LIB.invert(good_leps)
    
    elif cuts["type"] == "mu":
        passes_leading_iso = leps.pfRelIso04_all < cuts["subleading_iso"]
        passes_subleading_iso = leps.pfRelIso04_all < cuts["leading_iso"]
        passes_id = leps.tightId == 1

        #select muons
        good_leps = passes_eta & passes_leading_pt & passes_leading_iso & passes_id
        veto_leps = passes_eta & passes_subleading_pt & passes_subleading_iso & passes_id & NUMPY_LIB.invert(good_leps)
    
    return good_leps, veto_leps

# function for jet selection
def jet_selection(jets, leps, mask_leps, cuts):

    jets_pass_dr = ha.mask_deltar_first(jets, jets.masks["all"], leps, mask_leps, cuts["dr"])
    jets.masks["pass_dr"] = jets_pass_dr
    good_jets = (jets.pt > cuts["pt"]) & (NUMPY_LIB.abs(jets.eta) < cuts["eta"]) & (jets.jetId >= cuts["jetId"]) & (jets.puId>=cuts["puId"]) & jets_pass_dr

    return good_jets

#This function will be called for every file in the dataset
def analyze_data(data, NUMPY_LIB=None, parameters={}):
    #Output structure that will be returned and added up among the files.
    #Should be relatively small.
    ret = Results()

    # get basic objetcs
    muons = data["Muon"]
    electrons = data["Electron"]
    scalars = data["eventvars"]
    jets = data["Jet"]

    nEvents = muons.numevents()

    # prepare event cuts
    mask_events = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.bool)

    # apply event cleaning, PV selection and trigger selection
    flags = [
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter"]
    for flag in flags:
        mask_events = mask_events & scalars[flag]
    trigger = scalars["HLT_Ele35_WPTight_Gsf"] | scalars["HLT_Ele28_eta2p1_WPTight_Gsf_HT150"] | scalars["HLT_IsoMu24_eta2p1"] | scalars["HLT_IsoMu27"] 
    mask_events = mask_events & trigger & scalars["PV_npvsGood"]>0
    
    # apply object selection for muons, electrons, jets
    good_muons, veto_muons = lepton_selection(muons, parameters["muons"])
    good_electrons, veto_electrons = lepton_selection(electrons, parameters["electrons"])
    good_jets = jet_selection(jets, muons, good_muons, parameters["jets"]) & jet_selection(jets, electrons, good_electrons, parameters["jets"])
    bjets = good_jets & (jets.btagDeepB > 0.4941)

    # apply event selection
    nleps = (ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8) == 1) ^ (ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8) == 1)
    lepton_veto = (ha.sum_in_offsets(muons, veto_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8) < 1) & (ha.sum_in_offsets(electrons, veto_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8) < 1)  
    njets = ha.sum_in_offsets(jets, good_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8) >= 4
    btags = ha.sum_in_offsets(jets, bjets, mask_events, jets.masks["all"], NUMPY_LIB.int8) >= 2
    met = (scalars["MET_pt"] > 20)

    mask_events = mask_events & nleps & lepton_veto & njets & btags & met

    #TODO: calculate XS, gen, pu, btagging, lepton weights 

    #TODO: add Histograms for control variables

    #TODO: add DNN evaluation

    #TODO: think about how to split up different tt+jets backgrounds

    #TODO: implement JECs

    return ret
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Runs a simple array-based analysis')
    parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--from-cache', action='store_true', help='Load from cache (otherwise create it)')
    parser.add_argument('--nthreads', action='store', help='Number of CPU threads to use', type=int, default=4, required=False)
    parser.add_argument('--files-per-batch', action='store', help='Number of files to process per batch', type=int, default=1, required=False)
    parser.add_argument('--cache-location', action='store', help='Path prefix for the cache, must be writable', type=str, default=os.path.join(os.getcwd(), 'cache'))
    parser.add_argument('--filelist', action='store', help='List of files to load', type=str, default=None, required=False)
    parser.add_argument('filenames', nargs=argparse.REMAINDER)
    args = parser.parse_args()
 
    NUMPY_LIB, ha = choose_backend(args.use_cuda)
    NanoAODDataset.numpy_lib = NUMPY_LIB
   
    #define arrays to load: these are objects that will be kept together 
    arrays_objects = [
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId", "Jet_puId",
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso04_all", "Muon_tightId", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge", "Electron_deltaEtaSC", "Electron_cutBased", "Electron_dz", "Electron_dxy",
        "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id",
    ]
    #these are variables per event
    arrays_event = [
        "PV_npvsGood",
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter",
        "HLT_Ele35_WPTight_Gsf", "HLT_Ele28_eta2p1_WPTight_Gsf_HT150",
        "HLT_IsoMu24_eta2p1", "HLT_IsoMu27",
        "MET_pt", "MET_phi", "MET_sumEt",
        "run", "luminosityBlock", "event"
    ]


    parameters = {
        "muons": {
                    "type": "mu",
                    "leading_pt": 29,
                    "subleading_pt": 15,
                    "eta": 2.4,
                    "leading_iso": 0.15,
                    "subleading_iso": 0.25,
                    },

        "electrons" : {
                        "type": "el",
                        "leading_pt": 30,
                        "subleading_pt": 15,
                        "eta": 2.4
                        },
        "jets": {
                "dr": 0.4,
                "pt": 30,
                "eta": 2.4,
                "jetId": 2,
                "puId": 4
        }
    }

    filenames = None
    if not args.filelist is None:
        filenames = [l.strip() for l in open(args.filelist).readlines()]
    else:
        filenames = args.filenames

    for fn in filenames:
        if not fn.endswith(".root"):
            raise Exception("Must supply ROOT filename, but got {0}".format(fn))

    results = Results()
    for ibatch, files_in_batch in enumerate(chunks(filenames, args.files_per_batch)): 
        #define our dataset
        dataset = NanoAODDataset(files_in_batch, arrays_objects + arrays_event, "Events", ["Jet", "Muon", "Electron","TrigObj"], arrays_event)
        dataset.get_cache_dir = lambda fn,loc=args.cache_location: os.path.join(loc, fn)

        if not args.from_cache:
            #Load data from ROOT files
            dataset.preload(nthreads=args.nthreads, verbose=True)

            #prepare the object arrays on the host or device
            dataset.make_objects()

            print("preparing dataset cache")
            #save arrays for future use in cache
            dataset.to_cache(verbose=True, nthreads=args.nthreads)

        #Optionally, load the dataset from an uncompressed format
        else:
            print("loading dataset from cache")
            dataset.from_cache(verbose=True, nthreads=args.nthreads)
        if ibatch == 0:
            print(dataset.printout())

        #Run the analyze_data function on all files
        results += dataset.analyze(analyze_data, NUMPY_LIB=NUMPY_LIB, parameters=parameters)
            
    print(results)
    #print("Efficiency of dimuon events: {0:.2f}".format(results["events_dimuon"]/results["num_events"]))
    
    #Save the results 
    results.save_json("out.json")
