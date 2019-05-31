import os, glob
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import argparse
import json
import numpy as np

import uproot
import hepaccelerate
from hepaccelerate.utils import Results, NanoAODDataset, Histogram, choose_backend

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

### function for lepton selection
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
        passes_leading_iso = (leps.pfRelIso04_all < cuts["subleading_iso"])
        passes_subleading_iso = (leps.pfRelIso04_all < cuts["leading_iso"])
        passes_id = (leps.tightId == 1)

        #select muons
        good_leps = passes_eta & passes_leading_pt & passes_leading_iso & passes_id
        veto_leps = passes_eta & passes_subleading_pt & passes_subleading_iso & passes_id & NUMPY_LIB.invert(good_leps)
    
    return good_leps, veto_leps

### function for jet selection
def jet_selection(jets, leps, mask_leps, cuts):

    jets_pass_dr = ha.mask_deltar_first(jets, jets.masks["all"], leps, mask_leps, cuts["dr"])
    jets.masks["pass_dr"] = jets_pass_dr
    good_jets = (jets.pt > cuts["pt"]) & (NUMPY_LIB.abs(jets.eta) < cuts["eta"]) & (jets.jetId >= cuts["jetId"]) & (jets.puId>=cuts["puId"]) & jets_pass_dr

    return good_jets

### get number of processed events weighted
def count_weighted(filenames):
    sumw = 0
    for fi in filenames:
        ff = uproot.open(fi)
        bl = ff.get("Runs")
        sumw += bl.array("genEventSumw").sum()
    return sumw

### pileup weight
def compute_pu_weights(pu_corrections_target, weights, mc_nvtx, reco_nvtx):
    pu_edges, (values_nom, values_up, values_down) = pu_corrections_target

    src_pu_hist = get_histogram(mc_nvtx, weights, pu_edges)
    norm = sum(src_pu_hist.contents)
    src_pu_hist.contents = src_pu_hist.contents/norm
    src_pu_hist.contents_w2 = src_pu_hist.contents_w2/norm

    ratio = values_nom / src_pu_hist.contents
    remove_inf_nan(ratio)
    pu_weights = NUMPY_LIB.zeros_like(weights)
    ha.get_bin_contents(reco_nvtx, NUMPY_LIB.array(pu_edges), NUMPY_LIB.array(ratio), pu_weights)
    #fix_large_weights(pu_weights)

    return pu_weights


def load_puhist_target(filename):
    fi = uproot.open(filename)

    h = fi["pileup"]
    edges = np.array(h.edges)
    values_nominal = np.array(h.values)
    values_nominal = values_nominal / np.sum(values_nominal)

    h = fi["pileup_plus"]
    values_up = np.array(h.values)
    values_up = values_up / np.sum(values_up)

    h = fi["pileup_minus"]
    values_down = np.array(h.values)
    values_down = values_down / np.sum(values_down)
    return edges, (values_nominal, values_up, values_down)

def get_histogram(data, weights, bins):
    return Histogram(*ha.histogram_from_vector(data, weights, bins))

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0


#This function will be called for every file in the dataset
def analyze_data(data, sample, NUMPY_LIB=None, parameters={}, samples_info={}, is_mc=True):
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
    trigger = (scalars["HLT_Ele35_WPTight_Gsf"] | scalars["HLT_Ele28_eta2p1_WPTight_Gsf_HT150"] | scalars["HLT_IsoMu24_eta2p1"] | scalars["HLT_IsoMu27"]) 
    mask_events = mask_events & trigger & scalars["PV_npvsGood"]>0
    
    # apply object selection for muons, electrons, jets
    good_muons, veto_muons = lepton_selection(muons, parameters["muons"])
    good_electrons, veto_electrons = lepton_selection(electrons, parameters["electrons"])
    good_jets = jet_selection(jets, muons, good_muons, parameters["jets"]) & jet_selection(jets, electrons, good_electrons, parameters["jets"])
    bjets = good_jets & (jets.btagDeepB > 0.4941)

    # apply event selection
    SL = (ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8) == 1) ^ (ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8) == 1)
    lepton_veto = (ha.sum_in_offsets(muons, veto_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8) < 1) & (ha.sum_in_offsets(electrons, veto_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8) < 1)  
    njets = ha.sum_in_offsets(jets, good_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
    btags = ha.sum_in_offsets(jets, bjets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
    met = (scalars["MET_pt"] > 20)

    mask_events = mask_events & SL & lepton_veto & (njets >= 4) & (btags >=2) & met


    # further variables
    nMuons = ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8)
    nElectrons = ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8)

    # calculate weights for MC samples
    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"] * parameters["lumi"] * samples_info[sample]["XS"] / samples_info[sample]["ngen_weight"]
        pu_corrections_target = load_puhist_target("data/pileup_Cert_294927-306462_13TeV_PromptReco_Collisions17_withVar.root")
        pu_weights = compute_pu_weights(pu_corrections_target, weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
        weights["nominal"] = weights["nominal"] * pu_weights

    processes = {}
    if sample.startswith("TT"):
        ttCls = scalars["genTtbarId"]%100
        processes["ttbb"] = mask_events & (ttCls >=53) & (ttCls <=56) 
        processes["tt2b"] = mask_events & (ttCls ==52) 
        processes["ttb"] = mask_events & (ttCls ==51) 
        processes["ttcc"] = mask_events & (ttCls >=41) & (ttCls <=45) 
        ttHF =  ((ttCls >=53) & (ttCls <=56)) | (ttCls ==52) | (ttCls ==51) | ((ttCls >=41) & (ttCls <=45))
        processes["ttlf"] = mask_events & (~ttHF)
    else:
        processes["unsplit"] = mask_events
        
    for p in processes.keys():

        mask_events = processes[p]

        # get control variables
        inds = NUMPY_LIB.zeros(nEvents, dtype=NUMPY_LIB.int32)
        leading_jet_pt = ha.get_in_offsets(jets.pt, jets.offsets, inds, mask_events, good_jets)
        import pdb
        pdb.set_trace()
        #leading_lep_pt = ha.maximum(ha.get_in_offsets(muons.pt, muons.offsets, inds, mask_events, good_muons), ha.get_in_offsets(electrons.pt, electrons.offsets, inds, mask_events, good_electrons))
 
        # create histograms filled with weighted events
        hist_njets = Histogram(*ha.histogram_from_vector(njets[mask_events], weights["nominal"], NUMPY_LIB.linspace(0,18,19)))
        hist_nElectrons = Histogram(*ha.histogram_from_vector(nElectrons[mask_events], weights["nominal"], NUMPY_LIB.linspace(0,3,4)))
        hist_nMuons = Histogram(*ha.histogram_from_vector(nMuons[mask_events], weights["nominal"], NUMPY_LIB.linspace(0,3,4)))
        hist_nbtags = Histogram(*ha.histogram_from_vector(btags[mask_events], weights["nominal"], NUMPY_LIB.linspace(0,10,11)))
        hist_leading_jet_pt = Histogram(*ha.histogram_from_vector(leading_jet_pt[mask_events], weights["nominal"], NUMPY_LIB.linspace(0,1000,101)))
        hist_ttCls = Histogram(*ha.histogram_from_vector(ttCls, NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.float32), NUMPY_LIB.linspace(0,60,61)))
        #hist_genWeights = Histogram(*ha.histogram_from_vector(scalars["genWeight"][mask_events], NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.float32), NUMPY_LIB.linspace(-1,1,21)))
        #hist_leading_lepton_pt = Histogram(*ha.histogram_from_vector(leading_lepton_pt[mask_events], weights["nominal"], NUMPY_LIB.linspace(0,300,101)))

        if p=="unsplit":
            name = samples_info[sample]["process"]
        else:
            name = p   
 
        ret["hist_{0}_njets".format(name)] = hist_njets
        ret["hist_{0}_nElectrons".format(name)] = hist_nElectrons
        ret["hist_{0}_nMuons".format(name)] = hist_nMuons
        ret["hist_{0}_nbtags".format(name)] = hist_nbtags
        ret["hist_{0}_leading_jet_pt".format(name)] = hist_leading_jet_pt
        ret["hist_{0}_ttCls".format(name)] = hist_ttCls
        #ret["hist_{0}_genWeights".format(name)] = hist_genWeights
        #ret["hist_leading_lepton_pt"] = hist_leading_lepton_pt
 
    #TODO: think about how to split up different tt+jets backgrounds
    #TODO: add DNN evaluation

    #TODO: implement JECs, btagging, lepton+trigger weights

    return ret
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Runs a simple array-based analysis')
    parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--from-cache', action='store_true', help='Load from cache (otherwise create it)')
    parser.add_argument('--nthreads', action='store', help='Number of CPU threads to use', type=int, default=4, required=False)
    parser.add_argument('--files-per-batch', action='store', help='Number of files to process per batch', type=int, default=1, required=False)
    parser.add_argument('--cache-location', action='store', help='Path prefix for the cache, must be writable', type=str, default=os.path.join(os.getcwd(), 'cache'))
    parser.add_argument('--filelist', action='store', help='List of files to load', type=str, default=None, required=False)
    parser.add_argument('--sample', action='store', help='sample name', type=str, default=None, required=True)
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
        "PV_npvsGood", "Pileup_nTrueInt",
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter",
        "HLT_Ele35_WPTight_Gsf", "HLT_Ele28_eta2p1_WPTight_Gsf_HT150",
        "HLT_IsoMu24_eta2p1", "HLT_IsoMu27",
        "MET_pt", "MET_phi", "MET_sumEt",
        "genWeight",
        "run", "luminosityBlock", "event"
    ]

    if args.sample.startswith("TT"):
        arrays_event.append("genTtbarId")

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
        },
        "lumi":  41529.0,
    }

    samples_info = {
        "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8" : {
            "process" : "ttHTobb",
            "XS" : 0.2934045,
        },
        "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8": {
            "XS": 365.45736135,
        }
    }

    filenames = None
    if not args.filelist is None:
        filenames = [l.strip() for l in open(args.filelist).readlines()]
    else:
        filenames = args.filenames

    for fn in filenames:
        if not fn.endswith(".root"):
            print(fn)
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

            samples_info[args.sample]["ngen_weight"] = count_weighted(filenames)
            with open("samples_info.json", "w") as conf_file:
                json.dump(samples_info, conf_file)   

            print("preparing dataset cache")
            #save arrays for future use in cache
            dataset.to_cache(verbose=True, nthreads=args.nthreads)

        #Optionally, load the dataset from an uncompressed format
        else:
            print("loading dataset from cache")
            dataset.from_cache(verbose=True, nthreads=args.nthreads)
            
            with open("samples_info.json") as json_file:
                settings = json.load(json_file)
            samples_info[args.sample]["ngen_weight"] = settings[args.sample]["ngen_weight"]

        if ibatch == 0:
            print(dataset.printout())

        #Run the analyze_data function on all files
        results += dataset.analyze(analyze_data, NUMPY_LIB=NUMPY_LIB, parameters=parameters, is_mc = True, sample=args.sample, samples_info=samples_info)
            
    print(results)
    #print("Efficiency of dimuon events: {0:.2f}".format(results["events_dimuon"]/results["num_events"]))
    
    #Save the results 
    results.save_json("out_{}.json".format(args.sample))
