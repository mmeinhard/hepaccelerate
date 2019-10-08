import os, glob
#os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
#os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
os.environ['KERAS_BACKEND'] = "tensorflow"
import argparse
import json
import numpy as np

import uproot
import hepaccelerate
from hepaccelerate.utils import Results, NanoAODDataset, Histogram, choose_backend

import tensorflow as tf
from keras.models import load_model
import itertools
from lib_analysis import mse0,mae0,r2_score0

from definitions_analysis import histogram_settings

import lib_analysis
from lib_analysis import vertex_selection, lepton_selection, jet_selection, load_puhist_target, compute_pu_weights, compute_lepton_weights, compute_btag_weights, chunks, evaluate_DNN, calculate_variable_features

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#This function will be called for every file in the dataset
def analyze_data(data, sample, NUMPY_LIB=None, parameters={}, samples_info={}, is_mc=True, lumimask=None, cat=False, DNN=False, DNN_model=None):
    #Output structure that will be returned and added up among the files.
    #Should be relatively small.
    ret = Results()

    muons = data["Muon"]
    electrons = data["Electron"]
    scalars = data["eventvars"]
    jets = data["Jet"]

    nEvents = muons.numevents()

    mask_events = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.bool)

    # apply event cleaning, PV selection and trigger selection
    flags = [
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter", "Flag_ecalBadCalibFilter"]
    if not is_mc:
        flags.append("Flag_eeBadScFilter")
    for flag in flags:
        mask_events = mask_events & scalars[flag]
    if args.year.startswith('2016'):
        trigger = (scalars["HLT_Ele27_WPTight_Gsf"] | scalars["HLT_IsoMu24"]  | scalars["HLT_IsoTkMu24"])
    else:
        trigger = (scalars["HLT_Ele35_WPTight_Gsf"] | scalars["HLT_Ele28_eta2p1_WPTight_Gsf_HT150"] | scalars["HLT_IsoMu27"])
    mask_events = mask_events & trigger
    mask_events = mask_events & (scalars["PV_npvsGood"]>0)
    #mask_events = vertex_selection(scalars, mask_events)

    # apply object selection for muons, electrons, jets
    good_muons, veto_muons = lepton_selection(muons, parameters["muons"])
    good_electrons, veto_electrons = lepton_selection(electrons, parameters["electrons"])
    good_jets = jet_selection(jets, muons, (veto_muons | good_muons), parameters["jets"]) & jet_selection(jets, electrons, (veto_electrons | good_electrons) , parameters["jets"])
    bjets = good_jets & (getattr(jets, parameters["btagging algorithm"]) > parameters["btagging WP"])

    # apply basic event selection -> individual categories cut later
    nleps =  NUMPY_LIB.add(ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8), ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8))
    lepton_veto = NUMPY_LIB.add(ha.sum_in_offsets(muons, veto_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8), ha.sum_in_offsets(electrons, veto_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8))
    njets = ha.sum_in_offsets(jets, good_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)

    btags = ha.sum_in_offsets(jets, bjets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
    met = (scalars["MET_pt"] > 20)

    mask_events = mask_events & (nleps == 1) & (lepton_veto == 0) & (njets >= 4) & (btags >=2) & met

    ### calculation of all needed variables
    var = {}

    var["njets"] = njets
    var["btags"] = btags
    var["nleps"] = nleps

    indices = {}    
    indices["leading"] = NUMPY_LIB.zeros(nEvents, dtype=NUMPY_LIB.int32)
    indices["subleading"] = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.int32)

    variables = [
        ("jet", jets, good_jets, "leading", ["pt", "eta"]),
        ("bjet", jets, bjets, "leading", ["pt", "eta"]),
    ]

    # special role of lepton
    var["leading_lepton_pt"] = NUMPY_LIB.maximum(ha.get_in_offsets(muons.pt, muons.offsets, indices["leading"], mask_events, good_muons), ha.get_in_offsets(electrons.pt, electrons.offsets, indices["leading"], mask_events, good_electrons))
    var["leading_lepton_eta"] = NUMPY_LIB.maximum(ha.get_in_offsets(muons.eta, muons.offsets, indices["leading"], mask_events, good_muons), ha.get_in_offsets(electrons.eta, electrons.offsets, indices["leading"], mask_events, good_electrons))

    # all other variables
    for v in variables:
        calculate_variable_features(v, mask_events, indices, var)

    # calculate weights for MC samples
    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"] * parameters["lumi"] * samples_info[sample]["XS"] / samples_info[sample]["ngen_weight"]

        # pu corrections
        pu_weights = compute_pu_weights(parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
        weights["nominal"] = weights["nominal"] * pu_weights

        # lepton SF corrections
        electron_weights = compute_lepton_weights(electrons, electrons.pt, (electrons.deltaEtaSC + electrons.eta), mask_events, good_electrons, evaluator, ["el_triggerSF", "el_recoSF", "el_idSF"])
        muon_weights = compute_lepton_weights(muons, muons.pt, NUMPY_LIB.abs(muons.eta), mask_events, good_muons, evaluator, ["mu_triggerSF", "mu_isoSF", "mu_idSF"])
        weights["nominal"] = weights["nominal"] * muon_weights * electron_weights

        # btag SF corrections
        btag_weights = compute_btag_weights(jets, mask_events, good_jets, evaluator)
        weights["nominal"] = weights["nominal"] * btag_weights

    #in case of data: check if event is in golden lumi file
    if not is_mc and not (lumimask is None):
        mask_lumi = lumimask(scalars["run"], scalars["luminosityBlock"])
        mask_events = mask_events & mask_lumi

    #evaluate DNN
    if DNN:
        DNN_pred = evaluate_DNN(jets, good_jets, electrons, good_electrons, muons, good_muons, scalars, mask_events, nEvents, DNN, DNN_model)

    # in case of tt+jets -> split in ttbb, tt2b, ttb, ttcc, ttlf
    processes = {}
    if sample.startswith("TT"):
        ttCls = scalars["genTtbarId"]%100
        processes["ttbb"] = mask_events & (ttCls >=53) & (ttCls <=56)
        processes["tt2b"] = mask_events & (ttCls ==52)
        processes["ttb"] = mask_events & (ttCls ==51)
        processes["ttcc"] = mask_events & (ttCls >=41) & (ttCls <=45)
        ttHF =  ((ttCls >=53) & (ttCls <=56)) | (ttCls ==52) | (ttCls ==51) | ((ttCls >=41) & (ttCls <=45))
        processes["ttlf"] = mask_events & NUMPY_LIB.invert(ttHF)
    else:
        processes["unsplit"] = mask_events

    for p in processes.keys():

        mask_events_split = processes[p]

        # Categories
        categories = {}
        categories["sl_jge4_tge2"] = mask_events_split
        categories["sl_jge4_tge3"] = mask_events_split & (btags >=3)

        categories["sl_j4_tge3"] = mask_events_split & (njets ==4) & (btags >=3)
        categories["sl_j5_tge3"] = mask_events_split & (njets ==5) & (btags >=3)
        categories["sl_jge6_tge3"] = mask_events_split & (njets >=6) & (btags >=3)

        categories["sl_j4_t3"] = mask_events_split & (njets ==4) & (btags ==3)
        categories["sl_j4_tge4"] = mask_events_split & (njets ==4) & (btags >=4)
        categories["sl_j5_t3"] = mask_events_split & (njets ==5) & (btags ==3)
        categories["sl_j5_tge4"] = mask_events_split & (njets ==5) & (btags >=4)
        categories["sl_jge6_t3"] = mask_events_split & (njets >=6) & (btags ==3)
        categories["sl_jge6_tge4"] = mask_events_split & (njets >=6) & (btags >=4)
        
        if not isinstance(cat, list):
            cat = [cat] 
        for c in cat:
            cut = categories[c]
            cut_name = c

            if p=="unsplit":
                if "Run" in sample:
                    name = "data" + "_" + cut_name
                else:
                    name = samples_info[sample]["process"] + "_" + cut_name
            else:
                name = p + "_" + cut_name

            # create histograms filled with weighted events
            for k in var.keys():
                if not k in histogram_settings.keys():
                    raise Exception("please add variable {0} to config_analysis.py".format(k))
                hist = Histogram(*ha.histogram_from_vector(var[k][cut], weights["nominal"][cut], NUMPY_LIB.linspace(histogram_settings[k][0], histogram_settings[k][1], histogram_settings[k][2])))
                ret["hist_{0}_{1}".format(name, k)] = hist

            if DNN:
                if DNN=="mass_fit":
                    print("Dijet_masses:", DNN_pred.shape, DNN_pred[0])
                    hist_DNN = Histogram(*ha.histogram_from_vector(DNN_pred[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0.,300.,30)))
                    hist_DNN_zoom = Histogram(*ha.histogram_from_vector(DNN_pred[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0.,170.,30)))
                else:
                    hist_DNN = Histogram(*ha.histogram_from_vector(DNN_pred[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0.,1.,16)))
                ret["hist_{0}_DNN".format(name)] = hist_DNN
                ret["hist_{0}_DNN_zoom".format(name)] = hist_DNN_zoom


    #TODO: implement JECs

    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs a simple array-based analysis')
    parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
    parser.add_argument('--from-cache', action='store_true', help='Load from cache (otherwise create it)')
    parser.add_argument('--nthreads', action='store', help='Number of CPU threads to use', type=int, default=4, required=False)
    parser.add_argument('--files-per-batch', action='store', help='Number of files to process per batch', type=int, default=1, required=False)
    parser.add_argument('--cache-location', action='store', help='Path prefix for the cache, must be writable', type=str, default=os.path.join(os.getcwd(), 'cache'))
    parser.add_argument('--outdir', action='store', help='directory to store outputs', type=str, default=os.getcwd())
    parser.add_argument('--filelist', action='store', help='List of files to load', type=str, default=None, required=False)
    parser.add_argument('--sample', action='store', help='sample name', type=str, default=None, required=True)
    parser.add_argument('--DNN', action='store', choices=['save-arrays','cmb_binary', 'cmb_multiclass', 'ffwd_binary', 'ffwd_multiclass',False, 'mass_fit'], help='options for DNN evaluation / preparation', default=False)
    parser.add_argument('--categories', nargs='+', help='categories to be processed (default: sl_jge4_tge2)', default="sl_jge4_tge2")
    parser.add_argument('--path-to-model', action='store', help='path to DNN model', type=str, default=None, required=False)
    parser.add_argument('--year', action='store', choices=['2016', '2017', '2018'], help='Year of data/MC samples', default='2017')
    parser.add_argument('filenames', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # set CPU or GPU backend
    NUMPY_LIB, ha = choose_backend(args.use_cuda)
    lib_analysis.NUMPY_LIB, lib_analysis.ha = NUMPY_LIB, ha
    NanoAODDataset.numpy_lib = NUMPY_LIB

    if args.use_cuda:
        os.environ["HEPACCELERATE_CUDA"] = "1"
    else:
        os.environ["HEPACCELERATE_CUDA"] = "0"

    from coffea.util import USE_CUPY
    from coffea.lumi_tools import LumiMask, LumiData
    from coffea.lookup_tools import extractor

    # load definitions
    from definitions_analysis import parameters, eraDependentParameters, samples_info
    parameters.update(eraDependentParameters[args.year])
    print(parameters)

    outdir = args.outdir
    if not os.path.exists(outdir):
        print(os.getcwd())
        os.makedirs(outdir)

    if "Run" in args.sample:
        is_mc = False
        lumimask = LumiMask(parameters["lumimask"])
    else:
        is_mc = True
        lumimask = None


    #define arrays to load: these are objects that will be kept together
    arrays_objects = [
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId", "Jet_puId", "Jet_mass", "Jet_hadronFlavour",
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso04_all", "Muon_tightId", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge", "Electron_deltaEtaSC", "Electron_cutBased", "Electron_dz", "Electron_dxy",
    ]
    #these are variables per event
    arrays_event = [
        "PV_npvsGood", "PV_ndof", "PV_npvs", "PV_score", "PV_x", "PV_y", "PV_z", "PV_chi2",
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter",
        "MET_pt", "MET_phi", "MET_sumEt",
        "run", "luminosityBlock", "event",
        "nGenPart"
    ]

    if args.year.startswith('2016'): arrays_event += [ "HLT_Ele27_WPTight_Gsf", "HLT_IsoMu24", "HLT_IsoTkMu24" ]
    else: arrays_event += [ "HLT_Ele35_WPTight_Gsf", "HLT_Ele28_eta2p1_WPTight_Gsf_HT150", "HLT_IsoMu27" ]

    if args.sample.startswith("TT"):
        arrays_event.append("genTtbarId")

    if is_mc:
        arrays_event += ["PV_npvsGood", "Pileup_nTrueInt", "genWeight"]

    filenames = None
    if not args.filelist is None:
        filenames = [l.strip() for l in open(args.filelist).readlines()]
    else:
        filenames = args.filenames

    print("Number of files:", len(filenames))

    for fn in filenames:
        if not fn.endswith(".root"):
            print(fn)
            raise Exception("Must supply ROOT filename, but got {0}".format(fn))

    results = Results()


    for ibatch, files_in_batch in enumerate(chunks(filenames, args.files_per_batch)):
        #define our dataset
        structs = ["Jet", "Muon", "Electron"]
        dataset = NanoAODDataset(files_in_batch, arrays_objects + arrays_event, "Events", structs, arrays_event)
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

        if is_mc:

            # add information needed for MC corrections
            parameters["pu_corrections_target"] = load_puhist_target(parameters["pu_corrections_file"])

            ext = extractor()
            for corr in parameters["corrections"]:
                ext.add_weight_sets([corr])
            ext.finalize()
            evaluator = ext.make_evaluator()


        if ibatch == 0:
            print(dataset.printout())

        # in case of DNN evaluation: load model
        model = None
        if args.DNN:
            model = load_model(args.path_to_model, custom_objects=dict(itertools=itertools, mse0=mse0, mae0=mae0, r2_score0=r2_score0))

        print(args.categories)
        #### this is where the magic happens: run the main analysis
        results += dataset.analyze(analyze_data, NUMPY_LIB=NUMPY_LIB, parameters=parameters, is_mc = is_mc, lumimask=lumimask, cat=args.categories, sample=args.sample, samples_info=samples_info, DNN=args.DNN, DNN_model=model)


    print(results)

    #Save the results
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    results.save_json(os.path.join(outdir,"out_{}.json".format(args.sample)))
