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
from losses import mse0,mae0,r2_score0


from fnal_column_analysis_tools.lumi_tools import LumiMask, LumiData
from fnal_column_analysis_tools.lookup_tools import extractor

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#session = tf.Session(config=config)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

### function for primary vertex selection
def vertex_selection(scalars, mask_events):

    PV_isfake = (scalars["PV_score"] == 0) & (scalars["PV_chi2"] == 0)
    PV_rho = NUMPY_LIB.sqrt(scalars["PV_x"]**2 + scalars["PV_y"]**2) 
    mask_events = mask_events & (~PV_isfake) & (scalars["PV_ndof"] > 4) & (scalars["PV_z"]<24) & (PV_rho < 2)

    return mask_events 

### function for lepton selection
def lepton_selection(leps, cuts):

    passes_eta = (NUMPY_LIB.abs(leps.eta) < cuts["eta"])
    passes_subleading_pt = (leps.pt > cuts["subleading_pt"])
    passes_leading_pt = (leps.pt > cuts["leading_pt"])

    if cuts["type"] == "el":
        sca = NUMPY_LIB.abs(leps.deltaEtaSC + leps.eta)
        passes_id = (leps.cutBased >= 4)
        passes_SC = NUMPY_LIB.invert((sca >= 1.4442) & (sca <= 1.5660))
        # cuts taken from: https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2#Working_points_for_92X_and_later
        passes_impact = ((leps.dz < 0.10) & (sca <= 1.479)) | ((leps.dz < 0.20) & (sca > 1.479)) | ((leps.dxy < 0.05) & (sca <= 1.479)) | ((leps.dxy < 0.1) & (sca > 1.479))      
        #passes_id = ( (leps.cutBased >= 4) & NUMPY_LIB.invert( (sca>=1.4442) & (sca<1.5669)) & ( ((leps.dz < 0.10) & (sca <= 1.479)) | ((leps.dz < 0.20) & (sca > 1.479)) ) & ( ((leps.dxy < 0.05) & (sca <= 1.478)) | ((leps.dxy < 0.1) & (sca > 1.479)) ))

        #select electrons
        good_leps = passes_eta & passes_leading_pt & passes_id & passes_SC & passes_impact
        veto_leps = passes_eta & passes_subleading_pt & NUMPY_LIB.invert(good_leps) & passes_id & passes_SC & passes_impact
    
    elif cuts["type"] == "mu":
        passes_leading_iso = (leps.pfRelIso04_all < cuts["leading_iso"])
        passes_subleading_iso = (leps.pfRelIso04_all < cuts["subleading_iso"])
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


# function to calculate lepton & trigger SFs
def calculate_lepton_weights(lepton_pt, lepton_SuperClusterEta, lepton_eta, leptonFlavour, evaluator):

    if not isinstance(lepton_pt, np.ndarray):
        lepton_pt = NUMPY_LIB.asnumpy(lepton_pt)
        lepton_eta = NUMPY_LIB.asnumpy(lepton_eta)
        lepton_SuperClusterEta = NUMPY_LIB.asnumpy(lepton_SuperClusterEta)

    # calculate all electron SFs
    SF_trigger_el = NUMPY_LIB.array(evaluator["el_triggerSF"](lepton_pt, lepton_SuperClusterEta))
    SF_ID_el = NUMPY_LIB.array(evaluator["el_idSF"](lepton_pt, lepton_SuperClusterEta))
    SF_reco_el = NUMPY_LIB.array(evaluator["el_recoSF"](lepton_pt, lepton_SuperClusterEta))

    # calculate all muon SFs
    SF_trigger_mu = NUMPY_LIB.array(evaluator["mu_triggerSF"](lepton_pt, np.abs(lepton_eta)))
    SF_ID_mu = NUMPY_LIB.array(evaluator["mu_idSF"](lepton_pt, np.abs(lepton_eta)))
    SF_iso_mu = NUMPY_LIB.array(evaluator["mu_isoSF"](lepton_pt, np.abs(lepton_eta)))

    # combine all SFs to events weights
    SF_trigger = NUMPY_LIB.add(NUMPY_LIB.where((leptonFlavour == 13), SF_trigger_mu, 0), NUMPY_LIB.where((leptonFlavour == 11), SF_trigger_el, 0))
    SF_ID = NUMPY_LIB.add(NUMPY_LIB.where((leptonFlavour == 13), SF_ID_mu, 0), NUMPY_LIB.where((leptonFlavour == 11), SF_ID_el, 0))
    SF_iso = NUMPY_LIB.add(NUMPY_LIB.where((leptonFlavour == 13), SF_iso_mu, 0), NUMPY_LIB.where((leptonFlavour == 11), SF_reco_el, 0))

    lepton_SF = (SF_trigger * SF_ID * SF_iso)
    return lepton_SF

# function to calculate btwg weights
def calculate_btagSF_jets(jets_eta, jets_pt, jets_discr, jets_hadronFlavour, evaluator):

    if not isinstance(jets_pt, np.ndarray):
        jets_pt = NUMPY_LIB.asnumpy(jets_pt)
        jets_eta = NUMPY_LIB.asnumpy(jets_eta)
        jets_discr = NUMPY_LIB.asnumpy(jets_discr)

    SF_btag_0 = NUMPY_LIB.array(evaluator["BTagSFDeepCSV_3_iterativefit_central_0"](jets_eta, jets_pt, jets_discr))
    SF_btag_1 = NUMPY_LIB.array(evaluator["BTagSFDeepCSV_3_iterativefit_central_1"](jets_eta, jets_pt, jets_discr))
    SF_btag_2 = NUMPY_LIB.array(evaluator["BTagSFDeepCSV_3_iterativefit_central_2"](jets_eta, jets_pt, jets_discr))

    SF_btag = NUMPY_LIB.where(jets_hadronFlavour == 5, SF_btag_0, 0) + NUMPY_LIB.where(jets_hadronFlavour == 4, SF_btag_1, 0) + NUMPY_LIB.where(jets_hadronFlavour == 0, SF_btag_2, 0)

    return SF_btag


# function to calculate Btagging weights

def get_histogram(data, weights, bins):
    return Histogram(*ha.histogram_from_vector(data, weights, bins))

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0


#This function will be called for every file in the dataset
def analyze_data(data, sample, NUMPY_LIB=None, parameters={}, samples_info={}, is_mc=True, lumimask=None, cat=False, DNN=False, DNN_model=None):
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
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter", "Flag_ecalBadCalibFilter"]
    if not is_mc:
        flags.append("Flag_eeBadScFilter")
    for flag in flags:
        mask_events = mask_events & scalars[flag]
    trigger = (scalars["HLT_Ele35_WPTight_Gsf"] | scalars["HLT_Ele28_eta2p1_WPTight_Gsf_HT150"] | scalars["HLT_IsoMu27"]) 
    mask_events = mask_events & trigger 
    mask_events = mask_events & (scalars["PV_npvsGood"]>0)
    #mask_events = vertex_selection(scalars, mask_events) 
    
    # apply object selection for muons, electrons, jets
    good_muons, veto_muons = lepton_selection(muons, parameters["muons"])
    good_electrons, veto_electrons = lepton_selection(electrons, parameters["electrons"])
    good_jets = jet_selection(jets, muons, veto_muons, parameters["jets"]) & jet_selection(jets, electrons, veto_electrons, parameters["jets"]) & jet_selection(jets, muons, good_muons, parameters["jets"]) & jet_selection(jets, electrons, good_electrons, parameters["jets"])
    bjets = good_jets & (jets.btagDeepB > 0.4941)

    # apply event selection
    #nleps =  NUMPY_LIB.add(ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8), ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8))
    nleps =  NUMPY_LIB.add(ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8), ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8))

    #SL = (ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8) == 1) ^ (ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8) == 1)
    lepton_veto = NUMPY_LIB.add(ha.sum_in_offsets(muons, veto_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8), ha.sum_in_offsets(electrons, veto_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8))
    njets = ha.sum_in_offsets(jets, good_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
    btags = ha.sum_in_offsets(jets, bjets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
    met = (scalars["MET_pt"] > 20)

    mask_events = mask_events & (nleps == 1) & (lepton_veto == 0) & (njets >= 4) & (btags >=2) & met

    # in this section, we calculate all needed variables
    # get control variables
    inds = NUMPY_LIB.zeros(nEvents, dtype=NUMPY_LIB.int32)
    leading_jet_pt = ha.get_in_offsets(jets.pt, jets.offsets, inds, mask_events, good_jets)
    leading_lepton_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, mask_events, good_muons) + ha.get_in_offsets(electrons.pt, electrons.offsets, inds, mask_events, good_electrons)
    leading_lepton_eta = ha.get_in_offsets(muons.eta, muons.offsets, inds, mask_events, good_muons) + ha.get_in_offsets(electrons.eta, electrons.offsets, inds, mask_events, good_electrons)

    # variables for trigger SFs
    SuperClusterEta = (electrons.deltaEtaSC + electrons.eta)
    leading_lepton_SuperClusterEta = ha.get_in_offsets(SuperClusterEta, electrons.offsets, inds, mask_events, good_electrons)
    nMuons =  ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8)
    nElectrons = ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8)
    event_leptonFlavour = NUMPY_LIB.zeros(nEvents)
    event_leptonFlavour[nMuons == 1] = 13
    event_leptonFlavour[nElectrons == 1] = 11


    # calculate weights for MC samples
    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"] * parameters["lumi"] * samples_info[sample]["XS"] / samples_info[sample]["ngen_weight"]
        
        # calc pu corrections
        pu_weights = compute_pu_weights(parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
        weights["nominal"] = weights["nominal"] * pu_weights

        # calc trigger SF
        leptonSF = calculate_lepton_weights(leading_lepton_pt, leading_lepton_SuperClusterEta, leading_lepton_eta, event_leptonFlavour, evaluator)
        weights["nominal"] = weights["nominal"] * leptonSF

        # calc btag weights
        btagSF_jets = calculate_btagSF_jets(jets.eta, jets.pt, jets.btagDeepB, jets.hadronFlavour, evaluator)
        btagSF = ha.multiply_in_offsets(jets, btagSF_jets, mask_events, good_jets)
        weights["nominal"] = weights["nominal"] * btagSF

    #in case of data: check if event is in golden lumi file
    if not is_mc and not (lumimask is None):
        mask_lumi = lumimask(scalars["run"], scalars["luminosityBlock"])
        mask_events = mask_events & mask_lumi

    if DNN:

        import time
        start = time.time()

        jets_feats = ha.make_jets_inputs(jets, jets.offsets, 10, ["pt","eta","phi","en","px","py","pz", "btagDeepB"], mask_events, good_jets)
        met_feats = ha.make_met_inputs(scalars, nEvents, ["phi","pt","sumEt","px","py"], mask_events)
        leps_feats = ha.make_leps_inputs(electrons, muons, nEvents, ["pt","eta","phi","en","px","py","pz"], mask_events, good_electrons, good_muons)

        inputs = [jets_feats, leps_feats, met_feats]

        if DNN.startswith("ffwd"):
            inputs = [NUMPY_LIB.reshape(x, (x.shape[0], -1)) for x in inputs]
            inputs = NUMPY_LIB.hstack(inputs)           
            inputs = NUMPY_LIB.asnumpy(inputs)
 
        if DNN.startswith("cmb"):
            if not isinstance(jets_feats, np.ndarray):
                inputs = [NUMPY_LIB.asnumpy(x) for x in inputs]

        if jets_feats.shape[0] == 0:
            DNN_pred = NUMPY_LIB.zeros(nEvents, dtype=NUMPY_LIB.float32)
        else:
            DNN_pred = DNN_model.predict(inputs, batch_size = 10000)
            DNN_pred = NUMPY_LIB.array(DNN_model.predict(inputs, batch_size = 10000))
            if DNN.endswith("binary"):
                DNN_pred = NUMPY_LIB.reshape(DNN_pred, DNN_pred.shape[0])
            print("time needed to pred model:", (time.time() - start))

    # in case of tt+jets -> split in ttbb, tt2b, ttb, ttcc, ttlf
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

        mask_events_split = processes[p]
    
        sl_jge4_tge3 = mask_events_split & (njets >= 4) & (btags >= 3)
        sl_j4_tge3 = mask_events_split & (njets == 4) & (btags >=3) 
        sl_j5_tge3 = mask_events_split & (njets == 5) & (btags >=3) 
        sl_jge6_tge3 = mask_events_split & (njets >= 6) & (btags >=3) 
 
        if not cat:
            list_cat = zip([mask_events_split, sl_jge4_tge3, sl_j4_tge3, sl_j5_tge3, sl_jge6_tge3], ["sl_jge4_tge2", "sl_jge4_tge3", "sl_j4_tge3", "sl_j5_tge3", "sl_jge6_tge3"])
        if cat == "sl_j4_tge3":
            list_cat = zip([sl_j4_tge3], ["sl_j4_tge3"])
        if cat == "sl_j5_tge3":
            list_cat = zip([sl_j5_tge3], ["sl_j5_tge3"])
        if cat == "sl_jge6_tge3":
            list_cat = zip([sl_jge6_tge3], ["sl_jge6_tge3"])
 
        #for cut, cut_name in zip([mask_events_split, sl_4j_geq3t, sl_5j_geq3t, sl_geq6j_geq3t], ["sl_geq4_geq3t", "sl_4j_geq3t","sl_5j_geq3t", "sl_geq6_geq3t"]): 
        for cut, cut_name in list_cat: 

            if p=="unsplit":
                if "Run" in sample:
                    name = "data" + "_" + cut_name
                else:
                    name = samples_info[sample]["process"] + "_" + cut_name
            else:
                name = p + "_" + cut_name  
        
            # create histograms filled with weighted events
            hist_njets = Histogram(*ha.histogram_from_vector(njets[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,30,31)))
            ret["hist_{0}_njets".format(name)] = hist_njets
            hist_nleps = Histogram(*ha.histogram_from_vector(nleps[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,10,11)))
            ret["hist_{0}_nleps".format(name)] = hist_nleps
            hist_nbtags = Histogram(*ha.histogram_from_vector(btags[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,8,9)))
            ret["hist_{0}_nbtags".format(name)] = hist_nbtags
            hist_leading_jet_pt = Histogram(*ha.histogram_from_vector(leading_jet_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,500,31)))
            ret["hist_{0}_leading_jet_pt".format(name)] = hist_leading_jet_pt
            hist_leading_lepton_pt = Histogram(*ha.histogram_from_vector(leading_lepton_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,500,31)))
            ret["hist_{0}_leading_lepton_pt".format(name)] = hist_leading_lepton_pt

            if DNN:
                if DNN.endswith("multiclass"):
                    class_pred = NUMPY_LIB.argmax(DNN_pred, axis=1)
                    for n, n_name in zip([0,1,2,3,4,5], ["ttH", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]):
                        node = (class_pred == n)
                        DNN_node = DNN_pred[:,n]
                        hist_DNN = Histogram(*ha.histogram_from_vector(DNN_node[(cut & node)], weights["nominal"][(cut & node)], NUMPY_LIB.linspace(0.,1.,16)))
                        ret["hist_{0}_DNN_{1}".format(name, n_name)] = hist_DNN
                        hist_DNN_ROC = Histogram(*ha.histogram_from_vector(DNN_node[(cut & node)], weights["nominal"][(cut & node)], NUMPY_LIB.linspace(0.,1.,1000)))
                        ret["hist_{0}_DNN_ROC_{1}".format(name, n_name)] = hist_DNN_ROC
                        
                else:
                    hist_DNN = Histogram(*ha.histogram_from_vector(DNN_pred[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0.,1.,16)))
                    ret["hist_{0}_DNN".format(name)] = hist_DNN
                    hist_DNN_ROC = Histogram(*ha.histogram_from_vector(DNN_pred[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0.,1.,1000)))
                    ret["hist_{0}_DNN_ROC".format(name)] = hist_DNN_ROC

 
    #TODO: implement JECs, btagging, lepton+trigger weights

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
    parser.add_argument('--DNN', action='store', choices=['save-arrays','cmb_binary', 'cmb_multiclass', 'ffwd_binary', 'ffwd_multiclass',False], help='options for DNN evaluation / preparation', default=False)
    parser.add_argument('--categories', action='store', choices=['sl_j4_tge3','sl_j5_tge3', 'sl_jge6_tge3',False], help='categories to be processed (default: False -> all categories)', default=False)
    parser.add_argument('--path-to-model', action='store', help='path to DNN model', type=str, default=None, required=False)
    parser.add_argument('filenames', nargs=argparse.REMAINDER)
    args = parser.parse_args()
 
    NUMPY_LIB, ha = choose_backend(args.use_cuda)
    NanoAODDataset.numpy_lib = NUMPY_LIB

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if "Run" in args.sample:
        is_mc = False
        lumimask = LumiMask("data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt") 
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
        #"PV_npvsGood",
        "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter", "Flag_BadChargedCandidateFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter",
        "HLT_Ele35_WPTight_Gsf", "HLT_Ele28_eta2p1_WPTight_Gsf_HT150",
        "HLT_IsoMu24_eta2p1", "HLT_IsoMu27",
        "MET_pt", "MET_phi", "MET_sumEt",
        "run", "luminosityBlock", "event"
    ]

    if args.sample.startswith("TT"):
        arrays_event.append("genTtbarId")

    if is_mc:
        arrays_event += ["PV_npvsGood", "Pileup_nTrueInt", "genWeight"]  

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
        },
        "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8" : {
            "process" : "ttHToNonbb",
            "XS" : 0.2150955,
        },
        "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8" : {
            "XS" : 88.341903326,
        },
        "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8" : {
            "XS" : 377.9607353256,
        }
    }

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

    """
    if is_mc and not args.use_cuda:
        with open("samples_info.json", "r") as jsonFile:
            samples_info = json.load(jsonFile)

        samples_info[args.sample]["ngen_weight"] = count_weighted(filenames)

        print("Sum of events (weighted by generator weight:)", samples_info[args.sample]["ngen_weight"])
        with open("samples_info.json", "w") as jsonFile:
            json.dump(samples_info, jsonFile)
    """


    for ibatch, files_in_batch in enumerate(chunks(filenames, args.files_per_batch)): 
        #define our dataset
        dataset = NanoAODDataset(files_in_batch, arrays_objects + arrays_event, "Events", ["Jet", "Muon", "Electron"], arrays_event)
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

            with open("samples_info.json") as json_file:
                samples_info = json.load(json_file)

            parameters["pu_corrections_target"] = load_puhist_target("data/pileup_Cert_294927-306462_13TeV_PromptReco_Collisions17_withVar.root")

            # add information for SF calculation
            ext = extractor()
            ext.add_weight_sets(["el_triggerSF SFs_ele_pt_ele_sceta_ele28_ht150_OR_ele35_2017BCDEF ./data/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb_Data_MC_v5.0.histo.root"])
            ext.add_weight_sets(["el_recoSF EGamma_SF2D ./data/egammaEffi_EGM2D_runBCDEF_passingRECO.histo.root"])
            ext.add_weight_sets(["el_idSF EGamma_SF2D ./data/egammaEffi_EGM2D_runBCDEF_passingTight94X.histo.root"])
            ext.add_weight_sets(["mu_triggerSF IsoMu27_PtEtaBins/pt_abseta_ratio ./data/EfficienciesAndSF_RunBtoF_Nov17Nov2017.histo.root"])
            ext.add_weight_sets(["mu_isoSF NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta ./data/RunBCDEF_SF_ISO.histo.root"])
            ext.add_weight_sets(["mu_idSF NUM_TightID_DEN_genTracks_pt_abseta ./data/RunBCDEF_SF_ID.histo.root"])

            ext.add_weight_sets(["BTagSF * ./data/deepCSV_sfs_v2.btag.csv"])
            ext.finalize()
            evaluator = ext.make_evaluator()


        if ibatch == 0:
            print(dataset.printout())

        #Run the analyze_data function on all files
        if args.DNN:
            model = load_model(args.path_to_model, custom_objects=dict(itertools=itertools, mse0=mse0, mae0=mae0, r2_score0=r2_score0))
            
            import time
            start = time.time()            
            results += dataset.analyze(analyze_data, NUMPY_LIB=NUMPY_LIB, parameters=parameters, is_mc = is_mc, lumimask=lumimask, cat=args.categories, sample=args.sample, samples_info=samples_info, DNN=args.DNN, DNN_model=model)
            print("time needed for file:", time.time()-start)       
             
        else:
            import time
            start = time.time()
            results += dataset.analyze(analyze_data, NUMPY_LIB=NUMPY_LIB, parameters=parameters, is_mc = is_mc, lumimask=lumimask, cat=args.categories, sample=args.sample, samples_info=samples_info, DNN=args.DNN, DNN_model=None)
            print("time needed for file:", time.time()-start)       
            
    print(results)
    #print("Efficiency of dimuon events: {0:.2f}".format(results["events_dimuon"]/results["num_events"]))
    
    #Save the results 
    results.save_json(outdir + "/out_{}.json".format(args.sample))
