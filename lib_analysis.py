import os, glob
import argparse
import json
import numpy as np

import uproot
import hepaccelerate

from hepaccelerate.utils import Results, NanoAODDataset, Histogram, choose_backend

NUMPY_LIB = None
ha = None

############################################## OBJECT SELECTION ################################################

### Primary vertex selection
def vertex_selection(scalars, mask_events):

    PV_isfake = (scalars["PV_score"] == 0) & (scalars["PV_chi2"] == 0)
    PV_rho = NUMPY_LIB.sqrt(scalars["PV_x"]**2 + scalars["PV_y"]**2)
    mask_events = mask_events & (~PV_isfake) & (scalars["PV_ndof"] > 4) & (scalars["PV_z"]<24) & (PV_rho < 2)

    return mask_events


### Lepton selection
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

### Jet selection
def jet_selection(jets, leps, mask_leps, cuts):

    jets_pass_dr = ha.mask_deltar_first(jets, jets.masks["all"], leps, mask_leps, cuts["dr"])
    jets.masks["pass_dr"] = jets_pass_dr
    good_jets = (jets.pt > cuts["pt"]) & (NUMPY_LIB.abs(jets.eta) < cuts["eta"]) & (jets.jetId >= cuts["jetId"]) & jets_pass_dr
    if cuts["type"] == "jet":
      good_jets &= (jets.puId>=cuts["puId"]) 

    return good_jets


###################################################### WEIGHT / SF CALCULATION ##########################################################

### PileUp weight
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


# lepton scale factors
def compute_lepton_weights(leps, lepton_x, lepton_y, mask_rows, mask_content, evaluator, SF_list):

    weights = NUMPY_LIB.ones(len(lepton_x))

    for SF in SF_list:
        weights *= evaluator[SF](lepton_x, lepton_y)
    
    per_event_weights = ha.multiply_in_offsets(leps, weights, mask_rows, mask_content)
    return per_event_weights


# btagging scale factor 
def compute_btag_weights(jets, mask_rows, mask_content, evaluator):

    pJet_weight = NUMPY_LIB.ones(len(mask_content))

    for tag in ["BTagSFDeepCSV_3_iterativefit_central_0", "BTagSFDeepCSV_3_iterativefit_central_1", "BTagSFDeepCSV_3_iterativefit_central_2"]:
        SF_btag = evaluator[tag](jets.eta, jets.pt, jets.btagDeepB)
        if tag.endswith("0"):
            SF_btag[jets.hadronFlavour != 5] = 1.
        if tag.endswith("1"):
            SF_btag[jets.hadronFlavour != 4] = 1.
        if tag.endswith("2"):
            SF_btag[jets.hadronFlavour != 0] = 1.

        pJet_weight *= SF_btag

    per_event_weights = ha.multiply_in_offsets(jets, pJet_weight, mask_rows, mask_content)
    return per_event_weights

############################################# HIGH LEVEL VARIABLES (DNN evaluation, ...) ############################################

def evaluate_DNN(jets, good_jets, electrons, good_electrons, muons, good_muons, scalars, mask_events, DNN, DNN_model):
    
        # make inputs (defined in backend (not extremely nice))
        jets_feats = ha.make_jets_inputs(jets, jets.offsets, 10, ["pt","eta","phi","en","px","py","pz", "btagDeepB"], mask_events, good_jets)
        met_feats = ha.make_met_inputs(scalars, nEvents, ["phi","pt","sumEt","px","py"], mask_events)
        leps_feats = ha.make_leps_inputs(electrons, muons, nEvents, ["pt","eta","phi","en","px","py","pz"], mask_events, good_electrons, good_muons)

        inputs = [jets_feats, leps_feats, met_feats]

        if DNN.startswith("ffwd"):
            inputs = [NUMPY_LIB.reshape(x, (x.shape[0], -1)) for x in inputs]
            inputs = NUMPY_LIB.hstack(inputs)
            # numpy transfer needed for keras
            inputs = NUMPY_LIB.asnumpy(inputs)

        if DNN.startswith("cmb"):
            # numpy transfer needed for keras
            if not isinstance(jets_feats, np.ndarray):
                inputs = [NUMPY_LIB.asnumpy(x) for x in inputs]

        # fix in case inputs are empty
        if jets_feats.shape[0] == 0:
            DNN_pred = NUMPY_LIB.zeros(nEvents, dtype=NUMPY_LIB.float32)
        else:
            # run prediction (done on GPU)
            DNN_pred = DNN_model.predict(inputs, batch_size = 10000)
            # in case of NUMPY_LIB is cupy: transfer numpy output back to cupy array for further computation
            DNN_pred = NUMPY_LIB.array(DNN_model.predict(inputs, batch_size = 10000))
            if DNN.endswith("binary"):
                DNN_pred = NUMPY_LIB.reshape(DNN_pred, DNN_pred.shape[0])

        return DNN_pred

# calculate simple object variables
def calculate_variable_features(z, mask_events, indices, var):

    name, coll, mask_content, inds, feats = z
    idx = indices[inds]

    for f in feats:
        var[inds+"_"+name+"_"+f] = ha.get_in_offsets(getattr(coll, f), getattr(coll, "offsets"), idx, mask_events, mask_content)
    
####################################################### Simple helpers  #############################################################

def get_histogram(data, weights, bins):
    return Histogram(*ha.histogram_from_vector(data, weights, bins))

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

import keras.backend as K
import keras.losses
import keras.utils.generic_utils

def mse0(y_true,y_pred):
    return K.mean( K.square(y_true[:,0] - y_pred[:,0]) )

def mae0(y_true,y_pred):
    return K.mean( K.abs(y_true[:,0] - y_pred[:,0]) )

def r2_score0(y_true,y_pred):
    return 1. - K.sum( K.square(y_true[:,0] - y_pred[:,0]) ) / K.sum( K.square(y_true[:,0] - K.mean(y_true[:,0]) ) )

