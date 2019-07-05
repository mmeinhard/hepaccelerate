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
    good_jets = (jets.pt > cuts["pt"]) & (NUMPY_LIB.abs(jets.eta) < cuts["eta"]) & (jets.jetId >= cuts["jetId"]) & (jets.puId>=cuts["puId"]) & jets_pass_dr

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

