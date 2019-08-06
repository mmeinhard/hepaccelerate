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

import lib_analysis
from lib_analysis import vertex_selection, lepton_selection, jet_selection, load_puhist_target, compute_pu_weights, compute_lepton_weights, compute_btag_weights, chunks, evaluate_DNN

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#This function will be called for every file in the dataset
def analyze_data(data, sample, NUMPY_LIB=None, parameters={}, samples_info={}, is_mc=True, lumimask=None, cat=False, boosted=False, DNN=False, DNN_model=None):
    #Output structure that will be returned and added up among the files.
    #Should be relatively small.
    ret = Results()

    muons = data["Muon"]
    electrons = data["Electron"]
    scalars = data["eventvars"]
    jets = data["Jet"]

    if boosted:
      fatjets = data["FatJet"]
      genparts = data["GenPart"]

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
    bjets = good_jets & (jets.btagDeepB > 0.4941)

    if boosted:
      good_fatjets = jet_selection(fatjets, muons, (veto_muons | good_muons), parameters["fatjets"]) & jet_selection(fatjets, electrons, (veto_electrons | good_electrons), parameters["fatjets"])
      bfatjets = good_fatjets & (fatjets.btagHbb > .8) # Higgs to BB tagger discriminator, working point medium2

    # apply basic event selection -> individual categories cut later
    nleps =  NUMPY_LIB.add(ha.sum_in_offsets(muons, good_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8), ha.sum_in_offsets(electrons, good_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8))
    lepton_veto = NUMPY_LIB.add(ha.sum_in_offsets(muons, veto_muons, mask_events, muons.masks["all"], NUMPY_LIB.int8), ha.sum_in_offsets(electrons, veto_electrons, mask_events, electrons.masks["all"], NUMPY_LIB.int8))
    njets = ha.sum_in_offsets(jets, good_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
    btags = ha.sum_in_offsets(jets, bjets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
    met = (scalars["MET_pt"] > 20)

    # apply basic event definition (inverted for boosted analysis)
    if boosted:
      mask_events = mask_events & (nleps == 1) & (lepton_veto == 0) & NUMPY_LIB.invert( (njets >= 4) & (btags >=2) ) & met
    else:
      mask_events = mask_events & (nleps == 1) & (lepton_veto == 0) & (njets >= 4) & (btags >=2) & met

    ### check overlap between AK4 and AK8 jets: if (based on tau32 and tau21) the AK8 jet is a t/H/W candidate remove the AK4 jet, otherwise remove the AK8 jet
    if boosted:
      fatjets.tau32 = NUMPY_LIB.divide(fatjets.tau3, fatjets.tau2)
      fatjets.tau21 = NUMPY_LIB.divide(fatjets.tau2, fatjets.tau1)
      tau32cut = 0.4
      tau21cut = 0.4
      jets_to_keep = ha.mask_overlappingAK4(jets, good_jets, fatjets, good_fatjets, 1.2, tau32cut=tau32cut, tau21cut=tau21cut)
      non_overlapping_fatjets = ha.mask_deltar_first(fatjets, good_fatjets, jets, good_jets, 1.2)

      good_jets &= jets_to_keep
      good_fatjets &= non_overlapping_fatjets | (fatjets.tau32 < 0.4) | (fatjets.tau21 < 0.4) #we keep fat jets which are not overlapping, or if they are either a top or W/H candidate

      top_candidates = (fatjets.tau32 < tau32cut)
      WH_candidates = (fatjets.tau32 > tau32cut) & (fatjets.tau21 < tau21cut)
      bjets = good_jets & (jets.btagDeepB > 0.4941)
      njets = ha.sum_in_offsets(jets, good_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)
      btags = ha.sum_in_offsets(jets, bjets, mask_events, jets.masks["all"], NUMPY_LIB.int8)

      bbtags = ha.sum_in_offsets(fatjets, bfatjets, mask_events, fatjets.masks["all"], NUMPY_LIB.int8)
      ntop_candidates = ha.sum_in_offsets(fatjets, top_candidates, mask_events, fatjets.masks["all"], NUMPY_LIB.int8)
      nWH_candidates = ha.sum_in_offsets(fatjets, WH_candidates, mask_events, fatjets.masks["all"], NUMPY_LIB.int8)

      ### 2 fat jets from H and W, 2 b jets from the tops
      #mask_events &= (nWH_candidates > 1) & (btags > 1)
      ### 1 top candidate and 1 H candidate, and 1 b jet from the leptonic top
      mask_events &= (ntop_candidates > 0) & (nWH_candidates > 0) & (btags > 0)

    # calculation of all needed variables
    # get control variables
    inds = NUMPY_LIB.zeros(nEvents, dtype=NUMPY_LIB.int32)
    leading_jet_pt = ha.get_in_offsets(jets.pt, jets.offsets, inds, mask_events, good_jets)
    leading_jet_eta = ha.get_in_offsets(jets.eta, jets.offsets, inds, mask_events, good_jets)
    leading_lepton_pt = NUMPY_LIB.maximum(ha.get_in_offsets(muons.pt, muons.offsets, inds, mask_events, good_muons), ha.get_in_offsets(electrons.pt, electrons.offsets, inds, mask_events, good_electrons))
    leading_lepton_eta = NUMPY_LIB.maximum(ha.get_in_offsets(muons.eta, muons.offsets, inds, mask_events, good_muons), ha.get_in_offsets(electrons.eta, electrons.offsets, inds, mask_events, good_electrons))
    leading_bjet_pt = ha.get_in_offsets(jets.pt,jets.offsets,inds,mask_events,bjets)
    if boosted:
      nfatjets = ha.sum_in_offsets(fatjets, good_fatjets, mask_events, fatjets.masks["all"], NUMPY_LIB.int8)
      leading_fatjet_pt = ha.get_in_offsets(fatjets.pt, fatjets.offsets, inds, mask_events, good_fatjets)
      leading_fatjet_eta = ha.get_in_offsets(fatjets.eta, fatjets.offsets, inds, mask_events, good_fatjets)
      leading_fatjet_mass = ha.get_in_offsets(fatjets.mass, fatjets.offsets, inds, mask_events, good_fatjets)
      leading_fatjet_SDmass = ha.get_in_offsets(fatjets.msoftdrop, fatjets.offsets, inds, mask_events, good_fatjets)
      leading_fatjet_tau32 = ha.get_in_offsets(fatjets.tau32, fatjets.offsets, inds, mask_events, good_fatjets)
      leading_fatjet_tau21 = ha.get_in_offsets(fatjets.tau21, fatjets.offsets, inds, mask_events, good_fatjets)
      leading_topcandidate_SDmass = ha.get_in_offsets(fatjets.msoftdrop, fatjets.offsets, inds, mask_events, top_candidates)
      leading_topcandidate_tau32 = ha.get_in_offsets(fatjets.tau32, fatjets.offsets, inds, mask_events, top_candidates)
      leading_topcandidate_tau21 = ha.get_in_offsets(fatjets.tau21, fatjets.offsets, inds, mask_events, top_candidates)
      inds_WHcandidates = ha.index_in_offsets(fatjets.btagHbb, fatjets.offsets, 1, mask_events, WH_candidates)
      leading_WHcandidate_SDmass = ha.get_in_offsets(fatjets.msoftdrop, fatjets.offsets, inds_WHcandidates, mask_events, WH_candidates)
      leading_WHcandidate_tau32 = ha.get_in_offsets(fatjets.tau32, fatjets.offsets, inds_WHcandidates, mask_events, WH_candidates)
      leading_WHcandidate_tau21 = ha.get_in_offsets(fatjets.tau21, fatjets.offsets, inds_WHcandidates, mask_events, WH_candidates)

      higgs = (genparts.pdgId == 25) & (genparts.status==62)
      tops  = ( (genparts.pdgId == 6) | (genparts.pdgId == -6) ) & (genparts.status==62)
      higgs_pt = ha.get_in_offsets(genparts.pt, genparts.offsets, inds, mask_events, higgs)
      higgs_eta = ha.get_in_offsets(genparts.eta, genparts.offsets, inds, mask_events, higgs)
      top_pt = ha.get_in_offsets(genparts.pt, genparts.offsets, inds, mask_events, tops)
      top_eta = ha.get_in_offsets(genparts.eta, genparts.offsets, inds, mask_events, tops)

    inds = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.int32)
    subleading_bjet_pt = ha.get_in_offsets(jets.pt,jets.offsets,inds,mask_events,bjets)
    if boosted:
      subleading_fatjet_pt = ha.get_in_offsets(fatjets.pt, fatjets.offsets, inds, mask_events, good_fatjets)
      subleading_fatjet_eta = ha.get_in_offsets(fatjets.eta, fatjets.offsets, inds, mask_events, good_fatjets)
      subleading_fatjet_mass = ha.get_in_offsets(fatjets.mass, fatjets.offsets, inds, mask_events, good_fatjets)
      subleading_fatjet_SDmass = ha.get_in_offsets(fatjets.msoftdrop, fatjets.offsets, inds, mask_events, good_fatjets)

    # calculate weights for MC samples
    weights = {}
    weights["nominal"] = NUMPY_LIB.ones(nEvents, dtype=NUMPY_LIB.float32)

    if is_mc:
        weights["nominal"] = weights["nominal"] * scalars["genWeight"] * eraDependentParameters[args.year]["lumi"] * samples_info[sample]["XS"] / samples_info[sample]["ngen_weight"]

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
        DNN_pred = evaluate_DNN(jets, good_jets, electrons, good_electrons, muons, good_muons, scalars, mask_events, DNN, DNN_model)

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

        if boosted:
          list_cat = zip([mask_events_split],["boosted"])
        else:
          list_cat = zip([mask_events_split], ["sl_jge4_tge2"])

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
            hist_leading_bjet_pt = Histogram(*ha.histogram_from_vector(leading_bjet_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,500,31)))
            ret["hist_{0}_leading_bjet_pt".format(name)] = hist_leading_bjet_pt

            if boosted: 
              hist_subleading_bjet_pt = Histogram(*ha.histogram_from_vector(subleading_bjet_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,500,31)))
              ret["hist_{0}_subleading_bjet_pt".format(name)] = hist_subleading_bjet_pt
              hist_leading_jet_eta = Histogram(*ha.histogram_from_vector(leading_jet_eta[cut], weights["nominal"][cut], NUMPY_LIB.linspace(-2.4,2.4,31)))
              ret["hist_{0}_leading_jet_eta".format(name)] = hist_leading_jet_eta
              hist_leading_lepton_eta = Histogram(*ha.histogram_from_vector(leading_lepton_eta[cut], weights["nominal"][cut], NUMPY_LIB.linspace(-2.4,2.4,31)))
              ret["hist_{0}_leading_lepton_eta".format(name)] = hist_leading_lepton_eta
              hist_higgs_pt = Histogram(*ha.histogram_from_vector(higgs_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,500,31)))
              ret["hist_{0}_higgs_pt".format(name)] = hist_higgs_pt
              hist_higgs_eta = Histogram(*ha.histogram_from_vector(higgs_eta[cut], weights["nominal"][cut], NUMPY_LIB.linspace(-2.4,2.4,31)))
              ret["hist_{0}_higgs_eta".format(name)] = hist_higgs_eta
              hist_top_pt = Histogram(*ha.histogram_from_vector(top_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,500,31)))
              ret["hist_{0}_top_pt".format(name)] = hist_top_pt
              hist_top_eta = Histogram(*ha.histogram_from_vector(top_eta[cut], weights["nominal"][cut], NUMPY_LIB.linspace(-2.4,2.4,31)))
              ret["hist_{0}_top_eta".format(name)] = hist_top_eta

              hist_nfatjets = Histogram(*ha.histogram_from_vector(nfatjets[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,5,6)))
              ret["hist_{0}_nfatjets".format(name)] = hist_nfatjets
              hist_nbbtags = Histogram(*ha.histogram_from_vector(bbtags[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,4,5)))
              ret["hist_{0}_nbbtags".format(name)] = hist_nbbtags
              hist_ntop_candidates = Histogram(*ha.histogram_from_vector(ntop_candidates[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,5,6)))
              ret["hist_{0}_ntop_candidates".format(name)] = hist_ntop_candidates
              hist_nWH_candidates = Histogram(*ha.histogram_from_vector(nWH_candidates[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,5,6)))
              ret["hist_{0}_nWH_candidates".format(name)] = hist_nWH_candidates
              hist_leading_fatjet_pt = Histogram(*ha.histogram_from_vector(leading_fatjet_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(200,500,31)))
              ret["hist_{0}_leading_fatjet_pt".format(name)] = hist_leading_fatjet_pt
              hist_leading_fatjet_eta = Histogram(*ha.histogram_from_vector(leading_fatjet_eta[cut], weights["nominal"][cut], NUMPY_LIB.linspace(-2.4,2.4,31)))
              ret["hist_{0}_leading_fatjet_eta".format(name)] = hist_leading_fatjet_eta
              hist_leading_fatjet_mass = Histogram(*ha.histogram_from_vector(leading_fatjet_mass[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,300,31)))
              ret["hist_{0}_leading_fatjet_mass".format(name)] = hist_leading_fatjet_mass
              hist_leading_fatjet_SDmass = Histogram(*ha.histogram_from_vector(leading_fatjet_SDmass[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,300,31)))
              ret["hist_{0}_leading_fatjet_SDmass".format(name)] = hist_leading_fatjet_SDmass
              hist_subleading_fatjet_pt = Histogram(*ha.histogram_from_vector(subleading_fatjet_pt[cut], weights["nominal"][cut], NUMPY_LIB.linspace(200,500,31)))
              ret["hist_{0}_subleading_fatjet_pt".format(name)] = hist_subleading_fatjet_pt
              hist_subleading_fatjet_mass = Histogram(*ha.histogram_from_vector(subleading_fatjet_mass[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,300,31)))
              ret["hist_{0}_subleading_fatjet_mass".format(name)] = hist_subleading_fatjet_mass
              hist_subleading_fatjet_SDmass = Histogram(*ha.histogram_from_vector(subleading_fatjet_SDmass[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,300,31)))
              ret["hist_{0}_subleading_fatjet_SDmass".format(name)] = hist_subleading_fatjet_SDmass
              hist_leading_WHcandidate_SDmass = Histogram(*ha.histogram_from_vector(leading_WHcandidate_SDmass[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,300,31)))
              ret["hist_{0}_leading_WHcandidate_SDmass".format(name)] = hist_leading_WHcandidate_SDmass
              hist_leading_topcandidate_SDmass = Histogram(*ha.histogram_from_vector(leading_topcandidate_SDmass[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,300,31)))
              ret["hist_{0}_leading_topcandidate_SDmass".format(name)] = hist_leading_topcandidate_SDmass
              hist_tau32_fatjets = Histogram(*ha.histogram_from_vector(leading_fatjet_tau32[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,1,31)))
              ret["hist_{0}_tau32_fatjets".format(name)] = hist_tau32_fatjets
              hist_tau32_topcandidates = Histogram(*ha.histogram_from_vector(leading_topcandidate_tau32[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,1,31)))
              ret["hist_{0}_tau32_topcandidates".format(name)] = hist_tau32_topcandidates
              hist_tau32_WHcandidates = Histogram(*ha.histogram_from_vector(leading_WHcandidate_tau32[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,1,31)))
              ret["hist_{0}_tau32_WHcandidates".format(name)] = hist_tau32_WHcandidates
              hist_tau21_fatjets = Histogram(*ha.histogram_from_vector(leading_fatjet_tau21[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,1,31)))
              ret["hist_{0}_tau21_fatjets".format(name)] = hist_tau21_fatjets
              hist_tau21_topcandidates = Histogram(*ha.histogram_from_vector(leading_topcandidate_tau21[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,1,31)))
              ret["hist_{0}_tau21_topcandidates".format(name)] = hist_tau21_topcandidates
              hist_tau21_WHcandidates = Histogram(*ha.histogram_from_vector(leading_WHcandidate_tau21[cut], weights["nominal"][cut], NUMPY_LIB.linspace(0,1,31)))
              ret["hist_{0}_tau21_WHcandidates".format(name)] = hist_tau21_WHcandidates


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
    parser.add_argument('--DNN', action='store', choices=['save-arrays','cmb_binary', 'cmb_multiclass', 'ffwd_binary', 'ffwd_multiclass',False], help='options for DNN evaluation / preparation', default=False)
    parser.add_argument('--categories', action='store', choices=['sl_j4_tge3','sl_j5_tge3', 'sl_jge6_tge3',False], help='categories to be processed (default: False -> all categories)', default=False)
    parser.add_argument('--path-to-model', action='store', help='path to DNN model', type=str, default=None, required=False)
    parser.add_argument('--boosted', action='store_true', help='Flag to include boosted objects', default=False)
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

    outdir = args.outdir
    if not os.path.exists(outdir):
        print(os.getcwd())
        os.makedirs(outdir)

    if "Run" in args.sample:
        is_mc = False
        lumimask = LumiMask(eraDependentParameters["lumimask"])
    else:
        is_mc = True
        lumimask = None


    #define arrays to load: these are objects that will be kept together
    arrays_objects = [
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepB", "Jet_jetId", "Jet_puId", "Jet_mass", "Jet_hadronFlavour",
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pfRelIso04_all", "Muon_tightId", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge", "Electron_deltaEtaSC", "Electron_cutBased", "Electron_dz", "Electron_dxy",
    ]
    if args.boosted:
      arrays_objects += [
        "GenPart_eta","GenPart_genPartIdxMother","GenPart_mass","GenPart_pdgId","GenPart_phi","GenPart_pt","GenPart_status","GenPart_statusFlags",
        "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_btagHbb", "FatJet_deepTagMD_HbbvsQCD", "FatJet_deepTagMD_ZHbbvsQCD", "FatJet_deepTagMD_TvsQCD", "FatJet_deepTag_H", "FatJet_deepTag_TvsQCD", "FatJet_jetId", "FatJet_mass", "FatJet_msoftdrop", "FatJet_tau1", "FatJet_tau2", "FatJet_tau3", "FatJet_tau4"]

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
        if args.boosted:
          structs.append(["FatJet", "GenPart"])
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
            parameters["pu_corrections_target"] = load_puhist_target(eraDependentParameters[args.year]["pu_corrections_file"])

            ext = extractor()
            for corr in eraDependentParameters[args.year]["corrections"]:
                ext.add_weight_sets([corr])
            ext.finalize()
            evaluator = ext.make_evaluator()


        if ibatch == 0:
            print(dataset.printout())

        # in case of DNN evaluation: load model
        model = None
        if args.DNN:
            model = load_model(args.path_to_model, custom_objects=dict(itertools=itertools, mse0=mse0, mae0=mae0, r2_score0=r2_score0))

        #### this is where the magic happens: run the main analysis
        results += dataset.analyze(analyze_data, NUMPY_LIB=NUMPY_LIB, parameters=parameters, is_mc = is_mc, lumimask=lumimask, cat=args.categories, sample=args.sample, samples_info=samples_info, boosted=args.boosted, DNN=args.DNN, DNN_model=model)


    print(results)

    #Save the results
    if not os.path.isdir(args.outdir):
      os.makedirs(args.outdir)
    results.save_json(os.path.join(outdir,"out_{}.json".format(args.sample)))
