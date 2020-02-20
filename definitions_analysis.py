

######################################## Selection criteria / Inputs for corrections ############################################

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
            "type": "jet",
            "dr": 0.4,
            "pt": 30,
            "eta": 2.4,
            #"jetId": 2,
            "jetId": 4,
            "puId": 4
    },
    "fatjets": {
               "type": "fatjet",
               "dr": 0.8,
               "pt": 200,
               "eta": 2.4,
               "jetId": 2,
               "tau32cut": 0.4,
               "tau21cut": 0.4,
    },
}

eraDependentParameters = {
    "2016" : {
        "lumi":  35922.0,
        "lumimask": "data/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt",
        "pu_corrections_file" : "data/puData2017_withVar.root",
        "corrections" : [
            "el_triggerSF Ele27_WPTight_Gsf data/TriggerSF_Run2016All_v1.root",
            "el_recoSF EGamma_SF2D data/egammaEffi.txt_EGM2D.root",
            "el_idSF EGamma_SF2D data/egammaEffi.txt_EGM2D.root",
            "mu_triggerSF IsoMu27_PtEtaBins/pt_abseta_ratio data/EfficienciesAndSF_RunBtoF_Nov17Nov2017.histo.root",
            "mu_isoSF NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta data/RunBCDEF_SF_ISO.histo.root",
            "mu_idSF NUM_TightID_DEN_genTracks_pt_abseta data/RunBCDEF_SF_ID.histo.root",
            "BTagSF * data/DeepCSV_Moriond17_B_H.csv"
        ]
    },
    "2017" : {
        "lumi":  41529.0,
        "lumimask": "data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt",
        "pu_corrections_file" : "data/pileup_Cert_294927-306462_13TeV_PromptReco_Collisions17_withVar.root",
        "corrections" : [
            #"el_triggerSF ele28_ht150_OR_ele32_ele_pt_ele_sceta ./data/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb2017_v3.histo.root",
            "el_triggerSF SFs_ele_pt_ele_sceta_ele28_ht150_OR_ele35_2017BCDEF ./data/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb_Data_MC_v5.0.histo.root",
            "el_recoSF EGamma_SF2D ./data/egammaEffi_EGM2D_runBCDEF_passingRECO_v2.histo.root",
            "el_idSF EGamma_SF2D ./data/2017_ElectronTight.histo.root",
            "mu_triggerSF IsoMu27_PtEtaBins/pt_abseta_ratio ./data/EfficienciesAndSF_RunBtoF_Nov17Nov2017.histo.root",
            "mu_isoSF NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta ./data/RunBCDEF_SF_ISO.histo.root",
            "mu_idSF NUM_TightID_DEN_genTracks_pt_abseta ./data/RunBCDEF_SF_ID.histo.root",
            #"BTagSF * ./data/DeepCSV_94XSF_V5_B_F.btag.csv"
            "BTagSFbtagCSVV2 * ./data/CSVv2_Moriond17_B_H.btag.csv",
            #"BTagSFbtagCSVV2 * ./data/CSVv2_94XSF_V2_B_F_2017.btag.csv",
            "BTagSFbtagDeepB * ./data/deepCSV_sfs_v2.btag.csv",
	        #"BTagSF * ./data/DeepCSV_94XSF_V4_B_F.btag.csv"
        ],
        "btagging algorithm" : "btagCSVV2",
        "btagging WP" : 
            {"btagDeepB": 0.4941, # medium working point for btagDeepB
            "btagCSVV2": 0.8484 # medium working point for btagCSVV2
            },
        "bbtagging WP" : 0.8, # medium 2 working point for DeepDoubleB tagger
    }

}


#################################### Samples info ##############################################################################

genweights = {
    "maren": {
        "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8": 4163245.9264759924,
        "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8": 32426751447.698845,
        "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8": 4371809.996849993,
        "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8": 4720387516.446639,
        "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8": 27550924865.573532,
    },
    "maren_v2":{
        "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8": 4163307.8224659907,
        "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8": 30587299080.771355,
        "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8": 4388463.910001993,
        "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8": 4723736912.791826,
        "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8": 27606592686.468067,
    },
    "nanoAOD_central":{
        "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8": 4216319.315883999,
        "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8": 720253370.0403845, #not full statistics
        "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8": 5722756.565262001,
        "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8": 283000430.5968169, #not full statistics
        "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8": 1647945788.3386502, #not full statistics
    },
}

samples = "maren_v2"

samples_info = {
    "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8": {
            "process": "ttHTobb",
            "XS": 0.2934045,
            "ngen_weight": genweights[samples]["ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8"],
            },
    "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8": {
            "XS": 365.45736135,
            "ngen_weight": genweights[samples]["TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8"],
            },
    "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8": {
            "process": "ttHToNonbb",
            "XS": 0.2150955,
            "ngen_weight": genweights[samples]["ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8"],
            },
    "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8": {
            "XS": 88.341903326,
            "ngen_weight": genweights[samples]["TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8"],
            },
    "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8": {
            "XS": 377.9607353256,
            "ngen_weight": genweights[samples]["TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8"],
            },
}


############################################################### Histograms ########################################################

histogram_settings = {

    "njets" : (0,14,15),
    "nleps" : (0,10,11),
    "btags" : (0,8,9),
    "pu_weights" : (0,4,21),
    "btag_weights" : (0,2,50),
    "leading_jet_pt" : (0,500,31),
    "leading_jet_pt_nom" : (0,500,31),
    "leading_jet_eta" : (-2.4,2.4,31),
    "leading_lepton_pt" : (0,500,31),
    "leading_lepton_eta" : (-2.4,2.4,31),
    "leading_bjet_pt" : (0,500,31),
    "leading_bjet_pt_nom" : (0,500,31),
    "leading_bjet_eta" : (-2.4,2.4,31),
    "subleading_bjet_pt" : (0,500,31),
    "subleading_bjet_pt_nom" : (0,500,31),
    "subleading_bjet_eta" : (-2.4,2.4,31),

    "higgs_pt": (0,500,31),
    "higgs_eta": (-2.4,2.4,31),
    "top_pt" : (0,500,31),
    "top_eta": (-2.4,2.4,31),
    "nfatjets": (0,5,6),
    "nbbtags": (0,4,5),
    "ntop_candidates": (0,5,6),
    "nWH_candidates": (0,5,6),
    "leading_fatjet_pt": (200,500,31),
    "leading_fatjet_eta": (-2.4,2.4,31),
    "leading_fatjet_mass": (0,300,31),
    "leading_fatjet_SDmass": (0,300,31),
    "subleading_fatjet_pt": (200,500,31),
    "subleading_fatjet_mass": (0,300,31),
    "subleading_fatjet_SDmass": (0,300,31),
    "leading_WHcandidate_SDmass": (0,300,31),
    "leading_topcandidate_SDmass": (0,300,31),
    "tau32_fatjets": (0,1,31),
    "tau32_topcandidates": (0,1,31),
    "tau32_WHcandidates": (0,1,31),
    "tau21_fatjets": (0,1,31),
    "tau21_topcandidates": (0,1,31),
    "tau21_WHcandidates": (0,1,31)
}
