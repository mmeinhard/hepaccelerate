

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
            "jetId": 2,
            "puId": 4
    },
    "fatjets": {
               "type": "fatjet",
               "dr": 0.8,
               "pt": 200,
               "eta": 2.4,
               "jetId": 2
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
            "el_triggerSF SFs_ele_pt_ele_sceta_ele28_ht150_OR_ele35_2017BCDEF ./data/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb_Data_MC_v5.0.histo.root",
            "el_recoSF EGamma_SF2D ./data/egammaEffi_EGM2D_runBCDEF_passingRECO.histo.root",
            "el_idSF EGamma_SF2D ./data/egammaEffi_EGM2D_runBCDEF_passingTight94X.histo.root",
            "mu_triggerSF IsoMu27_PtEtaBins/pt_abseta_ratio ./data/EfficienciesAndSF_RunBtoF_Nov17Nov2017.histo.root",
            "mu_isoSF NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta ./data/RunBCDEF_SF_ISO.histo.root",
            "mu_idSF NUM_TightID_DEN_genTracks_pt_abseta ./data/RunBCDEF_SF_ID.histo.root",
            "BTagSF * ./data/deepCSV_sfs_v2.btag.csv"
        ]
    }

}


#################################### Samples info ##############################################################################

samples_info = {
    "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8": {
            "process": "ttHTobb",
            "XS": 0.2934045,
            "ngen_weight": 4216319.315883999
            },
    "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8": {
            "XS": 365.45736135,
            "ngen_weight": 720253370.0403845
            },
    "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8": {
            "process": "ttHToNonbb",
            "XS": 0.2150955,
            "ngen_weight": 4484065.542378001},
    "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8": {
            "XS": 88.341903326,
            "ngen_weight": 283000430.5968169
            },
    "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8": {
            "XS": 377.9607353256,
            "ngen_weight": 1647945788.3386502
            },

    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8": {
            "XS": 88.341903326,
            "ngen_weight": 283000430.5968169
            },
    "TTToHadronic_TuneCP5_13TeV-powheg-pythia8": {
            "XS": 377.9607353256,
            "ngen_weight": 1647945788.3386502
            },
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8": {
            "XS": 365.45736135,
            "ngen_weight": 720253370.0403845
            },
}


############################################################### Histograms ########################################################
