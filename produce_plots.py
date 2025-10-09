import uproot
import numpy as np
import pandas as pd
from Validation.fit import *
from Validation.hist_helpers import *
from Validation.plots import *
import pdb

class validationPlotter():
    def __init__(self,file_ideal,file_real,outDir,legacy=True):
        self.file_ideal = file_ideal
        self.file_real = file_real
        self.outDir = outDir
        self.pt_low = 10
        self.pt_high = 600
        self.etaCutVal = 3
        self.fittype="Cruijff"
        if legacy:
            self.loadData()
        else:
            self.loadDataFromTrees()
            #self.loadDataFromTrees(trees=["egRegDataEcalV1","egRegDataHGCALV1"])
        self.applyCuts()

    def applyCuts(self):
        self.df_ideal_cut = self.df_ideal[(self.df_ideal["eg_gen_pt"] > self.pt_low) & (self.df_ideal["eg_gen_pt"] < self.pt_high) & (self.df_ideal["eg_gen_eta"] < 3)]
        self.df_real_cut = self.df_real[(self.df_real["eg_gen_pt"] > self.pt_low) & (self.df_real["eg_gen_pt"] < self.pt_high) & (self.df_real["eg_gen_eta"] < 3)]
        return 0


    def loadDataFromTrees(self,trees = ["egRegDataEcalHLTV1","egRegDataHGCALHLTV1"]):
        f_ideal = uproot.open(self.file_ideal)
        f_real = uproot.open(self.file_real)

        df_ideal = []
        df_real = []

        for tree in trees:
            df = f_ideal[tree].arrays(library="pd")
            if "Ecal" in tree:
                df = df.rename(columns = {"regEBMean":"regMean", "regEBSigma":"regSigma", "regEBInvTar":"regInvTar"})
                df["eg_isEB"] = 1
                df["eg_isEE"] = 0
            else:
                df = df.rename(columns = {"regEEMean":"regMean", "regEESigma":"regSigma", "regEEInvTar":"regInvTar"})
                df["eg_isEB"] = 0
                df["eg_isEE"] = 1

            df["regInvTar*regMean"] = df["regInvTar"]*df["regMean"]
            df["eg_energy/eg_gen_energy"] = df["rawEnergy"]/df["eg_gen_energy"]
            df["eg_rawEnergy/eg_gen_energy"] = df["rawEnergy"]/df["eg_gen_energy"]
            df["eg_regressedEnergyOld/eg_gen_energy"] = df["regressedEnergy"]/df["eg_gen_energy"]


            df_ideal.append(df)

            df = f_real[tree].arrays(library="pd")
            if "Ecal" in tree:
                df = df.rename(columns = {"regEBMean":"regMean", "regEBSigma":"regSigma", "regEBInvTar":"regInvTar"})
                df["eg_isEB"] = 1
                df["eg_isEE"] = 0
            else:
                df = df.rename(columns = {"regEEMean":"regMean", "regEESigma":"regSigma", "regEEInvTar":"regInvTar"})
                df["eg_isEB"] = 0
                df["eg_isEE"] = 1

            df["regInvTar*regMean"] = df["regInvTar"]*df["regMean"]
            df["eg_energy/eg_gen_energy"] = df["rawEnergy"]/df["eg_gen_energy"]
            df["eg_rawEnergy/eg_gen_energy"] = df["rawEnergy"]/df["eg_gen_energy"]
            df["eg_regressedEnergyOld/eg_gen_energy"] = df["regressedEnergy"]/df["eg_gen_energy"]


            df_real.append(df)
        

        self.df_real = pd.concat(df_real)
        self.df_ideal = pd.concat(df_ideal)
        return 0


    def loadData(self,tree="egHLTRun3Tree"):
        # Load data
        print(self.file_ideal, self.file_real)
        f_ideal = uproot.open(self.file_ideal)
        f_real = uproot.open(self.file_real)

        df_ideal = f_ideal[tree].arrays(library="pd")
        df_ideal["regInvTar*regMean"] = df_ideal["regInvTar"]*df_ideal["regMean"]
        df_ideal["eg_energy/eg_gen_energy"] = df_ideal["eg_energy"]/df_ideal["eg_gen_energy"]
        df_ideal["eg_rawEnergy/eg_gen_energy"] = df_ideal["eg_rawEnergy"]/df_ideal["eg_gen_energy"]
        df_ideal["eg_regressedEnergyOld/eg_gen_energy"] = df_ideal["eg_regressedEnergy"]/df_ideal["eg_gen_energy"]

        df_real = f_real[tree].arrays(library="pd")
        df_real["regInvTar*regMean"] = df_real["regInvTar"]*df_real["regMean"]
        df_real["eg_energy/eg_gen_energy"] = df_real["eg_energy"]/df_real["eg_gen_energy"]
        df_real["eg_rawEnergy/eg_gen_energy"] = df_real["eg_rawEnergy"]/df_real["eg_gen_energy"]
        df_real["eg_regressedEnergyOld/eg_gen_energy"] = df_real["eg_regressedEnergy"]/df_real["eg_gen_energy"]

        self.df_real = df_real
        self.df_ideal = df_ideal
        return 0 

    def doFullValidation(self,mode="CombinedEEandEB"):
        if mode == "CombinedEEandEB":
            df_ideal_cut = self.df_ideal_cut
            df_real_cut = self.df_real_cut
            energy_cuts = ENERGY_CUTS["CombinedEEandEB"]
            eta_cuts = ETA_CUTS["CombinedEEandEB"]
        elif mode == "EBonly":
            df_ideal_cut = self.df_ideal_cut[self.df_ideal_cut["eg_isEB"]==1]
            df_real_cut = self.df_real_cut[self.df_real_cut["eg_isEB"]==1]
            energy_cuts = ENERGY_CUTS["EBonly"]
            eta_cuts = ETA_CUTS["EBonly"]
        elif mode == "EEonly":
            df_ideal_cut = self.df_ideal_cut[self.df_ideal_cut["eg_isEE"]==1]
            df_real_cut = self.df_real_cut[self.df_real_cut["eg_isEE"]==1]
            energy_cuts = ENERGY_CUTS["EEonly"]
            eta_cuts = ETA_CUTS["EEonly"]
        else:
            print("Mode not defined")

        outDir = self.outDir + "/%s/"%mode
        make_overall_plots(df_ideal_cut, df_real_cut,outDir,self.fittype)
        make_energy_plots(df_ideal_cut, df_real_cut,outDir,["h_regressed_ratio_ideal","h_old_regressed_ratio_real"],energy_cuts,self.fittype) # ,"h_old_regressed_ratio_real"
        make_eta_plots(df_ideal_cut, df_real_cut,outDir,["h_regressed_ratio_ideal","h_old_regressed_ratio_ideal"],eta_cuts,self.fittype) # ,"h_old_regressed_ratio_real"

        return 0

def make_overall_plots(df_ideal_cut, df_real_cut,outDir,fittype="DSCB"):
    hists = create_histograms_legacy()
    hists = fill_histograms_legacy(hists,df_ideal_cut, df_real_cut)
    fits = {}
    for key in hists.keys():
        if fittype=="DSCB":
            fits[key] = getDSCBParams(hists,key)[0]
        elif fittype=="Cruijff":
            fits[key] = getCruijffParams(hists,key)[0]

    labels = hists.keys()
    hists = [val for val in hists.values()]
    fits = [val for val in fits.values()]

    plot_fitted_histograms(hists,fits,labels,COLORS[:len(hists)],outDir,"Overall",log=False,fittype=fittype)
    return 0


def make_energy_plots(df_ideal_cut, df_real_cut, outDir, histtypes,cuts,fittype="DSCB"):
    # TODO: Make only one function for the make_energies_plots, make_eta_plots so that we can extend this to other features easily
    histkeys = {"h_regressed_ratio_ideal":"regressed energy ratio",
        #"h_eg_energy_ratio_ideal":"eg energy ratio",
        "h_SC_rawEnergy_ratio_ideal":"SC rawEnergy ratio",
        "h_old_regressed_ratio_ideal": "old regressed energy ratio"}

    feature = "eg_gen_energy"
    hists_energies = create_subspace_histograms_legacy(df_ideal_cut,df_real_cut,feature,cuts)
    fit_energies = fit_subspace_histograms(hists_energies,fittype)

    make_fitted_histograms(hists_energies, fit_energies,histkeys, outDir,fittype=fittype)
    for histtype in histtypes:
        name = histtype+"_energies"
        make_fitted_histograms_combined(hists_energies,fit_energies,histtype,outDir,name,"E [GeV]",fittype=fittype)
    
    make_parameter_plots(fit_energies,histkeys,cuts,outDir,"energies","Gen Energy [GeV]",r"$\mu$ (DSCB)",fittype=fittype)   
    return 0

def make_eta_plots(df_ideal_cut, df_real_cut, outDir, histtypes,cuts,fittype="DSCB"):
    # TODO: Make only one function for the make_energies_plots, make_eta_plots so that we can extend this to other features easily
    histkeys = {"h_regressed_ratio_ideal":"regressed energy ratio",
        #"h_eg_energy_ratio_ideal":"eg energy ratio",
        "h_SC_rawEnergy_ratio_ideal":"SC rawEnergy ratio",
        "h_old_regressed_ratio_ideal": "old regressed energy ratio"}
    feature = "eg_gen_eta"

    hists_eta = create_subspace_histograms_legacy(df_ideal_cut,df_real_cut,feature,cuts)
    fit_eta = fit_subspace_histograms(hists_eta,fittype)

    make_fitted_histograms(hists_eta, fit_eta,histkeys, outDir,fittype=fittype)
    for histtype in histtypes:
        name = histtype+"_etas"
        make_fitted_histograms_combined(hists_eta,fit_eta,histtype,outDir,name,r"$\eta$",fittype=fittype)
    make_parameter_plots(fit_eta,histkeys,cuts,outDir,"etas",r"$\eta_{Gen}$",r"$\mu$ (DSCB)",fittype=fittype)
    return 0

# Setup

# TODO: 
# For HLT, REAL file doesn't exist. It is therefore not used in this script and would have to manually be added into the plots. Find better solution to process both Run3 and Run4 samples

OUTDIR = "/eos/home-m/mmatthew/www/test_workflow/Plots"
FILE_REAL = "/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/Data/Reg/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"
FILE_IDEAL = "/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/Data/Reg/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"


ETA_CUTS = {"CombinedEEandEB":np.array([list(np.linspace(-3,2.5,12)), list(np.linspace(-2.5,3,12))]),
            "EEonly": np.array([[-3,-2.6,-2.2,-1.8,1.5,1.8,2.2,2.6], [-2.6,-2.2,-1.8,-1.5,1.8,2.2,2.6,3]]),
            "EBonly": np.array([[-1.5,-1.2,-.9,-.6,-.3,0, 0.3,.6,.9,1.2], [-1.2,-.9,-.6,-.3,0, 0.3,.6,.9,1.2,1.5]])}
ENERGY_CUTS = {"CombinedEEandEB":np.array([[50,75, 100, 200, 400], [75, 100, 200, 400,600]]),
                "EEonly": np.array([[0, 100, 175, 250, 350,450],[100, 175, 250, 350,450,1000]]),
                "EBonly":np.array([[0, 35, 55, 75, 95,125], [35, 55, 75, 95,125,400]])}


if __name__ == "__main__":
    vP = validationPlotter(FILE_IDEAL,FILE_REAL,OUTDIR,False)
    #vP.doFullValidation("CombinedEEandEB")
    vP.doFullValidation("EEonly")
    vP.doFullValidation("EBonly")
