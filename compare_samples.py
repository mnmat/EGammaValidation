import pandas as pd
import numpy as np
import os
import sys

sys.path.append("/eos/home-m/mmatthew/SWAN_projects/BDT")

from utils.utils import *
from produce_plots import *
from Validation.plots import *
from Validation.hist_helpers import *

import argparse

def create_histograms(label="IdealIC"):
    min_frac = 0.7
    max_frac = 1.2
    min_energy = 0 
    max_energy = 600
    bins = 50
    
    hists = {
        "h_regressed_ratio_%s"%label : hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="%s, regInvTar*regMean"%label, label="regInvTar")),
        #h_eg_energy_ratio_ideal = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="IdealIC, eg_energy/eg_gen_energy", label="regInvTar")),
        "h_SC_rawEnergy_ratio_%s"%label : hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="%s, eg_rawEnergy/eg_gen_energy"%label, label="regInvTar")),
        "h_old_regressed_ratio_%s"%label : hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="%s, eg_regressedEnergy_old/eg_gen_energy"%label, label="regInvTar")),
    }
    return hists

def fill_histograms(hists, df_cut, label):
    hists["h_regressed_ratio_%s"%label].fill(df_cut["regInvTar*regMean"])
    #hists["h_eg_energy_ratio_ideal"].fill(df_ideal_cut["eg_energy/eg_gen_energy"])
    hists["h_SC_rawEnergy_ratio_%s"%label].fill(df_cut["eg_rawEnergy/eg_gen_energy"])
    hists["h_old_regressed_ratio_%s"%label].fill(df_cut["eg_regressedEnergyOld/eg_gen_energy"])
    return hists

def create_subspace_histograms(dfs,labels,feature, cuts):
    # Cuts should be imported as an array of pairs for the left and right boundaries
    hists_subspace = {}  
    print(cuts.shape)
    
    
    for i in range(cuts.shape[1]):
        hists = {}
        for j,df in enumerate(dfs):
        # Write Histograms
            print(cuts[0][i], cuts[1][i])
            tmp = df[(df[feature]>cuts[0][i]) & (df[feature]<cuts[1][i])]
            h = create_histograms(labels[j])
            h = fill_histograms(h,tmp,labels[j])
            hists.update(h)

        hists_subspace["%s_%s"%(cuts[0][i], cuts[1][i])] = hists
    return hists_subspace

def loadData(file,tree="egHLTRun3Tree"):
    # Load data
    f_ideal = uproot.open(file)

    df_ideal = f_ideal[tree].arrays(library="pd")
    df_ideal["regInvTar*regMean"] = df_ideal["regInvTar"]*df_ideal["regMean"]
    df_ideal["eg_energy/eg_gen_energy"] = df_ideal["eg_energy"]/df_ideal["eg_gen_energy"]
    df_ideal["eg_rawEnergy/eg_gen_energy"] = df_ideal["eg_rawEnergy"]/df_ideal["eg_gen_energy"]
    df_ideal["eg_regressedEnergyOld/eg_gen_energy"] = df_ideal["eg_regressedEnergy"]/df_ideal["eg_gen_energy"]

    return df_ideal 

def applyCuts(df, pt_low = 10, pt_high = 600, eta=3):
    df = df[(df["eg_gen_pt"] > pt_low) & (df["eg_gen_pt"] < pt_high) & (df["eg_gen_eta"] < eta)]
    return df

def loadDataFromTrees(file,trees = ["egRegDataEcalHLTV1","egRegDataHGCALHLTV1"]):
    f_ideal = uproot.open(file)

    df_ideal = []

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

    df_ideal = pd.concat(df_ideal)
    return df_ideal



f_1 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/GenSim/Combined_DoublePhoton_DoubleElectron_FlatPt-1To100-gun/s5Reg_genMatched/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied_photon.root"
f_2 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/GenSim/DoubleElectron_FlatPt-1To100-gun/s5Reg_genMatched_DoublePhoton_FlatPt-1To100-gun/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"
f_3 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/GenSim/DoublePhoton_FlatPt-1To100-gun/s5Reg_genMatched/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"

OUTDIR = "/eos/home-m/mmatthew/www/BDT/HLT_baseline/Comparison/Samples/TICLv4/unseeded_genMatched/GenSim/DoublePhoton"


OUTDIR = "/eos/home-m/mmatthew/www/test_workflow/CompareSamples"
f_1 = "/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/Data/Reg/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"
f_2 = "/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/Data/Reg/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--outDir",type=str,default="/eos/home-m/mmatthew/www/test_workflow/CompareSamples")
    parser.add_argument("--files",nargs='+',default = ["/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/Data/Reg/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root","/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/Data/Reg/Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"])
    parser.add_argument("--labels",nargs='+',default = ["A","B"])
    parser.add_argument("--colors",nargs="+",default = ["blue","red"])
    args = parser.parse_args()

    outDir = args.outDir
    fs = args.files
    names = args.labels
    colors = args.colors

    fittype = "Cruijff"

    # Load data

    sample = []
    dfs = []
    for name,f in zip(names,fs):
        sample.append(name) 
        df = loadDataFromTrees(f)
        df = applyCuts(df)
        dfs.append(df)

    for key in ["eg_isEE", "eg_isEB"]:
        dfs_subdetector = [df[df[key]==1] for df in dfs]

        # # Settings

        # df_ideal_1 = df_1[df_1["eg_isEE"]==1]
        # df_ideal_2 = df_2[df_2["eg_isEE"]==1]
        # df_ideal_3 = df_3[df_3["eg_isEE"]==1]


        # dfs = [df_ideal_1, df_ideal_2, df_ideal_3]

        if key == "eg_isEE":
            energy_cuts = ENERGY_CUTS["EEonly"]
            eta_cuts = ETA_CUTS["EEonly"]
            subdetector = "EE"
        else:
            energy_cuts = ENERGY_CUTS["EBonly"]
            eta_cuts = ETA_CUTS["EBonly"]
            subdetector = "EB"

        out = os.path.join(OUTDIR,subdetector)

        # Fit

        feature = "eg_gen_energy"
        hists = create_subspace_histograms(dfs_subdetector,names,feature,energy_cuts)
        fits = fit_subspace_histograms(hists,fittype)


        # Plot regressed energy

        histkeys = {"h_regressed_ratio_%s"%names[0]:"%s"%names[0],
            "h_regressed_ratio_%s"%names[1]:"%s"%names[1]}

        outDir = os.path.join(out,"RegressedEnergy")
        create_dirs(outDir,add_php=True)
        yscale=[[.98,1.04],[0.01,.08],[0.01,.06]]
        make_fitted_histograms(hists, fits,histkeys, outDir,fittype=fittype)
        make_parameter_plots(fits,histkeys,energy_cuts,outDir,"energies","Gen Energy [GeV]",r"$\mu$ (DSCB)",fittype=fittype,set_yscale=yscale)

        # Plot old regressed energy

        histkeys = {"h_old_regressed_ratio_%s"%names[0]:"%s"%names[0],
            "h_old_regressed_ratio_%s"%names[1]:"%s"%names[1]}

        outDir = os.path.join(out,"OldRegressedEnergy")
        create_dirs(outDir,add_php=True)
        yscale=[[.9,1.04],[0.01,.08],[0.01,.06]]
        make_fitted_histograms(hists, fits,histkeys, outDir,fittype=fittype)
        make_parameter_plots(fits,histkeys,energy_cuts,outDir,"energies","Gen Energy [GeV]",r"$\mu$ (DSCB)",fittype=fittype,set_yscale=yscale)

        # Plot SC rawEnergy

        histkeys = {"h_SC_rawEnergy_ratio_%s"%names[0]:"%s"%names[0],
            "h_SC_rawEnergy_ratio_%s"%names[1]:"%s"%names[1]}

        outDir = os.path.join(out,"RawEnergy")
        create_dirs(outDir,add_php=True)
        yscale=[[.8,1.04],[0.01,.08],[0.01,.06]]
        make_fitted_histograms(hists, fits,histkeys, outDir,fittype=fittype)
        make_parameter_plots(fits,histkeys,energy_cuts,outDir,"energies","Gen Energy [GeV]",r"$\mu$ (DSCB)",fittype=fittype,set_yscale=yscale)
