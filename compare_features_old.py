import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from utils.utils import *
from Validation.feature_plots import *

import sys


def compare_features():
    outdir = "/eos/home-m/mmatthew/www/BDT/HLT_baseline/GenSim/unseeded_genMatched/Features/Comparison_Electron_Photon"
    #outdir = "/eos/home-m/mmatthew/www/BDT/CMSSW_15_0_0_pre3/Spring24/Ticlv5/DoubleElectron/Features"

    create_dirs(outdir,True)

    # Samples
    training_samples = []
    test_samples = []

    path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/GenSim/DoubleElectron_FlatPt-1To100-gun"
    #path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/CMSSW_15_0_0_pre3/debug/TICLv5/ele/"


    train,test = get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s5Reg_genMatched")
    training_samples.append(train)
    test_samples.append(test)

    # path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/Spring23/DoubleElectron_FlatPt-1To100-gun/"
    path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/GenSim/DoublePhoton_FlatPt-1To100-gun/"
    train,test = get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s5Reg_genMatched")
    training_samples.append(train)
    test_samples.append(test)
    names = ["Electron","Photon"]

    plot_event_sizes(training_samples,test_samples,outdir,names)

    # Create Dataframes

    dfs = []
    key = "egRegDataHGCALHLTV1"

    for file in test_samples:
        f = uproot.open(file)
        df = f[key].arrays(library="pd")
        dfs.append(modify_tree(df))

    # Plot Energies
    for df,name in zip(dfs,names):
        plot_energies(df,outdir,name)
        scatter(df,outdir,name,ls=["raw","reg"])
        plot_fractions(df,outdir,name)

    # Plot features
    plot_feature_hists(dfs,names,outdir)

    # Plot correlation
    plot_feature_correlation(dfs,names,outdir)

if __name__ == "__main__":
    compare_features()
