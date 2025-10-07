
import pandas as pd
import numpy as np
import os
import sys
import pdb 
import pickle
import hist
import matplotlib.pyplot as plt

from utils.feature_mapping import *
from Validation.feature_plots import *

def get_dataframes(fnames):
    dfs_list = []
    keys = ["egRegDataHGCALHLTV1","egRegDataEcalHLTV1"]
    for key in keys:
        dfs = []
        for file in fnames:
            f = uproot.open(file)
            df = f[key].arrays(library="pd")
            df["tgt"] = df["rawEnergy"]/df["eg_gen_energy"]
            dfs.append(df)
        dfs_list.append(pd.concat(dfs,ignore_index=True))
    return dfs_list


def get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s5Reg_genMatched"):
    training_path = os.path.join(path,training_dir)
    test_path = os.path.join(path,test_dir)

    training_file = "HLTAnalyzerTree_IDEAL_Flat_train.root"
    test_file = "HLTAnalyzerTree_IDEAL_Flat_test.root"

    return [os.path.join(training_path,training_file)], [os.path.join(test_path,test_file)]


def make_feature_plots(fnames,FEATURES_RUN4_HLT,min_max_ee,min_max_eb,outDir,truth="cp_energy"):

    # Training sample
    hist2D_ee_training = create_histograms(FEATURES_RUN4_HLT,min_max_ee,truth=truth)
    hist2D_eb_training = create_histograms(FEATURES_RUN4_HLT,min_max_eb,truth=truth)
    for fname in fnames:

        f = uproot.open(fname)
        features_ee, features_eb = get_variables(f)
        gen_ee, gen_eb = get_gen_energy(f)

        hist2D_ee_training = fill_histograms(features_ee,gen_ee,FEATURES_RUN4_HLT,hist2D_ee_training,truth=truth)
        hist2D_eb_training = fill_histograms(features_eb,gen_eb,FEATURES_RUN4_HLT,hist2D_eb_training,truth=truth)

    write_split_histograms_2d(hist2D_ee_training,"clusterMaxDR",os.path.join(outDir,"EE"),save=True,truth=truth)
    write_split_histograms_2d(hist2D_eb_training,"clusterMaxDR",os.path.join(outDir,"EB"),save=True,truth=truth)

    write_histograms(hist2D_ee_training,FEATURES_RUN4_HLT,os.path.join(outDir,"EE"),truth=truth,save=True)
    write_histograms(hist2D_eb_training,FEATURES_RUN4_HLT,os.path.join(outDir,"EB"),truth=truth,save=True)


def analyse_features():
    outDir = "/eos/home-m/mmatthew/www/BDT/HLT_baseline/GenSim/unseeded_genMatched/DoublePhoton/Features_new"
    path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/GenSim/DoublePhoton_FlatPt-1To100-gun"

    train,test = get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s4Flat_genMatched")

    min_max_ee, min_max_eb = get_min_max(train+test,FEATURES_RUN4_HLT,FEATURES_RUN4_HLT,"Phase2",save=False,load=False)

    # Training

    outDir_ = os.path.join(outDir,"Training")
    make_feature_plots(train,FEATURES_RUN4_HLT,min_max_ee,min_max_eb,outDir_)
    make_feature_plots(train,FEATURES_RUN4_HLT,min_max_ee,min_max_eb,outDir_,truth="tgt")

    keys_base  = ["rawEnergy" if x=="sc_rawEnergy" else x for x in FEATURES_RUN4_HLT.keys()]

    dfs = get_dataframes(train)
    keys = ["eg_gen_energy"] + keys_base
    plot_corr_matrix(dfs[0][keys],os.path.join(outDir_,"EE"),True,name="gen_energy")
    plot_corr_matrix(dfs[1][keys],os.path.join(outDir_,"EB"),True,name="gen_energy")

    keys = ["tgt"] + keys_base
    plot_corr_matrix(dfs[0][keys],os.path.join(outDir_,"EE"),True,name="tgt")
    plot_corr_matrix(dfs[1][keys],os.path.join(outDir_,"EB"),True,name="tgt")

    # Testing

    outDir_ = os.path.join(outDir,"Testing")
    make_feature_plots(test,FEATURES_RUN4_HLT,min_max_ee,min_max_eb,outDir_)
    make_feature_plots(test,FEATURES_RUN4_HLT,min_max_ee,min_max_eb,outDir_,truth="tgt")

    dfs = get_dataframes(test)
    keys = ["eg_gen_energy"] + keys_base
    plot_corr_matrix(dfs[0][keys],os.path.join(outDir_,"EE"),True,name="gen_energy")
    plot_corr_matrix(dfs[1][keys],os.path.join(outDir_,"EB"),True,name="gen_energy")

    keys = ["tgt"] + keys_base
    plot_corr_matrix(dfs[0][keys],os.path.join(outDir_,"EE"),True,name="tgt")
    plot_corr_matrix(dfs[1][keys],os.path.join(outDir_,"EB"),True,name="tgt")


SMALL_SIZE = 28
MEDIUM_SIZE = 36
BIGGER_SIZE = 38

if __name__ == "__main__":

    hep.style.use("CMS")

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams['xtick.major.pad'] = "12"         # fontsize of the figure title
    plt.rcParams['ytick.major.pad'] = "12"          # fontsize of the figure title

    analyse_features()
    


