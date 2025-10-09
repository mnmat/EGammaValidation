import pandas as pd
import numpy as np
import os
import sys
import pdb 
import pickle
import hist
import matplotlib.pyplot as plt

from utils.utils import *
from Validation.feature_plots import *
from utils.feature_mapping import *

import mplhep as hep

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

def get_histograms(FEATURES_RUN4_HLT,training_samples,min_max_ee,min_max_eb):
    # Plot all features and correlations (gen_energy)
    hist2D_ee_ele = create_histograms(FEATURES_RUN4_HLT,min_max_ee)
    hist2D_eb_ele = create_histograms(FEATURES_RUN4_HLT,min_max_eb)
    for fname in [training_samples[0]]:

        f = uproot.open(fname)
        features_ee, features_eb = get_variables(f)
        gen_ee, gen_eb = get_gen_energy(f)

        hist2D_ee_ele = fill_histograms(features_ee,gen_ee,FEATURES_RUN4_HLT,hist2D_ee_ele)
        hist2D_eb_ele = fill_histograms(features_eb,gen_eb,FEATURES_RUN4_HLT,hist2D_eb_ele)

    return hist2D_ee_ele, hist2D_eb_ele

def get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s5Reg_genMatched"):
    training_path = os.path.join(path,training_dir)
    test_path = os.path.join(path,test_dir)

    training_file = "HLTAnalyzerTree_IDEAL_Flat_train.root"
    test_file = "HLTAnalyzerTree_IDEAL_Flat_test.root"

    return [os.path.join(training_path,training_file)], [os.path.join(test_path,test_file)]

def compare_histograms(hs,key,outDir,labels,colors,save=True):
    fig = plt.figure()
    for i,h in enumerate(hs):
        h[key].project(key).plot(label=labels[i],color=colors[i],density=True,flow="sum")
    plt.legend()
    plt.ylabel("a.u.")
    plt.tight_layout()
    if save:
        create_dirs(outDir,True)
        plt.savefig(os.path.join(outDir,"h_%s.png"%(key)))
        plt.savefig(os.path.join(outDir,"h_%s.pdf"%(key)))
    plt.close(fig)


def plot_feature_scatterplot(dfs,labels,outdir,name="",density=True,save=True):

    bins = 50
    keys = dfs[0].keys()

    for key in keys:
        rng = find_range(dfs,key)
        if "frac" in key:
            rng = [0.75,1.5]
        fig = plt.figure()
        for df,label in zip(dfs,labels):
            sns.scatterplot(data=df, x=key, y="eg_gen_energy",label=label,alpha=.5)
        plt.legend()
        plt.xlabel(key)
        plt.ylabel("a.u.")
        plt.title(key)
        if save == True:
            create_dirs(outdir,True)
            plt.savefig(os.path.join(outdir,"scatter_%s_%s.pdf"%(key,name)))
            plt.savefig(os.path.join(outdir,"scatter_%s_%s.png"%(key,name)))
        plt.close()

def plot_feature_kde(dfs,features,labels,outdir,name="",density=True,save=True):
    bins = 50

    for key in features:
        rng = find_range(dfs,key)
        if "frac" in key:
            rng = [0.75,1.5]
        fig = plt.figure()
        for df,label in zip(dfs,labels):
            print(key)
            sns.kdeplot(data=df, x=key, y="eg_gen_energy",label=label,alpha=.5)
        plt.legend()
        plt.xlabel(key)
        plt.ylabel("a.u.")
        plt.title(key)
        if save == True:
            create_dirs(outdir,True)
            plt.savefig(os.path.join(outdir,"kde_%s_%s.pdf"%(key,name)))
            plt.savefig(os.path.join(outdir,"kde_%s_%s.png"%(key,name)))
        plt.close()



def plot_diff_mesh(dfs,outdir,name="",save=True):
    
    keys = dfs[0].keys()
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="white")
    
    for key in keys:
        min_ = 999
        max_ = -999
        for df in dfs:
            tmp_min = df[key].min()
            tmp_max = df[key].max()
            
            if tmp_min < min_:
                min_ = tmp_min
            if tmp_max > max_:
                max_ = tmp_max
        
        rng = [[0,1000],[min_,max_]]
        H_ele,yedges,xedges = np.histogram2d(dfs[0]["eg_gen_energy"],dfs[0][key],bins=50,range=rng,density=True)
        H_pho,yedges,xedges = np.histogram2d(dfs[1]["eg_gen_energy"],dfs[1][key],bins=50,range=rng,density=True)
        mask_ele = H_ele == 0
        mask_pho = H_pho == 0
        
        H_ele_masked = H_ele.copy()
        H_pho_masked = H_pho.copy()
        
        H_ele_masked[mask_ele] = np.nan
        H_pho_masked[mask_pho] = np.nan

        H = H_ele-H_pho
        mask = H == 0
        H[mask]=np.nan
        
        fig,ax = plt.subplots(3,1)

        mesh = ax[0].pcolormesh(xedges, yedges, H, cmap=cmap)
        fig.colorbar(mesh,ax=ax[0])
        ax[0].set_title(r"$\Delta_{Electron,Photon}$")
        mesh = ax[1].pcolormesh(xedges, yedges, H_ele_masked, cmap=cmap)
        fig.colorbar(mesh,ax=ax[1])
        ax[1].set_title("Electron")
        mesh = ax[2].pcolormesh(xedges, yedges, H_pho_masked, cmap=cmap)
        fig.colorbar(mesh,ax=ax[2])
        ax[2].set_title("Photon")
        plt.xlabel(key)
        plt.ylabel("eg_gen_energy")
        plt.tight_layout()
        
        if save:
            create_dirs(outdir,True)
            plt.savefig(os.path.join(outdir,"mesh_%s_%s.pdf"%(key,name)))
            plt.savefig(os.path.join(outdir,"mesh_%s_%s.png"%(key,name)))
        plt.close()



def compare_features():

    outDir = "/eos/home-m/mmatthew/www/test_workflow/CompareFeatures"

    paths = [
        "/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/",
        "/afs/cern.ch/work/m/mmatthew/private/test_workflow/cms-egamma-hlt-reg/"
    ]

    labels = ["A","B"]
    colors = ["blue","red"]

    train = []
    test = []
    for path in paths:
        train_,test_ = get_training_test_files(path,training_dir="Flat",test_dir="Flat")
        train = train + train_
        test = test + test_
    
    min_max_ee, min_max_eb = get_min_max(train+test,FEATURES_RUN4_HLT,FEATURES_RUN4_HLT,"Phase2",save=False,load=False)

    # Training samples

    outDir_ = os.path.join(outDir,"Training")

    h_ee = []
    h_eb = []
    for t in train:
        h_ee_, h_eb_ = get_histograms(FEATURES_RUN4_HLT,[t],min_max_ee,min_max_eb)
        h_ee.append(h_ee_)
        h_eb.append(h_eb_)

    for key in FEATURES_RUN4_HLT:
        compare_histograms(h_ee,key,os.path.join(outDir_,"EE"),labels,colors,save=True)
        compare_histograms(h_eb,key,os.path.join(outDir_,"EB"),labels,colors,save=True)

    write_split_histograms(h_ee,"clusterMaxDR",labels,colors,os.path.join(outDir_,"EE"))
    write_split_histograms(h_eb,"clusterMaxDR",labels,colors,os.path.join(outDir_,"EB"))


    # More computationally demanding plots, use test sample for now

    outDir_ = os.path.join(outDir,"Testing")

    dfs_ee = []
    dfs_eb = []
    for t in test:
        ee, eb = get_dataframes([t])
        dfs_ee.append(ee)
        dfs_eb.append(eb)

    
    
    keys_base  = ["rawEnergy" if x=="sc_rawEnergy" else x for x in FEATURES_RUN4_HLT.keys()]



    test_ee = [df[:10000] for df in dfs_ee]
    test_eb = [df[:10000] for df in dfs_eb]

    # The following plots are time intensive and don't add much. Feel free to play around if you're interested

    # plot_feature_correlation(dfs_ee,labels,os.path.join(outDir_,"EE/2DHist"),save=True)
    # plot_feature_correlation(dfs_eb,labels,os.path.join(outDir_,"EB/2DHist"),save=True)

    # plot_feature_scatterplot(dfs_ee,labels,os.path.join(outDir_,"EE/2DScatter"),save=True)
    # plot_feature_scatterplot(dfs_eb,labels,os.path.join(outDir_,"EB/2DScatter"),save=True)

    # plot_diff_mesh(dfs_ee,os.path.join(outDir_,"EE/Mesh"),save=True)
    # plot_diff_mesh(dfs_eb,os.path.join(outDir_,"EB/Mesh"),save=True)

    # plot_feature_kde(dfs_ee,keys,labels,os.path.join(outDir_,"EE/KDE"),save=True)
    # plot_feature_kde(dfs_eb,keys,labels,os.path.join(outDir_,"EB/KDE"),save=True)

    # plot_feature_correlation(test_ee,labels,os.path.join(outDir_,"EE/2DHist"),save=True)
    # plot_feature_correlation(test_eb,labels,os.path.join(outDir_,"EB/2DHist"),save=True)

    # plot_feature_scatterplot(test_ee,labels,os.path.join(outDir_,"EE/2DScatter"),save=True)
    # plot_feature_scatterplot(test_eb,labels,os.path.join(outDir_,"EB/2DScatter"),save=True)

    # plot_diff_mesh(test_ee,os.path.join(outDir_,"EE/Mesh"),save=True)
    # plot_diff_mesh(test_eb,os.path.join(outDir_,"EB/Mesh"),save=True)


    # keys_base  = ["rawEnergy" if x=="sc_rawEnergy" else x for x in FEATURES_RUN4_HLT.keys()]
    # plot_feature_kde(test_ee,keys_base,labels,os.path.join(outDir_,"EE/KDE"),save=True)
    # plot_feature_kde(test_eb,keys_base,labels,os.path.join(outDir_,"EB/KDE"),save=True)



SMALL_SIZE = 34
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
        # mpl.rcParams['xtick.major.pad'] = 8
        # mpl.rcParams['ytick.major.pad'] = 8
    compare_features()
