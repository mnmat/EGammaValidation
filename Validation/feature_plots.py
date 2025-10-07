import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches
import hist
import mplhep as hep
from .utils import *
import pdb
import copy



def create_dirs(path,add_php=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if add_php:
        cmd = "cp /eos/home-m/mmatthew/www/index/index.php %s"%path
        os.system(cmd)


def plot_event_sizes(training_samples,test_samples,outdir,names,keys = ["egRegDataEcalHLTV1","egRegDataHGCALHLTV1"],labels = ["Ecal","HGCAL"]):
    for train,test,name in zip(training_samples,test_samples,names):
        y_train = []
        y_test = []
    
        f_train = uproot.open(train)
        for key in keys:
            y_train.append(f_train[key].num_entries)

     
        f_test = uproot.open(test)
        for key in keys:
            y_test.append(f_test[key].num_entries)

        x_train = np.linspace(0,len(y_train)-1,len(y_train))
        x_test = np.linspace(0,len(y_test)-1,len(y_test))
        width=0.25
        offset = width/2

        fig, ax = plt.subplots()
        r1=ax.bar(x_train-offset,y_train, width=width,label="Training Sample: %i"%np.array(y_train).sum())
        ax.bar_label(r1,padding=-15,color="white")
        r2 = ax.bar(x_test+offset,y_test, width=width,label="Test Sample: %i"%np.array(y_test).sum())
        ax.bar_label(r2,padding=+3)

        ax.set_xticks(ticks=[0,1],labels=labels)
        plt.legend()

        plt.savefig(os.path.join(outdir,"events_%s.pdf"%name))
        plt.savefig(os.path.join(outdir,"events_%s.png"%name))
        
def plot_energies(df,outdir,name,bins=50,rng=(0,800)):
    fig = plt.figure()
    df["rawEnergy"].hist(bins=bins,range=rng,histtype="step",label="rawEnergy")
    df["regressedEnergy"].hist(bins=bins,range=rng,histtype="step",label="regressedEnergy")
    df["eg_gen_energy"].hist(bins=bins,range=rng,histtype="step",label="genEnergy")
    df["old_regressedEnergy"].hist(bins=bins,range=rng,histtype="step",label="old_regressedEnergy")
    
    plt.legend()
    plt.savefig(os.path.join(outdir,"energies_%s.pdf"%name))
    plt.savefig(os.path.join(outdir,"energies_%s.png"%name))
    plt.close()
    
def scatter(df,outdir,name,ls=["raw","old"]):
    gen = df["eg_gen_energy"]
    raw = df["rawEnergy"]
    reg = df["regressedEnergy"]
    old = df["old_regressedEnergy"]
    fig = plt.figure()
    for l in ls:
        if l == "old":
            plt.scatter(gen,old,label="old")
        if l == "raw":
            plt.scatter(gen,raw,label="raw")
        if l == "reg":
            plt.scatter(gen,reg,label="reg")
    
    plt.xlabel(r"$E_{Gen} [GeV]$")
    plt.ylabel(r"$E_{Reco} [GeV]$")
    plt.legend()
    plt.savefig(os.path.join(outdir,"reco_vs_gen_%s.pdf"%name))
    plt.savefig(os.path.join(outdir,"reco_vs_gen_%s.png"%name))
    plt.close()
    
    
def plot_fractions(df,outdir,name,ls=["raw","old"]):
    frac_raw = df["frac_rawEnergy_genEnergy"]
    frac_reg = df["frac_regEnergy_genEnergy"]
    frac_old_reg = df["frac_old_regEnergy_genEnergy"]
    gen = df["eg_gen_energy"]

    fig = plt.figure()
    for l in ls:
        if l == "raw":
            plt.scatter(gen,frac_raw,label="rawEnergy")
        if l == "old":
            plt.scatter(gen,frac_old_reg,label="old_regEnergy")
        if l == "reg":
            plt.scatter(gen,frac_old_reg,label="regEnergy")
    #plt.ylim([0,3])
    plt.xlim([0,1000])
    plt.legend()
    plt.savefig(os.path.join(outdir,"frac_vs_gen_%s.pdf"%name))
    plt.savefig(os.path.join(outdir,"frac_vs_gen_%s.png"%name))
    plt.close
    
    
def find_range(dfs,key):
    minimum = 9999999
    maximum = -9999999
    
    for df in dfs:
        if minimum > df[key].min():
            minimum = df[key].min()
        if maximum < df[key].max():
            maximum = df[key].max()
    return (minimum,maximum)

def plot_feature_hists(dfs,labels,outdir,name="",density=True):
    bins = 50
    keys = dfs[0].keys()

    for key in keys:
        rng = find_range(dfs,key)
        if "frac" in key:
            rng = [0.75,1.5]
        fig = plt.figure()
        for df,label in zip(dfs,labels):
            df[key].hist(bins=bins,range=rng,density=density,label=label,histtype="step")
        plt.legend()
        plt.xlabel(key)
        plt.ylabel("a.u.")
        plt.title(key)
        plt.savefig(os.path.join(outdir,"hist_%s_%s.pdf"%(key,name)))
        plt.savefig(os.path.join(outdir,"hist_%s_%s.png"%(key,name)))
        plt.close()
        
COLORS = ["red","blue"]

def plot_feature_correlation(dfs,labels,outdir,name="",target="eg_gen_energy",keys=None,save=True):

    create_dirs(outdir,True)

    if keys == None:
        keys = dfs[0].keys()
        
    for key in keys:
        fig = plt.figure()
        rng = find_range(dfs,key)

        fig, ax = plt.subplots()
        patches = []
        for i,df in enumerate(dfs):
            sns.histplot(data=df,y=target,x=key,ax=ax,color=COLORS[i],alpha=0.5,bins=(50,50),binrange=((rng),(0,1000)),label=labels[i])
        # Manually create legend
            patches.append(mpatches.Patch(color=COLORS[i], alpha=0.5, label=labels[i]))
        plt.legend(handles=patches)
        
        if save == True:
            plt.savefig(os.path.join(outdir,"corr_%s_%s.pdf"%(key,name)))
            plt.savefig(os.path.join(outdir,"corr_%s_%s.png"%(key,name)))

def plot_corr_matrix(df,outdir,save=True,name=""):
    create_dirs(outdir,True)
    
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    if save == True:
        plt.savefig(os.path.join(outdir,"correlation_matrix_%s.pdf"%name))
        plt.savefig(os.path.join(outdir,"correlation_matrix_%s.png"%name))
    plt.close()

def get_variables(f):
    
    # EE
    tree = "egRegDataHGCALHLTV1"
    ee_nrHitsThreshold = f[tree]["nrHitsThreshold"]
    ee_eta = f[tree]["eta"]
    ee_sc_rawEnergy = f[tree]["rawEnergy"]
    ee_phiWidth = f[tree]["phiWidth"]
    ee_rvar = f[tree]["rvar"]
    ee_numberOfSubClusters = f[tree]["numberOfSubClusters"]
    ee_clusterMaxDR = f[tree]["clusterMaxDR"]
    
    # EB
    tree = "egRegDataEcalHLTV1"
    eb_nrHitsThreshold = f[tree]["nrHitsThreshold"]
    eb_eta = f[tree]["eta"]
    eb_sc_rawEnergy = f[tree]["rawEnergy"]
    eb_phiWidth = f[tree]["phiWidth"]
    eb_rvar = f[tree]["rvar"]
    eb_numberOfSubClusters = f[tree]["numberOfSubClusters"]
    eb_clusterMaxDR = f[tree]["clusterMaxDR"]
    
    features_ee = np.array(
        [ee_nrHitsThreshold,
         ee_eta,
         ee_sc_rawEnergy,
         ee_phiWidth,
         ee_rvar,
         ee_numberOfSubClusters,
         ee_clusterMaxDR
        ]
    ).T
    
    features_eb = np.array(
        [eb_nrHitsThreshold,
         eb_eta,
         eb_sc_rawEnergy,
         eb_phiWidth,
         eb_rvar,
         eb_numberOfSubClusters,
         eb_clusterMaxDR
        ]
    ).T
    
    
    return features_ee, features_eb
    
    
def get_gen_energy(f):
    
    target_arr_ee = f["egRegDataHGCALHLTV1"]["eg_gen_energy"].array(library="np")
    target_arr_eb = f["egRegDataEcalHLTV1"]["eg_gen_energy"].array(library="np")
    
    return np.array(target_arr_ee.tolist()), np.array(target_arr_eb.tolist())
    
    

def get_min_max(fnames,feature_map_ee, feature_map_eb,mode,load=True,save=True):
    if load:
        try:
            with open('min_max_ee_%s.p'%mode, 'rb') as fp:
                min_max_ee = pickle.load(fp)
            with open('min_max_eb_%s.p'%mode, 'rb') as fp:
                min_max_eb = pickle.load(fp)
            
            return min_max_ee, min_max_eb
        except:
            print("Load from storage file failed. Continue to read from root files")
        
    min_max_ee = create_min_max(feature_map_ee)
    min_max_eb = create_min_max(feature_map_eb)

    for fname in fnames:

        f = uproot.open(fname)

        features_ee, features_eb = get_variables(f)
        gen_ee, gen_eb = get_gen_energy(f)

        arr1 = features_ee[:,feature_map_ee["clusterMaxDR"]]
        arr1[arr1>990]=np.nan

        arr2 = features_eb[:,feature_map_eb["clusterMaxDR"]]
        arr2[arr2>990]=np.nan

        min_max_ee = fill_min_max(feature_map_ee,gen_ee,features_ee,min_max_ee)
        min_max_eb = fill_min_max(feature_map_eb,gen_eb,features_eb,min_max_eb)

    if save:
        with open('min_max_ee_%s.p'%mode, 'wb') as fp:
            pickle.dump(min_max_ee, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open('min_max_eb_%s.p'%mode, 'wb') as fp:
            pickle.dump(min_max_eb, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return min_max_ee, min_max_eb


def fill_min_max(feature_map,gen,features,min_max):
    names = feature_map.keys()
    for name in names:
        min_ = np.nanmin(features[:,feature_map[name]])
        max_ = np.nanmax(features[:,feature_map[name]])
        if min_ < min_max[name]["min"]:
            min_max[name]["min"] = min_
        if max_ > min_max[name]["max"]:
            min_max[name]["max"] = max_
    
    min_ = gen.min()
    max_ = gen.max()
    if min_ < min_max["cp_energy"]["min"]:
        min_max["cp_energy"]["min"] = min_ 
    if max_ > min_max["cp_energy"]["max"]:
        min_max["cp_energy"]["max"] = max_
        
    
    tgt = features[:,feature_map["sc_rawEnergy"]]/gen
    
    min_ = np.nanmin(tgt[np.isfinite(tgt)])
    max_ = np.nanmax(tgt[np.isfinite(tgt)])
    if min_ < min_max["tgt"]["min"]:
        min_max["tgt"]["min"] = min_ 
    if max_ > min_max["tgt"]["max"]:
        min_max["tgt"]["max"] = max_
    
    return min_max

def create_min_max(feature_map):
    min_max = {}
    names = feature_map.keys()
    for name in names:
        min_max[name] = {"min":999,
                        "max":-999}
    min_max["cp_energy"] = {"min":999,
                        "max":-999}
    min_max["tgt"] = {"min":999,
                     "max":-999}
    return min_max


def create_histograms(features,min_max,truth="cp_energy"):

    bins=50
    hist2D = {}
    names = features
    for name in names:
        if "nergy" in name and not "/" in name and truth == "cp_energy":
            if min_max[name]["min"] < min_max[truth]["min"]:
                ax_min = min_max[name]["min"]*.99
            else:
                ax_min =  min_max["cp_energy"]["min"]*.99
            if min_max[name]["max"] > min_max[truth]["max"]:
                ax_max = min_max[name]["max"]*1.01
            else:
                ax_max =  min_max["cp_energy"]["max"]*1.01
            hist2D[name] = hist.Hist(
                hist.axis.Regular(bins,ax_min,ax_max,name=name,overflow=True),
                hist.axis.Regular(bins,ax_min,ax_max,name=truth,overflow=True)
            )

        elif "mberOfSubClusters" in name:
            min_truth = min_max[truth]["min"]
            if min_truth >= 0:
                min_truth = 0 

            hist2D[name] = hist.Hist(
                hist.axis.Integer(int(min_max[name]["min"]),int(min_max[name]["max"])+2,name=name,overflow=True),
                hist.axis.Regular(bins,min_max[truth]["min"]*.99,min_max[truth]["max"]*1.01,name=truth,overflow=True)
            )

        else:
#             hist2D[name] = hist.Hist(
#                 hist.axis.Regular(bins,min_max[name]["min"]*.99,min_max[name]["max"]*1.01,name=name,overflow=True),
#                 hist.axis.Regular(bins,min_max[truth]["min"]*.99,min_max[truth]["max"]*1.01,name=truth,overflow=True)
#             )

            min_feature = min_max[name]["min"]
            if min_feature >= 0:
                min_feature = 0
        
            min_truth = min_max[truth]["min"]
            if min_truth >= 0:
                min_truth = 0         
                
            hist2D[name] = hist.Hist(
                hist.axis.Regular(bins,min_feature,min_max[name]["max"]*1.01,name=name,overflow=True),
                hist.axis.Regular(bins,min_truth,min_max[truth]["max"]*1.01,name=truth,overflow=True)
            )
    return hist2D

def fill_histograms(features,gen,feature_map,hist2D,truth="cp_energy"):
    target_arr = features[:,feature_map["sc_rawEnergy"]]/gen
    mask = np.where(~np.isinf(target_arr))
    
    features = features[mask]
    gen = gen[mask]
    tgt = target_arr[mask]
    
    names = feature_map.keys()
    for name in names:
        if truth == "cp_energy":
            hist2D[name].fill(features[:,feature_map[name]],gen) 
        else:
            hist2D[name].fill(features[:,feature_map[name]],tgt) 
    return hist2D

def write_histograms(hist2D,feature_map,root,truth="cp_energy",save=True):


    if truth == "cp_energy":
        outdir = os.path.join(root,"Features")
    else:
        outdir = os.path.join(root,"FeaturesTgt")
    create_dirs(outdir,True)
    
    names = feature_map.keys()
    for name in names:
        fig,ax = plt.subplots()
        hist2D[name].project(name).plot(ax=ax,density=True)
        plt.ylabel("a.u.")
        plt.tight_layout()
        if "/" in name:
            name = name.split("/")[0]+"_o_"+name.split("/")[1]
        
        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))
        plt.close()
            
            
    # TODO: Find better solution to plot the tgt variables
    if truth == "cp_energy":
        fig,ax = plt.subplots()
        hist2D[name].project("cp_energy").plot(ax=ax,density=True)
        name = "cp_energy"


        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))
        plt.close()
    elif truth == "tgt":
        fig,ax = plt.subplots()
        hist2D[name].project("tgt").plot(ax=ax,density=True)
        name = "tgt"

        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))
        plt.close()
        
            

    if truth == "cp_energy":
        outdir = os.path.join(root,"Correlation")
    else:
        outdir = os.path.join(root,"CorrelationTgt2")
    create_dirs(outdir,True)
    save = True

    for name in names:
        fig,ax = plt.subplots()
        hist2D[name].plot(cmap="plasma",cmin=1,ax=ax)
        if "/" in name:
            name = name.split("/")[0]+"_o_"+name.split("/")[1]

        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))
        plt.close()


def write_split_histograms(hs,key,labels,colors,outdir,save=True):


    hproj = hs[0][key].project(key)
    ax = hproj.axes[0]
    edges = ax.edges      # regular edges
    width = edges[1]-edges[0]

    # Create two subplots sharing the y-axis
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10,10), gridspec_kw={'width_ratios':[3,1]}
    )

    for i, h in enumerate(hs):
        # Project histogram and plot in both axes
        hproj = h[key].project(key)
        hproj.plot(ax=ax1, label=labels[i], color=colors[i], density=True, flow="sum")
        hproj.plot(ax=ax2, label=labels[i], color=colors[i], density=True, flow="sum")

    # Set the y-axis limits
    ax1.set_ylim(0,int(hproj.density().max()+1))

    # Remove xlabel
    ax1.set_xlabel("")

    # Set the x-axis limits
    epsilon = width*1.13 # 1.1 to 1.11
    ax1.set_xlim(0, edges[-3]+epsilon/2)   # physical region
    ax2.set_xlim(edges[-1]-epsilon, edges[-1]+width)  # focus on the 999 bin

    # Hide spines between the two plots
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax1.yaxis.tick_left()
    ax1.tick_params(labelright=False)
    ax2.yaxis.tick_right()

    # Add diagonal lines to indicate the break
    d = .015  # size of break mark
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d,1+d), (-d,+d), **kwargs)
    ax1.plot((1-d,1+d), (1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the right axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,d), **kwargs)
    ax2.set_xticks([edges[-1]-width/2],[999])


    # Add legend only once
    ax1.legend()
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(outdir,"h_split_%s.png"%key))
        fig.savefig(os.path.join(outdir,"h_split_%s.pdf"%key))

def write_split_histograms_2d(h,key,outdir,truth="cp_energy",save=True):

    if truth == "cp_energy":
        outdir = os.path.join(outdir,"Correlation")
    else:
        outdir = os.path.join(outdir,"CorrelationTgt2")
    create_dirs(outdir,True)

    hproj = h[key].project(key)
    ax = hproj.axes[0]
    edges = ax.edges      # regular edges
    width = edges[1]-edges[0]

    # Create two subplots sharing the y-axis
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10,10), gridspec_kw={'width_ratios':[3,1]}
    )

    # Project histogram and plot in both axes
    h1 = copy.deepcopy(h[key])
    h2 = copy.deepcopy(h[key])
    h1.plot(ax=ax1,cmap="plasma",cmin=1)
    h2.plot(ax=ax2,cmap="plasma",flow="sum")

    # Set the y-axis limits
    # ax1.set_ylim(0,int(hproj.density().max()+1))

    # Remove xlabel
    ax1.set_xlabel("")
    ax2.set_ylabel("")

    # Set the x-axis limits
    epsilon = width *1.0445
    ax1.set_xlim(0, edges[-1]-epsilon)   # physical region
    ax2.set_xlim(edges[-1]-epsilon/2, edges[-1])  # focus on the 999 bin

    # Hide spines between the two plots
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)

    ax1.yaxis.tick_left()
    ax1.tick_params(labelright=False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_visible(False)

    # # Add diagonal lines to indicate the break
    # d = .015  # size of break mark
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((1-d,1+d), (-d,+d), **kwargs)
    # ax1.plot((1-d,1+d), (1-d,1+d), **kwargs)

    # kwargs.update(transform=ax2.transAxes)  # switch to the right axes
    # ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    # ax2.plot((-d,+d), (-d,d), **kwargs)
    ax2.set_xticks([edges[-1]-width/4],[999])
    plt.tight_layout()
    
    if save:
        fig.savefig(os.path.join(outdir,"h_split_%s.png"%key))
        fig.savefig(os.path.join(outdir,"h_split_%s.pdf"%key))
