import matplotlib.pyplot as plt
import numpy as np
import hist
import mplhep as hep
import os
import sys
import pdb

#sys.path.append("/eos/home-m/mmatthew/SWAN_projects/BDT/utils")
#sys.path.append("/eos/home-m/mmatthew/SWAN_projects/BDT/Validation")

from utils.utils import *
from Validation.fit import *

plt.style.use(hep.style.CMS) 
COLORS = ["black","red","blue","magenta","green","orange","turquoise"]

def plot_parameter(x,ys,xerr,yerr,outDir,name,xlabel,ylabel,y_scale=None,plot_error=False):
    create_dirs(outDir)
    fig = plt.figure(figsize = (12,8))
    for i,key in enumerate(ys.keys()):
        if plot_error:
            plt.errorbar(x,ys[key], xerr=xerr,yerr=yerr[key], linestyle="",marker=".",label=key,color=COLORS[i],capsize=5)
        else:
            plt.errorbar(x,ys[key], xerr=xerr,linestyle="",marker=".",label=key,color=COLORS[i])

    if "mu" in name:
        plt.axhline(1,linestyle="--",color="grey")
    if y_scale:
        plt.ylim(y_scale)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig("%s/%s.png"%(outDir,name))
    plt.savefig("%s/%s.pdf"%(outDir,name))


def get_parameters_per_histtype(fit, histkeys, param):
    f = {}
    for val in histkeys.values():
        f[val] = [] 

    for key in fit.keys():
        for histtype,name in histkeys.items():
            f[name].append(fit[key][histtype]["values"][param])
    return f

def get_uncertainties_per_histtype(fit,histkeys, param):
    err = {}
    for val in histkeys.values():
        err[val] = [] 

    for key in fit.keys():
        for histtype,name in histkeys.items():
            err[name].append(fit[key][histtype]["uncertainties"][param])
    return err


def get_mu_sigma(fit,histkeys,fittype):
    if fittype=="DSCB":
        mu = get_parameters_per_histtype(fit,histkeys,0)
        sigma = get_parameters_per_histtype(fit,histkeys,1)
    elif fittype=="Cruijff":
        mu = get_parameters_per_histtype(fit,histkeys,1)
        sigmaR = get_parameters_per_histtype(fit,histkeys,2)
        sigmaL = get_parameters_per_histtype(fit,histkeys,3)
        sigma = {key:(np.array(sigmaR[key])+np.array(sigmaL[key]))/2 for key in sigmaR.keys()}
    return mu, sigma


def get_err_mu_sigma(fit,histkeys,fittype):
    if fittype=="DSCB":
        mu_ = get_uncertainties_per_histtype(fit,histkeys,(0,0))
        mu = {key: np.sqrt(mu_[key]) for key in mu_.keys()}
        sigma_ = get_uncertainties_per_histtype(fit,histkeys,(1,1))
        sigma = {key: np.sqrt(sigma_[key]) for key in sigma_.keys()}
        cov_ = get_uncertainties_per_histtype(fit,histkeys,(0,1))
        cov = cov_
        #cov =  {key: np.sqrt(cov_[key]) for key in cov_.keys()}

    elif fittype=="Cruijff":
        mu_ = get_uncertainties_per_histtype(fit,histkeys,(1,1))
        mu = {key: np.sqrt(mu_[key]) for key in mu_.keys()}
        sigmaR = get_uncertainties_per_histtype(fit,histkeys,(2,2))
        sigmaL = get_uncertainties_per_histtype(fit,histkeys,(3,3))
        cov_sigma = get_uncertainties_per_histtype(fit,histkeys,(2,3))
        print(sigmaR,sigmaL,cov_sigma)
        sigma = {key:np.sqrt((np.array(sigmaR[key])+np.array(sigmaL[key])+2*np.array(cov_sigma[key]))/4) for key in sigmaR.keys()}
        covR_ = get_uncertainties_per_histtype(fit,histkeys,(1,2))
        covL_ = get_uncertainties_per_histtype(fit,histkeys,(1,3))
        cov =  {key: (np.array(covR_[key])+np.array(covL_[key]))/2 for key in covR_.keys()}
    return mu, sigma, cov

def make_parameter_plots(fit,histkeys,cuts,outDir,name,xlabel,ylabel,fittype="DSCB",set_yscale=None):
    x_center = (cuts[0,:]+cuts[1,:])*0.5
    x_width = (-cuts[0,:]+cuts[1,:])*0.5

    mu, sigma = get_mu_sigma(fit,histkeys,fittype)
    err_mu, err_sigma, cov = get_err_mu_sigma(fit,histkeys,fittype)
    res = {}
    res_err = {}
    for key in mu.keys():
        s = np.array(np.abs(sigma[key]))
        s[s<0]=np.nan
        sigma[key]=s
        res[key] = np.array(sigma[key])/np.array(mu[key])
        res_err[key] = np.array(sigma[key])/np.array(mu[key])*np.sqrt((np.array(err_mu[key])/np.array(mu[key]))**2+(np.array(err_sigma[key])/np.array(sigma[key]))**2-2*np.array(cov[key])/(np.array(sigma[key])*np.array(mu[key])))
    print(res_err)
    if set_yscale:
        plot_parameter(x_center, mu, x_width,err_mu,outDir, "%s_mu_%s"%(fittype,name), xlabel, r"$\mu$ (%s)"%(fittype),set_yscale[0],plot_error=True)
        plot_parameter(x_center, sigma, x_width,err_sigma,outDir, "%s_sigma_%s"%(fittype,name), xlabel, r"$\sigma$ (%s)"%(fittype),set_yscale[1],plot_error=True)
        plot_parameter(x_center, res, x_width,res_err,outDir, "%s_res_%s"%(fittype,name), xlabel, r"$\sigma / \mu $ (%s)"%(fittype),set_yscale[2],plot_error=True)
    else:
        plot_parameter(x_center, mu, x_width,err_mu,outDir, "%s_mu_%s"%(fittype,name), xlabel, r"$\mu$ (%s)"%(fittype),plot_error=True)
        plot_parameter(x_center, sigma, x_width,err_sigma,outDir, "%s_sigma_%s"%(fittype,name), xlabel, r"$\sigma$ (%s)"%(fittype),plot_error=True)
        plot_parameter(x_center, res, x_width,res_err,outDir, "%s_res_%s"%(fittype,name), xlabel, r"$\sigma / \mu $ (%s)"%(fittype),plot_error=True)  
    
def plot_fitted_histograms(hists,popts,labels,colors,outDir,name,xlabel=r"$E_{reco}/E_{gen}$",fit=True,normalize=True,log=True,fittype="DSCB"): 
    # TODO: Change this hack for the labels
    labels = [label.replace("h_","") for label in labels]
    labels = [label.replace("_", " ") for label in labels]
    labels = [label.replace("eg ", "") for label in labels]
    labels = [label+" IC" for label in labels]
    labels = [label.replace("ratio","") for label in labels]

    create_dirs(outDir,True)
    if normalize: hists = [hist/hist.sum() for hist in hists]
    fig = plt.figure(figsize = (8,8))
    if fit:
        for color,hist,popt in zip(colors,hists,popts):
            x = hist.axes.centers[0]
            y = hist.values()
            y_fit = performfit(x,popt,fittype)
            y_fit = y_fit*y.sum()
            plt.plot(x,y_fit,color)
        if fittype=="DSCB":
            labels = [r"%s: $\mu$ = %s, $\sigma$ = %s"%(label,round(popt[0],3),round(popt[1],3)) for label,popt in zip(labels,popts)]
        elif fittype=="Cruijff":
            labels = [r"%s: $\mu$ = %s, $\sigma$ = %s"%(label,round(popt[1],3),round((popt[2]+popt[3])/2,3)) for label,popt in zip(labels,popts)]
    hep.histplot(hists,label=labels,color=colors,yerr=False)
    plt.xlabel(xlabel)
    plt.ylabel("a.u.")
    if log: 
        plt.yscale("log")
        plt.ylim([0.001,1])
    else:
        plt.ylim([0,0.075])
    plt.grid()
    plt.legend(fontsize="14")
    plt.savefig("%s/%s.png"%(outDir,name))
    plt.savefig("%s/%s.pdf"%(outDir,name))

def make_fitted_histograms(hists,fits,histtype,outDir,fit=True, normalize=True,log=True,fittype="DSCB"):
    for key in hists.keys():
        hs = [hists[key][histtype] for histtype in histtype.keys()]
        fs = [fits[key][histtype]["values"] for histtype in histtype.keys()]
        errs = [fits[key][histtype]["uncertainties"] for histtype in histtype.keys()]
        labels = [key for key in histtype.keys()]

        plot_fitted_histograms(hs,fs,labels,COLORS[:len(labels)],outDir,key,xlabel=r"$E_{reco}/E_{gen}$",fit=fit,normalize=normalize,log=log,fittype=fittype)    

def make_fitted_histograms_combined(hists,fits,histtype,outDir,name,param,fittype="DSCB"):
    labels = ["%s < %s < %s" %(label.split("_")[0],param, label.split("_")[1]) for label in labels]
    hists = [hists[r][histtype] for r in hists]
    fit_values = [fits[r][histtype]["values"] for r in fits]
    fit_uncertainties = [fits[r][histtype]["uncertainties"] for r in fits]

    plot_fitted_histograms(hists,fit_values,labels,COLORS[:len(hists)],outDir,name,fittype=fittype)
    

"""
def plot_fit_parameter(fits,label,cuts):
    gen_energy_mean = (cuts[0:-1]+cuts[1:])*0.5
    gen_energy_sigma = (cuts[1:]-cuts[0:-1])*0.5
    plot_parameter(gen_energy_mean, d_mean, gen_energy_sigma,outDir, "m")
"""