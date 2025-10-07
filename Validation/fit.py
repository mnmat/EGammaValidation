from scipy.stats import crystalball
from scipy.optimize import curve_fit
import numpy as np
import pdb
from scipy.special import erf


def crystalball_fit_scipy(x,beta,m,loc,scale):
    return crystalball.pdf(x,beta=beta,m=m,loc=loc,scale=scale)

def dscb(x, m, s, aL, nL, aR, nR,N):
    n_over_alphaL = nL/aL
    n_over_alphaR = nR/aR
    zeta = (x - m)/s
    
    mask_g = np.where((zeta>=-aL) & (zeta<=aR))
    mask_pL = np.where(zeta<-aL)
    mask_pR = np.where(zeta>aR)

    gaussian = N*np.exp(-zeta[mask_g]**2*0.5)
    powerlawL = N*np.exp(-aL**2*0.5)*(1/n_over_alphaL*(n_over_alphaL - aL - zeta[mask_pL]))**(-nL)
    powerlawR = N*np.exp(-aR**2*0.5)*(1/n_over_alphaR*(n_over_alphaR - aR + zeta[mask_pR]))**(-nR)
    
    result = np.zeros_like(x)
    result[mask_g] = gaussian
    result[mask_pL] = powerlawL 
    result[mask_pR] = powerlawR
    
    return result

def crystal_ball(x, alpha, n, xbar, sigma):
    n_over_alpha = n/abs(alpha)
    exp = np.exp(-0.5*alpha ** 2)
    A = (n_over_alpha)**n*exp
    B =  n_over_alpha - abs(alpha)
    C = n_over_alpha/(n-1)*exp
    D = np.sqrt(0.5*np.pi)*(1 + erf(abs(alpha)/np.sqrt(2)))
    N = 1/(sigma*(C + D))

    mask = (x - xbar)/sigma > -alpha

    gaussian = N*np.exp(-0.5*((x[mask]-xbar)/sigma)**2)
    powerlaw = N*A*(B - (x[~mask]-xbar)/sigma)**-n

    result = np.zeros_like(x)
    result[mask] = gaussian
    result[~mask] = powerlaw
    # or
    # result = np.concatenate((powerlaw, gaussian))

    return result/result.sum()

def getCBParams(hists,key):
    # Get normalized Values
    x = hists[key].axes.centers[0]
    y = hists[key].values()
    y = np.array(y/np.sum(y))

    # Fit CB
    p0 = [0.75,4,1,0.02]
    popt, pcov = curve_fit(crystal_ball, x, y, p0 = p0)
    return [popt,pcov]


def getDSCBParams(hists,key):
    # Get normalized Values
    x = hists[key].axes.centers[0]
    y = hists[key].values()
    y = np.array(y/np.sum(y))
    
    # Fit CB
    p0 = [1,0.02,0.95,20,20,10,0.075]
    try:
        popt, pcov = curve_fit(dscb, x, y, p0 = p0,
                   bounds=np.transpose([(-np.inf, np.inf), (0., np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf),(0., np.inf)]))
    except RuntimeError:
        popt = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        pcov = np.ones([7,7])*np.inf
    except ValueError:
        popt = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        pcov = np.ones([7,7])*np.inf

    return [popt, pcov]

def fit_subspace_histograms(hists_subspace,method="DSCB"):
    fit_subspace = {}
    for subspace in hists_subspace.keys():
        hists = hists_subspace[subspace]
        fit_subspace[subspace] = {}
        for key in hists.keys():
            fit_subspace[subspace][key] = {}
            if method == "DSCB":
                popt, pcov = getDSCBParams(hists,key)
                fit_subspace[subspace][key]["values"] = popt
                fit_subspace[subspace][key]["uncertainties"] = pcov
            elif method == "Cruijff":
                try:
                    popt, pcov = getCruijffParams(hists,key)
                    fit_subspace[subspace][key]["values"] = popt
                    fit_subspace[subspace][key]["uncertainties"] = pcov
                except ValueError:
                    fit_subspace[subspace][key]["values"] = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
                    fit_subspace[subspace][key]["uncertainties"] = np.ones([6,6])*np.inf
    return fit_subspace

def performfit(x,popt,fittype="DSCB"):
    if fittype == "DSCB":
        return dscb(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
    elif fittype == "Cruijff":
        return cruijff(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])

def cruijff(x, A, m, sigmaL,sigmaR, alphaL, alphaR):
    dx = (x-m)
    SL = np.full(x.shape, sigmaL)
    SR = np.full(x.shape, sigmaR)
    AL = np.full(x.shape, alphaL)
    AR = np.full(x.shape, alphaR)
    sigma = np.where(dx<0, SL,SR)
    alpha = np.where(dx<0, AL,AR)
    f = 2*sigma*sigma + alpha*dx*dx

    return A* np.exp(-dx*dx/f)

def getCruijffParams(hists,key):
    if hists[key].values().sum() == 0:
        return np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
    mean = np.average(hists[key].axes[0].centers, weights=hists[key].values())
    stdDev = np.average((hists[key].axes[0].centers - mean)**2, weights=hists[key].values())
    #y = hists[key].values()/hists[key].values().sum()
    y = hists[key].values()
    #pdb.set_trace()
    param_optimised,param_covariance_matrix = curve_fit(cruijff, hists[key].axes[0].centers, y, p0=[np.max(y), mean, stdDev, stdDev,  0.1, 0.05], sigma=np.maximum(np.sqrt(y), 1.8), absolute_sigma=True, maxfev=500000, 
        #bounds=np.transpose([(0, 1), (0.1, 2), (0,2), (0, 2), (-np.inf, np.inf), (-np.inf, np.inf)])
        bounds=np.transpose([(np.max(y)*.1, np.inf), (0.1, 2), (0,np.inf), (0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
        )
    param_optimised[0] = param_optimised[0]/y.sum()
    return param_optimised,param_covariance_matrix
