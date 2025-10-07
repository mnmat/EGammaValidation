import hist

def create_histograms_legacy():
    min_frac = 0.7
    max_frac = 1.2
    min_energy = 0 
    max_energy = 600
    bins = 50
    
    hists = dict(
        h_regressed_ratio_ideal = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="IdealIC, regInvTar*regMean", label="regInvTar")),
        #h_eg_energy_ratio_ideal = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="IdealIC, eg_energy/eg_gen_energy", label="regInvTar")),
        h_SC_rawEnergy_ratio_ideal = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="IdealIC, eg_rawEnergy/eg_gen_energy", label="regInvTar")),
        h_old_regressed_ratio_ideal = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="IdealIC, eg_regressedEnergy_old/eg_gen_energy", label="regInvTar")),
        h_regressed_ratio_real = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="RealIC, regInvTar*regMean", label="regInvTar")),
        #h_eg_energy_ratio_real = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="RealIC, eg_energy/eg_gen_energy", label="regInvTar")),
        h_SC_rawEnergy_ratio_real = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="RealIC, eg_rawEnergy/eg_gen_energy", label="regInvTar")),
        h_old_regressed_ratio_real = hist.Hist(hist.axis.Regular(bins, min_frac, max_frac, name="RealIC, eg_regressedEnergy_old/eg_gen_energy", label="regInvTar")),
    )
    return hists

def fill_histograms_legacy(hists, df_ideal_cut, df_real_cut):
    hists["h_regressed_ratio_ideal"].fill(df_ideal_cut["regInvTar*regMean"])
    #hists["h_eg_energy_ratio_ideal"].fill(df_ideal_cut["eg_energy/eg_gen_energy"])
    hists["h_SC_rawEnergy_ratio_ideal"].fill(df_ideal_cut["eg_rawEnergy/eg_gen_energy"])
    hists["h_old_regressed_ratio_ideal"].fill(df_ideal_cut["eg_regressedEnergyOld/eg_gen_energy"])

    hists["h_regressed_ratio_real"].fill(df_real_cut["regInvTar*regMean"])
    #hists["h_eg_energy_ratio_real"].fill(df_real_cut["eg_energy/eg_gen_energy"])
    hists["h_SC_rawEnergy_ratio_real"].fill(df_real_cut["eg_rawEnergy/eg_gen_energy"])
    hists["h_old_regressed_ratio_real"].fill(df_real_cut["eg_regressedEnergyOld/eg_gen_energy"])
    return hists

def create_subspace_histograms_legacy(df_ideal, df_real,feature, cuts):
    # Cuts should be imported as an array of pairs for the left and right boundaries
    hists_subspace = {}  
    print(cuts.shape)
    for i in range(cuts.shape[1]):
        # Write Histograms
        print(cuts[0][i], cuts[1][i])
        tmp_ideal = df_ideal[(df_ideal[feature]>cuts[0][i]) & (df_ideal[feature]<cuts[1][i])]
        tmp_real = df_real[(df_real[feature]>cuts[0][i]) & (df_real[feature]<cuts[1][i])]
        hists = create_histograms_legacy()
        hists = fill_histograms_legacy(hists,tmp_ideal,tmp_real)
        hists_subspace["%s_%s"%(cuts[0][i], cuts[1][i])] = hists
    return hists_subspace
