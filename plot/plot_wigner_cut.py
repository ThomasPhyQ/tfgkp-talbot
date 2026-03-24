import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

################################################################################
### Coincidence vs time shift only at each fsr/2 shift for each talbot steps ###
################################################################################

cut_tab = ["m2", "m1", "0", "1", "2"]
step_tab = ["zero", "half", "one", "one_half", "two"]

def plot_cut(wigner,step,cut,sampling_rate,n_total_sample,n_sample,fsr,param):
    width_prop = (7/8)*(1/3)*(1/2) 
    width_A4 = 210
    width_mm = width_prop*width_A4
    width_inch = width_mm/25.4

    golden_ratio = (1+np.sqrt(5))/2
    
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = "Serif"
    plt.rcParams['text.usetex'] = True

    fig,ax = plt.subplots(figsize=(width_inch,width_inch/golden_ratio))

    f_cut = fsr*(cut-2)/2
    f_bin = n_sample//2 + round(f_cut/sampling_rate*n_total_sample)

    n_peaks = 2
    n_display_t = int(n_peaks*1/fsr*sampling_rate)
    t_min = -n_display_t/sampling_rate
    t_max = n_display_t/sampling_rate

    plt.plot(
        np.linspace(t_min,t_max,2*n_display_t)*fsr,
        1/2 - 1/2*np.flip(wigner[step,n_sample//2-n_display_t:n_sample//2+n_display_t:,f_bin],axis=0),
        linewidth = 0.5
    )
    plt.ylim(0,1)

    plt.savefig(
        "data/plot/minimal/wigner_cut_"+cut_tab[cut]+"_"+step_tab[step]+"_"+param+".pdf",
        dpi = 1000,
        bbox_inches="tight"
        )
    plt.close()


def plot_wigner(filename,param):
    with open(filename, 'rb') as f:
        dict_gkp = pickle.load(f)
        
        bandwidth_fwhm = dict_gkp["bandwidth"]
        linewidth_fwhm = dict_gkp["linewidth"]
        fsr = dict_gkp["fsr"]
        sampling_rate = dict_gkp["sampling_rate"]
        beta_T = dict_gkp["beta_T"]
        wigner_vs_beta = dict_gkp["wigner_vs_beta"]
        n_total_sample = dict_gkp["n_sample"]

        n_sample = np.shape(wigner_vs_beta)[1]

        for cut in range(5):
            for step in range(5):
                plot_cut(wigner_vs_beta,step,cut,sampling_rate,n_total_sample,n_sample,fsr,param)

        
plot_wigner("data/wigner_10_001.pkl", "10_001")
