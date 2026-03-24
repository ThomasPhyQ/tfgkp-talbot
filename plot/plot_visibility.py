import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

#####################################################
### Visibility of each peaks at each Talbot steps ###
#####################################################

step_tab=["zero", "half", "one", "one_half", "two"]
cut_tab=["m2","m1","0","1","2"]
peak_tab=["m2","m1","0","1","2"]

def plot_visibility_cut_peak(visibility,step,cut,peak,fsr,lw_tab,bw_tab):
    width_prop = 1/4
    width_A4 = 210
    width_mm = width_prop*width_A4
    width_inch = width_mm/25.4

    fig,ax = plt.subplots(figsize=(width_inch,width_inch))
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = "Serif"

    plt.pcolormesh(
        lw_tab/fsr,
        bw_tab/fsr,
        np.abs(visibility[:,:,step,cut,peak]),
        vmin=0,
        vmax=1
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.colorbar(pad = 0.1)

    ax.set_aspect('equal', adjustable='box')  # ensures square cells

    plt.savefig(
        "data/plot/minimal/visibility_"+step_tab[step]+"_"+cut_tab[cut]+"_"+peak_tab[peak]+"_bw_lw_max.jpeg",
        dpi = 1000,
        bbox_inches="tight"
    )
    plt.close()



def plot_visibility(filename):
    with open(filename, 'rb') as f:
        dict_gkp = pickle.load(f)
        
        bandwidth_fwhm_tab = dict_gkp["bandwidth"]
        linewidth_fwhm_tab = dict_gkp["linewidth"]
        fsr = dict_gkp["fsr"]
        visibility = dict_gkp["visibility_vs_beta_vs_cut"]

        for step in range(5):
            for cut in range(5):
                for peak in range(5):
                    plot_visibility_cut_peak(visibility,step,cut,peak,fsr,linewidth_fwhm_tab,bandwidth_fwhm_tab)
       

plot_visibility("data/visibility_beta_cut_bw_lw_max.pkl")
