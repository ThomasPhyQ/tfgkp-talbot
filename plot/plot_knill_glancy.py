import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

##########################################################
### Knill glancy p_no_error vs envelope and peka width ###
##########################################################

def plot_p_error_state(p_no_error,bw_axis,lw_axis):

    width_prop= 1/2
    width_A4 = 210
    width_mm = width_prop*width_A4
    width_inch = width_mm/25.4

    fig,ax = plt.subplots(figsize=(width_inch,width_inch))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = "Serif"

    plt.pcolormesh(
        lw_axis,
        bw_axis,
        1-p_no_error[:,:],
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.colorbar(pad = 0.1)

    ax.set_aspect('equal', adjustable='box')  

    plt.savefig(
            "data/plot/minimal/knill_glancy_bw_lw.jpeg",
            dpi = 1000,
            bbox_inches="tight"
            )
    plt.close()

def plot_overlap(filename):
    with open(filename, 'rb') as f:
        dict_gkp = pickle.load(f)
        
        bandwidth_fwhm_tab = dict_gkp["bandwidth_fwhm_tab"]
        linewidth_fwhm_tab = dict_gkp["linewidth_fwhm_tab"]
        fsr = dict_gkp["fsr"]
        p_no_error = dict_gkp["p_no_error"]

        n_bw = np.shape(bandwidth_fwhm_tab)[0]
        n_lw = np.shape(linewidth_fwhm_tab)[0]

        plot_p_error_state(p_no_error,bandwidth_fwhm_tab/fsr,linewidth_fwhm_tab/fsr)


plot_overlap("data/knill_glancy_bw_lw.pkl")
