import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

####################################################################
### Overlap between all logical state vs envelope and peka width ###
####################################################################

states = ["0", "1", "+", "-", "+i", "-i"]

def plot_overlap_state(overlap,bw_axis,lw_axis,state_1,state_2):
    print(f"Plotting overlap for state {state_1+1}/{6} vs {state_2+1}/6")

    width_prop=1/4
    width_A4 = 210
    width_mm = width_prop*width_A4
    width_inch = width_mm/25.4

    fig,ax = plt.subplots(figsize=(width_inch,width_inch))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = "Serif"

    plt.pcolormesh(
        lw_axis,
        bw_axis,
        overlap[:,:,state_1,state_2],
    )
    plt.yticks(np.arange(0, 2.001,0.5))
    plt.xticks(np.arange(0, 2.001 ,0.5))

    plt.colorbar(pad = 0.1)

    ax.set_aspect('equal', adjustable='box')  

    n_bw = np.shape(overlap)[0]
    n_lw = np.shape(overlap)[1]

    plt.savefig(
            "data/plot/minimal/overlap_"+states[state_1]+"_vs_"+states[state_2]+"_bw_lw.jpeg",
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
        overlap = dict_gkp["overlap_tab"]

        n_bw = np.shape(bandwidth_fwhm_tab)[0]
        n_lw = np.shape(linewidth_fwhm_tab)[0]

        for state_1 in range(6):
            for state_2 in range(state_1,6):
                plot_overlap_state(overlap,bandwidth_fwhm_tab/fsr,linewidth_fwhm_tab/fsr,state_1,state_2)


plot_overlap("data/overlap_bw_lw.pkl")
