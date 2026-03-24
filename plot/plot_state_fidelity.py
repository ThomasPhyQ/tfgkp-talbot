import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

##############################################################
### State fidelity for each multiple of half Talbot length ###
##############################################################

states = ["0", "1", "+", "-", "+i", "-i"]
steps = ["zero", "half", "one", "three_half", "two"]

filename = "data/fidelity_one_lw_bw.pkl"

def plot_fidelity_state(param,fid,bw_axis,lw_axis,beta_T,state,step):
    print(f"Plotting fid for state {state+1}/{6}")
    print(f"Plotting fid for step {step+1}/{5}")

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
        fid[step,:,:,state],
        vmin = 0,
        vmax = 1
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.colorbar(pad = 0.1)

    ax.set_aspect('equal', adjustable='box')  

    n_bw = np.shape(fid)[1]
    n_lw = np.shape(fid)[2]

    plt.savefig(
            "data/plot/minimal/fidelity_"+param+"_"+states[state]+"_"+steps[step]+"_bw_lw.jpeg",
            dpi = 1000,
            bbox_inches="tight"
            )
    plt.close()

def plot_fidelity(filename,param):
    with open(filename, 'rb') as f:
        dict_fidelity = pickle.load(f)
        
        bandwidth_fwhm_tab = dict_fidelity["bandwidth_fwhm_tab"]
        linewidth_fwhm_tab = dict_fidelity["linewidth_fwhm_tab"]
        fidelity = dict_fidelity["fidelity_tab"]
        fsr = dict_fidelity["fsr"]

        n_bw = np.shape(bandwidth_fwhm_tab)[0]
        n_lw = np.shape(linewidth_fwhm_tab)[0]


        beta_T = dict_fidelity["beta_T"]
        for beta_idx in range(5):
            for state in range(6):
                plot_fidelity_state(param,fidelity,bandwidth_fwhm_tab/fsr,linewidth_fwhm_tab/fsr,beta_T,state,beta_idx)


plot_fidelity("data/fidelity_single_beta_bw_lw.pkl","single_beta")
plot_fidelity("data/fidelity_single_beta_bw_lw_high_res.pkl","single_beta_high_res")
