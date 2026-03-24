import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

################################################
### Gate fidelity vs envelope and peak width ###
################################################

def plot_gate_fidelity(filename,param):
    with open(filename, 'rb') as f:
        dict_gkp = pickle.load(f)
        
        bandwidth_fwhm_tab = dict_gkp["bandwidth_fwhm_tab"]
        linewidth_fwhm_tab = dict_gkp["linewidth_fwhm_tab"]
        fsr = dict_gkp["fsr"]
        gate_fidelity = dict_gkp["gate_fidelity"]

        width_prop = 1/4
        width_A4 = 210
        width_mm = width_prop*width_A4
        width_inch = width_mm/25.4

        fig,ax = plt.subplots(figsize=(width_inch,width_inch))
        plt.rcParams['font.size'] = 9
        plt.rcParams['font.family'] = "Serif"

        plt.pcolormesh(
            linewidth_fwhm_tab/fsr,
            bandwidth_fwhm_tab/fsr,
            gate_fidelity[:,:],
            vmax=1
        )
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.colorbar(pad = 0.1)

        ax.set_aspect('equal', adjustable='box')  

        plt.savefig(
            "data/plot/minimal/gate_fidelity_"+param+"_bw_lw.jpeg",
            dpi = 1000,
            bbox_inches="tight"
            )
        plt.close()

       


plot_gate_fidelity("data/gate_fidelity_one_bw_lw.pkl","one")
plot_gate_fidelity("data/gate_fidelity_half_bw_lw.pkl","half")
