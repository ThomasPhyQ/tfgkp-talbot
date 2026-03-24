import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

############################################
### Wigner function at each Talbot steps ###
############################################

step_tab = ["zero", "half", "one", "one_half", "two"]

def plot_step(wigner,step,sampling_rate,n_total_sample,n_sample,fsr,param):
    width_prop = (7/8)*(1/3) 
    width_A4 = 210
    width_mm = width_prop*width_A4
    width_inch = width_mm/25.4
    golden_ratio = (1+np.sqrt(5))/2

    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = "Serif"
    plt.rcParams['text.usetex'] = True

    fig,ax = plt.subplots(figsize=(width_inch,width_inch/golden_ratio))

    n_peaks = 1.5
    n_display_t = int(n_peaks*1/fsr*sampling_rate)
    n_display_f = int(n_peaks*fsr*n_total_sample/sampling_rate)
    t_min = -n_display_t/sampling_rate
    t_max = n_display_t/sampling_rate
    n_yticks = 5
    step_yticks = round(n_display_t/n_yticks)
    y_labels = np.flip(np.vectorize(lambda x: f"{x:.1f}")(np.linspace(t_min,t_max,n_yticks)/(1/fsr)))
    y_positions = (np.linspace(-0.5,0.5,n_yticks)+0.5)*2*n_display_t
    f_min = -sampling_rate*n_display_f/n_total_sample
    f_max = +sampling_rate*n_display_f/n_total_sample
    n_xticks = 5
    step_xticks = round(n_display_f/n_xticks)
    x_labels = np.vectorize(lambda x: f"{x:.1f}")(np.linspace(f_min,f_max,n_xticks)/fsr)
    x_positions = (np.linspace(-0.5,0.5,n_xticks)+0.5)*2*n_display_f


    plt.imshow(
        np.flip(wigner[step,n_sample//2-n_display_t:n_sample//2+n_display_t:,n_sample//2-n_display_f:n_sample//2+n_display_f:],axis=0),
        cmap="RdBu",
        aspect = "auto",
        interpolation = "nearest"
    )
    plt.yticks(y_positions, y_labels)
    plt.xticks(x_positions, x_labels)

    plt.colorbar()

    plt.savefig(
        "data/plot/minimal/wigner_"+step_tab[step]+"_"+param+".jpeg",
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

        for step in range(5):
            plot_step(wigner_vs_beta,step,sampling_rate,n_total_sample,n_sample,fsr,param)


plot_wigner("data/wigner_30_01.pkl", "30_01")
plot_wigner("data/wigner_10_001.pkl", "10_001")
plot_wigner("data/wigner_2_004.pkl","2_004")
