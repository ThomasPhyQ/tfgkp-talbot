import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#################################
### JSI vs dispersion heatmap ###
#################################

filename = "data/plus_vs_talbot_30_01.pkl"

with open(filename,"rb") as f:
    dict_gkp = pickle.load(f)

    sampling_rate = dict_gkp["sampling_rate"]

    fsr = dict_gkp["fsr"]
    beta_T = np.pi/(fsr**2)

    beta_tab = dict_gkp["beta_param"]
    n_parameters = (np.shape(beta_tab))[0]

    gkp_vs_beta = dict_gkp["wavefunction_vs_beta"]

    n_sample = np.shape(gkp_vs_beta)[0]
    time_vector = np.linspace(-n_sample/sampling_rate/2,n_sample/sampling_rate/2,n_sample)

    intensity_vs_beta = np.zeros((n_sample,n_parameters))
    for param_index in range(n_parameters):
        intensity_vs_beta[:,param_index] = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(gkp_vs_beta[:,param_index]))))**2

    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = "Serif"
    plt.rcParams['text.usetex'] = True

    n_peaks = 5
    n_display = int(n_peaks*1/fsr/2*sampling_rate)
    t_min = -n_display/sampling_rate
    t_max = n_display/sampling_rate
    n_yticks = 3
    step_yticks = round(n_display/n_yticks)
    y_labels = np.vectorize(lambda x: f"{x:.2f}")(np.linspace(t_min,t_max,n_yticks)*fsr)
    y_positions = (np.linspace(-0.5,0.5,n_yticks)+0.5)*n_display

    golden_ratio = (1+np.sqrt(5))/2

    n_beta = int(2*n_display*golden_ratio)
    beta_step = round(n_parameters/n_beta)
    n_beta = n_parameters//beta_step
    n_xticks = 5
    step_xticks = int(n_beta/n_xticks)
    x_labels = np.vectorize(lambda x: f"{x:.2f}")(np.linspace(-1/8,2+1/8,n_xticks))
    x_positions = (np.linspace(-0.5,0.5,n_xticks)+0.5)*n_beta

    intensity_vs_beta_display = intensity_vs_beta[n_sample//2-n_display//2:n_sample//2+n_display//2,::beta_step]


    width_prop = 1/2
    width_A4 = 210
    width_mm = width_prop*width_A4
    width_inch = width_mm/25.4

    fig,ax = plt.subplots(figsize=(width_inch/2,width_inch/golden_ratio))

    plt.imshow(
        intensity_vs_beta_display/np.max(intensity_vs_beta_display),
        cmap = "RdBu"
    )

    plt.yticks(y_positions, y_labels)
    plt.xticks(x_positions, x_labels)

    plt.colorbar(fraction = 0.02)

    plt.savefig("data/plot/plus_vs_talbot_30_01.png",dpi=1000,bbox_inches="tight")
