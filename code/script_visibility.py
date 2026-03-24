import os
os.environ["OMP_NUM_THREADS"] = "70"
os.environ["OPENBLAS_NUM_THREADS"] = "70"
from src import *
import numpy as np
import scipy as sp
import pickle as pic

############################################################################################
### Visibility of the coincidences  computed through Gaussian fits or maximum heuristics ###
############################################################################################

fsr = 40e9
time_period = 1/2/fsr
n_param = 100 
bandwidth_tab_FWHM = np.logspace(0,2,n_param)*fsr
linewidth_tab_FWHM = np.logspace(-2,0,n_param)*fsr

visibility_vs_beta_vs_cut = np.zeros((n_param,n_param,5,5,5))

for bw in range(n_param):
    for lw in range(n_param):
        bandwidth = FWHM_std(bandwidth_tab_FWHM[bw])
        linewidth = FWHM_std(linewidth_tab_FWHM[lw])
        sampling_rate = 4*(np.max((bandwidth,linewidth,fsr))//fsr+1)*2*fsr
        frequency_resolution =1/np.max((fsr//bandwidth,fsr//linewidth,1)) *fsr /2 /4
        n_sample = int(sampling_rate/frequency_resolution)
        print(f"bw : {bw+1}/{n_param}, lw : {lw+1}/{n_param}, n_sample : {n_sample}")
        freq_vector = make_freq_vector(sampling_rate,n_sample)

        gkp = make_freq_gkp_plus_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm = True)/np.sqrt(sampling_rate)*np.sqrt(n_sample)

        n_saved_sample = n_sample

        beta_T = np.pi/(fsr**2)

        for beta_idx in range(5):
            gkp_disp = dispersion(gkp,beta_idx*1/2*beta_T,sampling_rate)
            for cut_idx in range(5):
                wigner = np.zeros((3,n_sample))
                for k in range(3):
                    wigner[k,:] = wigner_transform_single_cut(gkp_disp,(cut_idx*fsr/2-fsr) + (k-1)*frequency_resolution,sampling_rate)
                for peak_idx in range(5):
                    t_bin = int((peak_idx-2)*time_period*sampling_rate) + n_sample//2

                    y_data = np.abs(wigner[:,t_bin-1:t_bin+2])
                    visibility_vs_beta_vs_cut[bw,lw,beta_idx,cut_idx,peak_idx] = y_data.max()



dict_gkp = {"sampling_rate" : sampling_rate, "bandwidth" : bandwidth_tab_FWHM, "linewidth" : linewidth_tab_FWHM, "fsr" : fsr, "beta_T": beta_T, "visibility_vs_beta_vs_cut" : visibility_vs_beta_vs_cut}

with open("data/visibility_beta_cut_bw_lw_max_mult.pkl","wb") as f:
    pic.dump(dict_gkp,f)   
