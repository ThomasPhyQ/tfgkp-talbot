import os
os.environ["OMP_NUM_THREADS"] = "50"
os.environ["OPENBLAS_NUM_THREADS"] = "50"
from src import *
import numpy as np
import scipy as sp
import pickle as pic

#np.seterr(invalid='raise')

##################################################################################################################
### Fidelity vs all logical states and stabilizer of the propagating state for varying peak width and envelope ###
##################################################################################################################

fsr = 40e9
n_param = 200
bandwidth_tab_FWHM = np.logspace(0,2,n_param)*fsr
linewidth_tab_FWHM = np.logspace(-2,0,n_param)*fsr

beta_T = np.pi/(fsr**2)

fidelity_tab = np.zeros((5,n_param,n_param,6))
stabilizer_tab = np.zeros((5,n_param,n_param,2))

for bw in range(n_param):
    bandwidth = FWHM_std(bandwidth_tab_FWHM[bw])
    for lw in range(n_param):  
        linewidth = FWHM_std(linewidth_tab_FWHM[lw])
        sampling_rate = 10*np.max((bandwidth,linewidth,fsr))
        frequency_resolution = np.min((bandwidth,linewidth,fsr))/10
        n_sample = int(sampling_rate/frequency_resolution)
        if n_sample%2 == 0:
            n_sample = n_sample+1

        print(f"bw : {bw+1}/{n_param}, lw : {lw+1}/{n_param}, n_sample : {n_sample}")

        freq_vector = make_freq_vector(sampling_rate,n_sample)

        gkp = make_freq_gkp_plus_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm = True)
        
        even_comb = make_freq_gaussian_comb_opt(0,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
        odd_comb = make_freq_gaussian_comb_opt(fsr,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
        envelope = make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)

        gkp_ref = np.zeros((6,n_sample), dtype = complex)
        gkp_ref[0,:] = even_comb*envelope
        gkp_ref[0,:] = gkp_ref[0,:]/np.linalg.norm(gkp_ref[0,:])
        gkp_ref[1,:] = odd_comb*envelope
        gkp_ref[1,:] = gkp_ref[1,:]/np.linalg.norm(gkp_ref[1,:])
        gkp_ref[2,:] = (odd_comb+even_comb)*envelope
        gkp_ref[2,:] = gkp_ref[2,:]/np.linalg.norm(gkp_ref[2,:])
        gkp_ref[3,:] = (odd_comb-even_comb)*envelope
        gkp_ref[3,:] = gkp_ref[3,:]/np.linalg.norm(gkp_ref[3,:])
        gkp_ref[4,:] = (1j*odd_comb+even_comb)*envelope
        gkp_ref[4,:] = gkp_ref[4,:]/np.linalg.norm(gkp_ref[4,:])
        gkp_ref[5,:] = (-1j*odd_comb+even_comb)*envelope
        gkp_ref[5,:] = gkp_ref[5,:]/np.linalg.norm(gkp_ref[5,:])

        for beta_idx in range(5):
            gkp_dispersion = dispersion(gkp,beta_idx*1/2*beta_T,sampling_rate)
            fidelity_current = np.abs(gkp_ref@gkp_dispersion)**2
            fidelity_tab[beta_idx,bw,lw,:] = fidelity_current
            
dict_gkp = {"bandwidth_fwhm_tab" : bandwidth_tab_FWHM, "linewidth_fwhm_tab" : linewidth_tab_FWHM, "fsr" : fsr, "beta_T": beta_T, "fidelity_tab" : fidelity_tab}

with open("data/fidelity_single_beta_bw_lw_high_res.pkl","wb") as f:
    pic.dump(dict_gkp,f)
