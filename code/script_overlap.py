import os
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
from src import *
import numpy as np
import scipy as sp
import pickle as pic

#np.seterr(invalid='raise')

##############################################################################
### Overlap between all logical states for varying envelope and peak width ###
##############################################################################

fsr = 40e9
n_param = 100 
bandwidth_tab_FWHM = np.linspace(0.01,2.01,n_param)*fsr
linewidth_tab_FWHM = np.linspace(0.01,2.01,n_param)*fsr

beta_T = np.pi/(fsr**2)

overlap_tab = np.zeros((n_param,n_param,6,6))

for bw in range(n_param):
    bandwidth = FWHM_std(bandwidth_tab_FWHM[bw])
    for lw in range(n_param):  
        linewidth = FWHM_std(linewidth_tab_FWHM[lw])
        sampling_rate = 20*np.max((bandwidth,linewidth,fsr))
        frequency_resolution = np.min((bandwidth,linewidth,fsr))/20
        n_sample = int(sampling_rate/frequency_resolution)
        if n_sample%2 == 0:
            n_sample = n_sample+1

        print(f"bw : {bw+1}/{n_param}, lw : {lw+1}/{n_param}, n_sample : {n_sample}")

        freq_vector = make_freq_vector(sampling_rate,n_sample)

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

        overlap_tab[bw,lw,:,:] = np.abs(gkp_ref@(gkp_ref.conj().T))**2


dict_gkp = {"bandwidth_fwhm_tab" : bandwidth_tab_FWHM, "linewidth_fwhm_tab" : linewidth_tab_FWHM, "fsr" : fsr, "beta_T": beta_T, "overlap_tab" : overlap_tab}

with open("data/overlap_bw_lw.pkl","wb") as f:
    pic.dump(dict_gkp,f)
