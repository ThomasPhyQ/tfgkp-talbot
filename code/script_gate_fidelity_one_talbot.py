import os
os.environ["OMP_NUM_THREADS"] = "30"
os.environ["OPENBLAS_NUM_THREADS"] = "30"
from src import *
import numpy as np
import scipy as sp
import pickle as pic
#import matplotlib.pyplot as plt

#np.seterr(invalid='raise')

############################################################
### Gate fidelity for different envelope and peak widths ###
############################################################

fsr = 40e9
n_param = 100
bandwidth_tab_FWHM = np.logspace(0,2,n_param)*fsr
linewidth_tab_FWHM = np.logspace(-2,0,n_param)*fsr

beta_T = np.pi/(fsr**2)

gate_fidelity = np.zeros((n_param,n_param))

def T_op(x, gkp_ref):
    return np.dot(gkp_ref[0,:].conj(),x)*gkp_ref[1,:] + np.dot(gkp_ref[1,:].conj(),x)*gkp_ref[0,:]

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

        even_comb = make_freq_gaussian_comb_opt(0,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
        odd_comb = make_freq_gaussian_comb_opt(fsr,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
        envelope = make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)

        gkp_ref = np.zeros((2,n_sample), dtype = complex)
        gkp_ref[0,:] = (odd_comb+even_comb)*envelope
        gkp_ref[0,:] = gkp_ref[0,:]/np.linalg.norm(gkp_ref[0,:])
        gkp_ref[1,:] = (odd_comb-even_comb)*envelope
        gkp_ref[1,:] = gkp_ref[1,:]/np.linalg.norm(gkp_ref[1,:])

        basis_ref = np.zeros((2,n_sample), dtype = complex)
        basis_ref[0,:] = gkp_ref[0,:]
        tmp_vector = gkp_ref[1,:] - np.dot(gkp_ref[0,:].conj(),gkp_ref[1,:])*gkp_ref[0,:]
        basis_ref[1,:] = tmp_vector/np.linalg.norm(tmp_vector)

        disp = np.exp(1j*beta_T*freq_vector**2)
        conj_disp = np.conj(disp)
        
        #WT
        T_basis_0 = T_op(basis_ref[0,:],gkp_ref)
        disp_T_basis_0 = conj_disp*T_basis_0

        T_basis_1 = T_op(basis_ref[1,:],gkp_ref)
        disp_T_basis_1 = conj_disp*T_basis_1

        tr_WT = np.dot(basis_ref[0,:].conj(), disp_T_basis_0) + np.dot(basis_ref[1,:].conj(), disp_T_basis_1)

        #TT
        T_T_basis_0 = T_op(T_basis_0,gkp_ref)
        T_T_basis_1 = T_op(T_basis_1,gkp_ref)

        tr_TT = np.dot(T_T_basis_0,basis_ref[0,:].conj()) + np.dot(T_T_basis_1,basis_ref[1,:].conj())

        #WW
        tr_WW = 2
        
        gate_fidelity[bw,lw] = np.abs(np.abs(tr_WT)**2/(tr_WW*tr_TT))
        if gate_fidelity[bw,lw]>1:
            print(gate_fidelity[bw,lw])

dict_gkp = {"bandwidth_fwhm_tab" : bandwidth_tab_FWHM, "linewidth_fwhm_tab" : linewidth_tab_FWHM, "fsr" : fsr, "beta_T": beta_T, "gate_fidelity" : gate_fidelity}

with open("data/gate_fidelity_one_bw_lw.pkl","wb") as f:
    pic.dump(dict_gkp,f)
