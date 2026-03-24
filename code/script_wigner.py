import os
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
from src import *
import numpy as np
import pickle as pic

############################################
### Wigner function at each Talbot steps ###
############################################

fsr = 40e9
bandwidth_FWHM = 10*fsr
linewidth_FWHM = 0.01*fsr

bandwidth = FWHM_std(bandwidth_FWHM)
linewidth = FWHM_std(linewidth_FWHM)

sampling_rate = 10*np.max((bandwidth,linewidth,fsr))
frequency_resolution = np.min((bandwidth,linewidth,fsr))/5

n_sample = int(sampling_rate/frequency_resolution)
if n_sample%2 == 0:
    n_sample = n_sample+1


n_saved_sample = n_sample//10

beta_T = np.pi/(fsr**2)

wigner_vs_beta = np.zeros((5,n_saved_sample,n_saved_sample))

gkp = make_freq_gkp_plus(fsr,linewidth,bandwidth,n_sample,sampling_rate)

for beta_idx in range(5):
    print(f"Calculating for step {beta_idx}/4")
    gkp_disp = dispersion(gkp,beta_idx*1/2*beta_T,sampling_rate)
    wigner_vs_beta[beta_idx,:,:] = wigner_transform(gkp_disp)[int(n_sample/2)-int(n_saved_sample/2):int(n_sample/2)+int(n_saved_sample/2),int(n_sample/2)-int(n_saved_sample/2):int(n_sample/2)+int(n_saved_sample/2)]*n_sample


dict_gkp = {"n_sample" : n_sample, "n_saved_sample" : n_saved_sample, "sampling_rate" : sampling_rate, "bandwidth" : bandwidth_FWHM, "linewidth" : linewidth_FWHM, "fsr" : fsr, "beta_T": beta_T, "wigner_vs_beta" : wigner_vs_beta}

with open("data/wigner_10_001.pkl","wb") as f:
    pic.dump(dict_gkp,f)   
