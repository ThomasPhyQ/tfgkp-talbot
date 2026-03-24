from src import *
import numpy as np
import scipy as sp
import pickle as pic

###########################################
### Wavefunction for varying dispersion ###
###########################################

fsr = 40e9
bandwidth_FWHM = 30  *fsr
linewidth_FWHM = 0.05*fsr
bandwidth = FWHM_std(bandwidth_FWHM)
linewidth = FWHM_std(linewidth_FWHM)

sampling_rate = 11*bandwidth
frequency_resolution = linewidth/10
n_sample = int(sampling_rate/frequency_resolution)
if n_sample%2 == 0:
    n_sample = n_sample+1

print(n_sample)

beta_T = np.pi/(fsr**2)
n_parameters = 4000

n_peaks = 5
n_display = int(n_peaks*1/fsr/2*sampling_rate)
golden_ratio = (1+np.sqrt(5))/2
n_beta = int(2*n_display*golden_ratio)
beta_step = max(round(n_parameters/n_beta),1)
n_beta=n_parameters//beta_step
print(n_beta/(2*n_display))

beta_tab = np.linspace(-1/8*beta_T,(2+1/8)*beta_T,n_parameters)

gkp_vs_beta = np.zeros((n_sample,n_parameters),dtype=complex)

gkp = make_freq_gkp_plus(fsr,linewidth,bandwidth,n_sample,sampling_rate)

for beta_index in range(n_parameters):
    gkp_vs_beta[:,beta_index] = dispersion(gkp,beta_tab[beta_index],sampling_rate)

dict_gkp = {"sampling_rate" : sampling_rate, "bandwidth" : bandwidth_FWHM, "linewidth" : linewidth_FWHM, "fsr" : fsr, "beta_param" : beta_tab , "wavefunction_vs_beta" : gkp_vs_beta}

with open("data/plus_vs_talbot_30_005.pkl","wb") as f:
    pic.dump(dict_gkp,f)
