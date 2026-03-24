import os
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
from src import *
import numpy as np
import scipy as sp
import pickle as pic

#np.seterr(invalid='raise')

###################################################################
### Knill Glancy P_no_error for several envelope and peak width ###
###################################################################

fsr = 40e9
n_param = 100
bandwidth_tab_FWHM = np.logspace(0,2,n_param)*fsr
linewidth_tab_FWHM = np.logspace(-2,0,n_param)*fsr
bandwidth_tab = bandwidth_tab_FWHM/2/np.log(2)
linewidth_tab = linewidth_tab_FWHM/2/np.log(2)

beta_T = np.pi/(fsr**2)

cut_off = 51

p_no_error = np.zeros((n_param,n_param))

u = np.arange(-cut_off//2,cut_off//2+1)[:,None]
v = np.arange(-cut_off//2,cut_off//2+1)[None,:]
n = np.arange(-cut_off//2,cut_off//2+1)[:,None]
m = np.arange(-cut_off//2,cut_off//2+1)[None,:]

factor_1 = np.zeros((cut_off,cut_off))
factor_2 = np.zeros((cut_off,cut_off))
factor_3 = np.zeros((cut_off,cut_off))
factor_4 = np.zeros((cut_off,cut_off))

for bw in range(n_param):
    for lw in range(n_param):
        bandwidth = bandwidth_tab[bw]
        linewidth = linewidth_tab[lw]
        print(f"bw : {bw+1}/{n_param}, lw : {lw+1}/{n_param}")
        factor_1 = np.exp(-(u-v)**2*np.pi**2*(bandwidth/fsr)**2/4)
        factor_2 = np.exp(-(n-m)**2*(fsr/linewidth)**2)
        factor_3 = sp.special.erf((np.pi/2)*(bandwidth/fsr)*(u+v+1/3))-sp.special.erf((np.pi/2)*(bandwidth/fsr)*(u+v-1/3))
        factor_4 = sp.special.erf((fsr/linewidth)*(m+1/6))-sp.special.erf((fsr/linewidth)*(m-1/6))[None,:]
        p_no_error[bw,lw] = 1/4* np.sum(factor_1[:,:,None,None]*factor_2[None,None,:,:]*factor_3[:,:,None,None]*factor_4[None,None,:,:])

dict_gkp = {"bandwidth_fwhm_tab" : bandwidth_tab_FWHM, "linewidth_fwhm_tab" : linewidth_tab_FWHM, "fsr" : fsr, "beta_T": beta_T, "p_no_error" : p_no_error}

with open("data/knill_glancy_bw_lw.pkl","wb") as f:
    pic.dump(dict_gkp,f)
