import numpy as np

# Make frequency axis
def make_freq_vector(sampling_rate,n_sample):
    return np.linspace(-sampling_rate/2,sampling_rate/2,n_sample)

# Gaussian centered at mu with sigma svp L2 normalized
def make_freq_gaussian(mu,sigma,n_sample,sampling_rate):
    freq_vector = np.linspace(-sampling_rate/2,sampling_rate/2,n_sample)
    freq_step = sampling_rate/n_sample
    return np.sqrt(freq_step)/np.sqrt(np.sqrt(np.pi)*sigma)*np.exp(-(freq_vector-mu)**2/(2*sigma**2))

# Gaussian centered at mu with sigma svp L2 normalized optimized (no generation of frequency vector)
def make_freq_gaussian_opt(mu,sigma,n_sample,sampling_rate,freq_vector):
    freq_step = sampling_rate/n_sample
    return np.sqrt(freq_step)/np.sqrt(np.sqrt(np.pi)*sigma)*np.exp(-(freq_vector-mu)**2/(2*sigma**2))

# Infinite Gaussian comb
def make_freq_gaussian_comb(offset,fsr,sigma,n_sample,sampling_rate):
    n_peak=sampling_rate/fsr
    current_peak=1
    comb = make_freq_gaussian(offset,sigma,n_sample,sampling_rate)
    while 2*current_peak<=n_peak:
        comb= comb + make_freq_gaussian(offset+current_peak*fsr,sigma,n_sample,sampling_rate) + make_freq_gaussian(offset-current_peak*fsr,sigma,n_sample,sampling_rate)
        current_peak+=1
    return comb/np.sqrt(n_peak)

# Infinite Gaussian comb optimized (no generation of frequency vector)
def make_freq_gaussian_comb_opt(offset,fsr,sigma,n_sample,sampling_rate,freq_vector):
    n_peak=int(sampling_rate/fsr)
    freq = freq_vector[:,None]
    if n_peak%2==0:
        n_peak=n_peak+1
    centers = (np.arange(-(n_peak//2),n_peak//2+1)*fsr)[None,:]
    return (np.exp(-(freq-centers-offset)**2/(2*sigma**2))/np.sqrt(n_peak)/np.sqrt(np.sqrt(np.pi)*sigma)*np.sqrt(sampling_rate/n_sample)).sum(axis=1)

# GKP states (centered envelope)
def make_freq_gkp_0(fsr,linewidth,bandwidth,n_sample,sampling_rate):
    comb = make_freq_gaussian_comb(0,2*fsr,linewidth,n_sample,sampling_rate)
    gkp = comb*make_freq_gaussian(0,bandwidth,n_sample,sampling_rate)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_0_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm=False):
    comb = make_freq_gaussian_comb_opt(0,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    gkp = comb*make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)
    if norm:
        return gkp/np.linalg.norm(gkp)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_1(fsr,linewidth,bandwidth,n_sample,sampling_rate):
    comb = make_freq_gaussian_comb(fsr,2*fsr,linewidth,n_sample,sampling_rate)
    gkp = comb*make_freq_gaussian(0,bandwidth,n_sample,sampling_rate)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_1_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm=False):
    comb = make_freq_gaussian_comb_opt(fsr,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    gkp = comb*make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)
    if norm:
        return gkp/np.linalg.norm(gkp)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_plus(fsr,linewidth,bandwidth,n_sample,sampling_rate):
    comb0 = make_freq_gaussian_comb(0,2*fsr,linewidth,n_sample,sampling_rate)
    comb1 = make_freq_gaussian_comb(fsr,2*fsr,linewidth,n_sample,sampling_rate)
    gkp = (comb0+comb1)*make_freq_gaussian(0,bandwidth,n_sample,sampling_rate)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_plus_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm=False):
    comb = make_freq_gaussian_comb_opt(0,fsr,linewidth,n_sample,sampling_rate,freq_vector)
    gkp = comb*make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)
    if norm:
        return gkp/np.linalg.norm(gkp)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_minus(fsr,linewidth,bandwidth,n_sample,sampling_rate):
    comb0 = make_freq_gaussian_comb(0,2*fsr,linewidth,n_sample,sampling_rate)
    comb1 = make_freq_gaussian_comb(fsr,2*fsr,linewidth,n_sample,sampling_rate)
    gkp = (comb0-comb1)*make_freq_gaussian(0,bandwidth,n_sample,sampling_rate)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_minus_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm=False):
    comb0 = make_freq_gaussian_comb_opt(0,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    comb1 = make_freq_gaussian_comb_opt(fsr,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    gkp = (comb0-comb1)*make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)
    if norm:
        return gkp/np.linalg.norm(gkp)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_plus_i(fsr,linewidth,bandwidth,n_sample,sampling_rate):
    comb0 = make_freq_gaussian_comb(0,2*fsr,linewidth,n_sample,sampling_rate)
    comb1 = make_freq_gaussian_comb(fsr,2*fsr,linewidth,n_sample,sampling_rate)
    gkp = (comb0+1j*comb1)*make_freq_gaussian(0,bandwidth,n_sample,sampling_rate)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_plus_i_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm=False):
    comb0 = make_freq_gaussian_comb_opt(0,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    comb1 = make_freq_gaussian_comb_opt(fsr,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    gkp = (comb0+1j*comb1)*make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)
    if norm:
        return gkp/np.linalg.norm(gkp)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_minus_i(fsr,linewidth,bandwidth,n_sample,sampling_rate):
    comb0 = make_freq_gaussian_comb(0,2*fsr,linewidth,n_sample,sampling_rate)
    comb1 = make_freq_gaussian_comb(fsr,2*fsr,linewidth,n_sample,sampling_rate)
    gkp = (comb0-1j*comb1)*make_freq_gaussian(0,bandwidth,n_sample,sampling_rate)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

def make_freq_gkp_minus_i_opt(fsr,linewidth,bandwidth,n_sample,sampling_rate,freq_vector,norm=False):
    comb0 = make_freq_gaussian_comb_opt(0,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    comb1 = make_freq_gaussian_comb_opt(fsr,2*fsr,linewidth,n_sample,sampling_rate,freq_vector)
    gkp = (comb0-1j*comb1)*make_freq_gaussian_opt(0,bandwidth,n_sample,sampling_rate,freq_vector)
    if norm:
        return gkp/np.linalg.norm(gkp)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

# IFFT for wigner function
def custom_ifft(signal):
    n_sample = np.shape(signal)[0]
    large_signal = np.zeros(2*n_sample,dtype=complex)
    large_signal[::2] = signal
    ifft_large_signal = np.fft.ifft(np.fft.ifftshift(large_signal),norm="backward")
    return ifft_large_signal[n_sample-int(n_sample/2):n_sample+int(n_sample/2)+1]

# Wigner transform
def wigner_transform(signal):
    n_sample = np.shape(signal)[0]
    half = n_sample//2
    correl = np.zeros((n_sample,n_sample),dtype=complex)
    f_idx = np.arange(n_sample)[:,None]
    F_idx = np.arange(n_sample)[None,:]
    padded_signal = np.pad(signal,(half,half),mode="constant")
    correl = padded_signal[F_idx-f_idx+n_sample-1]*np.conj(padded_signal)[f_idx+F_idx]
    ifft_large = np.fft.ifft(
            np.fft.ifftshift(correl,axes = 0),
            axis = 0,
            norm = "backward"
            )
    wigner=np.fft.fftshift(np.real(ifft_large),axes=0)
    return (wigner/np.sum(wigner,dtype = np.float128)).astype(np.float32)

# Wigner transform for a single frequency shift (bin index)
def wigner_transform_single_cut_bin(signal,f_bin,sampling_rate):
    n_sample = np.shape(signal)[0]
    half = n_sample//2
    f_idx = np.arange(n_sample)[:]
    padded_signal = np.pad(signal,(half,half),mode="constant")
    correl = padded_signal[f_bin-f_idx+n_sample-1]*np.conj(padded_signal)[f_bin+f_idx]
    ifft_large = np.fft.ifft(
            np.fft.ifftshift(correl),
            norm = "backward"
            )
    return np.fft.fftshift(np.real(ifft_large))*sampling_rate

# Wigner transform for a single frequency shift (in Hz)
def wigner_transform_single_cut(signal,f_cut,sampling_rate):
    n_sample = np.shape(signal)[0]
    f_bin = int(f_cut/sampling_rate*n_sample)+n_sample//2
    return wigner_transform_single_cut_bin(signal,f_bin,sampling_rate)

# Dispersion of beta value
def dispersion(fsignal,beta,sampling_rate):
    n_sample = np.shape(fsignal)[0]
    freq_vector = np.linspace(-sampling_rate/2,sampling_rate/2,n_sample)
    return fsignal*np.exp(1j*beta*freq_vector**2)
    
# Dispersion on a matrix of state
def dispersion_matrix(fsignal,beta,sampling_rate):
    #rows contain the vectors
    n_sample = np.shape(fsignal)[0]
    n_state = np.shape(fsignal)[0]
    freq_vector = np.linspace(-sampling_rate/2,sampling_rate/2,n_sample)
    out_sig = fsignal[:,:]
    for state in range(n_state):
        out_sig[state,:] = out_sig[state,:]*np.exp(1j*beta*freq_vector**2)
    return out_sig

def dispersion_matrix_opt(fsignal,beta,freq_vector):
    #rows contain the vectors
    n_state = np.shape(fsignal)[0]
    n_sample = np.shape(fsignal)[1]
    out_sig = np.zeros((n_state,n_sample),dtype=complex)
    for state in range(n_state):
        out_sig[state,:] = fsignal[state,:]*np.exp(1j*beta*freq_vector**2)
    return out_sig

# Wigner function (only time shift)
def wigner_transform_cut(signal):
    n_sample = np.shape(signal)[0]
    correl = np.multiply(np.flip(signal),np.conj(signal)) 
    wigner_cut = np.real(custom_ifft(correl))
    return (wigner_cut/np.sum(wigner_cut,dtype = np.float128)).astype(np.float32)

# Fidelity between two states
def fidelity(state_1,state_2):
    return np.abs(np.dot(np.conj(state_1),state_2))**2/(np.linalg.norm(state_1)**2*np.linalg.norm(state_2)**2)

# Not working
def make_freq_gkp_hex_0(fsr,linewidth,bandwidth,n_sample,sampling_rate):
    n_peak=sampling_rate/fsr
    current_peak=1
    comb = make_freq_gaussian(0,linewidth,n_sample,sampling_rate)
    while 2*current_peak<=n_peak:
        phase = 1j if current_peak%2 == 1 else 1
        comb= comb + phase*make_freq_gaussian(current_peak*fsr,linewidth,n_sample,sampling_rate) + make_freq_gaussian(-current_peak*fsr,linewidth,n_sample,sampling_rate)
        current_peak+=1
    gkp = comb*make_freq_gaussian(0,bandwidth,n_sample,sampling_rate)
    return gkp/np.linalg.norm(gkp)/np.sqrt(sampling_rate/n_sample)

# Time shift of duration delta
def time_shift(x,sampling_rate,delta):
    delta_bin = np.rint(delta*sampling_rate)
    X = np.fft.ifftshift(np.fft.ifft(x))
    return np.fft.fft(np.fft.fftshift(np.roll(X,delta_bin)))

# frequency shift of frequency delta
def freq_shift(x,sampling_rate,delta):
    n = np.shape(x)[0]
    delta_bin = np.rint(delta/sampling_rate*n)
    return np.roll(x,delta_bin)

# FWHM to standard deviation for Gaussians
def FWHM_std(fwhm):
    # In intensity
    return fwhm/2/np.log(2)

# Stabilizer (frequency shift)
def stabilizer_freq(x,sampling_rate,fsr):
    return fidelity(freq_shift(x,sampling_rate,2*fsr),x)

# Stabilizer (time shift)
def stabilizer_time(x,sampling_rate,fsr):
    return fidelity(time_shift(x,sampling_rate,1/fsr),x)
