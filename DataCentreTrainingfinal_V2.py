import matplotlib.pyplot as plt
from six.moves.urllib import request
import numpy as np
from pycbc.filter import matched_filter
from pycbc.filter import sigma
import pycbc.frame
import pylab as py
import pycbc
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from random import randint
from numpy import pi
from random import random
import numpy.polynomial.polynomial as poly
import Harmonic_functions_and_waveform as hf
from pandas import DataFrame
import pandas as pd
from random import randint, choice, randrange, uniform
import decimal
from pycbc.waveform.waveform_modes import jframe_to_l0frame
from pycbc import noise

delta_f = 1/256
delta_t = 1/2048
flow  = 15
nyquist = 2048
tlen = 256
flen = 1+tlen*nyquist
f_ref=15
f_lower=15
m1 = 10
m2 = 1.4
chi1 = 0.9
k = 0.5



def waveform_spin(incl, psi, kappa, alpha0=0, phase=0):
    lframe = jframe_to_l0frame(m1, m2, f_ref=f_ref,
                    thetajn=incl,
                    spin1_a=chi1, spin1_polar=np.arccos(kappa),
                    phijl=alpha0)
    hp, hc = get_fd_waveform(approximant='IMRPhenomPv2', mass1=m1, mass2=m2,
                                delta_f = 1./tlen, f_lower=f_lower, f_ref=f_ref,
                                coa_phase=phase, distance=475,
                                **lframe)
    
    h = (hp*np.cos(2*psi) + hc*np.sin(2*psi))
    h = h.cyclic_time_shift(h.start_time + tlen/2)
    
    return h[:flen]



def waveform_spin_template(incl, kappa, alpha0=pi/2, phase=0):
    lframe = jframe_to_l0frame(m1, m2, f_ref=f_ref,
                    thetajn=incl,
                    spin1_a=chi1, spin1_polar=np.arccos(kappa),
                    phijl=alpha0)
    hp, hc = get_fd_waveform(approximant='IMRPhenomPv2', mass1=m1, mass2=m2,
                                delta_f = 1./tlen, f_lower=f_lower, f_ref=f_ref,
                                coa_phase=phase,
                                **lframe)
    
    hp = hp.cyclic_time_shift(hp.start_time)
    hc = hc.cyclic_time_shift(hc.start_time)
    
    return (hp[:flen], hc[:flen])



def isolated_harmonics():
    
    harm_val = np.array([hf.harmonics_p(0.),
                    hf.harmonics_p(pi/4),
                    hf.harmonics_p(pi/2),
                    hf.harmonics_x(pi/2),
                    hf.harmonics_x(pi)])
    
    h_inv = np.linalg.inv(harm_val)
    
    hAp, hAx  = waveform_spin_template(0, 0.5)
    hBp, hBx  = waveform_spin_template(pi/4, 0.5)
    hCp, hCx  = waveform_spin_template(pi/2, 0.5)
    hDp, hDx  = waveform_spin_template(pi, 0.5)
    
    h_m2 = h_inv[0,0]*hAp + h_inv[0,1]*hBp + h_inv[0,2]*hCp + h_inv[0,3]*hCx + h_inv[0,4]*hDx
    h_m1 = h_inv[1,0]*hAp + h_inv[1,1]*hBp + h_inv[1,2]*hCp + h_inv[1,3]*hCx + h_inv[1,4]*hDx
    h0   = h_inv[2,0]*hAp + h_inv[2,1]*hBp + h_inv[2,2]*hCp + h_inv[2,3]*hCx + h_inv[2,4]*hDx
    h1   = h_inv[3,0]*hAp + h_inv[3,1]*hBp + h_inv[3,2]*hCp + h_inv[3,3]*hCx + h_inv[3,4]*hDx
    h2   = h_inv[4,0]*hAp + h_inv[4,1]*hBp + h_inv[4,2]*hCp + h_inv[4,3]*hCx + h_inv[4,4]*hDx
    
    return h0, h1, h2, h_m1, h_m2




def search(x):
   
    theta  = []
    #psi    = []
    #alpha0 = []
    SNR0   = []
    SNR1   = []
    SNR2   = []
    SNR_1  = []
    SNR_2  = []
    SNRphase0 = []
    SNRphase1 = []
    SNRphase2 = []
    SNRphase_1 = []
    SNRphase_2 = []
    sigma_data = []
    coa_phase = []
    rt_sqrt = []
    mean    = []
    sqr_sum = []
    
    m_0 = isolated_harmonics()[0]
    m_1 = isolated_harmonics()[1]
    m_2 = isolated_harmonics()[2]
    h_m1 = isolated_harmonics()[3]
    h_m2 = isolated_harmonics()[4]
        
    psd      = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
    
    tsamples = int(1024 * 1024)
    
    for i in range(x):
        
        noise = pycbc.noise.noise_from_psd(tsamples, 1/4096, psd, seed = x)
        noise = noise.to_frequencyseries()
            
        t = np.arccos(uniform(-1,1))
        p = random()*pi
        a = random()*2*pi
        phase = random()*2*pi
        
        z = uniform(0,1)#(uniform(0.001,1)**(-1/3))#FIXME
        
        #print(t, p, a, phase)
        
        signal = waveform_spin(t,p,k,a,phase)*z
        
        sig   = sigma(signal, psd=psd, low_frequency_cutoff = 15)
        
        data = signal + noise
        
        ## Matched filter of 5 harmonics.
        #print('before match')
        
        snr2  = matched_filter(m_2, data, psd=psd, low_frequency_cutoff = 15).numpy()
        snr1  = matched_filter(m_1, data, psd=psd, low_frequency_cutoff = 15).numpy()
        snr0  = matched_filter(m_0, data, psd=psd, low_frequency_cutoff = 15).numpy()
        snr_1  = matched_filter(h_m1, data, psd=psd, low_frequency_cutoff = 15).numpy()
        snr_2  = matched_filter(h_m2, data, psd=psd, low_frequency_cutoff = 15).numpy()
        
        #print('after match')        
        
        for peak in [524288, 524288-8*4096, 524288+8*4096, 524288-12*4096, 524288+12*4096, 524288-16*4096, 524288+16*4096, 524288-20*4096, 524288+20*4096, 524288-24*4096, 524288+24*4096, 524288-28*4096, 524288+28*4096, 524288-32*4096, 524288+32*4096, 524288-36*4096, 524288+36*4096, 524288-40*4096, 524288+40*4096, 524288-44*4096, 524288+44*4096, 524288-48*4096, 524288+48*4096, 524288-52*4096, 524288+52*4096, 524288-56*4096, 524288+56*4096]:
        
            snrp2 = abs(snr2[peak])
            snr2_phase = np.angle(snr2[peak])

            snrp1 = abs(snr1[peak])
            snr1_phase = np.angle(snr1[peak]*snr2[peak].conj())

            snrp0 = abs(snr0[peak])
            snr0_phase = np.angle(snr0[peak]*snr2[peak].conj())

            snrp_1 = abs(snr_1[peak])
            snr_1_phase = np.angle(snr_1[peak]*snr2[peak].conj()) 

            snrp_2 = abs(snr_2[peak])
            snr_2_phase = np.angle(snr_2[peak]*snr2[peak].conj())

            root_sqr = np.sqrt(snrp0**2 + snrp1**2 + snrp2**2 + snrp_1**2 + snrp_2**2)
            mn = ((snrp0 + snrp1 + snrp2 + snrp_1 + snrp_2)/5)

            
            theta.append(t)

            SNR0.append(snrp0)
            SNR1.append(snrp1)
            SNR2.append(snrp2)
            SNR_1.append(snrp_1)
            SNR_2.append(snrp_2)

            SNRphase0.append(snr0_phase)
            SNRphase1.append(snr1_phase)
            SNRphase2.append(snr2_phase)
            SNRphase_1.append(snr_1_phase)
            SNRphase_2.append(snr_2_phase)
            
            if peak == 524288:
                sigma_data.append(sig)
            else:
                sigma_data.append(0)

            coa_phase.append(phase)
            rt_sqrt.append(root_sqr)
            mean.append(mn) 

    df = pd.DataFrame(data={"snr m=0": SNR0, "snr m=1": SNR1, "snr m=2": SNR2, "snr m=-1": SNR_1, "snr m=-2": SNR_2, "theta": theta,
                            "SNRphase0": SNRphase0, "SNRphase1": SNRphase1, "SNRphase2":SNRphase2, "SNRphase_1": SNRphase_1,
                            "SNRphase_2": SNRphase_2, "sigma": sigma_data, "phase": coa_phase, "square root": rt_sqrt,
                            "mean": mean}) #FIXME
    df.to_csv("FinalTrain_V3.csv", sep=',',index=False)
    
    print('Done')
    
    return

search(2000)