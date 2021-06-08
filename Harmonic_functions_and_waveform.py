from six.moves.urllib import request
import numpy as np
from pycbc.filter import matched_filter
import pycbc.frame
import pylab as py
import pycbc
from pycbc.waveform import get_td_waveform
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from random import randint
from numpy import pi, cos, sin
from random import random
import numpy.polynomial.polynomial as poly




def waveform_spin(incl, kappa, alpha0=pi/2, phase=0):
    lframe = jframe_to_l0frame(m1, m2, f_ref=f_ref,
                    thetajn=incl,
                    spin1_a=chi1, spin1_polar=np.arccos(kappa),
                    phijl=alpha0)
    hp, hc = get_fd_waveform(approximant='IMRPhenomPv2', mass1=m1, mass2=m2,
                                delta_f = 1./tlen, f_lower=f_lower, f_ref=f_ref,
                                coa_phase=phase, distance=945,
                                **lframe)
    
    return (hp[:flen], hc[:flen])




    def isolated_harmonics():

        harms = np.array([hf.harmonics_p(0.),
                        hf.harmonics_p(pi/4),
                        hf.harmonics_p(pi/2),
                        hf.harmonics_x(pi/2),
                        hf.harmonics_x(pi)])

        hinv = np.linalg.inv(harms)

        hAp, hAx  = waveform_spin_template(0, 0.5)
        hBp, hBx  = waveform_spin_template(pi/4, 0.5)
        hCp, hCx  = waveform_spin_template(pi/2, 0.5)
        hDp, hDx  = waveform_spin_template(pi, 0.5)

        h_m2 = hinv[0,0]*hAp + hinv[0,1]*hBp + hinv[0,2]*hCp + hinv[0,3]*hCx + hinv[0,4]*hDx
        h_m1 = hinv[1,0]*hAp + hinv[1,1]*hBp + hinv[1,2]*hCp + hinv[1,3]*hCx + hinv[1,4]*hDx
        h0   = hinv[2,0]*hAp + hinv[2,1]*hBp + hinv[2,2]*hCp + hinv[2,3]*hCx + hinv[2,4]*hDx
        h1   = hinv[3,0]*hAp + hinv[3,1]*hBp + hinv[3,2]*hCp + hinv[3,3]*hCx + hinv[3,4]*hDx
        h2   = hinv[4,0]*hAp + hinv[4,1]*hBp + hinv[4,2]*hCp + hinv[4,3]*hCx + hinv[4,4]*hDx

        return h0, h1, h2, h_m1, h_m2





def harmonics_p(theta):
    return np.array([0.5*(1+cos(theta)**2),
                        -sin(2*theta),
                        3*sin(theta)**2,
                        sin(2*theta),
                        0.5*(1+cos(theta)**2)])



def harmonics_x(theta):
    return np.array([1.j*cos(theta),
                        -2.j*sin(theta),
                        0.,
                        -2.j*sin(theta),
                        -1.j*cos(theta)])





def isolated_harmonics(theta1, theta2, theta3):
    a = np.array(hf.Harmonics_A(theta1))
    b = np.array(hf.Harmonics_A(theta2))
    c = np.array(hf.Harmonics_A(theta3))
    
    matrix = np.array([[a[2],a[1],a[0]], [b[2],b[1],b[0]], [c[2],c[1],c[0]]])
    x = np.linalg.inv(matrix)
    
    hA = waveform_spin_j(0,0,0.5)
    hB = waveform_spin_j(np.pi/4,0,0.5)
    hC = waveform_spin_j(np.pi/2,0,0.5)
    
    h2 = hA
    h1 = x[1,0]*hA + x[1,1]*hB + x[1,2]*hC
    h0 = x[2,0]*hA + x[2,1]*hB + x[2,2]*hC
    
    return h2, h1, h0




def Harmonics_A_j(theta):
    
    """Harmonics function of theta"""
    
    m_0 = np.sqrt(3/2) * np.sin(theta)**2
    
    m_1 = np.cos(theta)*np.sin(theta) - 1.j*np.sin(theta)
    
    m_2 = 1/4*(3 + np.cos(2*theta)) - 1.j*np.cos(theta)
    
    # these are same for negative m
    
    return m_0, m_1, m_2




def Harmonics_A(theta):
    
    """Harmonics function of theta"""
    
    m_0 = np.sqrt(3/2) * np.sin(theta)**2
    
    m_1 = np.cos(theta)*np.sin(theta)
    
    m_2 = 1/4*(3 + np.cos(2*theta))
    
    # these are same for negative m
    
    return m_0, m_1, m_2





def Harmonics_B(beta):
    """Harmonics function of beta"""
    
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    
    m_0 = 1/4*np.sqrt(15/(2*np.pi))*sin_beta**2
    
    m_1 = 1/4*np.sqrt(5/np.pi)*(1 + cos_beta)*sin_beta
    
    m_2 = 1/2 * np.sqrt(5/np.pi) * ((1+cos_beta)/2)**2
    
    return m_0, m_1, m_2

    
    



def Beta(kappa,chi,frequency):
    

    G = 6.67430e-11
    c = 299792458

    m1 = 10
    m2 = 1.4
    mt = ((m1 + m2)*1.98e30*(G/c**3))
    #S = m1**2 * chi
    v = (np.pi*mt*frequency)**(1/3)
    #L = (m1*m2)/v
    gamma = (m1*chi*v)/m2

    beta = np.arccos((1 + kappa*gamma)/(np.sqrt(1 + 2*kappa*gamma + gamma**2)))
    
    return beta


def Beta_simple(beta):
    
    cos_beta = np.cos(beta)

    sin_beta = np.sqrt(1-np.cos(beta)**2)
    
    return sin_beta, cos_beta




def phase_diff(incl2,psi2,kappa2):
    wave1 = waveform_no_spin(0,0,1)
    wave2 = waveform_spin(incl2,psi2,kappa2)
    
    wave1_freq = wave1.to_frequencyseries()
    wave2_freq = wave2.to_frequencyseries()
    
    frequency = wave1_freq.sample_frequencies[15*256:300*256]
    
    phase_change = wave2_freq[15*256:300*256] *       np.conjugate(wave1_freq[15*256:300*256])
    
    sub_phase = np.unwrap(np.angle(phase_change))
    
    x = frequency**(-1/3)
    y = sub_phase
    p = np.polyfit(x,y,2)
    f = np.poly1d(p)
    
    phase_diff = y - f(x)
    
    return frequency, phase_diff




