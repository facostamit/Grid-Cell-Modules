#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 9 2020
By Francisco Acosta

Processes Data from Kuramoto simulation (Kuramoto.py) and creates plots of relevant quantities

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


""".....................Helper Functions.........................."""

## Calculates numerical time derivative of oscillator phases at all times t 

def calc_eff_freq(system_t,dt):
    size = np.shape(system_t)
    eff_freqs_t = np.zeros((size[0]-2,size[1]))
    for t in range(1,size[0]-1):  
        eff_freqs = np.zeros(size[1])
        for i in range(size[1]):
            delta = (system_t[t+1,i]-system_t[t-1,i])
            #diff = min(delta,2*np.pi-delta)
            eff_freqs[i] = delta/(2*dt)
        eff_freqs_t[t-1,:] = eff_freqs.flatten() 
    return eff_freqs_t




## Calculate dominant frequency components through Fourier analysis
    
def fourier_freqs(system_t):
    
    dim1,dim2 = np.shape(system_t)
    
    n = int(dim1/4)
    
    freq_domain = []
    
    primary_freq = np.zeros((dim2,1))
    
    for i in range(dim2):
        freq_domain.append(np.abs(np.fft.rfft(system_t[-n:,i]-np.mean(system_t[-n:,i]))))
        peak_freqs, peak_info = find_peaks(freq_domain[-1],height = 0)
        
        freq = peak_freqs[np.argmax(peak_info["peak_heights"])]
        
        primary_freq[i] = freq
    
    freq_domain = np.array(freq_domain).T

    
    return (freq_domain,primary_freq)



## Calculates order parameter r (population phase-coherence)
def calc_order_parameter(phase_data):
    
    dims = np.shape(phase_data)
    n1 = dims[0]
    n2 = dims[1]
    
    r_t = np.zeros(n1)
    
    for t in range(n1):
        tot = 0
        for n in range(n2):
            tot += np.exp(1j*phase_data[t,n])
    
        r_t[t] = (1/n)*abs(tot)
    
    return r_t


def eff_freqs_std(eff_freqs):
    return np.std(eff_freqs)    



""".......................Load & Process Data........................"""


### Simulation parameters
params = np.loadtxt("simulation_params.dat")
N,freq_0,T,dt = params
#
### Phase evolution data
phase_data = np.loadtxt("phase_evolution.dat")
#
phases = np.loadtxt("total_phases.dat")
#
eff_freqs = calc_eff_freq(phases,dt)


# Simulation parameters
#params = np.loadtxt("KSsimulation_params.dat")
N,freq_0,T,dt = params

## Phase evolution data
#phase_data = np.loadtxt("KSphase_evolution.dat")

#phases = np.loadtxt("KStotal_phases.dat")

#eff_freqs = calc_eff_freq(phases,dt)



# keeps track of population effective frequencies in time
#eff_freqs = calc_eff_freq(phase_data)
    
## keeps track of standard deviation of effective frequencies in time
#freq_std_t = np.std(eff_freqs,axis=1)

## keeps track of system phase-coherence order parameter in time
#r_t = calc_order_parameter(phase_data)

## freq domain
#freq_domain = fourier_freqs(phase_data)
    

##2d pca
#from sklearn.decomposition import PCA
#eff_freqs.shape
#pca = PCA(n_components=2)
#pca.fit(eff_freqs)
#x = pca.transform(eff_freqs)
#x.shape
#
##3d pca
#import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#indices = np.arange(x.shape[0])
#plt.plot(x[0,1000:],x[1,1000:],x[2,1000:])

x = np.linspace(0,N,N)
y = np.mean(eff_freqs[20000:,:],axis=0)
yerr = np.std(eff_freqs[20000:,:],axis=0)

plt.errorbar(x,y,yerr,alpha = 0.3)


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    a = plt.figure(1,dpi=200)
    ax.plot(x, y, color=color,marker="o",figure=a)
    ax.fill_between(x, ymax, ymin, color="red", alpha=alpha_fill)
    ax.set_xlabel("Oscillator")
    ax.set_ylabel("⟨dθ/dt⟩")

errorfill(x,y,yerr,color = "black")

"........................Generate Plots....................."""


#plot1 = plt.figure(1,dpi=150)
#plt.plot(np.linspace(0,T,int(T/dt)),phase_data)
#plt.title("Population Phase Evolution")
#plt.xlabel("time")
#plt.ylabel("phase")
#plt.savefig("Phases.png")

#plot1 = plt.figure(1,dpi=150)
#plt.plot(phase_data)
#plt.title("Population Phase Evolution")
#plt.xlabel("time")
#plt.ylabel("phase")
#plt.savefig("Phases.png")


#plt2 = plt.figure(2,dpi=150)
#plt.plot(np.linspace(0,T,int(T/dt)),r_t)
#plt.title("Population Phase Coherence Evolution")
#plt.xlabel("time")
#plt.ylabel("phase coherence r")
#plt.ylim((0,1))
##plt.savefig("r.png")


#plot3 = plt.figure(3,dpi=150)
#plt.plot(np.linspace(1,T-1,int(T/dt)-2),eff_freqs)
#plt.title("Population Eff. Frequency Evolution")
#plt.xlabel("time")
#plt.ylabel("eff. freq.")

#plot3 = plt.figure(3,dpi=150)
#plt.plot(eff_freqs)
#plt.title("Population Eff. Frequency Evolution")
#plt.xlabel("time")
#plt.ylabel("eff. freq.")



#plot4 = plt.figure(4,dpi=100)
#plt.plot(np.linspace(1,T-1,int(T/dt)-2),freq_std_t)
#plt.title("Effective Frequency Standard Deviation Evolution")
#plt.xlabel("time")
#plt.ylabel("eff. freq. std")
#
#plt.show()

#bins = np.linspace(min(-freq_0,-1),max(freq_0,1),50)
#
#bins = np.linspace(1,6,80)
#
#plt.hist(eff_freqs[int(0.9*int(T/dt))],bins=bins)
#plt.title("Distribution of Effective frequencies at t = 0.9*T")
#plt.xlabel("frequency")
#plt.ylabel("number of oscillators")





#show eff. freq behavior during time (90,000 - 10,000) incrementing oscillator number


#for i in range(200):
#    plot = plt.figure()
#    plt.plot(eff_freqs[90000:100000,i])

    
#Why is this happening?!?!? Understand behavior of frequency
## look at 3D surface 


## formation of modules?!?!?! experiment with gradient steepness, coupling value.
# try delta_freq = 4, k = 2, no defects

#for i in range(0,120):
#    plot = plt.figure()
#    plt.hist(eff_freqs[int(0.01*i*int(T/dt))],bins=bins)
    
    
    
    
    






 
