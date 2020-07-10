"""
Created on Fri Jun 26 11:47:46 2020 
By Francisco Acosta

Simulates 1D Kuramoto model with local coupling

Things to add: (1) Animation of system evolution, (2) Play around with parameters: population size, coupling, freq distribution width,
(3) Explore non-uniform oscillator coupling (?) 

"""

import numpy as np
from scipy.sparse import csr_matrix
import time


""" ..........................Helper functions............................... """


## Creates NxN interaction matrix with local uniform coupling k. Aperiodic by default (set periodic = True to make periodic).
## radius determines range of local interactions. By deault radius = 1 (i.e., each oscilliator interacts only with its 2 closest neighbors)
def interaction_matrix(N,k,periodic = False, radius = 1):
   
    W = np.zeros((N,N))
    for i in range(-radius,radius+1):
        vec = (i!=0)*(k/(2**(abs(i)-1)))*np.ones(N-abs(i))
        W += np.diag(vec,i)

        if periodic:
            if i != 0:
                W[0,N-abs(i)] = k/(2**abs(i)-1)
                W[N-1,abs(i)-1] = k/(2**abs(i)-1)    
    return W

## defines oscillator object with attributes phase ([0,2π]) and frequency
class oscillator:
    def __init__(self,phase,frequency):
        self.phase = phase
        self.frequency = frequency



## creates population of oscillators with uniform random phases and frequencies drawn from a narrow normal distribution. 
## By default, uniform mean frequency (graditn = None). 
## set gradient = "linear","quadratic", or "exp" to introduce 
##      corresponding frequency gradient starting at freq_0 and ending at freq_final    
def create_population(N,freq_0,freq_std, gradient = "linear", delta_freq = 1):
    population = []

    for i in range(N):
        phase_i = 2*np.pi*np.random.rand()
        if gradient != None:
            if gradient == "linear":
                freq_i = freq_0+(i/(N-1))*delta_freq + np.random.normal(0,freq_std)
            if gradient == "quadratic":
                freq_i = freq_0 + (i/(N-1))**2*delta_freq+np.random.normal(0,freq_std)
            if gradient == "exponential":
                freq_i = freq_0 + (np.exp(i)-1)/(np.exp(N-1)-1)*delta_freq +np.random.normal(0,freq_std)
        else:
            freq_i = np.random.normal(freq_0,freq_std)
        population.append(oscillator(phase_i,freq_i))

    return population



## single time step update to oscillator population according to Kuramoto model 
def update_population(W,thetas,frequencies,dt):
    thetas += dt*(frequencies+np.cos(thetas)*(W.dot(np.sin(thetas)))-np.sin(thetas)*(W.dot(np.cos(thetas))))
    thetas = np.mod(thetas,2*np.pi)
    return thetas



## models evolution of population for time T
def update_system(W,thetas,frequencies,T,dt):
    
    ## keeps track of population pattern (phases) in time 
    system_t  = np.zeros((int(T/dt),N))
    
    ## keeps track of system phase-coherence order parameter in time
    r_t = np.zeros(int(T/dt))
    
    for iter in range(int(T/dt)):
        system_t[iter,:] = thetas.flatten()
        r_t[iter] = calc_order_parameter(thetas)
        thetas = update_population(W,thetas,frequencies,dt)
    
    return (system_t,r_t)



## Calculates numerical time derivative of oscillator phases at all times t 
def calc_eff_freq(system_t):
    size = np.shape(system_t)
    eff_freqs_t = np.zeros((size[0]-2,size[1]))
    for t in range(1,size[0]-1):  
        eff_freqs = np.zeros(size[1])
        for i in range(size[1]):
            delta = (system_t[t+1,i]-system_t[t-1,i])%(2*np.pi)
            diff = min(delta,2*np.pi-delta)
            eff_freqs[i] = diff/(2*dt)
        eff_freqs_t[t-1,:] = eff_freqs.flatten() 
    return eff_freqs_t



## Calculates order parameter r (population phase-coherence)
def calc_order_parameter(phases):
    n = len(phases)
    tot = 0
    for i in range(n):
        tot += np.exp(1j*phases[i])
    
    r = (1/n)*abs(tot)
    
    return r

def eff_freqs_std(eff_freqs):
    return np.std(eff_freqs)    


## Runs simulation. Produces plots. 
def simulate(N,k,radius,periodic,freq_0,delta_freq,freq_std,gradient,T,dt):
    
    start_time = time.time()
    
    ## Matrix of pair-wise oscillator couplings 
    W = interaction_matrix(N,k,periodic = periodic,radius = radius)
    
    ## population of N oscillators 
    population = create_population(N,freq_0,freq_std,gradient = gradient,delta_freq = delta_freq)
    
    ## Nx1 vectors containing the phase and frequency of each oscillator 
    phases = np.array([[oscillator.phase for oscillator in population]]).T
    frequencies = np.array([[oscillator.frequency for oscillator in population]]).T
    
    ## Interaction matrix in sparse format
    W = csr_matrix(W)
    
    ##updates all oscillators 
    system_t, r_t = update_system(W,phases,frequencies,T,dt)
    
    simulation_time = time.time() - start_time

    ## keeps track of population effective frequencies in time
    eff_freqs = calc_eff_freq(system_t)
    
    ## keeps track of standard deviation of effective frequencies in time
    freq_std_t = np.std(eff_freqs,axis=1)
    
    
    return (system_t,r_t,eff_freqs,freq_std_t,simulation_time)
    
    
""" ..........................Simulation................................. """


# N : number of oscillators
N = 100
# k : coupling constant
k = 1
# radius : radius of local interactions
radius = 5
# periodic : set to True for periodic topology, set to False for aperiodic topology
periodic = True
# freq_0 : initial center of frequency distribution
freq_0 = 0.0
# delta_freq : absolute change in frequency due to gradient. final freq = freq_0 + delta_freq
delta_freq = 1
# freq_std : std of frequency distribution
freq_std = 0.01
# gradient : sets functional form of population frequency gradient. gradient ∈ {None,"linear","quadratic","exponential"}
gradient = "linear"
# T : simulation time length
T = 1000
# dt : time step width
dt = 0.01

system_t,r_t,eff_freqs,freq_std_t,simulation_time = simulate(N,k,radius,periodic,freq_0,delta_freq,freq_std,gradient,T,dt)
    
np.savetxt('phase_evolution.dat',system_t)
np.savetxt('phase_coherence_evolution.dat',r_t)
np.savetxt('eff_freqs.dat',eff_freqs)
np.savetxt('freq_std_t',freq_std_t)
np.savetxt('simulation_time',np.array([simulation_time]))


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
    
    

    


        
    
            

        
        
        
    

    





        





    
    

    
    








    
    





