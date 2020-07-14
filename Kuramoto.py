"""
Created on Fri Jun 26 2020 
By Francisco Acosta

Simulates 1D Kuramoto model with local coupling

saves output data as array of phase evolutions in "phase_evolution.dat"

Things to add: (1) Play around with parameters: population size, coupling, freq distribution width,
(2) Explore non-uniform oscillator coupling (?) 

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
               for j in range(abs(i)):
                W[j,N-abs(i)+j] = k/(2**(abs(i)-1))
                W[N-1-j,abs(i)-1-j] = k/(2**(abs(i)-1))
    
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


## introduces a total of num_defects defects into population.
## Each defect is implemented as an oscillator that is cut off from interacting with the rest

def introduce_defects(W,num_defects,loc_variability = False):
    
    N = np.shape(W)[0]
    indices = np.linspace(0,N-1,num_defects)
    
    for i in indices:
        
        if loc_variability:
            x = int(np.random.normal(float(i),2))
        else:
            x = int(i)
        
        W[:,x] = np.zeros(N)
        W[x,:] = np.zeros(N)
    
    return W



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
    #r_t = np.zeros(int(T/dt))
    
    for iter in range(int(T/dt)):
        system_t[iter,:] = thetas.flatten()
        #r_t[iter] = calc_order_parameter(thetas)
        thetas = update_population(W,thetas,frequencies,dt)
    
    return system_t #,r_t)



## Runs simulation 
def simulate(N,k,radius,periodic,defects,num_defects,freq_0,delta_freq,freq_std,gradient,T,dt):
    
    start_time = time.time()
    
    ## Matrix of pair-wise oscillator couplings 
    W = interaction_matrix(N,k,periodic = periodic,radius = radius)
    
    if defects:
        W = introduce_defects(W,num_defects)
    
    ## population of N oscillators 
    population = create_population(N,freq_0,freq_std,gradient = gradient,delta_freq = delta_freq)
    
    ## Nx1 vectors containing the phase and frequency of each oscillator 
    phases = np.array([[oscillator.phase for oscillator in population]]).T
    frequencies = np.array([[oscillator.frequency for oscillator in population]]).T
    
    ## Interaction matrix in sparse format
    W = csr_matrix(W)
    
    ##updates all oscillators 
    system_t = update_system(W,phases,frequencies,T,dt)
    
    simulation_time = time.time() - start_time

    np.savetxt('phase_evolution.dat',system_t)
    
    
    print(simulation_time)
    
  
    

    
    
""" ..........................Simulation................................. """


# N : number of oscillators
N = 100
# k : coupling constant
k = 4
# radius : radius of local interactions
radius = 6
# periodic : set to True for periodic topology, set to False for aperiodic topology
periodic = True
# defects : set to True to introduce sparse uniformly ditributed defects 
defects = False
# num_defects : specify number of defects to introduce 
num_defects = int(0.05*N)
# freq_0 : initial center of frequency distribution
freq_0 = 0
# delta_freq : absolute change in frequency due to gradient. final freq = freq_0 + delta_freq
delta_freq = 6
# freq_std : std of frequency distribution
freq_std = 0.01
# gradient : sets functional form of population frequency gradient. gradient ∈ {None,"linear","quadratic","exponential"}
gradient = "linear" 
# T : simulation time length
T = 1000
# dt : time step width
dt = 0.01

params = np.array([N,freq_0,T,dt])
np.savetxt('simulation_params.dat',params)

simulate(N,k,radius,periodic,defects,num_defects,freq_0,delta_freq,freq_std,gradient,T,dt)












    
    

    


        
    
            

        
        
        
    

    





        





    
    

    
    








    
    





