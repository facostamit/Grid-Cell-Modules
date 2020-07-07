"""
Created on Fri Jun 26 11:47:46 2020 
By Francisco Acosta

Simulates 1D Kuramoto model with local coupling

Things to add: (1) Animation of system evolution, (2) Play around with parameters: population size, coupling, freq distribution width,
(3) Explore non-uniform oscillator coupling (?) 

"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time


""" ..........................Helper functions............................... """

def tridiag(a,b,c,k1 = 1, k2 = 0,k3 =-1):
    return np.diag(a,k1) + np.diag(b,k2)  + np.diag(c,k3)


## NxN interaction matrix with uniform coupling k. Aperiodic by default (set periodic = True to make periodic) 
def interaction_matrix(N,k,periodic = False):
    upper_diag = k*np.ones(N-1)
    diag = np.zeros(N)
    lower_diag = k*np.ones(N-1)
    W = tridiag(upper_diag,diag,lower_diag)
    if periodic:
        W[0,N-1] = k
        W[N-1,0] = k    
    
    return W

## defines oscillator object with attributes phase ([0,2π]) and frequency
class oscillator:
    def __init__(self,phase,frequency):
        self.phase = phase
        self.frequency = frequency
    
        
## creates population of oscillators with uniform random phases and frequencies drawn from a narrow normal distribution. 
## By default, uniform mean frequency.
## set freq_gradient = True to introduce linear frequency gradient starting at freq_0 and ending at freq_final    
def create_population(N,freq_0,freq_std,freq_gradient = False, gradient = "linear", delta_freq = 1):
    population = []

    for i in range(N):
        phase_i = 2*np.pi*np.random.rand()
        if freq_gradient:
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


start_time = time.time()


""" ..........................Simulation................................. """

# N : number of oscillators
# k : coupling constant
# mu : center of frequency distribution
# sigma : std of frequency distribution
# T : simulation time length
# dt : time step width
N = 100
k = 0.5
freq_0 = 0.0
freq_std = 0.01
T = 200
dt = 0.01


## Matrix of pair-wise oscillator couplings 
W = interaction_matrix(N,k,periodic = True)

## population of N oscillators 
population = create_population(N,freq_0,freq_std,freq_gradient=True)

## keeps track of population pattern (phases) in time 
system_t  = np.zeros((int(T/dt),N))

## keeps track of system phase-coherence order parameter in time
r_t = np.zeros(int(T/dt))


## Nx1 vectors containing the phase and frequency of each oscillator 
phases = np.array([[oscillator.phase for oscillator in population]]).T
frequencies = np.array([[oscillator.frequency for oscillator in population]]).T

## Interaction matrix in sparse format
W = csr_matrix(W)


## models evolution of population for time T
def update(W,thetas,frequencies,T,dt):
    for iter in range(int(T/dt)):
        system_t[iter,:] = thetas.flatten()
        r_t[iter] = calc_order_parameter(thetas)
        thetas = update_population(W,thetas,frequencies,dt)
        

update(W,phases,frequencies,T,dt)


## keeps track of population effective frequencies in time
eff_freqs = calc_eff_freq(system_t)

## keeps track of standard deviation of effective frequencies in time
freq_std_t = np.std(eff_freqs,axis=1)




## Plots system & order parameter evolving in time
plot1 = plt.figure(1,dpi=150)
plt.plot(np.linspace(0,T,int(T/dt)),system_t)
plt.title("Population Phase Evolution")
plt.xlabel("time")
plt.ylabel("phase")
#plt.savefig("Phases.png")

plt2 = plt.figure(2,dpi=150)
plt.plot(np.linspace(0,T,int(T/dt)),r_t)
plt.title("Population Phase Coherence Evolution")
plt.xlabel("time")
plt.ylabel("phase coherence r")
plt.ylim((0,1))
#plt.savefig("r.png")


plot3 = plt.figure(3,dpi=150)
plt.plot(np.linspace(1,T-1,int(T/dt)-2),eff_freqs)
plt.title("Population Eff. Frequency Evolution")
plt.xlabel("time")
plt.ylabel("eff. freq.")

plot4 = plt.figure(4,dpi=100)
plt.plot(np.linspace(1,T-1,int(T/dt)-2),freq_std_t)
plt.title("Effective Frequency Standard Deviation Evolution")
plt.xlabel("time")
plt.ylabel("eff. freq. std")

plt.show()

bins = np.linspace(min(-freq_0,-1),max(freq_0,1),50)

plt.hist(eff_freqs[int(0.9*np.shape(system_t)[0])],bins=bins)
plt.title("Distribution of Effective frequencies at t = 0.9*T")
plt.xlabel("frequency")
plt.ylabel("proportion of oscillators")



## Different plots

#fig, axs = plt.subplots(2,sharex = True)
#fig.suptitle("Population Phase Evolution")
#axs[0].plot(np.linspace(0,T,int(T/dt)), system_t)
#axs[0].set(ylabel = "Phases")
#axs[1].plot(np.linspace(0,T,int(T/dt)), r_t)
#a#xs[0].ylabel("phase")


#newpop = create_population(N,0,0.1)
#
#phases = [x.phase for x in newpop]
#
#freqs = [x.frequency for x in newpop]


tot_time = time.time()-start_time

print(tot_time)



        
    
            

        
        
        
    

    





        





    
    

    
    








    
    





