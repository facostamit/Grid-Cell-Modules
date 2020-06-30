#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:47:46 2020

@author: franciscoacosta
"""
import matplotlib.pyplot as plt
import numpy as np

## Creates simple Kuramoto model with uniform local coupling

#### Things to add: (1) Animation of system evolution, (2) Play around with parameters: population size, coupling, freq distribution width,
#### (3) Explore non-uniform oscillator coupling? ####

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

# N : number of oscillators
# k : coupling constant
# mu : center of frequency distribution
# sigma : std of frequency distribution
# T : simulation time length
# dt : time step width
N = 10
k = 0.3
freq_0 = 0.1
freq_std = 0.1
T = 100
dt = 0.01

## Matrix of pair-wise oscillator couplings 
W = interaction_matrix(N,k)


## creates population of oscillators with uniform random phases and frequencies drawn from a narrow normal distribution. By default, uniform mean frequency.
##      set freq_gradient = True to introduce linear frequency gradient starting at freq_0 and ending at freq_final

def create_population(N,freq_0,freq_std,freq_gradient = false,freq_final = 1):
    population = []

    for i in range(N):
        phase_i = 2*np.pi*np.random.rand()
        if freq_gradient:
            freq_i = np.random.normal(freq_0+(i/(N-1))*freq_final,freq_std)
        else:
            freq_i = np.random.normal(freq_0,freq_std)
        population.append(oscillator(phase_i,freq_i))

    return population

population = create_population(N,freq_0,freq_std)
    
## single time step update to individual oscillator according to Kuramoto model 
def update_oscillator(i,oscillator,dt):
    oscillator.phase += oscillator.frequency*dt
    for j in range(N):
        oscillator.phase += W[i,j]*np.sin(population[j].phase - oscillator.phase)*dt
    oscillator.phase = oscillator.phase%(2*np.pi)


## keeps track of population pattern in time 
phase_t  = np.zeros((int(T/dt),N))

## Models evolution of population for time T
def update(T,dt):
    for iter in range(int(T/dt)):
        for i in range(N):
            update_oscillator(i,population[i],dt)
            phase_t[iter,i] = population[i].phase
            #phase_t.append(population[i].phase)


update(T,dt)
plt.plot(np.linspace(0,T,round(T/dt)),phase_t)
plt.title("Population Phase Evolution")
plt.xlabel("time")
plt.ylabel("phase")





    
            

        
        
        
    

    





        





    
    

    
    








    
    





