#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:47:46 2020

@author: franciscoacosta
"""
import matplotlib.pyplot as plt
import numpy as np


def tridiag(a,b,c,k1 = 1, k2 = 0,k3 =-1):
    return np.diag(a,k1) + np.diag(b,k2)  + np.diag(c,k3)

## aperiodic interaction matrix with uniform coupling 
def interaction_matrix(N,k):
    upper_diag = k*np.ones(N-1)
    diag = np.zeros(N)
    lower_diag = k*np.ones(N-1)
    
    return tridiag(upper_diag,diag,lower_diag)

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
mu = 0.1
sigma = 0.1
T = 100
dt = 0.01

## Matrix of pair-wise oscillator couplings 
W = interaction_matrix(N,k)


## population: list of oscillators with uniform random phases and normal random frequencies
population = []
for i in range(N):
    phase_i = 2*np.pi*np.random.rand()
    freq_i = np.random.normal(mu,sigma)
    population.append(oscillator(phase_i,freq_i))
    
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





    
            

        
        
        
    

    





        





    
    

    
    








    
    





