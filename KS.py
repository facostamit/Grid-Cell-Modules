#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:04:05 2020

@author: franciscoacosta
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
import Kuramoto



def create_subpopulations(N_pop,N,freq_0,freq_std,delta_freq,gradient = "linear"):
    
    global_population = []
    
    for i in range(N_pop):
        if gradient == None:
            subpop_freq = freq_0 
        if gradient == "linear":
            subpop_freq = freq_0+(i/(N_pop-1))*delta_freq
        
        subpop = Kuramoto.create_population(N,subpop_freq,freq_std)
        
        global_population.append(subpop)
        
    
    return global_population



def update_populations(N_pop,N,Ws,alpha,g,total_phases,thetas,frequencies,dt):
    
    for j in range(N_pop):
        
        delta = dt*(frequencies[j]+np.cos(thetas[j]+alpha)*(Ws[j].dot(np.sin(thetas[j])))-np.sin(thetas[j]+alpha)*(Ws[j].dot(np.cos(thetas[j]))))
        if N_pop > 1:
            theta_mean1 = np.mean(thetas[j-1])
            delta += dt*(g*np.sin(theta_mean1-thetas[j]))
        if N_pop > 2:
            theta_mean2 = np.mean(thetas[(j+1)%N_pop])
            delta += dt*(g*np.sin(theta_mean2-thetas[j]))
        
        total_phases[j] += delta
        thetas[j] += delta
        thetas[j] = np.mod(thetas[j],2*np.pi)
        
        
    return (thetas, total_phases)


## models evolution of population for time T
def update_system(N_pop,N,Ws,alpha,g,total_phases,thetas,frequencies,T,dt):
    
    ## keeps track of population pattern (phases) in time 
    system_t  = np.zeros((int(T/dt),N*N_pop))
    
    system_t_total = np.zeros((int(T/dt),N*N_pop))

    
    for t in range(int(T/dt)):
        all_thetas = np.zeros(0)
        all_total_phases = np.zeros(0)
        for j in range(N_pop):
            all_thetas = np.append(all_thetas,thetas[j].flatten())
            all_total_phases = np.append(all_total_phases,total_phases[j].flatten())
        system_t[t,:] = all_thetas
        system_t_total[t,:] = all_total_phases

        thetas, total_phases = update_populations(N_pop,N,Ws,alpha,g,total_phases,thetas,frequencies,dt)
    
    return (system_t, system_t_total)




def simulate(N_pop,N,k,alpha,g,radius,periodic,defects,num_defects,freq_0,delta_freq,freq_std,gradient,T,dt):
    
    start_time = time.time()
    
    ## Matrix of pair-wise oscillator couplings 
    Ws = []
    for j in range(N_pop):
        W = Kuramoto.interaction_matrix(N,k,periodic = periodic,radius = radius)
        Ws.append(csr_matrix(W))
    
#    if defects:
#        W = introduce_defects(W,num_defects)
    
    ## global population of N_pop subpopulations of N oscillators each 
    global_population = create_subpopulations(N_pop,N,freq_0,freq_std,delta_freq,gradient)
    
    ## Nx1 vectors containing the phase and frequency of each oscillator 
    phases = []
    total_phases = []
    frequencies = []
    
    for j in range(N_pop):
        phases_j = np.array([[oscillator.phase for oscillator in global_population[j]]]).T
        phases.append(phases_j)
        total_phases_j = np.array([[oscillator.phase for oscillator in global_population[j]]]).T
        total_phases.append(total_phases_j)
        frequencies_j = np.array([[oscillator.frequency for oscillator in global_population[j]]]).T
        frequencies.append(frequencies_j)
    
    
    ##updates all oscillators 
    system_t, system_t_total = update_system(N_pop,N,Ws,alpha,g,total_phases,phases,frequencies,T,dt)
    
    simulation_time = time.time() - start_time

    np.savetxt('KSphase_evolution.dat',system_t)
    np.savetxt('KStotal_phases.dat',system_t_total)
    
    
    print(simulation_time)
    
    
  
""" ..........................Simulation................................. """

N_pop = 1

g = 0


alpha = 1.5

# N : number of oscillators
N = 50
# k : coupling constant
k = 0.1
# radius : radius of local interactions
radius = 1
# periodic : set to True for periodic topology, set to False for aperiodic topology
periodic = True
# defects : set to True to introduce sparse uniformly ditributed defects 
defects = False
# num_defects : specify number of defects to introduce 
num_defects = int(0.05*N)
# freq_0 : initial center of frequency distribution
freq_0 = 0
# delta_freq : absolute change in frequency due to gradient. final freq = freq_0 + delta_freq
delta_freq = 0.5
# freq_std : std of frequency distribution
freq_std = 0.00
# gradient : sets functional form of population frequency gradient. gradient ∈ {None,"linear","quadratic","exponential"}
gradient = None#"linear" 
# T : simulation time length
T = 88000
# dt : time step width
dt = 0.01

params = np.array([N,freq_0,T,dt])
np.savetxt('KSsimulation_params.dat',params)

simulate(N_pop,N,k,alpha,g,radius,periodic,defects,num_defects,freq_0,delta_freq,freq_std,gradient,T,dt)
    
    






















        
    