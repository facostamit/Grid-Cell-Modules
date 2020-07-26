
"""
Original MATLAB code by Ila Fiete and John Widloski (2014)
Adapted into Python by Francisco Acosta in 2020 
"""

import numpy as np



def compute_LNP_ionic_currents(g_L):
    
    '''
    This function computes the total ionic current (I_ion) based on:
    - ionic conductance amplitudes of the leak current (g_L)
    '''
    
    I_ion = g_L
    
    return I_ion



def compute_LNP_syn_currents(spk,s,dt,ind_EL,ind_ER,ind_I,G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I,tau_g):
    
    '''
     This function computes synaptic currents to all neurons (I_syn) and updates values of
     synaptic activations (s), based on:
        
    - current spike vector (spk) 
    - time step (dt)
    - synaptic activations from previous time step (s)
    - indices of neurons in each population (ind_EL, ind_ER, ind_I)
    - synaptic conductance amplitudes (G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I)
    - synaptic conductance time constants (tau_g)
    '''
    s_EL = s[ind_EL]
    s_ER = s[ind_ER]
    s_I = s[ind_I]
    
    ''' Update synaptic inputs into inhibitory pop. '''
    I_I_syn = G_I_EL*s_EL + G_I_ER*s_ER - G_I_I*s_I
    
    ''' Update synaptic inputs into excitatory pops. '''
    I_EL_syn = -G_EL_I*s_I
    I_ER_syn = -G_ER_I*s_I

    I_syn = np.array([[I_I_syn],[I_EL_syn],[I_ER_syn]])
    
    '''Update synaptic activations'''
    s_EL = s_EL + dt/tau_g*(-s_EL) + spk[ind_EL]
    s_ER = s_ER + dt/tau_g*(-s_ER) + spk[ind_ER]
    s_I = s_I + dt/tau_g*(-s_I) + spk[ind_I]

    s = np.array([[s_I],[s_EL],[s_ER]])
    
    return (I_syn,s)



def compute_LNP_output(dt,I_ion,I_syn,I_app,A_env):
    
    '''
     This function outputs the spike vector (spk), based on:
    
     - time step (dt)
     - ionic currents (I_ion)
     - synaptic currents (I_syn)
     - externally applied currents (I_app)
     - envelope function for suppressing edge neurons in aperiodic nets (A_env)
     '''
    
    '''total input current into all neurons'''
    I_in = -I_ion + A_env*(I_syn + I_app)
    
    '''apply threshold nonlinearity to input current'''
    I_in = I_in*(I_in>0)
    
    '''draw spikes with rate = I_in*dt'''
    spk = np.random.poisson(I_in*dt)
    
    return spk



def create_envelope(periodic,N):
    
    '''
    This function returns an envelope for network of size N; The envelope can
    either be suppressive (periodic = 0) or flat and equal to one (periodic = 1)
    '''
    
    kappa = 0.3     #controls width of main body of envelope
    a0 = 30     #controls steepness of envelope
    
    if not periodic:
        A = np.zeros((1,N))
        for m in range(1,N+1):
            r = abs(m-N/2)
            if r < kappa*N:
                A[0,m-1] = 1
            else:
                A[0,m-1] = np.exp(-a0*((r-kappa*N)/((1-kappa)*N))**2)
            
    else:
        A = np.ones((1,N))
        
    return A
    
    
    


    
        