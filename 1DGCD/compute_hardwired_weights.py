#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:20:28 2020

@author: franciscoacosta
"""


import numpy as np
from LNP_helper_functions import create_envelope


def compute_weights(rho,N_E,N_I,periodic):
    """
    rho: the scale of the synaptic profiles 
    (N_E, N_I): the size of the exctitatory and inhibitory pops 
    periodic:  boundary conditions of the network (periodic=1 for periodic b.c.s; periodic = 0 for aperiodic b.c.s)
    """

    
    """   
    The parameters below are arranged according to the following order of
    synaptic weights: EL->I, ER->I, I->EL, I->ER, I->I
    (see Supplementary Methods of PNAS paper for description of params below)
    """   
    weight_sizes = np.array([[N_I,N_E],[N_I,N_E],[N_E,N_I],[N_E,N_I],[N_I,N_I]])
    gamma_param = np.array([N_I/N_E,N_I/N_E,N_E/N_I,N_E/N_I,N_I/N_I])
    eta_param = np.array([1.5*21, 1.5*21, 8, 8, 24])          #controls overall strength of weights
    epsilon_param = np.array([0,0,0,0,1])                     #controls contribution of negative gaussian in diff. of Gaussians weights
    sigma_param = np.array([2,2,5,5,3])                       #controls width of weight profiles
    Delta_param = np.array([-2,2,8,-8,3])                     #controls weight asymmetries
    mu_param = np.array([0,0,-1,1,0])                         #controls weight asymmetries
    delta_param = np.array([0,0,3,3,3])                       #controls weight asymmetries
    
    
    #the for-loop below iterates through the 5 synaptic weight types:
    
    for k in range(0,5):
        
        #N_2 = size of projecting pop; N_1 = size of receiving pop.
        N_1 = weight_sizes[k,0]    
        N_2 = weight_sizes[k,1]
        
        #create envelopes based on pop. sizes
        A_1 = np.squeeze(create_envelope(periodic,N_1))
        A_2 = np.squeeze(create_envelope(periodic,N_2))
        
        #Create synaptic weight matrix
        G = np.zeros((N_1,N_2))
        
        for i in range(0,N_1):
            for j in range(0,N_2):
                
                x = (i+1) - gamma_param[k]*(j+1)
                
                
                c_left = min(N_1 - abs((x-Delta_param[k])%N_1),abs((x-Delta_param[k])%N_1))
                c_right = min(N_1 - abs((x+Delta_param[k])%N_1),abs((x+Delta_param[k])%N_1))
                c_0 = min(N_1-abs(x%N_1),abs(x%N_1))
                
                
                G[i,j] = eta_param[k]/rho*A_1[i]*A_2[j]*((c_0-delta_param[k]*rho)>=0)*(((-mu_param[k]*x)>=0)*((mu_param[k]*(x+mu_param[k]*N_1/2))>=0) + 
                  ((mu_param[k]*(x-mu_param[k]*N_1/2))>=0))*(np.exp(-c_left**2/(2*(sigma_param[k]*rho)**2)) + 
                  epsilon_param[k]*np.exp(-c_right**2/(2*(sigma_param[k]*rho)**2)));
                
        if k==1:
            G_I_EL = G
        elif k==2:
            G_I_ER = G
        elif k==3:
            G_EL_I = G
        elif k==4:
            G_ER_I = G
        else:
            G_I_I = G
            
        
    return G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I



#import os

#"""
#rho: the scale of the synaptic profiles 
#(N_E, N_I): the size of the exctitatory and inhibitory pops 
#periodic:  boundary conditions of the network (periodic=1 for periodic b.c.s; periodic = 0 for aperiodic b.c.s)
#"""

#rho = 2.2
#N_E = 400
#N_I = 160
#periodic = True

#G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I = compute_weights(rho,N_E,N_I,periodic)

#path = os.getcwd()
#os.mkdir(path + '/hardwired_weights')
#
#np.savetxt(str(path + "/hardwired_weights/1DGCD_G_I_EL.dat"),G_I_EL)
#np.savetxt(str(path + "/hardwired_weights/1DGCD_G_I_ER.dat"),G_I_ER)
#np.savetxt(str(path + "/hardwired_weights/1DGCD_G_EL_I.dat"),G_EL_I)
#np.savetxt(str(path + "/hardwired_weights/1DGCD_G_ER_I.dat"),G_ER_I)
#np.savetxt(str(path + "/hardwired_weights/1DGCD_G_I_I.dat"),G_I_I)


    
                 
                
    
    
    
    
    
    
    
    
    