#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:19:37 2020

@author: franciscoacosta
"""
import math
import numpy as np
#from scipy.optimize import minimize



#params:
#    N = 100
#    k = 0.12
#    alpha = 0.6
# delta_freq = 6

def find_uc(freqs,inc = 0.0000001,tolerance = 0.001, u0 = 0.01):
    
    N = len(freqs)
    
    def cost(u):
        tot = 0
        for j in range(N):
            tot += 2*np.sqrt(1-(freqs[j][0]/u)**2) - 1/(np.sqrt(1-(freqs[j][0]/u)**2))
        return tot
    
    u = u0

    y = cost(u)

    while math.isnan(abs(y)) or abs(y) > tolerance:
        u += inc
        y = cost(u)
        if u > 2:
            return "failure"
    
    return u



def compute_Kc(freqs):
    
    N = len(freqs)
    uc = find_uc(freqs)
    
    denom = 0
    
    for j in range(N):
        denom += (1/N)*np.sqrt(1-(freqs[j][0]/uc)**2)
    
    return uc/denom




    
    
    
    








    
