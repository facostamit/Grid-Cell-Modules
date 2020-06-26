#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix


#network parameters
N = 1000
periodic = True
m = 1;              #CV = 1/sqrt(m)
x_prefs = np.arange(1,N+1)/N; #inherited location preferences (m)
spiking= False

#FF input
beta_vel = 1.5;      #velocity gain
beta_0 = 70;         #uniform input
alpha = 1000;        #weight imbalance parameter
gamma = 1.05/100;    #Cennter Surround weight params
beta = 1/100;        #Cennter Surround weight params

#temporal parameters
eta = 10**(-6)
T = 10;            #length of integration time blocks (s)
dt = 1/2000;        #step size of numerical integration (s)
#tau_s = np.concatenate((np.linspace(1,2,N)*1000/30,np.linspace(1,2,N)*1000/30),axis=0);    #synaptic time constant (s)
#tau_s = np.expand_dims(tau_s,axis=1)


tau_s = 1000/30;

def period_calc(x,sigma):
    x_g1d = ndimage.gaussian_filter1d(x, sigma)
    peaks,_ = find_peaks(x_g1d)
    if(len(peaks)==0):
        return 0,0
    diifs = np.zeros(len(peaks)-1)
    for i in range(0,len(diifs)):
        diifs[i] = peaks[i+1]-peaks[i]
    return np.mean(diifs),x_g1d


#gradient of velocity feedforward input (applied on beta_vel)
grad = np.linspace(.1,10,N)
#grad = np.linspace(1,1,N)

#Graphing parameters
bins = np.linspace(0+.01,1-.01,50);


# Trajectory Data (Sinusoidal)
x = (np.sin(np.linspace(dt,T,int(T/dt))*2*np.pi/10)+1)/2;
v= np.zeros((int(T/dt)));
for i in range(0,int(T/dt)):
    v[i] = (x[i]-x[i-1])/dt;
    
v = np.ones(int(T/dt))*0.2

z = np.linspace(-N/2,N/2-1,N);
z = np.expand_dims(z,1);

# Feed forward network input
if (periodic == 0):
    # gaussian FF input for aperiodic network
    envelope = np.exp(-4*(z/(800))**2);
else:
    envelope = np.ones((N,1));


s_prev = np.zeros((2*N,1));  #Population activity
spk = np.zeros((2*N,int(T/dt)));  #Total spiking
spk_count = np.zeros((2*N,1)); #Current spiking

# Weight setup


w_RR = np.zeros((N,N));
w_LL = np.zeros((N,N));
w_RL = np.zeros((N,N));
w_LR = np.zeros((N,N));


W_RR = np.zeros((N,N));
W_LL = np.zeros((N,N));
W_RL = np.zeros((N,N));
W_LR = np.zeros((N,N));
Rates = np.zeros((int(T/dt),2*N))

for i in range(0,N):
    crossSection = alpha*(np.exp(-gamma*(grad[i])*(z**2))-np.exp(-beta*(grad[i])*(z**2))).flatten();
    crossSection = -17.5*(crossSection/np.min(crossSection))
    crossSection = np.roll(crossSection, int(N/2));

    
    
    W_RR[i,:] =  np.roll(crossSection,[0, i - 1]); # Right neurons to Right neurons
    W_LL[i,:] =  np.roll(crossSection,[0, i + 1]); # Left neurons to Left neurons
    W_RL[i,:] =  np.roll(crossSection,[0, i]);     # Left neurons to Right neurons
    W_LR[i,:] =  np.roll(crossSection,[0, i]);     # Right neurons to Left neurons


defects = np.random.randint(1,999,int(0.1*N))



Total_W_RR = W_RR + w_RR
Total_W_LL = W_LL + w_LL
Total_W_RL = W_RL + w_RL
Total_W_LR = W_LR + w_LR

## eliminate negligibly small nonzero elements 
def increase_sparsity(M,cutoff):
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            if np.abs(M[i][j]) < cutoff:
                M[i][j] = 0
    return M


## cutoff of element magnitude for increasing sparsity
cutoff = 0.001


## increase spar
Total_W_RR = csr_matrix(increase_sparsity(Total_W_RR,cutoff))
Total_W_LL = csr_matrix(increase_sparsity(Total_W_LL,cutoff))
Total_W_RL = csr_matrix(increase_sparsity(Total_W_RL,cutoff))
Total_W_LR = csr_matrix(increase_sparsity(Total_W_LR,cutoff))



#spiking dynamics
for t in range(1,int(T/dt)):

        s_prev[defects,0] = 0
        s_prev[defects+1000,0] = 0

        #LEFT population
        v_L = np.array([1 - beta_vel*v[t]])
          
        g_LL = Total_W_LL.dot(s_prev[0:N])                          #L->L
        g_LR = Total_W_LR.dot(s_prev[N:2*N])                        #R->L

        
        
        
        G_L = ((g_LL + g_LR) + v_L*envelope*beta_0);              #input conductance into Left population
        #G_L = v_L*((g_LL + g_LR) + envelope*beta_0);   

        #RIGHT population
        v_R = np.array([1  + beta_vel*v[t]]);
        
        g_RR = Total_W_RR.dot(s_prev[N:2*N])                        #R->R
        g_RL = Total_W_RL.dot(s_prev[0:N])                          #L->R

        
        G_R = ((g_RR + g_RL) + v_R*envelope*beta_0);              #input conductance into Right population
        #G_R = v_R*((g_RR + g_RL) + envelope*beta_0);

        G = np.concatenate([G_L,G_R]);
        R = np.zeros((2*N,1)) + G*(G>0);     #ReLU transfer function
        R[defects,0] = 0
        R[defects+1000,0] = 0

        #spiking model
        if(spiking==True):
            #subdivide interval m times
            spk_sub = np.random.poisson(np.repeat(R,m,1)*dt)
            spk_count = spk_count + np.expand_dims(np.sum(spk_sub,1),axis=1)
            spk[:,t] = np.floor(spk_count.flatten()/m)
            spk_count = np.remainder(spk_count.flatten(),m)
            spk_count  = np.expand_dims(spk_count,1)
    
            #update population activity
            s_new = s_prev + np.expand_dims(spk[:,t],1) - s_prev*dt*tau_s
        
        
        else:
            #Rate based model
            s_new = s_prev + (-tau_s*s_prev + R)*dt
                        
        s_prev = s_new
        Rates[t,:] = R[0:2*N].flatten()
        

periods = np.zeros((N,1))        
for i in range(0,N):
    periods[i,0],_ = period_calc(Rates[:,i],100)
    
plt.scatter(np.linspace(1,N,N),periods)
        
        #hebbian learning
        #w_LL += eta*(np.outer(R[0:N],s_prev[0:N]) - np.tril(np.outer(R[0:N],R[0:N]),k=0)@w_LL)
        #w_LR += eta*(np.outer(R[N:2*N],s_prev[0:N]) - np.tril(np.outer(R[N:2*N],R[N:2*N]),k=0)@w_LR)
        #w_RL += eta*(np.outer(R[0:N],s_prev[N:2*N]) - np.tril(np.outer(R[0:N],R[0:N]),k=0)@w_RL)
        #w_RR += eta*(np.outer(R[N:2*N],s_prev[N:2*N]) - np.tril(np.outer(R[N:2*N],R[N:2*N]),k=0)@w_RR)
        
        

'''
        if(np.mod(t,100)==0):
            
            plt.plot(x_prefs,R[0:N],'r')
            plt.title('Population Response')
            plt.show()
            
            
            fig = plt.figure(figsize=(10,4))
        
            plt.subplot(221)
            plt.plot(x_prefs,W_RR[:,int(N/2)],'r');plt.plot(x_prefs,W_LL[:,int(N/2)],'b')
            plt.title('')
            
            plt.subplot(222)
            plt.plot(x_prefs,R[0:N],'r')
            plt.title('Population Response')
            #plt.xlabel()
            #plt.ylabel()
            
            plt.subplot(223)
            plt.plot(x_prefs,np.exp(-(x_prefs - x[t])**2/0.001**2)/max(np.exp(-(x_prefs - x[t])**2/.001**2)),'g')
            plt.title('')
            
            plt.subplot(224)
            
            
            plt.tight_layout()
            plt.show()
        
        '''







