
import numpy as np
from LNP_helper_functions import create_envelope


def hcat(x,y):
    return np.concatenate((x,y),axis=0)

def zeros(size):
    return np.zeros((size))

def gridcelldevelopment1D():
     
    T_long = 14400
    T = 10
    tau_s = 30/1000
    dt = 1/2000
    
    N_E = 200
    N_I = 80
    
    N_tot = N_I + 2*N_E
    n_I = np.arange(0,N_I)
    n_EL = np.arange(N_I,N_I+N_E) 
    n_ER = np.arange(N_I+N_E,N_I+2*N_E)
    
    #external input 
    beta_vel = 0.9;     #velocity gain
    sigma_LS = 0.01;    #width of gaussian visual input (m)
    beta_LS_E = 10;     #magnitude of gaussian visual input to E pop
    beta_LS_I = 50;     #magnitude of gaussian visual input to I pop
    
    x_prefs_I = np.arange(1,N_I+1)/N_I; #location preferences of I pop
    x_prefs_E = np.arange(1,N_E+1)/N_E;
    
    #Suppressive envelope
    periodic = 0; #set to 0 for aperiodic net
    A_I = create_envelope(periodic,N_I) 
    A_E = create_envelope(periodic,N_E) 
    
    #STDP parameters
    l=0.1;          #length of STDP kernel (s)
    tau_STDP = 0.006;   #STDP time constant
    
    #STDP kernel for excitatory synapses
    K1_pos = 1.2*np.exp(-np.arange(start = l-dt,stop = -dt,step = -dt)/(4*tau_STDP));
    K1_neg = -1*np.exp(-np.arange(start = l-dt,stop = -dt,step = -dt)/(3*tau_STDP));
    
    #STDP kernel for inhibitory synapses
    K2_pos = 0.5*np.exp(-np.arange(start = l-dt,stop = -dt,step = -dt)/(4*tau_STDP));
    K2_neg = -np.exp(-np.arange(start = l-dt,stop = -dt,step = -dt)/(2*tau_STDP));
    
    #learning rates
    eta = 0.015;        #base learning rate factor 
    eta_II = 7*eta;     #I->I learning rate
    eta_IE = eta;       #E->I learning rate
    eta_EI = 2*eta;     #I->E learning rate
    
    #Initial weights
    g0 = 1e-3;        #scales initial size of weights
    G_I_EL = g0*(2*np.random.rand(N_I,N_E)-1);
    G_I_ER = g0*(2*np.random.rand(N_I,N_E)-1);
    G_EL_I = g0*(2*np.random.rand(N_E,N_I)-1);
    G_ER_I = g0*(2*np.random.rand(N_E,N_I)-1);
    G_I_I = g0*(2*np.random.rand(N_I,N_I)-1);
    
    
    #Sinusoidal animal trajectory for training
    x = (np.sin(np.arange(start = dt,stop = T+dt,step = dt)*2*np.pi/(T/2)-np.pi)+1)/2;
    v = np.diff(x)/dt;
    v = hcat([v[0]],v);
    
    #
    #DYNAMICS
    #
    NT = int(T/dt)
            
    for tt in range(0,T_long//T): #OUTER time loop (T_long sec)
        
        #initialize variables
        if tt==0:
            spk_prev = zeros((N_tot,int(l//dt)))
            s_prev = zeros((N_tot,1))
        
        spk = zeros((N_tot,NT));
    
        for t in range(1,NT): #INNER time loop (T sec)
        
            #Neural dynamics
            
        
            #I pop
            g_F_I = beta_LS_I*np.exp(-(x_prefs_I - x[t])**2/(2*sigma_LS)**2);   #loc-spec inputs
            G_I = A_I*g_F_I;                                               #total input conductance (no recurrents)
            
            #EL pop
            g_F_E = beta_LS_E*np.exp(-(x_prefs_E - x[t])**2/(2*sigma_LS)**2);   #loc-spec inputs
            v_L = (1 - beta_vel*v[t]); v_L = v_L*(v_L>0);                   #velocity input
            G_EL = v_L*A_E*g_F_E;                                          #total input conductance (no recurrents)
            
            #ER pop
            g_F_E = beta_LS_E*np.exp(-(x_prefs_E - x[t])**2/(2*sigma_LS)**2);   #loc-spec inputs
            v_R = (1 + beta_vel*v[t]); v_R = v_R*(v_R>0);                   #velocity input
            G_ER = v_R*A_E*g_F_E;                                          #total input conductance (no recurrents)           
            
        #pass conductance variables through nonlinearity to generate output rates, F
            G = np.expand_dims(np.concatenate((G_I,G_EL,G_ER),axis=1),axis=1)                             
            F = G*(G>=0);   #linear transfer function
        
        #subdivide interval m times and generate spikes (spk variable)
            spk[:,t] = np.random.poisson(F*dt);
            
        #update neural activation variables
            s_new = s_prev + spk[:,t] - s_prev*dt/tau_s; 
            s_prev = s_new;
    
        #Synaptic dynamics
              
            
        #Convolution of spks with STPD kernals
            if(t>=l/dt):    #here, l/dt is the size of the kernel
                
                spk_conv_1_pos = np.expand_dims(spk[:,t-int(l/dt):t].dot(K1_pos),axis=1)
                spk_conv_1_neg = np.expand_dims(spk[:,t-int(l/dt):t].dot(K1_neg),axis=1)
                spk_conv_2_pos = np.expand_dims(spk[:,t-int(l/dt):t].dot(K2_pos),axis=1)
                spk_conv_2_neg = np.expand_dims(spk[:,t-int(l/dt):t].dot(K2_neg),axis=1)
                
            else:              #if t<l/dt, must go back to previous spk matrix
                spk_temp = np.concatenate((spk_prev[:,t:],spk[:,0:t]),axis=1)
                spk_conv_1_pos = np.expand_dims(spk_temp.dot(K1_pos),axis=1) ##spk_temp*K1_pos.T
                spk_conv_1_neg = np.expand_dims(spk_temp.dot(K1_neg),axis=1)
                spk_conv_2_pos = np.expand_dims(spk_temp.dot(K2_pos),axis=1)
                spk_conv_2_neg = np.expand_dims(spk_temp.dot(K2_neg),axis=1)
            
            #compute weight updates
            delG_I_EL = np.dot(np.expand_dims(spk[n_I,t],axis=1),spk_conv_1_pos[n_EL].T) + np.dot(spk_conv_1_neg[n_I],np.expand_dims(spk[n_EL,t],axis=1).T);
            delG_I_ER = np.dot(np.expand_dims(spk[n_I,t],axis=1),spk_conv_1_pos[n_ER].T) + np.dot(spk_conv_1_neg[n_I],np.expand_dims(spk[n_ER,t],axis=1).T);
            delG_EL_I = np.dot(np.expand_dims(spk[n_EL,t],axis=1),spk_conv_2_neg[n_I].T) + np.dot(spk_conv_2_pos[n_EL],np.expand_dims(spk[n_I,t],axis=1).T);
            delG_ER_I = np.dot(np.expand_dims(spk[n_ER,t],axis=1),spk_conv_2_neg[n_I].T) + np.dot(spk_conv_2_pos[n_ER],np.expand_dims(spk[n_I,t],axis=1).T);
            
            delG_I_I = np.dot(np.expand_dims(spk[n_I,t],axis=1),spk_conv_2_neg[n_I].T) + np.dot(spk_conv_2_pos[n_I],np.expand_dims(spk[n_I,t],axis=1).T);
    
        #update weight matrices
            G_I_EL = G_I_EL + eta_IE*(delG_I_EL);
            G_I_ER = G_I_ER + eta_IE*(delG_I_ER);
            G_EL_I = G_EL_I + eta_EI*(delG_EL_I);
            G_ER_I = G_ER_I + eta_EI*(delG_ER_I);
            G_I_I = G_I_I + eta_II*(delG_I_I);

        #enforce Dale's law
            G_I_EL = G_I_EL*(G_I_EL>0);
            G_I_ER = G_I_ER*(G_I_ER>0);
            G_EL_I = G_EL_I*(G_EL_I>0);
            G_ER_I = G_ER_I*(G_ER_I>0);
            G_I_I = G_I_I*(G_I_I>0);
    
    return G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I



G_I_EL,G_I_ER,G_EL_I,G_ER_I,G_I_I = gridcelldevelopment1D()

np.savetxt("1DGCD_G_I_EL.dat",G_I_EL)
np.savetxt("1DGCD_G_I_ER.dat",G_I_ER)
np.savetxt("1DGCD_G_EL_I.dat",G_EL_I)
np.savetxt("1DGCD_G_ER_I.dat",G_ER_I)
np.savetxt("1DGCD_G_I_I.dat",G_I_I)

























