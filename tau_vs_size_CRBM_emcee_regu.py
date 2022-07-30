# Computes autocorrelation time as a function
# of linear size of square Ising lattice

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import importlib
import emcee
# import ising_analysis
# importlib.reload(ising_analysis)

''' ------------------------- Define Functions (Start) ------------------------- '''

def autocorrelation(data):
    '''Computes normalized autocorrelation function of sample data for each time'''
    N = data.shape[0]
    _autocorrelation = np.zeros(N)
    for Δt in range(N-1): # let the time separation be all possible distances
        c0 = np.mean(data[:N - Δt]**2) - np.mean(data[:N - Δt])**2 #Variance at t0
        ct = np.mean(data[:N - Δt]*data[Δt:]) - np.mean(data[:N - Δt])*np.mean(data[Δt:]) # unnormalized autocorrelation fn.
        _autocorrelation[Δt] = ct/c0 # normalized autocorrelation function for this 'radius' (actually time separation)
    return _autocorrelation

def autocorrelation_function(time,scale,autocorrelation_time):
    '''exponential form of the autocorrelation function'''
    return scale*np.exp(-time/autocorrelation_time)

''' ------------------------- Define Functions (End) ------------------------- '''

# Set how much initial samples to discard
throwaway = 10000

# Set physical parameters
L_list = [2**i for i in range(5,7)]
T = 1.0
J1 = -1.0
J2 = 0.0

kernel_dims = [2,2]

# Open file for writing
filename = './data_nnn/L_many_T_'+str(T)+'_J1_'+str(J1)+'_J2_'+str(J2)+'_kernelDims_'+str(kernel_dims[0])+'-'+str(kernel_dims[1])+'_autocorr_regu.dat'

file = open(filename, "w")
header = "#L=many, T=%.2f \n# L    tau_E          tau_E_err          tau_M          tau_M_err\n"%(T)
file.write(header)

for L in L_list:
    # Load data 
    
    data_correlated = np.loadtxt("./data_nnn/L_"+str(L)+"_T_"+str(T)+"_J1_"+str(J1)+"_J2_"+str(J2)+"_kernelDims_"+str(kernel_dims[0])+"-"+str(kernel_dims[1])+"_no_symmetries_regu.dat")
    
    E_data = data_correlated[:,0][throwaway:]
    M_data = data_correlated[:,1][throwaway:]
    
    tau_E = emcee.autocorr.integrated_time(E_data)
    tau_M = emcee.autocorr.integrated_time(M_data)
    # tau_M = 1

    file.write('%d %.8f %.8f\n'%(L,tau_E,tau_M))


    print("L: {}, τ_E: {}, τ_M: {}".format(L,tau_E,tau_M)) # from fitting
    # print("4τ_auto: {}".format(4*popt[1]))
    
# Close file if finished sampling
file.close()