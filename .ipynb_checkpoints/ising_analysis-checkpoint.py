# Useful functions for analysis of Ising model data in 2D

import numpy as np
from scipy.special import ellipk

def get_energy(ising_lattice, L):
    "Computes energy of LxL Ising configuration"

    E = 0
    for i in range(L):
        for j in range(L):
            # Accumulate 'horizontal' bond
            if i != L-1:
                E -= ising_lattice[i,j]*ising_lattice[i+1,j] 
            else:
                E -= ising_lattice[i,j]*ising_lattice[0,j]
            
             
            # Accumulate 'vertical' bond
            if j != L-1:
                E -= ising_lattice[i,j]*ising_lattice[i,j+1] 
            else:
                E -= ising_lattice[i,j]*ising_lattice[i,0]
    
    return E

"------------------------------------------------------------------------------"

def get_magnetization(ising_lattice, L):
    "Computes magnetization of LxL Ising configuration"

    M = 0
    for i in range(L):
        for j in range(L):
            M += ising_lattice[i,j]
    return M

"------------------------------------------------------------------------------"

def lattice_unflatten(ising_lattice):
    "If configuration is flattened, make it 2D"
    
    # won't work with torch.tensor objects (need to use detach())
    L = ising_lattice.shape[1]
    return ising_lattice.reshape((L,L))

"------------------------------------------------------------------------------"

def zeros_to_minus_ones(ising_lattice):
    "Replace 0 -> -1"
    
    ising_lattice[ising_lattice==0] = -1
    return ising_lattice

"------------------------------------------------------------------------------"

def minus_ones_to_zeros(ising_lattice):
    "Replace -1 -> 0"
    
    ising_lattice[ising_lattice==-1] = 0
    return ising_lattice 

"------------------------------------------------------------------------------"

def G(ising_lattice,r):
    "Compute spin-spin correlation function at r sites away"
    
    L = ising_lattice.shape[0]
    N = L**2
    
    G_accum = np.ones(N)
    idx = 0
    for i in range(L):
        for j in range(L):
            
            σ_0 = ising_lattice[i,j]
            
            σ_0σ_right = σ_0*ising_lattice[i,(j+r)%L]
            σ_0σ_left = σ_0*ising_lattice[i,(j-r)%L]
            σ_0σ_down = σ_0*ising_lattice[(i+r)%L,j]
            σ_0σ_up = σ_0*ising_lattice[(i-r)%L,j]
            
            G_accum[idx] = (σ_0σ_right+σ_0σ_left+σ_0σ_down+σ_0σ_up)/4
            idx += 1

    G_average = np.mean(G_accum)
    G_stderr = np.std(G_accum)/np.sqrt(N)
    
    return G_average,G_stderr

"------------------------------------------------------------------------------"

def e_exact_func(T):
    K = 1/T
    q = 2*np.sinh(2*K)/np.cosh(2*K)**2
    K1 = ellipk(q*q)
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*K1)

"------------------------------------------------------------------------------"

def m_exact_func(T):
    K = 1/T
    return (1-(1-np.tanh(K)**2)**4/(16*np.tanh(K)**4))**(1/8)

"------------------------------------------------------------------------------"
# Credit to Nathan Nichols

# import numpy as np
import matplotlib.pyplot as plt

## Bootstrap ###################################################################
def bootstrap(data,bootstrap_size=100):
    N = data.shape[0]
    _bootstrap_avg = data[np.random.randint(N, size=(bootstrap_size,N))].mean(axis=-1)

    bootstrap_avg = np.mean(_bootstrap_avg)
    bootstrap_err = np.sqrt((N/(N - 1))*(np.mean(_bootstrap_avg**2) - bootstrap_avg**2))
    return bootstrap_avg, bootstrap_err

def bootstrap_on_function(func,*argv,bootstrap_size=100,**kwargs):
    N = argv[0].shape[0]
    for data in argv:
        if data.shape[0] != N:
            msg = "Data must have same length"
            raise ValueError(msg)

    data_array = np.array(argv)

    _bootstrap_avg = data_array[:,np.random.randint(N, size=(N,bootstrap_size))].mean(axis=1)

    _func_result = func(*_bootstrap_avg,**kwargs)
    bootstrap_avg = np.mean(_func_result)
    bootstrap_avg2 = np.mean(_func_result**2)
    bootstrap_err = np.sqrt((N/(N - 1))*(bootstrap_avg2 - bootstrap_avg**2))
    return bootstrap_avg, bootstrap_err

## Jacknife ####################################################################
def jackknife(data):
    N = data.shape[0]
    _jackknife_avg = (data.sum() - data)/(N-1)
    
    jackknife_avg = np.mean(_jackknife_avg)
    jackknife_err = np.sqrt((N - 1)*(np.mean(_jackknife_avg**2) - jackknife_avg**2))
    return jackknife_avg, jackknife_err

def jackknife_on_function(func,*argv,**kwargs):
    N = argv[0].shape[0]
    for data in argv:
        if data.shape[0] != N:
            msg = "Data must have same length"
            raise ValueError(msg)
    data_array = np.array(argv)
    _jackknife_avg = ((data_array.sum(axis=1) - data_array.T)/(N-1)).T
    
    _func_result = func(*_jackknife_avg,**kwargs)
    jackknife_avg = np.mean(_func_result)
    jackknife_avg2 = np.mean(_func_result**2)
    jackknife_err = np.sqrt((N - 1)*(jackknife_avg2 - jackknife_avg**2))
    return jackknife_avg, jackknife_err


## Binning Analysis ############################################################
def binning_analysis_data(raw_data,skip=0,method="jackknife",bootstrap_size=100,running_window=1):
    if method not in ["jackknife","bootstrap","cumulative","running"]:
        msg = "Choose method from: jackknife, bootstrap, cumulative, running"
        raise ValueError(msg)
    if method == "running":
        print("WARNING: Binning analysis will use error from last window of running average.")
    N_raw = raw_data.shape[0] - skip
    max_bin_level = np.log2(N_raw).astype(int)
    binned_avg = np.zeros(max_bin_level)
    binned_err = np.zeros(max_bin_level)
    data_binned = raw_data[-2**(max_bin_level):]
    for i in range(max_bin_level):
        if method == "jackknife":
            binned_avg[i], binned_err[i] = jackknife(data_binned)
        if method == "bootstrap":
            binned_avg[i], binned_err[i] = bootstrap(data_binned,bootstrap_size=bootstrap_size)
        if method == "cumulative":
            binned_data_mean = np.mean(data_binned)
            binned_data_err = np.std(data_binned,ddof=0)/np.sqrt(data_binned.shape[0] - 1)
            binned_avg[i] = binned_data_mean
            binned_err[i] = binned_data_err
        if method == "running":
            if data_binned.shape[0] < running_window:
                binned_data_mean = np.mean(data_binned)
                binned_data_err = np.std(data_binned,ddof=0)/np.sqrt(data_binned.shape[0] - 1)
            else:
                binned_data_mean = np.mean(data_binned[-running_window:])
                binned_data_err = np.std(data_binned[-running_window:],ddof=0)/np.sqrt(running_window - 1)
            binned_avg[i] = binned_data_mean
            binned_err[i] = binned_data_err
        data_binned = (data_binned[::2] + data_binned[1::2])/2

    autocorrelation_time = ((binned_err[-1]/binned_err[0])**2 - 1)/2
    if binned_err.shape[0] > 1:
        convergence_factor = 2.0 - binned_err[-1]/binned_err[-2]
    else:
        convergence_factor = 0.0
    return autocorrelation_time, convergence_factor, max_bin_level, binned_avg, binned_err

def binning_analysis(raw_data,skip=0,method="jackknife",bootstrap_size=100,running_window=1,fig_ax=None,label=None):
    autocorrelation_time, convergence_factor, max_bin_level, binned_avg, binned_err = binning_analysis_data(raw_data,skip=skip,method=method,bootstrap_size=bootstrap_size,running_window=running_window)
    print(f"autocorrelation time: {autocorrelation_time}")
    print(f"convergence factor: {convergence_factor}")
    if fig_ax is not None:
        fig, ax = binning_analysis_plot_fig(max_bin_level,binned_err,*fig_ax,label=label)
    else:
        fig, ax = binning_analysis_plot(max_bin_level,binned_err,label=label)
    return fig, ax, autocorrelation_time, convergence_factor, max_bin_level, binned_avg, binned_err

def binning_analysis_on_function_data(func,*argv,skip=0,method="jackknife",bootstrap_size=100,running_window=1,propogated_error_func=None,**kwargs):
    if method not in ["jackknife","bootstrap","cumulative","running"]:
        msg = "Choose method from: jackknife, bootstrap, cumulative, running"
        raise ValueError(msg)
    if method in ["cumulative","running"]:
        print("WARNING: Reccomend either jackknife or bootstrap method as correlations in averages can cause misleading errorbars.")
        if method == "running":
            print("WARNING: Binning analysis will use error from last window of running average.")
        if propogated_error_func == None:
            msg = "Please provide propogated error function as <propogated_error_func>."
            raise ValueError(msg)
    N_raw = argv[0].shape[0]
    for data in argv:
        if data.shape[0] != N_raw:
            msg = "Data must have same length"
            raise ValueError(msg)
    N_raw -= skip
    if N_raw < 1:
        msg = f"skip greater than data length: {N_raw + skip}"
        raise ValueError(msg)

    max_bin_level = np.log2(N_raw).astype(int)
    binned_avg = np.zeros(max_bin_level)
    binned_err = np.zeros(max_bin_level)

    data_array = np.array(argv)
    data_binned = data_array[:,-2**(max_bin_level):]
    for i in range(max_bin_level):
        if method == "jackknife":
            binned_avg[i], binned_err[i] = jackknife_on_function(func,*data_binned,**kwargs)
        if method == "bootstrap":
            binned_avg[i], binned_err[i] = bootstrap_on_function(func,*data_binned,bootstrap_size=bootstrap_size,**kwargs)
        if method == "cumulative":
            binned_data_mean = np.mean(data_binned,axis=1)
            binned_data_err = np.std(data_binned,axis=1,ddof=0)/np.sqrt(data_binned.shape[1] - 1)
            binned_avg[i] = func(*binned_data_mean,**kwargs)
            binned_err[i] = propogated_error_func(*binned_data_mean,*binned_data_err,**kwargs)
        if method == "running":
            if data_binned.shape[1] < running_window:
                binned_data_mean = np.mean(data_binned,axis=1)
                binned_data_err = np.std(data_binned,axis=1,ddof=0)/np.sqrt(data_binned.shape[1] - 1)
            else:
                binned_data_mean = np.mean(data_binned[:,-running_window:],axis=1)
                binned_data_err = np.std(data_binned[:,-running_window:],axis=1,ddof=0)/np.sqrt(running_window - 1)
            binned_avg[i] = func(*binned_data_mean,**kwargs)
            binned_err[i] = propogated_error_func(*binned_data_mean,*binned_data_err,**kwargs)
        data_binned = (data_binned[:,::2] + data_binned[:,1::2])/2

    autocorrelation_time = ((binned_err[-1]/binned_err[0])**2 - 1)/2
    if binned_err.shape[0] > 1:
        convergence_factor = 2.0 - binned_err[-1]/binned_err[-2]
    else:
        convergence_factor = 0.0
    return autocorrelation_time, convergence_factor, max_bin_level, binned_avg, binned_err

def binning_analysis_on_function(func,*argv,skip=0,method="jackknife",bootstrap_size=100,running_window=1,propogated_error_func=None,fig_ax=None,label=None,**kwargs):
    autocorrelation_time, convergence_factor, max_bin_level, binned_avg, binned_err = binning_analysis_on_function_data(func,*argv,skip=skip,method=method,bootstrap_size=bootstrap_size,running_window=running_window,propogated_error_func=propogated_error_func,**kwargs)
    print(f"autocorrelation time: {autocorrelation_time}")
    print(f"convergence factor: {convergence_factor}")
    if fig_ax is not None:
        fig, ax = binning_analysis_plot_fig(max_bin_level,binned_err,*fig_ax,label=label)
    else:
        fig, ax = binning_analysis_plot(max_bin_level,binned_err,label=label)
    return fig, ax, autocorrelation_time, convergence_factor, max_bin_level, binned_avg, binned_err

def binning_analysis_plot(max_bin_level,binned_err,label=None):
    fig, ax = plt.subplots()
    return binning_analysis_plot_fig(max_bin_level,binned_err,fig,ax,label=label)

def binning_analysis_plot_fig(max_bin_level,binned_err,fig,ax,label=None):
    ax.plot(np.arange(max_bin_level),binned_err,label=label)
    ax.set_xlabel(r"Binning Level")
    ax.set_ylabel(r"$\sigma$")
    return fig,ax

## Pimcplot  ###################################################################
def averageplot_data(data,method="jackknife",bootstrap_size=100,running_window=1):
    if method not in ["jackknife","bootstrap","cumulative","running"]:
        msg = "Choose averageplot method from: jackknife, bootstrap, cumulative, running"
        raise ValueError(msg)
    N = data.shape[0]
    _avg = np.zeros(N)
    _err = np.zeros(N)
    
    _avg[0] = data[0]
    for i in range(N - 1):
        if method == "jackknife":
            _avg[i+1], _err[i+1] = jackknife(data[:i+2])
        if method == "bootstrap":
            _avg[i+1], _err[i+1] = bootstrap(data[:i+2],bootstrap_size=bootstrap_size)
        if method == "cumulative":
            data_mean = np.mean(data[:i+2])
            data_err = np.std(data[:i+2],ddof=0)/np.sqrt(i + 1)
            _avg[i+1] = data_mean
            _err[i+1] = data_err
        if method == "running":
            if i+1 < running_window:
                data_mean = np.mean(data[:i+2])
                data_err = np.std(data[:i+2],ddof=0)/np.sqrt(i+1)
            else:
                data_mean = np.mean(data[i+2-running_window:i+2])
                data_err = np.std(data[i+2-running_window:i+2],ddof=0)/np.sqrt(running_window - 1)
            _avg[i+1] = data_mean
            _err[i+1] = data_err

    return _avg, _err

def averageplot_method(raw_data,skip=0,bin_size=1,method="jackknife",bootstrap_size=100,running_window=1,fig_ax=None,label=None,**kwargs):
    data_binned = bin_data(raw_data,skip=skip,bin_size=bin_size)
    skip_binned = skip//bin_size + 1

    _avg, _err = averageplot_data(data_binned,method=method,bootstrap_size=bootstrap_size,running_window=running_window)

    if fig_ax is not None:
        fig, ax = averageplot_fig(_avg,_err,*fig_ax,skip=skip_binned,label=label,**kwargs)
    else:
        fig,ax = averageplot(_avg,_err,skip=skip_binned,label=label,**kwargs)
    return _avg, _err, fig, ax

def averageplot_on_function_data(func,*argv,method="jackknife",bootstrap_size=100,running_window=1,propogated_error_func=None,**kwargs):
    if method not in ["jackknife","bootstrap","cumulative","running"]:
        msg = "Choose method from: jackknife, bootstrap, cumulative, running"
        raise ValueError(msg)
    if method in ["cumulative","running"]:
        print("WARNING: Reccomend either jackknife or bootstrap method as correlations in averages can cause misleading errorbars.")
        if propogated_error_func == None:
            msg = "Please provide propogated error function as <propogated_error_func>."
            raise ValueError(msg)
    N = argv[0].shape[0]
    for data in argv:
        if data.shape[0] != N:
            msg = "Data must have same length"
            raise ValueError(msg)
    
    data_array = np.array(argv)
    _avg = np.zeros(N)
    _err = np.zeros(N)
    
    _avg[0] = func(*data_array[:,0],**kwargs)
    for i in range(N - 1):
        if method == "jackknife":
            _avg[i+1], _err[i+1] = jackknife_on_function(func,*data_array[:,:i+2],**kwargs)
        if method == "bootstrap":
            _avg[i+1], _err[i+1] = bootstrap_on_function(func,*data_array[:,:i+2],bootstrap_size=bootstrap_size,**kwargs)
        if method == "cumulative":
            data_array_mean = np.mean(data_array[:,:i+2],axis=1)
            data_array_err = np.std(data_array[:,:i+2],axis=1,ddof=0)/np.sqrt(i+1)
            _avg[i+1] = func(*data_array_mean,**kwargs)
            _err[i+1] = propogated_error_func(*data_array_mean,*data_array_err,**kwargs)
        if method == "running":
            if i+1 < running_window:
                data_array_mean = np.mean(data_array[:,:i+2],axis=1)
                data_array_err = np.std(data_array[:,:i+2],axis=1,ddof=0)/np.sqrt(i+1)
            else:
                data_array_mean = np.mean(data_array[:,i+2-running_window:i+2],axis=1)
                data_array_err = np.std(data_array[:,i+2-running_window:i+2],axis=1,ddof=0)/np.sqrt(running_window - 1)
            _avg[i+1] = func(*data_array_mean,**kwargs)
            _err[i+1] = propogated_error_func(*data_array_mean,*data_array_err,**kwargs)
    return _avg, _err

def averageplot_on_function_method(func,*argv,skip=0,bin_size=1,method="jackknife",bootstrap_size=100,running_window=1,propogated_error_func=None,fig_ax=None,label=None,**kwargs):
    N = argv[0].shape[0]
    for data in argv:
        if data.shape[0] != N:
            msg = "Data must have same length"
            raise ValueError(msg)
            
    data_binned = [bin_data(raw_data,skip=skip,bin_size=bin_size) for raw_data in argv]
    skip_binned = skip//bin_size + 1
    
    _avg, _err = averageplot_on_function_data(func,*data_binned,method=method,bootstrap_size=bootstrap_size,running_window=running_window,propogated_error_func=propogated_error_func,**kwargs)
    
    if fig_ax is not None:
        fig, ax = averageplot_fig(_avg,_err,*fig_ax,skip=skip_binned,label=label,**kwargs)
    else:
        fig,ax = averageplot(_avg,_err,skip=skip_binned,label=label,**kwargs)
    return _avg, _err, fig, ax

def averageplot(data_average, data_error, skip=0, label=None, **kwargs):
    fig, ax = plt.subplots()
    return averageplot_fig(data_average, data_error, fig, ax, skip=skip, label=label, **kwargs)

def averageplot_fig(data_average, data_error, fig, ax, skip=0, label=None, **kwargs):
    x,y,error = np.arange(data_average.shape[0]) + skip, data_average, data_error
    ax.plot(x, y, linewidth=.3, label=label)
    ax.fill_between(x, y-error, y+error, alpha=0.5)
    ax.set_xlabel(r'MC Bins')
    return fig, ax


## Helper Functions ############################################################
def integrated_autocorrelation_time_jackknife_avg(raw_data,bin_size=1,skip=0):
    data_binned = bin_data(raw_data,skip=skip,bin_size=bin_size)

    N = data_binned.shape[0]
    jackknife_avg, jackknife_err = jackknife(data_binned)
    return ((((data_binned[:,None] @ data_binned[:,None].T).sum(axis=1) - data_binned**2) - ((N - 1)*jackknife_avg**2))/(jackknife_err**2)).mean()

def integrated_autocorrelation_time_jackknife(raw_data,bin_size=1,skip=0,i=0):
    data_binned = bin_data(raw_data,skip=skip,bin_size=bin_size)

    N = data_binned.shape[0]
    jackknife_avg, jackknife_err = jackknife(data_binned)
    return ((((data_binned[i]*data_binned).sum() - data_binned[i]**2) - ((N - 1)*jackknife_avg**2))/(jackknife_err**2))

def bin_data(data,bin_size=1,skip=0):
    data_length = data.shape[0] - skip
    final_bin_size = data_length - (data_length//bin_size)*bin_size
    if final_bin_size > 0:
        #data_reshaped = data[skip:-final_bin_size].reshape(data_length//bin_size,bin_size)
        #np.mean(data_reshaped,axis=1)
        data_reshaped = data[skip:-final_bin_size].reshape(bin_size,data_length//bin_size)
        final_bin = data[-final_bin_size:]
        data_binned = np.zeros(data_length//bin_size + 1)
        data_binned[:-1] = np.mean(data_reshaped,axis=0)
        data_binned[-1] = np.mean(final_bin)
    else:
        if bin_size > 1:
            data_binned = np.mean(data[skip:].reshape(bin_size,data_length//bin_size),axis=0)
        else:
            data_binned = data[skip:]
    return data_binned

def running_mean(data,window=1):
    if data.shape[0] == 1:
        return data[0]
    if data.shape[0] < window:
        return np.mean(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def running_error(data,window=1,sem=True):
    if data.shape[0] == 1:
        return 0.0
    if data.shape[0] < window:
        return np.std(data,ddof=0)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    cumsum2 = np.cumsum(np.insert(data**2, 0, 0))
    c1 = (cumsum[window:] - cumsum[:-window])
    c2 = (cumsum2[window:] - cumsum2[:-window])
    err = np.sqrt(np.abs(window*c2-(c1*c1))/(window*(window-1)))
    if sem:
        return err/np.sqrt(window)
    return err

def cumulative_mean(data):
    cumsum = np.cumsum(data)
    return cumsum/(np.arange(len(cumsum))+1)

def cumulative_error(data,sem=True):
    c1 = np.cumsum(data)
    c2 = np.cumsum(data**2)
    N = np.arange(len(c1))+1
    err = np.sqrt(np.abs(N*c2-(c1*c1))/(N*(N-1)))
    if sem:
        return err/np.sqrt(N)
    return err

def fluctuations(N_avg,N2_avg,**kwargs):
    """Calculate fluctuations in N from averages <N> and <N^2>."""
    return N2_avg - N_avg**2

def fluctuations_error(N_avg,N2_avg,N_err,N2_err,**kwargs):
    """Calculate propogated error for fluctionations in data N from averages <N> and <N^2> and error in each average."""
    return np.sqrt(N2_err**2 + ((-2*N_avg)**2)*(N_err**2))
    return N2_avg - N_avg**2

def skew(N,N2,N3,**kwargs):
    s2 = N2 - N**2
    return (N3 - 3*N*s2 - N**3)/np.sqrt(s2**3)

def skew_error(N,N2,N3,N_err,N2_err,N3_err,**kwargs):
    s2 = N2 - N**2
    dN_ = (3*(N2**2 - N*N3))/(-s2*np.sqrt((s2)**3))
    dN2_ = (3*((s2)**2)*(N*N2 - N3))/(2*np.sqrt((s2)**9))
    dN3_ = 1/np.sqrt((s2)**3)
    return np.sqrt((dN_*N_err)**2 + (dN2_*N2_err)**2 + (dN3_*N3_err)**2)

def kurtosis(N,N2,N3,N4,**kwargs):
    """Calculate kurtosis in data N from averages <N^4>, <N^3>, <N^2>, and <N>."""
    return (-3*(N**4) + 6*(N**2)*N2 - 4*N*N3 + N4)/((N2 - (N**2))**2)

def kurtosis_error(N,N2,N3,N4,N_err,N2_err,N3_err,N4_err,**kwargs):
    """Calculate propogated error for kurtosis in data N from averages <N^4>, <N^3>, <N^2>, and <N> and error in each average."""
    dN_ = (-12*N**3 + 12*N*N2 - 4*N3)/(N2 - N**2)**2 + (4*N*(-3*N**4 + 6*N2*N**2 - 4*N*N3 + N4))/(N2 - N**2)**3
    dN2_ = (6*N**2)/(N2 - N**2)**2 - (2*(-3*N**4 + 6*N2*N**2 - 4*N*N3 + N4))/(N2 - N**2)**3
    dN3_ = (-4*N)/(N2 - N**2)**2
    dN4_ = (N2 - N**2)**(-2)
    return np.sqrt((dN_*N_err)**2 + (dN2_*N2_err)**2 + (dN3_*N3_err)**2 + (dN4_*N4_err)**2)

