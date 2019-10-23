#!/usr/bin/env python
"""Some functions to invert simple RT models
"""
import os
import sys

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import prosail


def fwd_model(x, rho_std, leaf, do_plot=True):
    wv, rho_pred, _ = prosail.run_prospect(x[0], x[1], x[2], x[3], x[4], x[5], ant=x[6])

    rho_noise = np.random.randn(len(wv))*rho_std
    rho_meas = rho_pred + rho_noise
    rho_std *= np.ones(2101)

    if do_plot:
        plot_spectra(rho_pred, rho_std, leaf)
    
    return rho_pred

def plot_spectra(rho_pred, rho_unc, leaf):
        fig_size = plt.rcParams['figure.figsize']
        fig_size[0] = 14
        fig_size[1] = 7
        plt.rcParams['figure.figsize'] = fig_size

        fig, ax1 = plt.subplots()
        wv = np.arange(400, 2501)
        ax1.scatter(leaf[0], leaf[1], s=10, c='black')
        ax1.plot(wv, rho_pred, '-', label=r'$\rho$')
        ax1.fill_between(wv, rho_pred - 1.96*rho_unc, rho_pred + 1.96*rho_unc, color="0.8", alpha=0.5)

        ax1.legend(loc="best")
        make_pretty(ax1)
        ax1.set_xlabel("Wavelength [nm]")

def make_pretty(axs):
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)    
 
 
def read_lopex_sample(sample_id, leaf, do_plot=True):
    os.chdir('C:/Users/SNMACIAS/Downloads/jose_gomez/barrax_2017-master/')
    
    if sample_id < 1 or sample_id > 116:
        raise ValueError("Only sample numbers between 1 and 116")
    
    rho = np.loadtxt("data/LOPEX93/refl.%03d.dat" % 
                     sample_id).reshape((5, 2101))
    tau = np.loadtxt("data/LOPEX93/trans.%03d.dat" % 
                     sample_id).reshape((5, 2101))
    if do_plot:
        plot_spectra(rho.mean(axis=0), rho.std(axis=0), leaf)
        
    return rho.mean(axis=0), rho.std(axis=0)

def prospect_lklhood(x, wl, rho, rho_unc):
    """Calculates the log-likelihood of leaf reflectance measurements
    assuming Gaussian additive noise. Can either use reflectance or
    transmittance or both."""
    wv, rho_pred, _ = prosail.run_prospect(x[0], x[1], x[2], x[3], x[4], x[5], ant=x[6])
    rho_unc_ndim = rho_unc.ndim
    
    if rho_unc_ndim == 1:
        cov_obs_rho_inv = 1.0/(rho_unc*rho_unc)
    elif rho_unc_ndim == 2:
        cov_obs_rho_inv = 1./rho_unc.diagonal()
    refl_lklhood = np.sum(0.5*cov_obs_rho_inv*(rho_pred[([150, 260, 335, 390])] - rho)**2)

    return refl_lklhood

def max_lklhood(x0, wl, rho, rho_unc): # do_plot=True
    bounds = [ [1.3, 1.9],
               [10, 80],
               [5., 20],
               [1e-2, 0.6], #0., 0.6
               [1e-3, 0.045], # 0.00005, 0.1
               [1e-3, 0.02], # 0.001, 0.03
               [1., 20]]

    opts = {'maxiter': 500, 'disp': True}
            
    def cost(x, wl, rho):
            return prospect_lklhood(x, wl, rho, rho_unc)
    retval = scipy.optimize.minimize(cost, x0, method="L-BFGS-B", jac=False, bounds=bounds, options=opts, args=(wl, rho))
    
#    if do_plot:
#        x = retval.x
#        wv, rho_pred, _ = prosail.run_prospect(x[0], x[1], x[2], x[3], x[4], x[5], ant=x[6])
#        plot_spectra(rho_pred, rho_unc, leaf)
        
    return retval

def least_squares(x0, wl, rho, rho_unc):
    bounds = ([1.3, 1., 1., 1e-2, 1e-3, 1e-3, 1.], [1.9, 80, 20, 0.6, 0.045, 0.02, 20])

    opts = {'maxiter': 10000}
            
    def cost(x, wl, rho):
            return prospect_lklhood(x, wl, rho, rho_unc)
    retval = scipy.optimize.least_squares(cost, x0, method="trf", jac="2-point", bounds=bounds, tr_options=opts, args=(wl, rho))
        
    return retval

def variational_prospect(mu_prior, prior_inv_cov, wl, rho, rho_unc, x0=None):
    bounds = [ [1.1, 3],
               [1., 100],
               [1., 30],
               [0., 0.6],
               [0.00005, 0.1],
               [ 0.001, 0.03],
               [0., 20]]

    opts = {'maxiter': 500, 'disp': True}
            
    def cost(x, wl, rho):
        return prospect_lklhood(x, wl, rho, rho_unc) + calculate_prior(x, mu_prior, prior_inv_cov)

    if x0 is None:
        x0 = mu_prior
    retval = scipy.optimize.minimize(cost, x0, method="L-BFGS-B", jac=False, 
                                     bounds=bounds, options=opts, args=(wl, rho))
    rho_unc_ndim = rho_unc.ndim

    if rho_unc_ndim == 1:
        cov_obs_rho_inv = 1.0/(rho_unc*rho_unc)
    elif rho_unc_ndim == 2:
        cov_obs_rho_inv = 1./rho_unc.diagonal()

    hess_obs = approx_hessian_prospect(retval.x, cov_obs_rho_inv)
    posterior_unc = np.linalg.inv(hess_obs + prior_inv_cov)
    posterior_corr = cov2corr(posterior_unc)

    print "%10s\t|\t%20s\t|\t%20s" %("Parameter", "Posterior Mean", "Posterior StdDev")
    d = np.sqrt(posterior_unc.diagonal())

    for i, par in enumerate(["N", "Cab","Car", "Cbrown", "Cw", "Cm", "Ant"]):
        print "%10s\t|\t%20.6f\t|\t%20.6f" %(par, retval.x[i], d[i]) 
        
    return retval, posterior_unc, posterior_corr

def calculate_prior (x, mu, inv_cov):
    dif = x-mu
    prior = (x-mu).dot((inv_cov.dot(dif)))
    return 0.5*prior
    
def approx_hessian_prospect(x, cov_rho):

    def func_rho(x):
        xt = x*1.
        grad = np.zeros((7, 4)) # 7, 2101
        for i in xrange(7):
            xt[i] = x[i] + 1e-5
            
            wv, r, t = prosail.run_prospect(xt[0], xt[1], xt[2], xt[3], xt[4], xt[5], ant=xt[6]) 
            grad[i, :] = r[([150, 260, 335, 390])]/1e-5
        return grad

    dH_rho = func_rho(x)
    Hess_rho = dH_rho.dot(cov_rho).dot(dH_rho.T)

    return Hess_rho 

def cov2corr(cov, return_std=False):
    '''convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    '''
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr 