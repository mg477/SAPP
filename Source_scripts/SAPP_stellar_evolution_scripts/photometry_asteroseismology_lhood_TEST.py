#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:26:41 2020

@author: gent
"""

import numpy as np
from numpy import e as EXP
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import psutil
import SAPP_spectroscopy.Payne.astro_constants_cgs as astroc
from numba import jit
# from IRFM_teff import IRFM_calc

#from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
#import warnings

"""
These warning suppressions aren't ideal.
These instead should be logged and saved as outputting them to the console in a cluster is also not a good idea.
"""

#warnings.simplefilter('ignore',category=NumbaDeprecationWarning)
#warnings.simplefilter('ignore',category=NumbaPendingDeprecationWarning)
#warnings.simplefilter('ignore',category=NumbaWarning)


plt.ioff()

def chis_sq_one(var,mu,sigma):
    
    
    ch2 = ((mu-var)/sigma) ** 2
    
    return ch2

def gaussian_one(var,mu,sigma):
    
    """
    var: Input variable 
    mu: Observed variable value
    sigma: Gaussian error in variable
    return: Gaussian distribution about mu normalised to 1
    purpose: Take in value or array of var and produce a gaussian distribution normalised to 1
    """
    
    g_one = EXP**(-(var-mu) ** 2/(2 * sigma ** 2))
    
    return g_one
    
def distance_modulus(distance): 
    
    """
    distance: distance of star [pc]
    return: distance modulus
    purpose: calculate distance modulus using parallax
    """
    
    dist_mod = 5 * np.log10(distance/10) # 10 is in parsecs
        
    return dist_mod

@jit
def Lhood_norm_red_chi_ind(var,mu,sigma,nu,N_k):
    
    if nu <= 0:
        
        nu = 1
        
        
    L_i = EXP ** (-((mu-var)/sigma) ** 2/(2 * nu)) 
    
    return L_i 

def DR2_GARSTEC_DR3_convert(G_DR2,Bp_DR2,Rp_DR2,Teff):
    
    """
    The GARSTEC stellar evolution models from photometry_stellar_track_mmap.npy have Gaia DR2 photometry
    and the Zero Points i.e. Solar Models haven't been corrected for. 
    
    This converter takes into account the corrections as well as converts DR2 --> DR3 within thousandths of a magnitude.
    
    This uses Teff from the models 
    
    'Note these transformations are very approximate, Note those transformations are very approximate, in a temperature range appropriate for the benchmark stars (but not giants), 
    roughly Teff > 4500K. I tested for Fe/H range down to -1'
    
    Models need to be re-run in the future, better than having corrections.
    """
    
    G_DR3 = G_DR2 -0.000001*(Teff-5000.) - 0.003

    Bp_DR3 = Bp_DR2 +0.0000065*(Teff-5000.)
    
    Rp_DR3 = Rp_DR2 +0.0000028*(Teff-5000.) - 0.0146
    
    
    return [G_DR3, Bp_DR3, Rp_DR3]

#@jit
def stellar_evo_likelihood(mag_arr,colour_arr,astrosize_arr,pax,spec_central_values,phot_ast_limits,DM,sigma_DM,magnitude_set): 
    
    """
    mag_arr: input observed photometry array; [magnitudes,error,extinction]
    astrosize_arr: input observed asteroseismic array; [[delta_nu,nu_max],[delta_nu_err,nu_max_error]]
    colour_arr: input observed photometry array; [colours,error]
    pax: parallax [mas]
    return: Three 3-D distribution of likelihoods for photometry, asteroseismology and combined of single star
    purpose: Takes in observational photometric data, asteroseismic data, parallax from Gaia and a file name string
    to calculate the likelihood function. This is done by loading up the model file line by line,
    calculate the likelihood probability and then save it as the new parameter space. 
    """
    
    teff_central = spec_central_values[0]
    logg_central = spec_central_values[1]
    feh_central = spec_central_values[2]
    
    # print(teff_central,logg_central,feh_central)
    
    teff_limit = phot_ast_limits[0]
    logg_limit = phot_ast_limits[1]
    feh_limit = phot_ast_limits[2]
    
    mag_obs = mag_arr[0] # 2 x 1D arrays containing magnitude observations for star in non-gaia: H,J,K,V,B and gaia: G,Bp,Rp
    non_gaia_mag_obs = mag_obs[0]
    gaia_mag_obs = mag_obs[1]
    mag_err = mag_arr[1] #  2 x 1D arrays containing error for magnitude observations of star in non-gaia and gaia
    non_gaia_mag_err = mag_err[0] # erH,erJ,erK,erV,erB 
    gaia_mag_err = mag_err[1] # erG,erBp,erRp
            
    col_obs = colour_arr[0] # 2 x 1D arrays containing colour observations for star in non-gaia: H-K,B-V,V-J,V-K and gaia: Bp-Rp,G-Rp
    non_gaia_col_obs = col_obs[0]
    gaia_col_obs = col_obs[1]
    col_err = colour_arr[1] # 2 x 1D arrays containing error for colour observations of star in non-gaia and gaia
    non_gaia_col_err = col_err[0] # in order er{H-K,B-V,V-J,V-K}
    gaia_col_err = col_err[1] # in order er{Bp-Rp,G-Rp,Bp-K}
    
    ast_obs = astrosize_arr[0] # array of stellar asteroseismic quantities: delta_nu, nu_max
    ast_err = astrosize_arr[1] # error of stellar asteroseismic quantities: delta_nu_err, err_nu_max
    
    d = 3 # dimensionality of non-probability parameter space ;  Teff, logg, [Fe/H]

    # initialisation of degrees of freedom

    nu_non_gaia_phot = 1
    nu_gaia_phot = 1
    nu_phot_non_gaia_col = 1
    nu_phot_gaia_col = 1
    nu_ast = 1

   # param_space = []
   
    # The file has already been read.
    mask = (evoTrackArr[3]<teff_central+teff_limit)&(evoTrackArr[3]>teff_central-teff_limit)&\
           (evoTrackArr[6]<logg_central+logg_limit)&(evoTrackArr[6]>logg_central-logg_limit)&\
           (evoTrackFeh<feh_central+feh_limit)&(evoTrackFeh>feh_central-feh_limit)&\
           (evoTrackArr[0] > 30)&(evoTrackArr[0]/1000 <= 16)
           # (evoTrackArr[0] > 30)&(evoTrackArr[0]/1000 <= 16)
  
    FeH = evoTrackFeh[mask]
                       
    ### Non-Gaia magnitudes ###

    # non_gaia_mag_model = [float(x[27]),float(x[26]),float(x[28]),float(x[23]),float(x[22])] # H,J,K,V,B

    ## OLD LARGE GARSTEC TRACKS ##    

    # non_gaia_mag_model  = [evoTrackArr[27][mask]+0.0047,
    #                         evoTrackArr[26][mask]-0.0357,
    #                         evoTrackArr[28][mask]-0.0414,
    #                         evoTrackArr[23][mask]+0.0479,
    #                         evoTrackArr[22][mask]+0.0497] # H,J,K,V,B

    ## NEW SMALLER GARSTEC TRACKS ##

    non_gaia_mag_model  = [evoTrackArr[16][mask]+0.0047,
                            evoTrackArr[15][mask]-0.0357,
                            evoTrackArr[17][mask]-0.0414,
                            evoTrackArr[12][mask]+0.0479,
                            evoTrackArr[11][mask]+0.0497] # H,J,K,V,B

    
    non_gaia_phot_var = non_gaia_mag_model # Gaussian input variable for non gaia magnitudes
    non_gaia_phot_sigma = non_gaia_mag_err # Assuming each factor has the same standard deviation
    N_non_gaia_phot = len(non_gaia_mag_model) # No. of gaussian factors for photometry
    nu_non_gaia_phot = N_non_gaia_phot - d # degrees of freedom
        
    if nu_non_gaia_phot <= 0:
            
        nu_non_gaia_phot = 1
        
    ### Gaia magnitudes ###
        
    # gaia_mag_model = [gaia_G_band_correction(float(x[34])),gaia_Bp_band_correction(float(x[35])),gaia_Rp_band_correction(float(x[36]))] # G,Bp,Rp
    
    # gaia_mag_model   = [gaia_G_band_correction(evoTrackArr[34][mask]),gaia_Bp_band_correction(evoTrackArr[35][mask]),gaia_Rp_band_correction(evoTrackArr[36][mask])] # G,Bp,Rp
    # gaia_mag_model   = [evoTrackArr[34][mask],evoTrackArr[35][mask],evoTrackArr[36][mask]] # G,Bp,Rp
    # gaia_mag_model  = [evoTrackArr[21][mask],evoTrackArr[22][mask],evoTrackArr[23][mask]] # G,Bp,Rp    

    ## OLD LARGE GARSTEC TRACKS ##
    
    # gaia_mag_model  = [evoTrackArr[34][mask]+0.01,
    #                    evoTrackArr[35][mask]+0.0201,
    #                    evoTrackArr[36][mask]-0.0078] # G,Bp,Rp    

    # gaia_mag_model = DR2_GARSTEC_DR3_convert(evoTrackArr[34][mask],evoTrackArr[34][mask],evoTrackArr[34][mask],evoTrackArr[3][mask])

    ## NEW SMALLER GARSTEC TRACKS ##

    gaia_mag_model  = [evoTrackArr[21][mask]+0.01,
                        evoTrackArr[22][mask]+0.0201,
                        evoTrackArr[23][mask]-0.0078] # G,Bp,Rp    
                
    # gaia_phot_var = gaia_mag_model # Gaussian input variable for gaia magnitudes 
    gaia_phot_sigma = gaia_mag_err # Assuming each factor has the same standard deviation
    # N_gaia_phot = len(gaia_mag_model) # No. of gaussian factors for photometry
    # nu_gaia_phot = N_gaia_phot - d # degrees of freedom

    # if nu_gaia_phot <= 0:
            
    #     nu_gaia_phot = 1
        
    ### Non-Gaia Colours ###
        
    # non_gaia_col_var = [non_gaia_mag_model[0]-non_gaia_mag_model[2],non_gaia_mag_model[4]-non_gaia_mag_model[3],non_gaia_mag_model[3]-non_gaia_mag_model[1],non_gaia_mag_model[3]-non_gaia_mag_model[2]] # H_K,B_V,V_J,V_K
    # non_gaia_col_sigma = non_gaia_col_err
    # N_non_gaia_phot_col = len(non_gaia_col_var) # No. of gaussian factors for photometry
    # nu_phot_non_gaia_col = N_non_gaia_phot_col - d # degrees of freedom

    # if nu_phot_non_gaia_col <= 0:
            
    #     nu_phot_non_gaia_col = 1
        
    ### Gaia Colours ###
    
    ## corrections need to be applied to these 

    gaia_col_var = [gaia_mag_model[1]-gaia_mag_model[2],gaia_mag_model[0]-gaia_mag_model[2]] # Bp_Rp,G_Rp
    # gaia_col_var = [gaia_mag_model[1]-gaia_mag_model[2],gaia_mag_model[0]-gaia_mag_model[2],gaia_mag_model[1]-non_gaia_mag_model[2]] # Bp_Rp,G_Rp,Bp_K                
    gaia_col_sigma = gaia_col_err
    # N_gaia_phot_col = len(gaia_col_var) # No. of gaussian factors for photometry
    # nu_phot_gaia_col = N_gaia_phot_col - d # degrees of freedom

    # if nu_phot_gaia_col <= 0:
            
    #     nu_phot_gaia_col = 1
        
    ### Asteroseismology ###
        
    # nu_max_model = astroc.nu_max_sol * 10 ** (float(x[6]))/astroc.surf_grav_sol * (astroc.teff_sol/float(x[3])) ** 0.5
    # delta_nu_model = float(x[16]) * astroc.delta_nu_sol/136.3 # 136.`3 uHz is Aldo's model solar value
    nu_max_model = astroc.nu_max_sol * 10 ** (evoTrackArr[6][mask])/astroc.surf_grav_sol * (astroc.teff_sol/evoTrackArr[3][mask]) ** 0.5
    delta_nu_model = evoTrackArr[16][mask] * astroc.delta_nu_sol/136.3 # 136.3 uHz is Aldo's model solar value
    ast_var = [delta_nu_model,nu_max_model]
    N_ast = len(ast_var)
    # nu_ast = N_ast-d # degrees of freedom 

    # if nu_ast <= 0:
            
    #     nu_ast = 1
                        
    # non_gaia_phot_like_line_sum = 0 # Initialisation of non-gaia photometric likelihood
    # gaia_phot_like_line_sum = 0 # Initialisation of gaia photometric likelihood
    # non_gaia_phot_col_like_line_sum = 0 # Initialisation of non-gaia photometric colour likelihood
    # gaia_phot_col_like_line_sum = 0 # Initialisation of gaia photometric colour likelilhood
    # ast_like_line_sum = 0 # Initialisation of asteroseismic likelihood
    
    ### MODIFIED DM 
    
    # M_mod_j (for all points)
    
    # create a grid of DM values based on the central DM value
    
    DM_sigma_range = 8
    DM_grid_num = 500
    
    # print("sigma DM",DM,sigma_DM)
    
    DM_grid = np.linspace(np.max([-5,DM-DM_sigma_range*sigma_DM]),DM+DM_sigma_range*sigma_DM,DM_grid_num) # this ensures the minimum DM is -5, sigma is currently hardcoded along with the steps
    non_gaia_mag_model = np.array(non_gaia_mag_model)
    if magnitude_set == 'all': # includes all photometric bands
        mag_model  = np.vstack([non_gaia_mag_model,gaia_mag_model])
        mag_obs    = np.hstack([non_gaia_mag_obs,gaia_mag_obs])
        phot_sigma = np.hstack([non_gaia_phot_sigma,gaia_phot_sigma])
    elif magnitude_set == 'gaia': # includes gaia bands only
        mag_model  = np.array(gaia_mag_model)
        mag_obs    = np.array(gaia_mag_obs)
        phot_sigma = np.array(gaia_phot_sigma)
    elif magnitude_set == 'non-gaia': # includes non-gaia bands only
        mag_model  = np.array(non_gaia_mag_model)
        mag_obs    = np.array(non_gaia_mag_obs)
        phot_sigma = np.array(non_gaia_phot_sigma)
    elif magnitude_set == 'gaia_col': # mix of gaia colours Bp-Rp, Bp-K, G-Rp
        mag_model  = np.array(gaia_col_var)
        mag_obs    = np.array(gaia_col_obs)
        phot_sigma = np.array(gaia_col_sigma)
        DM_grid    = np.zeros(2)
        DM         = 0 
    elif magnitude_set == 'gaia_bprp_col': # just Bp-Rp 
        mag_model  = np.array(gaia_col_var[:1])
        mag_obs    = np.array(gaia_col_obs[:1])
        phot_sigma = np.array(gaia_col_sigma[:1])
        DM_grid    = np.zeros(2)
        DM         = 0 

    
    ## for this formulation, we can only do it using probabilities, not easy to do it for chisq due to nature of exponents
    # for every point in the DM grid k, we calculate the probability for each band j which are summed

    L_tot = []
    for k in range(len(DM_grid)):
        L  = np.array([ np.nansum([-(mag_obs[j] - DM_grid[k] - mag_model[j])**2/(2*(phot_sigma[j])**2) for j in range(len(mag_model))],axis=0) ]) - (DM_grid[k]-DM)**2/(2*sigma_DM**2)
        L  = np.exp(L)
        L_tot.append(L)
        
    # This is now for a given model point i, a range of probabilities with a dimension in DM
    # next we marginalise over the DM dimension by summing the probabilities in the DM axis
    # N.B. should we multiply by the difference in DM? Its just a regular grid therefore shouldn't affect the shape
    
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("PHOTOMETRY Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    L_phot  = np.nansum(np.array(L_tot),axis=0)[-1]  #*(DM_grid[1]-DM_grid[0]) 
    
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("PHOTOMETRY Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    del L_tot
    
    # normalise probability by the maximum value such that we get 1   

    if len(L_phot) == 0:
        
        # raise PhotoSAPPError('Empty param space in photometry module detected.')
        print('Empty param space in photometry module detected.')
        
        phot_flag = False

        param_space = [] # added floats to here as numpy.save() was converting them into strings

        return param_space,phot_flag
    
    else:
 
        if np.max(L_phot) == 0.:
            L_phot = np.ones(len(L_phot))
            
        elif np.isnan(np.max(L_phot)): # for some reaosn we still get nans in this stage
            L_phot = np.ones(len(L_phot))
        
        else:
            L_phot /= np.max(L_phot)
            
        phot_flag = True
             
        # we're doing photometric band combination here because we marginalise afterwards, IF we create non-gaia and gaia separately then
        # combine, we're effectively doubling the imapct of DM.
        
        # what do we do with the rest of the stuff below?
        # these are the original formulations which contain the bug, so they should not be used at all
        
        # leave astroseismology, thats it 
        
        # ev_shape = evoTrackArr[27][mask].shape
        ev_shape = evoTrackArr[16][mask].shape

        ast_like_line_sum = np.zeros(ev_shape) # Initialisation of asteroseismic likelihood    
            
        ### combining Gaussian factors for Asteroseismology ###                    
        for j_ast in range(0,N_ast): 
                
            ast_sigma = ast_err[j_ast]
            ast_mu = ast_obs[j_ast]
    
            if np.isnan(ast_mu) == True or np.isnan(ast_sigma):
                continue
            
            else:
    
                ast_like_line_sum += chis_sq_one(ast_var[j_ast],ast_mu,ast_sigma)
                
                
        # normalise asteroseismology here
        
        L_ast = np.exp(-(ast_like_line_sum - min(ast_like_line_sum))/2)
           
        param_space = [evoTrackArr[3][mask],\
                       evoTrackArr[6][mask],\
                       FeH,\
                       L_phot,\
                       L_ast,\
                       evoTrackArr[0][mask]/1000.,\
                       evoTrackArr[2][mask],\
                       evoTrackArr[5][mask],\
                       evoTrackArr[1][mask]/1000,\
                       10**evoTrackArr[4][mask]] # added floats to here as numpy.save() was converting them into strings
    
        param_space = np.array(param_space).T.copy()
        
        # if len(param_space) == 0:
        #     raise PhotoSAPPError('Empty param space in photometry module detected.')
            
        return param_space, phot_flag


    ''

@jit    
def gaia_G_band_correction(G):
    
    """
    G: G band apparent magnitude from Gaia DR2
    return: G_corr, the corrected magnitude
    """
    
    if G <= 2:
        
        G_corr = G # No correction specified for G <= 2
    
    elif 2 < G <= 6:
        
        G_corr = -0.047344 + 1.16405 * G - 0.046799 * G ** 2 + 0.0035015 * G **3 # following saturation correction detailed in Evans et al 2018 Appendix B
    
    elif 6 < G <= 16:
        
        G_corr = G - 0.0032 * (G - 6) # following Maiz Apellaniz & Weiler 2018
        
    elif G > 16:
        
        G_corr = G - 0.0032 # following Maiz Apellaniz & Weiler 2018
        
    else:
        
        G_corr = G
        
#    return G_corr
    return G
@jit 
def gaia_Bp_band_correction(Bp):
    
    """
    Bp: Bp band apparent magnitude from Gaia DR2
    return: Bp_corr, the corrected magnitude
    purpose: applying corrections from Evans et al 2018 
    """

    if 2 < Bp < 4:
        
        Bp_corr = -2.0384 + 1.95282 * Bp - 0.011018 * Bp ** 2
    
    else:
        
        Bp_corr = Bp
        
#    return Bp_corr
    return Bp
@jit     
def gaia_Rp_band_correction(Rp):
    
    """
    Rp: Rp band apparent magnitude from Gaia DR2
    return: Rp_corr, the corrected magnitude
    purpose: applying corrections from Evans et al 2018 
    """

    
    if 2 < Rp < 3.5:
        
        Rp_corr = -13.946 + 14.239 * Rp - 4.23 * Rp ** 2 + 0.4532 * Rp ** 3
    
    else:
        
        Rp_corr = Rp
        
#    return Rp_corr
    return Rp

def extinction_multiple_bands(EBV,erEBV,Bp,Bp_err,Rp,Rp_err,G,G_err):
    
    """
    EBV: E(B-V) reddening calculated [mag]
    erEBV: E(B-V) reddening error associated with reddening [mag]
    return: array of extinction (A_zeta) values [mag]
    purpose: Calculates extinction based on Cassegrande et al. (2018) for different bands
    Note: It is possible to derive a EBV based on a separate measurement of Av
    
    This function requires Gaia information
    """
    
    extinction_err = erEBV
    
    A_vT = 3.24 * EBV # T stands for Tycho filter
    err_A_vT = 3.24 * extinction_err
    A_BT = 4.23 * EBV
    err_A_BT = 4.23 * extinction_err
    A_J = 0.86 * EBV
    err_A_J = 0.86 * extinction_err
    A_H = 0.5 * EBV
    err_A_H = 0.5 * extinction_err
    A_Ks = 0.3 * EBV # s stands for 2MASS
    err_A_Ks = 0.3 * extinction_err
    
    ### Gaia bands     

    Bp_Rp = Bp - Rp
    
    Bp_Rp_0 = Bp_Rp - 1.399 * EBV # de-reddened colour
    Bp_Rp_0_ERR = ((1) ** 2 * (Bp_err) ** 2  +  (1) ** 2 * (Rp_err) ** 2  +  (-1.399) ** 2 * (extinction_err) ** 2) ** 0.5
    
    Rg = 3.068 - 0.504 * Bp_Rp_0 + 0.053 * Bp_Rp_0 ** 2
    Rg_ERR = ((-0.504 + 2 * 0.053 * Bp_Rp_0) ** 2 * (Bp_Rp_0_ERR) ** 2) ** 0.5 
    A_G = Rg * EBV # check this, I don't think it should be E(B-V), should it?
    
    # print("A_G",A_G)
    err_A_G = ((EBV) ** 2 * (Rg_ERR) ** 2  +  (Rg) ** 2 * (extinction_err) ** 2) ** 0.5

    Rbp = 3.533 - 0.114 * Bp_Rp_0 - 0.219 * Bp_Rp_0 ** 2 + 0.07 * Bp_Rp_0 ** 3
    Rbp_ERR = ((- 0.114  - 2 * 0.219 * Bp_Rp_0 + 3 * 0.07 * Bp_Rp_0 ** 2) ** 2 * (Bp_Rp_0_ERR) ** 2) ** 0.5 
    A_Bp = Rbp * EBV # check this, I don't think it should be E(B-V), should it?
    err_A_Bp = ((EBV) ** 2 * (Rbp_ERR) ** 2  +  (Rbp) ** 2 * (extinction_err) ** 2) ** 0.5
    
    Rrp = 2.078 - 0.073 * Bp_Rp_0
    Rrp_ERR = ((- 0.073) ** 2 * (Bp_Rp_0_ERR) ** 2) ** 0.5 
    A_Rp = Rrp * EBV # check this, I don't think it should be E(B-V), should it?
    err_A_Rp = ((EBV) ** 2 * (Rrp_ERR) ** 2  +  (Rrp) ** 2 * (extinction_err) ** 2) ** 0.5
    
    # N.B. how does this change from Tycho V to UVES V? same with respect to 2MASS
    
    extinction_arr = [A_vT,A_BT,A_J,A_H,A_Ks,A_G,A_Bp,A_Rp]
    
    extinction_err_arr = [ err_A_vT, err_A_BT, err_A_J, err_A_H, err_A_Ks, err_A_G, err_A_Bp, err_A_Rp]
    
    # R values did not come with uncertainties
    
    # Assuming errors in E(B-V) are errors in A_zeta
    
    return [extinction_arr,extinction_err_arr]

def magnitude_converter(mag,DM,extinction):
    
    """
    mag: apparent magnitude in a certain band
    DM: distance modulus
    extinction: extinction in a certain band (corresponds to mag)
    """
    
    # absolute_mag = -(DM + extinction - mag)
    absolute_mag = -(extinction - mag)
    
    return absolute_mag
        
def photometry_asteroseismic_lhood_stellar_list(stellar_inp):
    
    """
    stellar_id: stellar id as stellar_names
    return: Nothing
    purpose: Takes in stellar id of star, finds the observational data, specifically colours associated with it,
    arranges the data in a specific format in accordance to isochrone_likelihood(), creates a photometric likelihood landscape
    in 4-D space (3 parameters Teff, logg, [M/H] and log10(P) i.e. probability)
    """
    
    stellar_id = stellar_inp[0]
    
    phot_ast_central_values = stellar_inp[1]
    
    phot_ast_limits = stellar_inp[2]
    
    star_field_name = stellar_inp[3]
    
    spec_obs_number = stellar_inp[4]
    
    magnitude_set = stellar_inp[5]
    
    save_phot_space_bool = stellar_inp[6]
    
    data_phot = stellar_inp[7]
    
    data_ast = stellar_inp[8]
    
    extra_save_string = stellar_inp[9]
    
                    
    mem = psutil.virtual_memory()
    threshhold = 0.8 * mem.total/2.**30/mp.cpu_count() # 80% of the total memory per core is noted as the "Threshold"
    
    data_phot[data_phot==''] = np.nan # because empty means nan
    data_ast[data_ast==''] = np.nan # because empty means nan
                
    G_corr = gaia_G_band_correction(float(data_phot[1]))
    Bp_corr = gaia_Bp_band_correction(float(data_phot[3]))
    Rp_corr = gaia_Rp_band_correction(float(data_phot[5]))
        
    distance = float(data_phot[7]) # distance [pc] FROM Bailer-Jones 21
    distance_err_upper = float(data_phot[8]) # distance upper error [pc] FROM Bailer-Jones 21
    distance_err_lower = float(data_phot[9]) # distance lower error [pc] FROM Bailer-Jones 21
    
    
    """
    N.B. the lines below are valid only for the PLATO bmk eDR3 file, the GES DR4 for example does not contain parallaxes from SIMBAD.
    
    Bmk stars are a lil problematic only  because of how close and bright they are, these shouldn't be needed for the more distant
    systems.

    Order of this photometry file should be the same as the cluster one except for the extra parallax columns :) 
    
    """
    
    ''
    if np.isnan(distance):
        
        # if distance from BJ 2021 is NaN, grab parallax from Gaia and do 1/pi
        
        # dividing by 1000 as parallax is presented in [mas] -- milliarcseconds
        
        # print(data_phot)
        
        parallax = float(data_phot[25])/1000 # from Gaia
        parallax_err = float(data_phot[26])/1000 # from Gaia
        
        if np.isnan(parallax):
            
            # if parallax from Gaia is NaN, grab parallax from SIMBAD (probably HIP data)
            
            parallax = float(data_phot[27])/1000 # from SIMBAD 
            parallax_err = float(data_phot[28])/1000 # from SIMBAD 
            
            # if these are still NaN, then lost cause, can't use magnitude, absolute mag will be NaN and therefore ignored by SAPP
            
        distance = 1/parallax
        distance_err_upper = np.sqrt(abs(-1/parallax**2)**2 * parallax_err ** 2)
        distance_err_lower = np.sqrt(abs(-1/parallax**2)**2 * parallax_err ** 2)
    ''    
        
    ### EXTINCTION AND DISTANCE MODULUS APPLIED TO MAGNITUDES + COLOURS ###     

#    magnitude_extinction = float(data[27]) # Av SFD # Just using Av for now, need to scale for other bands
    
    magnitude_extinction_Gaia = np.nan#float(data_phot[10]) # A_G queried from Gaia DR2, these sometimes do not exist 
    
    magnitude_reddening = float(data_phot[22]) # E(B-V)
    
    magnitude_reddening_err_pos = float(data_phot[23]) # these could be zero
    magnitude_reddening_err_neg = float(data_phot[24])
    
    # magnitude_reddening_err_ave = (magnitude_reddening_err_pos ** 2 + magnitude_reddening_err_neg ** 2) ** 0.5 # this is quadrature

    magnitude_reddening_err_ave = (magnitude_reddening_err_pos + magnitude_reddening_err_neg)/2 # this is average

    extinction_multiple_bands_arr = extinction_multiple_bands(magnitude_reddening,
                                                              magnitude_reddening_err_ave,
                                                              Bp_corr,
                                                              float(data_phot[4]),
                                                              Rp_corr,
                                                              float(data_phot[6]),
                                                              G_corr,
                                                              float(data_phot[2])) 
    
    extinction_bands = extinction_multiple_bands_arr[0] # In order of A_vT,A_BT,A_J,A_H,A_Ks,A_G,A_Bp,A_Rp
    A_vT = extinction_bands[0]
    A_BT = extinction_bands[1]
    A_J = extinction_bands[2]
    A_H = extinction_bands[3]
    A_Ks = extinction_bands[4]
    
    if np.isnan(magnitude_extinction_Gaia) == True:
        
        A_G = extinction_bands[5] 
        A_Bp = extinction_bands[6] 
        A_Rp = extinction_bands[7]
                
    else: 
        
        A_G = magnitude_extinction_Gaia
        A_Bp = magnitude_extinction_Gaia        
        A_Rp = magnitude_extinction_Gaia


    extinction_bands_errors = extinction_multiple_bands_arr[1] # likewise for magnitudes
    
    DM = distance_modulus(distance) # distance modulus to convert magnitudes to absolute magnitudes 
        
    H_mag = magnitude_converter(float(data_phot[16]),DM,A_H)
    J_mag = magnitude_converter(float(data_phot[18]),DM,A_J)
    K_mag = magnitude_converter(float(data_phot[20]),DM,A_Ks)
    V_mag = magnitude_converter(float(data_phot[14]),DM,A_vT)
    B_mag = magnitude_converter(float(data_phot[12]),DM,A_BT)
    G_mag = magnitude_converter(G_corr,DM,A_G)
    Bp_mag = magnitude_converter(Bp_corr,DM,A_Bp)
    Rp_mag = magnitude_converter(Rp_corr,DM,A_Rp)


    non_gaia_magnitude_arr = [H_mag,J_mag,K_mag,V_mag,B_mag] #This has H,J,K,V,B
    gaia_magnitude_arr = [G_mag,Bp_mag,Rp_mag] # This has G,Bp,Rp
    
    print("non_gaia_magnitude_arr",non_gaia_magnitude_arr)
    print("gaia_magnitude_arr",gaia_magnitude_arr)
    
    H_K = non_gaia_magnitude_arr[0] - non_gaia_magnitude_arr[2]
    V_K = non_gaia_magnitude_arr[3] - non_gaia_magnitude_arr[2]
    B_V = non_gaia_magnitude_arr[4] - non_gaia_magnitude_arr[3]
    V_J = non_gaia_magnitude_arr[3] - non_gaia_magnitude_arr[1]

    Bp_Rp = gaia_magnitude_arr[1] - gaia_magnitude_arr[2]
    G_Rp = gaia_magnitude_arr[0] -  gaia_magnitude_arr[2]
    
    Bp_K = gaia_magnitude_arr[1] - non_gaia_magnitude_arr[2] # Jeff recommended to add this
    Rp_K = gaia_magnitude_arr[2] - non_gaia_magnitude_arr[2] 
        
    non_gaia_magnitude_err_arr = [float(data_phot[17]),\
                                  float(data_phot[19]),\
                                  float(data_phot[21]),\
                                  float(data_phot[15]),\
                                  float(data_phot[13])] # This has erH,erJ,erK,erV,erB
        
    
    gaia_magnitude_err_arr = [float(data_phot[2]),float(data_phot[4]),float(data_phot[6])] # erG,erBp,erRp

    print("non_gaia_magnitude_err_arr",non_gaia_magnitude_err_arr)
    print("gaia_magnitude_err_arr",gaia_magnitude_err_arr)
    
    ### Checking magnitude errors for NaNs ### 
    
    mag_fud = 0.075 # Fudicial value of magnitude error, 0.1 - 0.2 is considered to be a large magnitude 
    
    non_gaia_error_nans = np.isnan(non_gaia_magnitude_err_arr)
    
    for non_gaia_error_index in range(len(non_gaia_magnitude_err_arr)):
        
        if non_gaia_error_nans[non_gaia_error_index] == True: 
            
            non_gaia_magnitude_err_arr[non_gaia_error_index] = mag_fud
            
    gaia_error_nans = np.isnan(gaia_magnitude_err_arr)
    
    for gaia_error_index in range(len(gaia_magnitude_err_arr)):
        
        if gaia_error_nans[gaia_error_index] == True:
            
            gaia_magnitude_err_arr[gaia_error_index] = mag_fud
           
    ### End of checking magnitude errors for NaNs
    
    ### COMPOUNDING PHOTOMETRY MAGNITUDE ERRORS WITH REDDENING AND DISTANCE MODULUS ###
    
    sigma_mag_non_gaia =  np.array(non_gaia_magnitude_err_arr)
    
    sigma_mag_gaia = np.array(gaia_magnitude_err_arr)
    
    # extinction_bands_errors in order of err{A_vT,A_BT,A_J,A_H,A_Ks,A_G,A_Bp,A_Rp}
    
    sigma_ext_non_gaia = np.array([extinction_bands_errors[3],extinction_bands_errors[2],extinction_bands_errors[4],extinction_bands_errors[0],extinction_bands_errors[1]]) # HJKVB
    sigma_ext_gaia = np.array([extinction_bands_errors[5],extinction_bands_errors[6],extinction_bands_errors[7]]) # GBpRp
    
    print("E(B-V) ",magnitude_reddening,"+",magnitude_reddening_err_pos,"-",magnitude_reddening_err_pos,", Gaia extinction [mag] ",extinction_bands_errors[5],extinction_bands_errors[6],extinction_bands_errors[7])

    
    sigma_DM_upper = 5 * (distance_err_upper) / (distance * np.log(10)) 
    sigma_DM_lower = 5 * (distance_err_lower) / (distance * np.log(10)) 

    
    # for now average the errors
    
    sigma_DM = (abs(sigma_DM_upper) + abs(sigma_DM_lower))/2
    
    sigma_photometry_non_gaia = (sigma_mag_non_gaia ** 2 + sigma_ext_non_gaia ** 2) ** 0.5 # order HJKVB 
    sigma_photometry_gaia = (sigma_mag_gaia ** 2 + sigma_ext_gaia ** 2) ** 0.5 # order GBpRp
    
    ### COMPOUNDING COLOUR ERRORS (WHICH ARE JUST MAGNITUDES) WITH REDDENING ###
    
    sigma_H_K = (non_gaia_magnitude_err_arr[0] ** 2 + non_gaia_magnitude_err_arr[2] ** 2 + sigma_ext_non_gaia[0] ** 2 + sigma_ext_non_gaia[2] ** 2)**0.5 # sqrt(errH^2 + errH_ext^2 + errK^2 + errK_ext^2)^1/2 
    sigma_V_K = (non_gaia_magnitude_err_arr[3] ** 2 + non_gaia_magnitude_err_arr[2] ** 2 + sigma_ext_non_gaia[3] ** 2 + sigma_ext_non_gaia[2] ** 2)**0.5
    sigma_B_V = (non_gaia_magnitude_err_arr[4] ** 2 + non_gaia_magnitude_err_arr[3] ** 2 + sigma_ext_non_gaia[4] ** 2 + sigma_ext_non_gaia[3] ** 2)**0.5
    sigma_V_J = (non_gaia_magnitude_err_arr[3] ** 2 + non_gaia_magnitude_err_arr[1] ** 2 + sigma_ext_non_gaia[3] ** 2 + sigma_ext_non_gaia[1] ** 2)**0.5

#    non_gaia_colour_err_arr = np.ones([4])*max(sigma_photometry_non_gaia)*0.2#[sigma_H_K,sigma_B_V,sigma_V_J,sigma_V_K]
    non_gaia_colour_err_arr = [sigma_H_K,sigma_B_V,sigma_V_J,sigma_V_K]

    sigma_Bp_Rp = (gaia_magnitude_err_arr[1] ** 2 + gaia_magnitude_err_arr[2] ** 2 + sigma_ext_gaia[1] ** 2 + sigma_ext_gaia[2] ** 2)**0.5
    sigma_G_Rp = (gaia_magnitude_err_arr[0] ** 2 + gaia_magnitude_err_arr[2] ** 2 + sigma_ext_gaia[0] ** 2 + sigma_ext_gaia[2] ** 2)**0.5
    sigma_Bp_K = (gaia_magnitude_err_arr[1] ** 2 + non_gaia_magnitude_err_arr[2] ** 2 + sigma_ext_gaia[1] ** 2 + sigma_ext_non_gaia[2] ** 2)**0.5   
    sigma_Rp_K = (gaia_magnitude_err_arr[2] ** 2 + non_gaia_magnitude_err_arr[2] ** 2 + sigma_ext_gaia[2] ** 2 + sigma_ext_non_gaia[2] ** 2)**0.5   
    
#    gaia_colour_err_arr = np.ones([3])*max(sigma_photometry_gaia)*0.7#[sigma_Bp_Rp,sigma_G_Rp,sigma_Bp_K]
    gaia_colour_err_arr = [sigma_Bp_Rp,sigma_G_Rp,sigma_Bp_K]
    
    ### PHOTOMETRY MAGNITUDE ### 
    
    magnitude_arr = [non_gaia_magnitude_arr,gaia_magnitude_arr]
    
    magnitude_err_arr = [sigma_photometry_non_gaia,sigma_photometry_gaia]
                
    mag_arr_input = [magnitude_arr,magnitude_err_arr] #array of magnitudes used
    
    ### PHOTOMETRY COLOUR ###
    
    gaia_colour_arr = [Bp_Rp,G_Rp,Bp_K]
    non_gaia_colour_arr = [H_K,B_V,V_J,V_K]
    
    colour_arr = [non_gaia_colour_arr,gaia_colour_arr]
    
    # Teff_IRFM_BpRp,Teff_IRFM_BpK,Teff_IRFM_RpK = IRFM_calc([Bp_Rp,sigma_Bp_Rp],[Bp_K,sigma_Bp_K],[Rp_K,sigma_Rp_K],phot_ast_central_values[1],phot_ast_central_values[2]) # requires

    # print("IRFM_TEFF : Bp-Rp, Bp-K, Rp-K",Teff_IRFM_BpRp,Teff_IRFM_BpK,Teff_IRFM_RpK)
    
    # from sbcr_plato_v1 import SCBR_VK
        
    non_gaia_colour_err = non_gaia_colour_err_arr # these are in same order as respective colour arrays 
    gaia_colour_err = gaia_colour_err_arr
    
    colour_err_input = [non_gaia_colour_err,gaia_colour_err]
        
    colour_arr_input = [colour_arr,colour_err_input] # This is used for the colour likelihood 
   
    ### ASTEROSEISMOLOGY ###
    
    astrosize_arr = [float(data_ast[4]),float(data_ast[1])] # delta_nu, nu_max
    astrosize_err_arr = [float(data_ast[5]),float(data_ast[2])] # delta_nu_err, err_nu_max
        
    astrosize_delta_err_input = astrosize_err_arr[0]
    astrosize_max_err_input = astrosize_err_arr[1]
    
    astrosize_delta_nu_sol_err = astroc.delta_nu_sol_err
    astrosize_nu_max_sol_err = astroc.nu_max_sol_err 
    
    astrosize_delta_err_fud = 1.5 # Fudicial value of delta_nu error [microhertz]
    
    astrosize_max_err_fud = 150 # Fudicial value of nu_max error [microhertz]
    
    if np.isnan(astrosize_delta_err_input) == True:
        
        astrosize_delta_err_input = astrosize_delta_err_fud
    
    if np.isnan(astrosize_max_err_input) == True:
        
        astrosize_max_err_input = astrosize_max_err_fud
            
    astrosize_arr_input = [astrosize_arr,[(astrosize_delta_err_input**2 + astrosize_delta_nu_sol_err**2) ** 0.5,(astrosize_max_err_input**2 + astrosize_nu_max_sol_err**2)**0.5]] # array of asteroseismic quantities used
                        
    print(f"starting photometry likelihood calculation for star {stellar_id}")
                
    phot_ast_probability_hbs = stellar_evo_likelihood(mag_arr_input,colour_arr_input,astrosize_arr_input,distance,phot_ast_central_values,phot_ast_limits,DM,sigma_DM,magnitude_set)

    Like_tot_param_space,phot_flag = phot_ast_probability_hbs #initialisation 
                            
    stellar_id = stellar_id.replace(" ","_") # replaces an empty space in the name with an underscore
            
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    if memoryUse >= threshhold:
        
        print("Warning, memory per core is at 80%, Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
        
    directory_photometry = star_field_name # name of directory is the name of the star
    directory_check = os.path.exists(f"../Output_data/Stars_Lhood_photometry/{directory_photometry}")
    
    if  directory_check == True:
    
        print(f"../Output_data/Stars_Lhood_photometry/{directory_photometry} directory exists")
        
    else:
        
        print(f"../Output_data/Stars_Lhood_photometry/{directory_photometry} directory does not exist")
        
        os.makedirs(f"../Output_data/Stars_Lhood_photometry/{directory_photometry}")
        
        print(f"../Output_data/Stars_Lhood_photometry/{directory_photometry} directory has been created")

    if save_phot_space_bool:

        np.savetxt(f'../Output_data/Stars_Lhood_photometry/{directory_photometry}/stellar_track_collection_{stellar_id}_OBS_NUM_{spec_obs_number + 1}_test_4_{extra_save_string}.txt',Like_tot_param_space,delimiter=",") 


    return Like_tot_param_space,phot_flag
    

#@jit    
def FeH_calc(Z_surf,X_surf):
    
    """
    Z_surf: surface metallicity of stellar model
    X_surf: surface hydrogen abundance of stellar model
    return: [Fe/H]
    """

    FeH = np.log10(Z_surf/(X_surf * 0.02439)) # This equation was given by Aldo for his most recent models
                                                # 0.02439 is Z_sol/X_sol
    
    return FeH

class PhotoSAPPError(Exception):
    pass

### N.B. These pathways are assuming this module is being run from main_TEST.py 

print('loading evo tracks')

# shape = (49,46255417) # this is the shape of the large stellar evolution file 
shape = (24,9453868) # this is the shape of the large stellar evolution file 

test_input_path = ""
# test_input_path = "../SAPP_v1.1_clean_thesis_use/"

if os.path.exists("../" + test_input_path + "Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap_v2.npy"):
# if os.path.exists("../" + test_input_path + "Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap.npy"): 

    evoTrackArr = np.memmap('../' + test_input_path + 'Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap_v2.npy',shape=shape,mode='r',dtype=np.float64)
    # evoTrackArr = np.memmap('../' + test_input_path + 'Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap.npy',shape=shape,mode='r',dtype=np.float64)
    
    # print(evoTrackArr[8],evoTrackArr[7])

    evoTrackFeh = evoTrackArr[7]    
    # evoTrackFeh = FeH_calc(evoTrackArr[8],evoTrackArr[7])

# else:
    
#     print('Memmap doesn\'t exsits: Creating new one. This loads it into RAM! If this is not possible for you consider a different approach')
    
#     filesize = os.path.getsize('../Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_collection_total.npy')/10 ** 9 # GB
    
#     mem_available = psutil.virtual_memory()[1]/2.**30
    
#     if filesize >= 0.8 * mem_available:
        
#         print("file too large!")
        
#         raise MemoryError
        
#     else:
       
#         # could check the file size and the local virtual mem and if the file size exceeds something like 80%
#         # then manually throw an errir

#         evoTrackArrmm = np.memmap('../Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap_v2.npy',shape=shape,mode='w+',dtype=np.float64)        
        
#         evoTrackArr = np.load('../Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_collection_total.npy',allow_pickle=True)
        
#         evoTrackArrmm[:,:] = evoTrackArr 
#         del evoTrackArrmm
#         del evoTrackArr
#         evoTrackArr = np.memmap('../Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap_v2.npy',shape=shape,mode='r',dtype=np.float64)
#         print('evo tracks loaded')

#         evoTrackFeh = FeH_calc(evoTrackArr[8],evoTrackArr[7])
