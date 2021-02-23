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
    
def distance_modulus(pax): 
    
    """
    pax: parallax of star [mas]
    return: distance modulus
    purpose: calculate distance modulus using parallax
    """

    distance = 10 ** 3/pax #for now, [pc] 
    
    dist_mod = 5 * np.log10(distance/10) # 10 is in parsecs
        
    return dist_mod

@jit
def Lhood_norm_red_chi_ind(var,mu,sigma,nu,N_k):
    
    if nu <= 0:
        
        nu = 1
        
        
    L_i = EXP ** (-((mu-var)/sigma) ** 2/(2 * nu)) 
    
    return L_i 

#@jit
def stellar_evo_likelihood(mag_arr,colour_arr,astrosize_arr,pax,hbs_index_i,spec_central_values,phot_ast_limits): 
    
    """
    mag_arr: input observed photometry array; [magnitudes,error,extinction]
    astrosize_arr: input observed asteroseismic array; [[delta_nu,nu_max],[delta_nu_err,nu_max_error]]
    colour_arr: input observed photometry array; [colours,error]
    pax: parallax [mas]
    hbs_index_i: specific stellar evolution file string
    return: Three 3-D distribution of likelihoods for photometry, asteroseismology and combined of single star
    purpose: Takes in observational photometric data, asteroseismic data, parallax from Gaia and a file name string
    to calculate the likelihood function. This is done by loading up the model file line by line,
    calculate the likelihood probability and then save it as the new parameter space. 
    """
    
    teff_central = spec_central_values[0]
    logg_central = spec_central_values[1]
    feh_central = spec_central_values[2]
    
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

    param_space = []
        
    with open("../Input_data/GARSTEC_stellar_evolution_models/evo_tracks/{}".format(hbs_index_i)) as infile:
        
        lis = [line.split() for line in infile]

        for i,x in enumerate(lis):
                        
            if x[0] == '#': # These are the comments from the stellar model headers, unfortunately it is tailored to the stellar model
                            # Although, it is convention for at least '#' to be used

                continue
            
            else:
                                
                # due to line.split(), the order of parameters is located in Aldo_stellar_evo/aldo_header.txt
                                
                if float(x[3]) > teff_central + teff_limit: # Temperature upper limit
                
                    continue
                
                if float(x[3]) < teff_central - teff_limit: # Temperature lower limit
                    
                    continue

                if float(x[6]) > logg_central + logg_limit: # logg upper limit
                    
                    continue

                if float(x[6]) < logg_central - logg_limit: # logg lower limit
                    
                    continue
                
                if float(x[0])/1000 > 14: # Age upper limit
                
                    continue
                
                FeH = FeH_calc(float(x[8]),float(x[7])) # Calculates [Fe/H] for given star from surface Z and X

                if FeH > feh_central + feh_limit: 
                        
                        continue
                    
                if FeH < feh_central - feh_limit:
                        
                        continue

                # print(float(x[3]),teff_central + teff_limit,teff_central - teff_limit)
                
                ### Non-Gaia magnitudes ###

                non_gaia_mag_model = [float(x[27]),float(x[26]),float(x[28]),float(x[23]),float(x[22])] # H,J,K,V,B
                non_gaia_phot_var = non_gaia_mag_model # Gaussian input variable for non gaia magnitudes
                non_gaia_phot_sigma = non_gaia_mag_err # Assuming each factor has the same standard deviation
                N_non_gaia_phot = len(non_gaia_mag_model) # No. of gaussian factors for photometry
                nu_non_gaia_phot = N_non_gaia_phot - d # degrees of freedom
                    
                if nu_non_gaia_phot <= 0:
                        
                    nu_non_gaia_phot = 1
                    
                ### Gaia magnitudes ###
                    
                gaia_mag_model = [gaia_G_band_correction(float(x[34])),gaia_Bp_band_correction(float(x[35])),gaia_Rp_band_correction(float(x[36]))] # G,Bp,Rp
                gaia_phot_var = gaia_mag_model # Gaussian input variable for gaia magnitudes 
                gaia_phot_sigma = gaia_mag_err # Assuming each factor has the same standard deviation
                N_gaia_phot = len(gaia_mag_model) # No. of gaussian factors for photometry
                nu_gaia_phot = N_gaia_phot - d # degrees of freedom

                if nu_gaia_phot <= 0:
                        
                    nu_gaia_phot = 1
                    
                ### Non-Gaia Colours ###
                    
                non_gaia_col_var = [non_gaia_mag_model[0]-non_gaia_mag_model[2],non_gaia_mag_model[4]-non_gaia_mag_model[3],non_gaia_mag_model[3]-non_gaia_mag_model[1],non_gaia_mag_model[3]-non_gaia_mag_model[2]] # H_K,B_V,V_J,V_K
                non_gaia_col_sigma = non_gaia_col_err
                N_non_gaia_phot_col = len(non_gaia_col_var) # No. of gaussian factors for photometry
                nu_phot_non_gaia_col = N_non_gaia_phot_col - d # degrees of freedom

                if nu_phot_non_gaia_col <= 0:
                        
                    nu_phot_non_gaia_col = 1
                    
                ### Gaia Colours ###
    
    #                gaia_col_var = [gaia_mag_model[1]-gaia_mag_model[2],gaia_mag_model[0]-gaia_mag_model[2]] # Bp_Rp,G_Rp
                gaia_col_var = [gaia_mag_model[1]-gaia_mag_model[2],gaia_mag_model[0]-gaia_mag_model[2],gaia_mag_model[1]-non_gaia_mag_model[2]] # Bp_Rp,G_Rp,Bp_K                
                gaia_col_sigma = gaia_col_err
                N_gaia_phot_col = len(gaia_col_var) # No. of gaussian factors for photometry
                nu_phot_gaia_col = N_gaia_phot_col - d # degrees of freedom

                if nu_phot_gaia_col <= 0:
                        
                    nu_phot_gaia_col = 1
                    
                ### Asteroseismology ###
                    
                nu_max_model = astroc.nu_max_sol * 10 ** (float(x[6]))/astroc.surf_grav_sol * (astroc.teff_sol/float(x[3])) ** 0.5
                delta_nu_model = float(x[16]) * astroc.delta_nu_sol/136.3 # 136.3 uHz is Aldo's model solar value
                ast_var = [delta_nu_model,nu_max_model]
                N_ast = len(ast_var)
                nu_ast = N_ast-d # degrees of freedom 

                if nu_ast <= 0:
                        
                    nu_ast = 1
                                    
                non_gaia_phot_like_line_sum = 0 # Initialisation of non-gaia photometric likelihood
                gaia_phot_like_line_sum = 0 # Initialisation of gaia photometric likelihood
                non_gaia_phot_col_like_line_sum = 0 # Initialisation of non-gaia photometric colour likelihood
                gaia_phot_col_like_line_sum = 0 # Initialisation of gaia photometric colour likelilhood
                ast_like_line_sum = 0 # Initialisation of asteroseismic likelihood
                    
                
                ### combining Gaussian factors for Non-Gaia magnitudes ###
                
                for j_non_gaia_phot in range(0,N_non_gaia_phot): # Multiply gaussian factors assuming they are independent to each other
                        
                    phot_non_gaia_mu = non_gaia_mag_obs[j_non_gaia_phot] # Gaussian mean, converting to absolute and de-reddening  
                    
                    # print(i,phot_non_gaia_mu)
                    
                    if np.isnan(phot_non_gaia_mu) == True:
                        
                        continue
                    
                    else:
                    
                        non_gaia_phot_like_line_sum +=  chis_sq_one(non_gaia_phot_var[j_non_gaia_phot],phot_non_gaia_mu,non_gaia_phot_sigma[j_non_gaia_phot])  
                        
                        # print(i,non_gaia_phot_like_line_sum)

                ### combining Gaussian factors for Non-Gaia colours ###
                '''
                for j_non_gaia_col in range(0,N_non_gaia_phot_col):
                    
                    phot_non_gaia_col_mu = non_gaia_col_obs[j_non_gaia_col] # Gaussian mean, for colours distance modulus and extinction are assumed to cancel

                    if np.isnan(phot_non_gaia_col_mu) == True:
                        
                        continue
                    
                    else:
    
                        non_gaia_phot_col_like_line_sum += chis_sq_one(non_gaia_col_var[j_non_gaia_col],phot_non_gaia_col_mu,non_gaia_col_sigma[j_non_gaia_col])
                '''
                
                ### here we want the last non-Gaia colour i.e. V-K
                
                j_non_gaia_col = N_non_gaia_phot_col - 1
                
                phot_non_gaia_col_mu = non_gaia_col_obs[j_non_gaia_col] # Gaussian mean, for colours distance modulus and extinction are assumed to cancel
                
                
                if np.isnan(phot_non_gaia_col_mu) == True:
                        
                    non_gaia_phot_col_like_line_sum = 0
                    
                else:
    
                    non_gaia_phot_col_like_line_sum = chis_sq_one(non_gaia_col_var[j_non_gaia_col],phot_non_gaia_col_mu,non_gaia_col_sigma[j_non_gaia_col])
                
                
                ### combining Gaussian factors for Gaia magnitudes ###

                for j_gaia_phot in range(0,N_gaia_phot): # Multiply gaussian factors assuming they are independent to each other
                                            
                    phot_gaia_mu = gaia_mag_obs[j_gaia_phot] # Gaussian mean, converting to absolute and de-reddening  

                    if np.isnan(phot_gaia_mu) == True:
                        
                        continue
                    
                    else:
                    
                        gaia_phot_like_line_sum += chis_sq_one(gaia_phot_var[j_gaia_phot],phot_gaia_mu,gaia_phot_sigma[j_gaia_phot])

                ### combining Gaussian factors for Gaia Colours ###
                
                
                '''            
                for j_gaia_col in range(0,N_gaia_phot_col):
    
                    phot_gaia_col_mu = gaia_col_obs[j_gaia_col] # Gaussian mean, for colours distance modulus and extinction are assumed to cancel

                    if np.isnan(phot_gaia_col_mu) == True:
                        
                        continue
                    
                    else:
                    
                        gaia_phot_col_like_line_sum += chis_sq_one(gaia_col_var[j_gaia_col],phot_gaia_col_mu,gaia_col_sigma[j_gaia_col])
                '''
                
                ### here we want the first Gaia colour i.e. Bp-Rp 
                
                j_gaia_col = 0
                
                phot_gaia_col_mu = gaia_col_obs[j_gaia_col]
                
                if np.isnan(phot_gaia_col_mu) == True:
                    
                    gaia_phot_col_like_line_sum = 0
                    
                else:
                
                    gaia_phot_col_like_line_sum = chis_sq_one(gaia_col_var[j_gaia_col],phot_gaia_col_mu,gaia_col_sigma[j_gaia_col])
                
                
                ### combining Gaussian factors for Asteroseismology ###                    
                ''
                for j_ast in range(0,N_ast): 
                        
                    ast_sigma = ast_err[j_ast]
                    ast_mu = ast_obs[j_ast]

                    if np.isnan(ast_mu) == True:
                        
                        continue
                    
                    else:

                        ast_like_line_sum += chis_sq_one(ast_var[j_ast],ast_mu,ast_sigma)
               
                ''
               
                ### here, we want the last asterosesimology component i.e. nu_max
               
                # j_ast =  N_ast - 1
               
                # ast_sigma = ast_err[j_ast]
                # ast_mu = ast_obs[j_ast]

                # if np.isnan(ast_mu) == True:
                        
                #    ast_like_line_sum = 0
                    
                # else:

                #     ast_like_line_sum = chis_sq_one(ast_var[j_ast],ast_mu,ast_sigma)
               
               
               
               
               
#                         # New param space Teff logg [Fe/H] Lhood_phot Lhood colour Lhood_ast Lhood_comb Age/Gyr M/Msol R/Rsol
                            
                param_space.append([float(x[3]),\
                                    float(x[6]),\
                                    FeH,\
                                    float(non_gaia_phot_like_line_sum),\
                                    float(gaia_phot_like_line_sum),\
                                    float(non_gaia_phot_col_like_line_sum),\
                                    float(gaia_phot_col_like_line_sum),\
                                    float(ast_like_line_sum),\
                                    float(x[0])/1000,\
                                    float(x[2]),\
                                    float(x[5]),\
                                    float(x[1])/1000,\
                                    10 ** float(x[4])])# added floats to here as numpy.save() was converting them into strings


                ''

                # param_space.append(10 ** float(x[4]))
                
                # print(non_gaia_phot_like_line_sum,gaia_phot_like_line_sum,non_gaia_phot_col_like_line_sum,gaia_phot_col_like_line_sum)
    
    param_space = np.array(param_space)
    
    ''
    
    ### change the degrees of freedom, just in case ###
    # this is because for both colours and asteroseismology, we only want one variable 
    
    nu_phot_non_gaia_col = 1
    nu_phot_gaia_col = 1
    nu_ast = 1
    
    nu_dof_arr = [nu_non_gaia_phot,\
                  nu_gaia_phot,\
                  nu_phot_non_gaia_col,\
                  nu_phot_gaia_col,\
                  nu_ast]
        
    nu_dof_arr = np.array(nu_dof_arr)
            
    return [param_space,nu_dof_arr]

    ''
    
    # return param_space

def choose_star_photometry(stellar_id):
    
    """
    stellar_id: stellar id name as a string,
    return: photometry data from combined photometry file for specific star
    """
        
    array = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/PLATO_bmk_phot_data/PLATO_photometry.csv',dtype=str,delimiter=",")

    star = []
       
    for i in range(len(array)):
        
        if stellar_id == array[i][0]:
            
            star = array[i]
            
            break
        
        else:
            
            continue
   
    return star

def choose_star_asteroseismology(stellar_id):
    
    """
    stellar_id: stellar id name as a string,
    return: photometry data from combined photometry file for specific star
    """
        
    array = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/PLATO_stars_seism.txt',dtype=str,delimiter=",")

    star = []
    
    for i in range(len(array)):
        
        if stellar_id == array[i][0]:
            
            star = array[i]
            
            break
        
        else:
            
            continue

   
    return star


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

def extinction_multiple_bands(EBV,erEBV):
    
    """
    EBV: E(B-V) reddening calculated [mag]
    erEBV: E(B-V) reddening error associated with reddening [mag]
    return: array of extinction (A_zeta) values [mag]
    purpose: Calculates extinction based on Cassegrande et al. (2018) for different bands
    Note: It is possible to derive a EBV based on a separate measurement of Av
    """
    
    if np.isnan(erEBV) == False:
        
        extinction_err = erEBV

    else:
        
        extinction_err = 0.15 * EBV # if an error doesn't exist, then fiducial value is calcd.

    
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
    
    # For now we're assuming AG (and so Bp, Rp) is = Av if one doesn't exist in Gaia database
    
    A_G = A_vT
    err_A_G = 3.24 * extinction_err
    A_Bp = A_vT
    err_A_Bp = 3.24 * extinction_err
    A_Rp = A_vT
    err_A_Rp = 3.24 * extinction_err
    
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
    
    absolute_mag = -(DM + extinction - mag)
    
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
            
    mem = psutil.virtual_memory()
    threshhold = 0.8 * mem.total/2.**30/mp.cpu_count() # 80% of the total memory per core is noted as the "Threshold"
    
    hbs_index = np.loadtxt('../Input_data/GARSTEC_stellar_evolution_models/models_list_new.txt',dtype='str')
        
    data_phot = choose_star_photometry(stellar_id) 
    data_ast = choose_star_asteroseismology(stellar_id) 

            
    G_corr = gaia_G_band_correction(float(data_phot[1]))
    Bp_corr = gaia_Bp_band_correction(float(data_phot[3]))
    Rp_corr = gaia_Rp_band_correction(float(data_phot[5]))
    
    # data has the current order : [Stars, RA+DEC, Mik_index, Parallax, Parallax_err, delta_nu, delta_nu_err, nu_max, err_nu_max, H, erH, J, erJ, K, erK, V, erV, B, erB, G, erG, Bp, erBp, Rp, erRp, A_G, E(B-V)_SFD, Av_SFD, Av_S&F]
    
    star_name = data_phot[0]
    
    parallax = float(data_phot[7]) # paralax [mas] FROM Gaia
    
    parallax_err = float(data_phot[8]) # parallax [mas] error from Gaia
    
    parallax_err_fiducial = 0.2 * parallax # fiducial error for parallax is 20% i.e. the maximum we could use to keep doing d = 1/p 
    
    if np.isnan(parallax) == True:
        
        parallax = float(data_phot[21]) # parallax from SIMBAD, sometimes there isn't a parallax (weird)
        
        parallax_err = float(data_phot[22]) # parallax error from SIMBAD
        
    ### EXTINCTION AND DISTANCE MODULUS APPLIED TO MAGNITUDES + COLOURS ###     

#    magnitude_extinction = float(data[27]) # Av SFD # Just using Av for now, need to scale for other bands
    
    magnitude_extinction_Gaia = float(data_phot[9]) # A_G queried from Gaia DR2, these sometimes do not exist 
    
    magnitude_reddening = float(data_phot[23]) # E(B-V)calculated from stilism
    
    magnitude_reddening_err_pos = float(data_phot[24])
    magnitude_reddening_err_neg = float(data_phot[25])
    
    magnitude_reddening_err_ave = (magnitude_reddening_err_pos ** 2 + magnitude_reddening_err_neg ** 2) ** 0.5 # this is quadrature

    # magnitude_reddening_err_ave = (magnitude_reddening_err_pos + magnitude_reddening_err_neg)/2 # this is average

    
    extinction_multiple_bands_arr = extinction_multiple_bands(magnitude_reddening,magnitude_reddening_err_ave) 
    
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
    
    DM = distance_modulus(parallax) # distance modulus to convert magnitudes to absolute magnitudes 
    
    H_mag = magnitude_converter(float(data_phot[15]),DM,A_H)
    J_mag = magnitude_converter(float(data_phot[17]),DM,A_J)
    K_mag = magnitude_converter(float(data_phot[19]),DM,A_Ks)
    V_mag = magnitude_converter(float(data_phot[13]),DM,A_vT)
    B_mag = magnitude_converter(float(data_phot[11]),DM,A_BT)
    G_mag = magnitude_converter(G_corr,DM,A_G)
    Bp_mag = magnitude_converter(Bp_corr,DM,A_Bp)
    Rp_mag = magnitude_converter(Rp_corr,DM,A_Rp)


    non_gaia_magnitude_arr = [H_mag,J_mag,K_mag,V_mag,B_mag] #This has H,J,K,V,B
    gaia_magnitude_arr = [G_mag,Bp_mag,Rp_mag] # This has G,Bp,Rp
    
    H_K = non_gaia_magnitude_arr[0] - non_gaia_magnitude_arr[2]
    V_K = non_gaia_magnitude_arr[3] - non_gaia_magnitude_arr[2]
    B_V = non_gaia_magnitude_arr[4] - non_gaia_magnitude_arr[3]
    V_J = non_gaia_magnitude_arr[3] - non_gaia_magnitude_arr[1]

    Bp_Rp = gaia_magnitude_arr[1] - gaia_magnitude_arr[2]
    G_Rp = gaia_magnitude_arr[0] -  gaia_magnitude_arr[2]
    
    Bp_K = gaia_magnitude_arr[1] - non_gaia_magnitude_arr[2] # Jeff recommended to add this
    
    non_gaia_magnitude_err_arr = [float(data_phot[16]),float(data_phot[18]),float(data_phot[19]),float(data_phot[14]),float(data_phot[12])] # This has erH,erJ,erK,erV,erB
    
    gaia_magnitude_err_arr = [float(data_phot[2]),float(data_phot[4]),float(data_phot[6])] # erG,erBp,erRp
    
    
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
    
    sigma_DM = 5 * (parallax_err/1000) / (parallax/1000 * np.log(10)) # this is assuming d = 1/p still holds 
    
    sigma_photometry_non_gaia = (sigma_mag_non_gaia ** 2 + sigma_ext_non_gaia ** 2 + (np.ones([len(sigma_mag_non_gaia)])*sigma_DM) ** 2) ** 0.5 # order HJKVB 
    sigma_photometry_gaia = (sigma_mag_gaia ** 2 + sigma_ext_gaia ** 2 + (np.ones([len(sigma_mag_gaia)])*sigma_DM) ** 2) ** 0.5 # order GBpRp
    
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
    
#    gaia_colour_err_arr = np.ones([3])*max(sigma_photometry_gaia)*0.7#[sigma_Bp_Rp,sigma_G_Rp,sigma_Bp_K]
    gaia_colour_err_arr = [sigma_Bp_Rp,sigma_G_Rp,sigma_Bp_K]
    
    ### PHOTOMETRY MAGNITUDE ### 
    
    magnitude_arr = [non_gaia_magnitude_arr,gaia_magnitude_arr]
    
    magnitude_err_arr = [sigma_photometry_non_gaia,sigma_photometry_gaia]
            
#    non_gaia_max_magnitude_error = max(non_gaia_magnitude_err_arr) #standard deviation for all colours, using maximum so far as don't have erB, erV
#    gaia_max_magnitude_error = max(gaia_magnitude_err_arr)
    
    mag_arr_input = [magnitude_arr,magnitude_err_arr] #array of magnitudes used
    
    ### PHOTOMETRY COLOUR ###
    
    gaia_colour_arr = [Bp_Rp,G_Rp,Bp_K]
    non_gaia_colour_arr = [H_K,B_V,V_J,V_K]
    
    colour_arr = [non_gaia_colour_arr,gaia_colour_arr]
    
#    gaia_colour_err = gaia_max_magnitude_error # errors in color are defined by errors in magnitdues
#    non_gaia_colour_err = non_gaia_max_magnitude_error
    
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
        
    
    N_hbs_index = len(hbs_index)
            
    Like_tot_param_space_list = []
    
    print(f"starting photometry likelihood calculation for star {stellar_id}")
            
    for iso_index in range(0,N_hbs_index):
                
        hbs_index_i = hbs_index[iso_index]
        
        # hbs_zero_metal_check = hbs_index_i.split("z")[1][:4]
        
        
        # """
        # GRABBING ONLY ZERO METALLICITY STARS
        # """
                
        # if hbs_zero_metal_check != "p000":
            
        #     if hbs_zero_metal_check != "p005":
            
        #         continue
            
        # else:
            
        #     pass
        
        # print(hbs_zero_metal_check)        
                
        phot_ast_probability_hbs = stellar_evo_likelihood(mag_arr_input,colour_arr_input,astrosize_arr_input,parallax,hbs_index_i,phot_ast_central_values,phot_ast_limits)
        
        Like_tot_param_space_i  = phot_ast_probability_hbs[0]
        # Like_tot_param_space_i  = phot_ast_probability_hbs
        
        nu_dof_arr = phot_ast_probability_hbs[1]
        
        if len(Like_tot_param_space_i) == 0: # If the variable is a non-populated list, then continue
            
            continue
        
        else:
            
            Like_tot_param_space_list.append(Like_tot_param_space_i)
    
    if len(Like_tot_param_space_list) == 0:
        
        print('Parameter space empty, loosen constraints')
        
        return Like_tot_param_space_list
    
    Like_tot_param_space = Like_tot_param_space_list[0] #initialisation 
    
    for build_index in range(1,len(Like_tot_param_space_list)): # The length of the list will be < length of hbs_index because of empty arrays being thrown out 
        
        Like_tot_param_space = np.vstack([Like_tot_param_space ,Like_tot_param_space_list[build_index]])   
        # Like_tot_param_space = np.hstack([Like_tot_param_space ,Like_tot_param_space_list[build_index]])   
            
#    stellar_id = stellar_id.replace("+","p").replace("-","m") # replaces +/- with p/m in filename respectively
    ''
    chi_2_non_gaia_mag = Like_tot_param_space[:,3]
    chi_2_gaia_mag = Like_tot_param_space[:,4]
    chi_2_non_gaia_col = Like_tot_param_space[:,5]
    chi_2_gaia_col = Like_tot_param_space[:,6]
    chi_2_ast = Like_tot_param_space[:,7]
            
    nu_non_gaia_phot = nu_dof_arr[0]
    nu_gaia_phot = nu_dof_arr[1]
    nu_non_gaia_col = nu_dof_arr[2]
    nu_gaia_col = nu_dof_arr[3]
    nu_ast  = nu_dof_arr[4]
        
    Lhood_non_gaia_mag_norm = chi_2_non_gaia_mag
    Lhood_gaia_mag_norm = chi_2_gaia_mag
    Lhood_non_gaia_col_norm = chi_2_non_gaia_col
    Lhood_gaia_col_norm = chi_2_gaia_col
    Lhood_ast_norm = chi_2_ast

                            
    for lhood_loop in range(len(Like_tot_param_space)):
        
        Like_tot_param_space[lhood_loop] = np.hstack((Like_tot_param_space[lhood_loop][:3],
                                                          Lhood_non_gaia_mag_norm[lhood_loop],
                                                          Lhood_gaia_mag_norm[lhood_loop],
                                                          Lhood_non_gaia_col_norm[lhood_loop],
                                                          Lhood_gaia_col_norm[lhood_loop],
                                                          Lhood_ast_norm[lhood_loop],                                                          
                                                          Like_tot_param_space[lhood_loop][8:]))
    
    ''
    
    
    stellar_id = stellar_id.replace(" ","_") # replaces an empty space in the name with an underscore
    
#    np.savetxt('Stars_Lhood_photometry/Gaia_benchmark/Lhood_space_photometry_best_params_GaiaBMS_starID{}'.format(stellar_id),best_fit_params,header=' Teff/K logg/dex [Fe/H]/dex Lhood_phot/dex Lhood_phot_col/dex Lhood_ast/dex Lhood_comb/dex Age/Gyr M/Msol') # Saving the best fit parameters from the combined likelihood plot

    # print(nu_dof_arr,type(nu_dof_arr),np.shape(nu_dof_arr))

        
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    if memoryUse >= threshhold:
        
        print("Warning, memory per core is at 80%, Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    directory_photometry = stellar_id # name of directory is the name of the star
    directory_check = os.path.exists(f"../Output_data/Stars_Lhood_photometry/{directory_photometry}")
    
    if  directory_check == True:
    
        print(f"../Output_data/Stars_Lhood_photometry/{directory_photometry} directory exists")
        
    else:
        
        print(f"../Output_data/Stars_Lhood_photometry/{directory_photometry} directory does not exist")
        
        os.makedirs(f"../Output_data/Stars_Lhood_photometry/{directory_photometry}")
        
        print(f"../Output_data/Stars_Lhood_photometry/{directory_photometry} directory has been created")
        

    # np.save(f'../Output_data/Stars_Lhood_photometry/{directory_photometry}/Lhood_space_photometry_tot_starID{stellar_id}',Like_tot_param_space,allow_pickle=True) 
    
    # print("TOTAL NO. POINTS EVO MODEL = ",len(Like_tot_param_space[:,0]))
    
    # print(max(Lhood_ast_norm))
    np.savetxt(f"../Output_data/Stars_Lhood_photometry/{directory_photometry}/degrees_freedom_phot_ast.txt",nu_dof_arr,delimiter=",") ### SAVE NU VALUES HERE
    
    np.savetxt(f'../Output_data/Stars_Lhood_photometry/{directory_photometry}/stellar_track_collection_{stellar_id}_test_3.txt',Like_tot_param_space,delimiter=",") 
    # np.savetxt(f'../Output_data/Stars_Lhood_photometry/{directory_photometry}/stellar_track_collection_{stellar_id}_test_3_non_stellar_evo_priors.txt',Like_tot_param_space,delimiter=",") 
    
    # np.savetxt(f'../Output_data/Stars_Lhood_photometry/{directory_photometry}/luminosity_{stellar_id}_test_3.txt',Like_tot_param_space,delimiter=",") 

    return Like_tot_param_space
    

@jit    
def FeH_calc(Z_surf,X_surf):
    
    """
    Z_surf: surface metallicity of stellar model
    X_surf: surface hydrogen abundance of stellar model
    return: [Fe/H]
    """

    FeH = np.log10(Z_surf/(X_surf * 0.02439)) # This equation was given by Aldo for his most recent models
                                                # 0.02439 is Z_sol/X_sol
    
    return FeH

