#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:25:17 2022

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

    
#@jit
def stellar_evo_likelihood(astrosize_arr,spec_central_values,phot_ast_limits): 
    
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
        
    ast_obs = astrosize_arr[0] # array of stellar asteroseismic quantities: delta_nu, nu_max
    ast_err = astrosize_arr[1] # error of stellar asteroseismic quantities: delta_nu_err, err_nu_max
    
    d = 3 # dimensionality of non-probability parameter space ;  Teff, logg, [Fe/H]

    # initialisation of degrees of freedom

    nu_ast = 1

   # param_space = []
   
    # The file has already been read.
    mask = (evoTrackArr[3]<teff_central+teff_limit)&(evoTrackArr[3]>teff_central-teff_limit)&\
           (evoTrackArr[6]<logg_central+logg_limit)&(evoTrackArr[6]>logg_central-logg_limit)&\
           (evoTrackFeh<feh_central+feh_limit)&(evoTrackFeh>feh_central-feh_limit)&\
           (evoTrackArr[0] > 30)&(evoTrackArr[0]/1000 <= 16)
           # (evoTrackArr[0] > 30)&(evoTrackArr[0]/1000 <= 16)
  
    FeH = evoTrackFeh[mask]
        
    ### Asteroseismology ###
        
    # nu_max_model = astroc.nu_max_sol * 10 ** (float(x[6]))/astroc.surf_grav_sol * (astroc.teff_sol/float(x[3])) ** 0.5
    # delta_nu_model = float(x[16]) * astroc.delta_nu_sol/136.3 # 136.`3 uHz is Aldo's model solar value
    nu_max_model = astroc.nu_max_sol * 10 ** (evoTrackArr[6][mask])/astroc.surf_grav_sol * (astroc.teff_sol/evoTrackArr[3][mask]) ** 0.5
    delta_nu_model = evoTrackArr[16][mask] * astroc.delta_nu_sol/136.3 # 136.3 uHz is Aldo's model solar value
    ast_var = [delta_nu_model,nu_max_model]
    N_ast = len(ast_var)
    # nu_ast = N_ast-d # degrees of freedom     
        
    ev_shape = evoTrackArr[27][mask].shape
    # ev_shape = evoTrackArr[16][mask].shape

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
    
    del ast_like_line_sum
    
    if len(L_ast) == 0:
        ast_flag = True
    else:
        ast_flag = False 
       
        param_space = [evoTrackArr[3][mask],\
                       evoTrackArr[6][mask],\
                       FeH,\
                       L_ast,\
                       evoTrackArr[0][mask]/1000.,\
                       evoTrackArr[2][mask],\
                       evoTrackArr[5][mask],\
                       evoTrackArr[1][mask]/1000,\
                       10**evoTrackArr[4][mask]] # added floats to here as numpy.save() was converting them into strings
    
        param_space = np.array(param_space).T.copy()
        
        del L_ast
        
        # if len(param_space) == 0:
        #     raise PhotoSAPPError('Empty param space in photometry module detected.')
            
        return param_space, ast_flag


    ''
        
def asteroseismic_lhood_stellar_list(stellar_inp):
    
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
    
    save_ast_space_bool = stellar_inp[5]
        
    data_ast = stellar_inp[6]
    
    extra_save_string = stellar_inp[7]
    
               
    mem = psutil.virtual_memory()
    threshhold = 0.8 * mem.total/2.**30/mp.cpu_count() # 80% of the total memory per core is noted as the "Threshold"
    
    data_ast[data_ast==''] = np.nan # because empty means nan
                
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
                        
    print(f"starting asteroseismology likelihood calculation for star {stellar_id}")
                
    ast_probability_hbs = stellar_evo_likelihood(astrosize_arr_input,phot_ast_central_values,phot_ast_limits)

    Like_tot_param_space,ast_flag = ast_probability_hbs #initialisation 
                            
    stellar_id = stellar_id.replace(" ","_") # replaces an empty space in the name with an underscore
            
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    if memoryUse >= threshhold:
        
        print("Warning, memory per core is at 80%, Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
        
    directory_asteroseismology = star_field_name # name of directory is the name of the star
    directory_check = os.path.exists(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology}")
    
    if  directory_check == True:
    
        print(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology} directory exists")
        
    else:
        
        print(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology} directory does not exist")
        
        os.makedirs(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology}")
        
        print(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology} directory has been created")

    if save_ast_space_bool:

        np.savetxt(f'../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology}/stellar_track_ast_collection_{stellar_id}_{extra_save_string}.txt',Like_tot_param_space,delimiter=",") 

    return Like_tot_param_space,ast_flag

def FeH_calc(Z_surf,X_surf):
    
    """
    Z_surf: surface metallicity of stellar model
    X_surf: surface hydrogen abundance of stellar model
    return: [Fe/H]
    """

    FeH = np.log10(Z_surf/(X_surf * 0.02439)) # This equation was given by Aldo for his most recent models
                                                # 0.02439 is Z_sol/X_sol
    
    return FeH

class AstSAPPError(Exception):
    pass

### N.B. These pathways are assuming this module is being run from main_TEST.py 

print('loading evo tracks')

shape = (49,46255417) # this is the shape of the large stellar evolution file 
# shape = (24,9453868) # this is the shape of the large stellar evolution file 

test_input_path = ""
# test_input_path = "../SAPP_v1.1_clean_thesis_use/"

# if os.path.exists("../" + test_input_path + "Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap_v2.npy"):
if os.path.exists("../" + test_input_path + "Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap.npy"): 

    # evoTrackArr = np.memmap('../' + test_input_path + 'Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap_v2.npy',shape=shape,mode='r',dtype=np.float64)
    evoTrackArr = np.memmap('../' + test_input_path + 'Input_data/GARSTEC_stellar_evolution_models/photometry_stellar_track_mmap.npy',shape=shape,mode='r',dtype=np.float64)

    # evoTrackFeh = evoTrackArr[7]    
    evoTrackFeh = FeH_calc(evoTrackArr[8],evoTrackArr[7])

    
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
