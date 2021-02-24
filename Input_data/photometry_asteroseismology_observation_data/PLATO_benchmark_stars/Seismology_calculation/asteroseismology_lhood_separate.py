#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:31:06 2019

@author: Matthew Gent

"""
import numpy as np
from numpy import e as EXP
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import psutil
import Payne.astro_constants_cgs as astroc
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
def stellar_evo_likelihood(astrosize_arr,hbs_index_i): 
    
    """
    astrosize_arr: input observed asteroseismic array; [[delta_nu,nu_max],[delta_nu_err,nu_max_error]]
    hbs_index_i: specific stellar evolution file string
    return: Three 3-D distribution of likelihoods for asteroseismology of single star
    purpose: Takes in observational asteroseismic data and a file name string
    to calculate the likelihood function. This is done by loading up the model file line by line,
    calculate the likelihood probability and then save it as the new parameter space. 
    """
        
    ast_obs = astrosize_arr[0] # array of stellar asteroseismic quantities: delta_nu, nu_max
    ast_err = astrosize_arr[1] # error of stellar asteroseismic quantities: delta_nu_err, err_nu_max --> within each error array there is a positive error and negative error
    
    d = 3 # dimensionality of non-probability parameter space ;  Teff, logg, [Fe/H]

    param_space = []
    
    with open("Aldo_stellar_evo/aldo_tracks_range/{}".format(hbs_index_i)) as infile:
        
        lis = [line.split() for line in infile]

        for i,x in enumerate(lis):
                        
            if x[0] == '#': # These are the comments from the stellar model headers, unfortunately it is tailored to the stellar model
                            # Although, it is convention for at least '#' to be used

                continue
            
            else:
                                
                # due to line.split(), the order of parameters is located in Aldo_stellar_evo/aldo_header.txt
                
                if float(x[3]) > 7500: # Temperature upper limit
                    
                    continue
                
                if float(x[3]) < 3500: # Temperature lower limit
                    
                    continue
                                
                if float(x[6]) < 3: # logg lower limit
                    
                    continue
                
                if float(x[6]) > 5: # logg upper limit
                    
                    continue
                
                if float(x[0])/1000 > 14: # Age upper limit
                
                    continue
                                
                nu_max_model = astroc.nu_max_sol * 10 ** (float(x[6]))/astroc.surf_grav_sol * astroc.teff_sol/float(x[3])
                delta_nu_model = float(x[16]) * astroc.delta_nu_sol/136.3 # 136.3 uHz is Aldo's model solar value
                ast_var = [delta_nu_model,nu_max_model]
                N_ast = len(ast_var)
                nu_ast = N_ast-d # degrees of freedom 
                                
                ast_like_line_prod = 1 # Initialisation of asteroseismic likelihood
                
                               
                for j_ast in range(0,N_ast): 
                    
                    ast_sigma = (ast_err[j_ast][0] + ast_err[j_ast][1])/2 # an average of the positive and negative errors are taken
                    ast_mu = ast_obs[j_ast]
                    
                    ast_like_line_prod = ast_like_line_prod * Lhood_norm_red_chi_ind(ast_var[j_ast],ast_mu,ast_sigma,nu_ast,N_ast) 
                
                else:

                    if ast_like_line_prod < 10 ** (-80): # astroseismology threshhold probability, this could probably be lower
                        
                        continue
                                        
                        # New param space Teff logg [Fe/H] Lhood_phot Lhood colour Lhood_ast Lhood_comb Age/Gyr M/Msol R/Rsol

                    FeH = FeH_calc(float(x[8]),float(x[7])) # Calculates [Fe/H] for given star from surface Z and X
                            
                    param_space.append([float(x[3]),float(x[6]),FeH,float(ast_like_line_prod),float(x[0])/1000,float(x[2]),float(x[5])]) # added floats to here as numpy.save() was converting them into strings
                    
    param_space = np.array(param_space)
        
    return param_space

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

def choose_star_seismology(stellar_id):
    
    """
    stellar_id: stellar id name as a string,
    return: asteroseismology data for specific star
    """
        
#    array = np.loadtxt('photometry_observation_data/Gaia_benchmark_stars_data_list.txt',dtype=str)
#    array = np.loadtxt('photometry_observation_data/GaiaESO_final_benchmark_list.txt',dtype=str,delimiter=",")

    array = np.loadtxt('PLATO_stars_seism.txt',dtype=str,delimiter=",")

    star = []
    
    with open('PLATO_stars_seism.txt') as infile:
    
        lis = [line.split(",") for line in infile]
        for i,x in enumerate(lis):
            
            if x[0] == '#':

                continue
        
            else:
                
                if x[0] == stellar_id: # Stellar name for this observation file is the 1st column
                    star = array[i-1] # There is one line of headers
                    break
   
    return star
        
def asteroseismic_lhood_stellar_list(stellar_id):
    
    """
    stellar_id: stellar id as stellar_names
    return: Nothing
    purpose: Takes in stellar id of star, finds the observational data, specifically colours associated with it,
    arranges the data in a specific format in accordance to isochrone_likelihood(), creates a photometric likelihood landscape
    in 4-D space (3 parameters Teff, logg, [M/H] and log10(P) i.e. probability)
    """
    
    mem = psutil.virtual_memory()
    threshhold = 0.8 * mem.total/2.**30/mp.cpu_count() # 80% of the total memory per core is noted as the "Threshold"
    
    hbs_index = np.loadtxt('Aldo_stellar_evo/range_models_list_new.txt',dtype='str')
    
    data = choose_star_seismology(stellar_id) 
                
    # data has the current order : [Stars,nu_max,err nu_max pos,err nu_max neg,d_nu,err d_nu pos,err d_nu neg]
    
    star_name = data[0]
    
    print(star_name)
    
    ### ASTEROSEISMOLOGY ###
    
    astrosize_arr = [float(data[4]),float(data[1])] # delta_nu, nu_max
    astrosize_err_arr = [[float(data[5]),float(data[6])],[float(data[2]),float(data[3])]] # delta_nu_err (pos,neg), err_nu_max (pos,neg)
    
    astrosize_delta_err_input = astrosize_err_arr[0] # pos and neg error for delta_nu
    astrosize_max_err_input = astrosize_err_arr[1] # pos and neg error for nu_max
    
    astrosize_delta_nu_sol_err = astroc.delta_nu_sol_err
    astrosize_nu_max_sol_err = astroc.nu_max_sol_err 
    
#    astrosize_delta_err_fud = 1.5 # Fudicial value of delta_nu error [microhertz]
#    
#    astrosize_max_err_fud = 150 # Fudicial value of nu_max error [microhertz]
#    
#    if np.isnan(astrosize_delta_err_input) == True:
#        
#        astrosize_delta_err_input = astrosize_delta_err_fud
#    
#    if np.isnan(astrosize_max_err_input) == True:
#        
#        astrosize_max_err_input = astrosize_max_err_fud
            
    astrosize_arr_input = [astrosize_arr,[(np.array(astrosize_delta_err_input)**2 + astrosize_delta_nu_sol_err**2) ** 0.5,(np.array(astrosize_max_err_input)**2 + astrosize_nu_max_sol_err**2)**0.5]] # array of asteroseismic quantities used
    
    N_hbs_index = len(hbs_index)
            
    Like_tot_param_space_list = []
    
    print(f"starting photometry likelihood calculation for star {stellar_id}")
            
    for iso_index in range(0,N_hbs_index):
                
        hbs_index_i = hbs_index[iso_index]
                
        Like_tot_param_space_i  = stellar_evo_likelihood(astrosize_arr_input,hbs_index_i)
        
        if len(Like_tot_param_space_i) == 0: # If the variable is a non-populated list, then continue
            
            continue
        
        else:
            
            Like_tot_param_space_list.append(Like_tot_param_space_i)
    
    if len(Like_tot_param_space_list) == 0:
        
        print('Parameter space empty, loosen constraints')
        
        return Like_tot_param_space_list
    
    Like_tot_param_space = Like_tot_param_space_list[0] #initialisation 
    
#    print(f"param_grid_section 1 done")
    
    for build_index in range(1,len(Like_tot_param_space_list)): # The length of the list will be < length of hbs_index because of empty arrays being thrown out 
        
        Like_tot_param_space = np.vstack([Like_tot_param_space ,Like_tot_param_space_list[build_index]])   
        
#        print(f"param_grid_section {build_index+1} done")
        
#    stellar_id = stellar_id.replace("+","p").replace("-","m") # replaces +/- with p/m in filename respectively
    
    stellar_id = stellar_id.replace(" ","_") # replaces an empty space in the name with an underscore
            
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    if memoryUse >= threshhold:
        
        print("Warning, memory per core is at 80%, Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    directory_seismology = stellar_id # name of directory is the name of the star
    directory_check = os.path.exists(f"seismology_lhood_results/{directory_seismology}")
    
    if  directory_check == True:
    
        print(f"seismology_lhood_results/{directory_seismology} directory exists")
        
    else:
        
        print(f"seismology_lhood_results/{directory_seismology} directory does not exist")
        
        os.makedirs(f"seismology_lhood_results/{directory_seismology}")
        
        print(f"seismology_lhood_results/{directory_seismology} directory has been created")
        

    np.save(f"seismology_lhood_results/{directory_seismology}/Lhood_space_asteroseismology_tot_PLATO_bmk_starID{stellar_id}",Like_tot_param_space,allow_pickle=True) 
    

    lhood_sorted_indicies = np.argsort(Like_tot_param_space[:,3])
    
    lhood_max_indice = lhood_sorted_indicies[len(lhood_sorted_indicies)-1]
    
    logg_best = Like_tot_param_space[:,1][lhood_max_indice]
    
    temp_best = Like_tot_param_space[:,0][lhood_max_indice]
    
    feh_best = Like_tot_param_space[:,2][lhood_max_indice]
    
    logg = Like_tot_param_space[:,1]
    teff = Like_tot_param_space[:,0]
    feh = Like_tot_param_space[:,2]
       
    sigma_1 = np.std(Like_tot_param_space[:,3]) # 1 sigma of 3d parameter space
    
    lhood_1_sigma_collect = Like_tot_param_space[:,3][Like_tot_param_space[:,3]>=max(Like_tot_param_space[:,3])-sigma_1]

    temp_1_sigma_collect = Like_tot_param_space[:,0][Like_tot_param_space[:,3]>=max(Like_tot_param_space[:,3])-sigma_1]

    feh_1_sigma_collect = Like_tot_param_space[:,2][Like_tot_param_space[:,3]>=max(Like_tot_param_space[:,3])-sigma_1]

    logg_1_sigma_collect = Like_tot_param_space[:,1][Like_tot_param_space[:,3]>=max(Like_tot_param_space[:,3])-sigma_1]
    
    logg_upper_error = max(logg_1_sigma_collect) - logg_best

    logg_lower_error = logg_best - min(logg_1_sigma_collect)
    
    save_logg_estimate = np.hstack([stellar_id,str(logg_best),str(logg_upper_error),str(logg_lower_error)])
    
    np.savetxt(f"seismology_lhood_results/{stellar_id}_seismic_logg.txt",save_logg_estimate,fmt='%s')
    
    return Like_tot_param_space
    
PLATO_stars_seism_list = np.loadtxt("PLATO_stars_seism.txt",dtype=str,delimiter=',')

PLATO_bmk_seism_names = PLATO_stars_seism_list[:,0]

import time

start_time = time.time()
    
#asteroseismic_lhood_stellar_list(PLATO_bmk_seism_names[9])

inp_index_list_new = np.hstack((PLATO_bmk_seism_names[9],PLATO_bmk_seism_names[12:19]))

num_processor = 8

#inp_index_list_new = PLATO_bmk_seism_names[19:23]
#
#num_processor = 4

if __name__ == '__main__':
#    
    p = mp.Pool(num_processor) # Pool distributes tasks to available processors using a FIFO schedulling (First In, First Out)
    p.map(asteroseismic_lhood_stellar_list,inp_index_list_new) # You pass it a list of the inputs and it'll iterate through them
    p.terminate() # Terminates process 

print(f"{(time.time()-start_time)/60} --- minutes elapsed")
