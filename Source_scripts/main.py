#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:26:18 2020

@author: gent
"""

### import python packages

import numpy as np
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.stats import iqr as IQR_SCIPY
from scipy import interpolate as INTERP
### import SAPP scripts

import os
import psutil

from SAPP_stellar_evolution_scripts.photometry_asteroseismology_lhood_TEST import photometry_asteroseismic_lhood_stellar_list as photometry_ast
import SAPP_spectroscopy.Payne.SAPP_best_spec_payne_v1p1 as mspec_new

np.set_printoptions(suppress=True,formatter={'float': '{: 0.2f}'.format})

def correlation_from_covariance(covariance):
    
    """
    source = https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    
    return correlation


def correlation_table(spec_data_inp,phot_ast_data_inp):
    
    """
    spec_data_inp: spec information from Input_spec_data array. First column
    must be the ID and second column must be the path from this script
    
    phot_ast_data_inp: photometry/asteroseismology information from 
    phot_ast_data_inp. First column must be ID. You don't need the rest of the
    info.
    
    purpose: This creates a "correlation" array to connect the spectra data
    to the photometry/asteroseismology. This is necessary as there tends to be
    multiple spectral observations (there of course can be multiple photometry/
    asteroseismology but typically the best is picked).
    
    Therefore the User must create a list in the text file spectra_list.txt,
    which has IDs in the first column (which match IDs they collected for other
    data) and the path to the spectra in the Input_data directory.
    
    The structure of the array will be N rows X M_N columns
    
    Each row represents a single star, each column represents a spectral 
    observation, hence why M_N as the number of observations can change with
    each star.
    
    return: correlation array
    
    """

    phot_ast_IDs = phot_ast_data_inp[:,0]
    
    spec_IDs = spec_data_inp[:,0]
        
    correlation_arr = []
    
    for star in phot_ast_IDs: # going through each row
    
        # print("star =",star)
        
        star_spec_list = []
        
        for spec_index in range(len(spec_IDs)):
                                    
            if spec_IDs[spec_index] == star:
                
                star_spec_list.append(spec_index)
                                                
            else: 
                
                continue

                # star_spec_list.append(np.nan)
        
        star_spec_list = np.array(star_spec_list)
        
        correlation_arr.append(star_spec_list)
        
    correlation_arr = np.array(correlation_arr)
    
    return correlation_arr

def spectroscopy_stellar_track_collect(inp_arr):
    
    """
    This function creates spectroscopy PDF based on the PDF created for Photometry  
    """
    
    inp_index = inp_arr[0] # index in ref. to star list
    
    best_fit_spectroscopy = inp_arr[1] # best fit spec params initial guess
    
    chi_2_red_bool = inp_arr[2] # if True, reduce the chi2, if False, do other normalisation
    
    spec_save_extension = inp_arr[3]
    
    
    import_path = "../Input_data/spectroscopy_model_data/Payne_input_data/"
    
    name="NN_results_RrelsigN20.npz"#NLTE NN trained spectra grid results
    
    temp=np.load(import_path+name)
    
    x_min = temp["x_min"] # minimum limits of grid
    x_max = temp["x_max"] # maximum limits of grid
    
    print(f"stellar_names[inp_index]")
    
    stellar_filename = stellar_names[inp_index].replace(" ","_")
    
    ### load photometry, asteroseismology stellar tracks
    param_space_phot = np.loadtxt(f"../Output_data/Stars_Lhood_photometry/{stellar_filename}/stellar_track_collection_{stellar_filename}_test_3.txt",delimiter=",")
    
    ### load photometry, asteroseismology stellar tracks with no stellar evo influence
    # i.e. photometry we care about is two colours: Bp-Rp, V-K and asteroseismology is nu_max, magnitudes and delta_nu require some form of stellar evolution information.
    # param_space_test = np.loadtxt(f"../Output_data/Stars_Lhood_photometry/{stellar_filename}/stellar_track_collection_{stellar_filename}_test_3_non_stellar_evo_priors.txt",delimiter=",")

    teff_phot = param_space_phot[:,0]
    logg_phot = param_space_phot[:,1]
    feh_phot = param_space_phot[:,2]
        
    best_params = best_fit_spectroscopy[0]
    
    best_params_err = best_fit_spectroscopy[1]

    cov_matrix = best_fit_spectroscopy[2]

    
    # ch2_best = best_fit_spectroscopy[4] # chi-square of best fit results
    # wvl_con = best_fit_spectroscopy[5] # wavelength of spectra post processing
    # wvl_obs_input = best_fit_spectroscopy[9] # original wavelength of obs spectra
    # obs_norm = best_fit_spectroscopy[6] # observation flux normalised
    # err_norm = best_fit_spectroscopy[10] # err of observation flux normalised

    
    # cov_matrix = best_fit_spectroscopy[11] # covariance matrix
    
    
    cov_matrix_new = cov_matrix[1:]
    
    cov_matrix_new = np.vstack((cov_matrix_new[:,1],\
                                cov_matrix_new[:,2],\
                                cov_matrix_new[:,3],\
                                cov_matrix_new[:,4],\
                                cov_matrix_new[:,5],\
                                cov_matrix_new[:,6],\
                                cov_matrix_new[:,7],\
                                cov_matrix_new[:,8])).T
        
    corr_matrix = correlation_from_covariance(cov_matrix_new) # correlation matrix
    
    sigma_arr = [best_params_err[0]*1000,best_params_err[1],best_params_err[2]] #Teff, logg, [Fe/H] error
        
    cov_matrix_norm = np.zeros([3,3]) 
    
    for i in range(len(sigma_arr)):
        
        for j in range(len(sigma_arr)):
                        
            cov_matrix_norm[i][j] = corr_matrix[i][j] * sigma_arr[i] * sigma_arr[j]
        
    cov_matrix_norm_inv = np.linalg.inv(cov_matrix_norm) # this is the inverse of the normalised covariance matrix
    
    
    params_fix = best_params[3:]
    
    MgFe = params_fix[2] # magnesium abundance
        
    print("NUMBER OF POINTS",len(teff_phot))
    
    chi2_spec_w_phot = []
    
    # this is calculating chi2 directly from model-obs space
    
    # ch2_spec_arr = []
    
    # this is calculating chi2 between spec best params and stellar tracks
    
    ch2_spec_second_arr = []
    
    # this is the above but taking into account covariance between the primary parameters
    
    ch2_spec_third_arr = []
    
    start_time_spec_tracks = time.time()
    
    for test_index in range(len(feh_phot)):
        
        # feh_stellar = feh_spec + (0.22/0.4 * MgFe)
        
        # feh_test are feh values from stellar tracks
        
        # need to de-enhance them first, then pass to spec
        
        ### first type of chi2
        ### creates chi2 grid sampling spectroscopy space directly
        
        feh_de_enhance = feh_phot[test_index] - (0.22/0.4 * MgFe) # Serenelli (in conversation)
    
        # spec_fix_values = [teff_test[test_index]/1000,logg_test[test_index],feh_de_enhance]
    
        # params_inp_new = np.hstack(([spec_fix_values[0],\
        #                              spec_fix_values[1],\
        #                              feh_de_enhance],params_fix))
            
        # params_norm_new = (params_inp_new-x_min)/(x_max-x_min)-0.5 # normalises parameters to 'Payne' values
            
        # model_norm_loop = mspec_new.restore(wvl_obs_input,*params_norm_new)
        
        # chi2_spec = np.sum(((obs_norm-model_norm_loop)/err_norm)**2)
        
        # ch2_spec_arr.append(chi2_spec)
        
        ### second type of chi2        
        ### creates chi2 grid sampling photometry space with best fit spec values
        
        chi2_teff_i = (((best_params[0]*1000)-teff_phot[test_index])/(best_params_err[0]*1000))**2
        chi2_logg_i = (((best_params[1])-logg_phot[test_index])/(best_params_err[1]))**2
        chi2_feh_i = (((best_params[2])-feh_phot[test_index])/(best_params_err[2]))**2
        chi2_spec_second = chi2_teff_i + chi2_logg_i + chi2_feh_i
        
        ch2_spec_second_arr.append(chi2_spec_second)
        
        ### third type of chi2
        ### creates chi2 grid sampling photometry space with best fit spec values while including co-variance
        
        diff_vector = np.array([teff_phot[test_index]-(best_params[0]*1000),logg_phot[test_index]-best_params[1],feh_de_enhance-best_params[2]])
        
        product_1 = np.matmul(diff_vector,cov_matrix_norm_inv)
        
        chi2_spec_third = np.matmul(product_1,diff_vector.T)
            
        ch2_spec_third_arr.append(chi2_spec_third)

    
    # ch2_spec_arr = np.array(ch2_spec_arr)
    ch2_spec_second_arr = np.array(ch2_spec_second_arr)
    ch2_spec_third_arr = np.array(ch2_spec_third_arr)

    for combine_index_loop in range(len(ch2_spec_second_arr)):
        
            # prob_spec_w_test.append(np.hstack((param_space_test[combine_index_loop],prob_spec_norm[combine_index_loop],prob_spec_norm_second[combine_index_loop])))
            
            # prob_spec_w_test.append(np.hstack((param_space_test[combine_index_loop],ch2_spec_arr[combine_index_loop],ch2_spec_second_arr[combine_index_loop],ch2_spec_third_arr[combine_index_loop])))
            
            chi2_spec_w_phot.append(np.hstack((param_space_phot[combine_index_loop],ch2_spec_second_arr[combine_index_loop],ch2_spec_third_arr[combine_index_loop])))

    chi2_spec_w_phot = np.array(chi2_spec_w_phot)
                                
    print(f"time elapsed --- {(time.time() - start_time_spec_tracks)/60} minutes ---")
    
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))


    directory_combined = stellar_filename # name of directory is the name of the star
    
    directory_check = os.path.exists(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined}")

    if  directory_check == True:
    
        print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined} directory exists")
        
    else:
        
        print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined} directory does not exist")
        
        os.makedirs(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined}")
        
        print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined} directory has been created")
        
    np.savetxt(f'../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{stellar_filename}/stellar_track_collection_w_spec_and_prob_{stellar_filename}_test_3{spec_save_extension}{extra_save_string}.txt',chi2_spec_w_phot,delimiter=",")
    
    return best_params


def IQR_NUMPY(data):
    
    """
    Calculates the IQR
    """
    
    return np.percentile(data, 75) - np.percentile(data, 25)

def bin_width_optimal(data):
    
    """
    Freedman-Diaconis rule
    
    source:https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram?fbclid=IwAR2zyw-XXJUV_FvwgP0de6L_bcP6oXBRzMiIOExBUwRRKKC_atwa_md7VwI
    """
    
    N = len(data)
    
    # iqr_output = IQR_NUMPY(data)
    iqr_output = IQR_SCIPY(data)

    bin_width = 2 * iqr_output * N ** (-1/3)
    
    return bin_width

def main_func_multi_obs(inp_index_arr):
    
    start_time = time.time()
    ### photometry, asteroseismology limit input
            
    phot_ast_limits = [Teff_resolution,logg_resolution,feh_resolution]
                            
    ### spectroscopy INITIALISATION ###
    
    # so far just picking the first observation for each star
    
    # need to develop a system for multiple observations w.r.t. Bayesian scheme
    
    stellar_filename = stellar_names[inp_index].replace(" ","_")
    
    for spec_obs_number in range(len(correlation_arr[inp_index])):
        
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Observation number {spec_obs_number + 1} for star {stellar_names[inp_index]} in spec mode: {mode_kw}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
   
        spec_type = Input_spec_data[:,1][correlation_arr[inp_index][spec_obs_number]]\
                    + "_" + str(spec_obs_number)
        spec_path = Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
        error_map_spec_path = Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
        error_mask_index = inp_index
        nu_max = Input_ast_data[inp_index][1]
        nu_max_err = Input_ast_data[inp_index][2]
        numax_input_arr = [float(nu_max),float(nu_max_err),niter_numax_MAX]
        recalc_metals_inp = [spec_fix_values[0],spec_fix_values[1],spec_fix_values[2],feh_recalc_fix_bool]
        logg_fix_load = np.loadtxt(f"../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/seismology_lhood_results/{stellar_filename}_seismic_logg.txt",dtype=str)
        logg_fix_input_arr = [float(logg_fix_load[1]),float(logg_fix_load[2]),float(logg_fix_load[3])]
        
        spec_init_run = [[spec_path,\
                    error_map_spec_path,\
                    error_mask_index,\
                    error_mask_recreate_bool,\
                    error_map_use_bool,\
                    cont_norm_bool,\
                    rv_shift_recalc,\
                    conv_instrument_bool,\
                    input_spec_resolution,\
                    numax_iter_bool,\
                    numax_input_arr,\
                    recalc_metals_bool,\
                    recalc_metals_inp,\
                    logg_fix_bool,\
                    logg_fix_input_arr],stellar_names[inp_index],spec_type]

        if best_spec_bool == True:
                
            best_fit_spectroscopy = mspec_new.find_best_val(spec_init_run[0])
            
            best_spec_params = best_fit_spectroscopy[0]   
            best_spec_params_err = best_fit_spectroscopy[1]   
            
            cov_matrix = best_fit_spectroscopy[11]   
                        
            N_pixels = len(best_fit_spectroscopy[6])
            N_params = len(best_spec_params)
            spec_dof = abs(N_pixels - N_params) # degrees of freedom

            stellar_filename = stellar_names[inp_index].replace(" ","_")
    
            directory_spectroscopy = stellar_filename # name of directory is the name of the star
            directory_check = os.path.exists(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")
            
            if  directory_check == True:
            
                print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory exists")
                
            else:
                
                print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory does not exist")
                
                os.makedirs(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")
                
                print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory has been created")
                    
            best_spec_params = np.hstack((best_spec_params,spec_dof))
                    
            ### Multiobs save
            
            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',best_spec_params)
            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',best_spec_params_err)    
            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',cov_matrix)
    
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0]/2.**30
            
            print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))

        ### SPEC BEST VALUES MULTIOBS ###
        
        # These have assumed to be calculated IF the first spec mode has ran

        best_spec_params = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt")
        best_spec_params_err = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt")

        
        ### photometry input    
        
        if phot_ast_track_bool == True:
        
            if phot_ast_central_values_bool == False:
            
                phot_ast_central_values = [best_spec_params[0]*1000,best_spec_params[1],best_spec_params[2]]
                
            elif phot_ast_central_values_arr == True:
                
                phot_ast_central_values = [phot_ast_central_values_arr[0],phot_ast_central_values_arr[1],phot_ast_central_values_arr[2]]
    
            stellar_inp_phot = [stellar_names[inp_index],phot_ast_central_values,phot_ast_limits]
                    
            start_time_phot = time.time()
            
            start_phot = time.time()    
            photometry_ast(stellar_inp_phot)
            print(f"Photometry and Asteroseismology calculation for star {stellar_names[inp_index]} time elapsed --- {(time.time()-start_phot)/60} minutes --- ")    
        
            print(f"photometry time = {time.time()-start_time_phot}")
        
            if spec_obs_number == 0:
                
                # this is because there is no point creating a photometry space for multiple spectra
                # the spaces will be very similar in extent
                # the only time this fails is if we have a bad spectra
                # so in blind mode this makes sense as it can be very inconsistent
                # but because the search range in photometry space is so large
                # just create it once for the first observation
                # you only need one set of central limits decided by spectroscopy.
                
                break

        if spec_track_bool == True:
            
            cov_matrix = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt')
                
            best_fit_spectroscopy_input = [best_spec_params,best_spec_params_err,cov_matrix]
 
            spec_extension = f"_OBS_NUM_{spec_obs_number+1}_SPEC_MODE_{mode_kw}"
            stellar_inp_spec = [inp_index,best_fit_spectroscopy_input,chi_2_red_bool,spec_extension]
    
            start_spec = time.time()
            spectroscopy_stellar_track_collect(stellar_inp_spec)
            print(f"Spectroscopy calculation for star {stellar_names[inp_index]} time elapsed --- {(time.time()-start_spec)/60} minutes --- ")
    
    
        if bayes_scheme_bool == True:
    
            ### load parameter space i.e. photometry, asteroseismology, spectroscopy, chi2 values
            spec_extension = f"_OBS_NUM_{spec_obs_number+1}_SPEC_MODE_{mode_kw}"

            param_space = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{stellar_filename}/stellar_track_collection_w_spec_and_prob_{stellar_filename}_test_3{spec_extension}{extra_save_string}.txt",delimiter=",")
            
            teff  = param_space[:,0]
            logg = param_space[:,1]
            feh = param_space[:,2]
            feh_spec = feh - (0.22/0.4 * best_spec_params[5])
            chi2_prob_mag = param_space[:,3]
            chi2_prob_mag_gaia = param_space[:,4]
            chi2_prob_col = param_space[:,5]
            chi2_prob_col_gaia = param_space[:,6]
            chi2_prob_ast = param_space[:,7]
            age = param_space[:,8]
            mass = param_space[:,9]
            radius = param_space[:,10]
            age_step = param_space[:,11]
            Luminosity = param_space[:,12]
            # chi2_prob_spec = param_space[:,13] 
            # chi2_prob_spec_2 = param_space[:,14]
            # chi2_prob_spec_3 = param_space[:,15]
            chi2_prob_spec_2 = param_space[:,13] # without covariance
            chi2_prob_spec_3 = param_space[:,14] # with covariance
                                    
           ### if _test_3 then these are pure chisq values, they need to be treated externally
                      
            if chi_2_red_bool == False:
            
                ### Spectroscopy PDF
                
                # chi2_prob_spec_norm = chi2_prob_spec - min(chi2_prob_spec)
                # prob_spec = np.exp(-chi2_prob_spec_norm/2)
                
                chi2_prob_spec_2_norm = chi2_prob_spec_2 - min(chi2_prob_spec_2)
                prob_spec_2 = np.exp(-chi2_prob_spec_2_norm/2)
        
                chi2_prob_spec_3_norm = chi2_prob_spec_3 - min(chi2_prob_spec_3)
                prob_spec_3 = np.exp(-chi2_prob_spec_3_norm/2)
        
                if spec_covariance_bool == True:

                    prob_spec_use = prob_spec_3
                    
                else:
                    
                    prob_spec_use = prob_spec_2    
                    
                ### Asteroseismology PDF
            
                chi2_prob_ast_norm = chi2_prob_ast - min(chi2_prob_ast)
                prob_ast = np.exp(-chi2_prob_ast_norm/2)
                
                ### Photometric PDFs
                
                chi2_prob_mag_norm = chi2_prob_mag - min(chi2_prob_mag) # non Gaia magnitudes
                prob_mag = np.exp(-chi2_prob_mag/2)
                
                chi2_prob_mag_gaia_norm = chi2_prob_mag_gaia - min(chi2_prob_mag_gaia) # Gaia magnitudes
                prob_mag_gaia = np.exp(-chi2_prob_mag_gaia_norm/2)
                
                chi2_prob_col_norm = chi2_prob_col - min(chi2_prob_col) # non Gaia colours
                prob_col = np.exp(-chi2_prob_col/2)
                
                chi2_prob_col_gaia_norm = chi2_prob_col_gaia - min(chi2_prob_col_gaia) # Gaia colours
                prob_col_gaia = np.exp(-chi2_prob_col_gaia/2)
                
                chi2_comb_with_gaia =  chi2_prob_spec_2_norm + chi2_prob_ast_norm + chi2_prob_mag_norm + chi2_prob_mag_gaia_norm
                prob_comb_sum_with_gaia = np.exp(-chi2_comb_with_gaia/2)
    
                chi2_comb_no_gaia =  chi2_prob_spec_2_norm + chi2_prob_ast_norm + chi2_prob_mag_norm
                prob_comb_sum_no_gaia = np.exp(-chi2_comb_no_gaia/2)
    
                
            elif chi_2_red_bool == True:
                
                ### load degrees of freedom ###
                
                spec_dof = int(best_spec_params[8])
                phot_ast_dof_arr = np.loadtxt(f"../Output_data/Stars_Lhood_photometry/{stellar_filename}/degrees_freedom_phot_ast.txt",delimiter=",")
                spec_dof_2 = 1 # because we fit 3 parameters and those are the ones we explore, so no change
                spec_dof_3 = 1 # because we fit 3 parameters and those are the ones we explore, so no change
                
                # chi2_prob_spec_norm = chi2_prob_spec/spec_dof
                # prob_spec = np.exp(-chi2_prob_spec_norm/2)
                
                chi2_prob_spec_2_norm = chi2_prob_spec_2/spec_dof_2
                prob_spec_2 = np.exp(-chi2_prob_spec_2_norm/2)
        
                chi2_prob_spec_3_norm = chi2_prob_spec_3/spec_dof_3
                prob_spec_3 = np.exp(-chi2_prob_spec_3_norm/2)
    
                if spec_covariance_bool == True:

                    prob_spec_use = prob_spec_3
                    
                else:
                    
                    prob_spec_use = prob_spec_2    
            
                chi2_prob_ast_norm = chi2_prob_ast/phot_ast_dof_arr[4]
                prob_ast = np.exp(-chi2_prob_ast_norm/2)
                
                chi2_prob_mag_norm = chi2_prob_mag/phot_ast_dof_arr[0]
                prob_mag = np.exp(-chi2_prob_mag_norm/2)
                
                chi2_prob_mag_gaia_norm = chi2_prob_mag_gaia/phot_ast_dof_arr[1]
                prob_mag_gaia = np.exp(-chi2_prob_mag_gaia_norm/2)
    
                chi2_prob_col_norm = chi2_prob_col/phot_ast_dof_arr[0]
                prob_col = np.exp(-chi2_prob_col_norm/2)
                
                chi2_prob_col_gaia_norm = chi2_prob_col_gaia/phot_ast_dof_arr[1]
                prob_col_gaia = np.exp(-chi2_prob_col_gaia_norm/2)
                
                
            prob_col_comb = prob_col * prob_col_gaia
            prob_mag_comb = prob_mag * prob_mag_gaia
            
            IMF_prior = mass ** (-2.35)
            Age_prior = age_step
            
            prior_PDF = IMF_prior * Age_prior
            
            prob_input = prob_spec_use * prob_ast * prob_mag_comb * prior_PDF
            prob_input_name = "prob_comb_3_w_prob_priors"
                                    
            savefig_bool = False
            makefig_bool = False
            
            ### bins for distributions
            
            teff_binwidth_prob_input = 25 # K
            logg_binwidth_prob_input = 0.025 # dex
            feh_binwidth_prob_input = 0.05 # dex
            feh_spec_binwidth_prob_input = 0.05 # dex
            age_binwidth_prob_input = 0.25 # Gyrs
            mass_binwidth_prob_input = 0.02 # Msol
            radius_binwidth_prob_input = 0.02 # Rsol
            lumin_binwidth_prob_input = 0.1 # Lsol
            
            ### probability array inputs to graph distributions
        
            # prob_input_name_arr = ["Spectroscopy","Photometry+Parallax","ast"]
            # prob_input_arr = [prob_spec_use,prob_mag_comb,prob_ast]
            # prob_col_arr = ["deepskyblue","tomato","k","g"]
            
            # prob_input_name_arr = ["Combined_no_Spec","Combined","Spectroscopy"]
            # prob_input_arr = [prob_input_without_spec,prob_input,prob_spec_use]
            # prob_col_arr = ["deepskyblue","tomato","k"]
            
            # Log_multiple_prob_distributions_1D(teff,prob_input_arr,"T$_{eff}$ [K]",teff_binwidth_prob_input,prob_input_name_arr,"temperature",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison") # 50
            # Log_multiple_prob_distributions_1D(logg,prob_input_arr,"Logg [dex]",logg_binwidth_prob_input,prob_input_name_arr,"logg",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison") # 0.005
            # Log_multiple_prob_distributions_1D(feh,prob_input_arr,"[Fe/H] [dex]",feh_binwidth_prob_input,prob_input_name_arr,"feh",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison") # 0.05
            # Log_multiple_prob_distributions_1D(feh_spec,prob_input_arr,"[Fe/H] spec [dex]",feh_spec_binwidth_prob_input,prob_input_name_arr,"feh_spec",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")    
            # Log_multiple_prob_distributions_1D(age,prob_input_arr,"Age [Gyrs]",age_binwidth_prob_input,prob_input_name_arr,"age",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison") # 0.5
            # Log_multiple_prob_distributions_1D(mass,prob_input_arr,"Mass [M$_\odot$]",mass_binwidth_prob_input,prob_input_name_arr,"mass",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison") # 0.02
            # Log_multiple_prob_distributions_1D(radius,prob_input_arr,"Radius [R$_\odot$]",radius_binwidth_prob_input,prob_input_name_arr,"radius",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")
            # Log_multiple_prob_distributions_1D(Luminosity,prob_input_arr,"Luminosity [L$_\odot$]",lumin_binwidth_prob_input,prob_input_name_arr,"lumin",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")
        
            ### BEST FIT PARAMETER ESTIMATION via fitting Log(Gaussian) of PDF from prob_input
            ''
            teff_fit = Log_prob_distributions_1D(teff,prob_input,"T$_{eff}$ [K]",teff_binwidth_prob_input,prob_input_name,"temperature",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 50
            logg_fit = Log_prob_distributions_1D(logg,prob_input,"Logg [dex]",logg_binwidth_prob_input,prob_input_name,"logg",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.005
            feh_fit = Log_prob_distributions_1D(feh,prob_input,"[Fe/H] [dex]",feh_binwidth_prob_input,prob_input_name,"feh",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.05
            feh_spec_fit = Log_prob_distributions_1D(feh_spec,prob_input,"[Fe/H] spec [dex]",feh_spec_binwidth_prob_input,prob_input_name,"feh_spec",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension)    
            age_fit = Log_prob_distributions_1D(age,prob_input,"Age [Gyrs]",age_binwidth_prob_input,prob_input_name,"age",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.5
            mass_fit = Log_prob_distributions_1D(mass,prob_input,"Mass [M$_\odot$]",mass_binwidth_prob_input,prob_input_name,"mass",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.02
            radius_fit = Log_prob_distributions_1D(radius,prob_input,"Radius [R$_\odot$]",radius_binwidth_prob_input,prob_input_name,"radius",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension)
            luminosity_fit = Log_prob_distributions_1D(Luminosity,prob_input,"Luminosity [L$_\odot$]",lumin_binwidth_prob_input,prob_input_name,"lumin",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension)

            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0]/2.**30
            
            print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))


            ### BEST FIT PARAMETER ESTIMATION via fitting Gaussian of PDF from prob_input
            
            # teff_fit = prob_distributions_1D(teff,prob_input,"T$_{eff}$ [K]",teff_binwidth,prob_input_name,"temperature",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool) # 50
            # logg_fit = prob_distributions_1D(logg,prob_input,"Logg [dex]",logg_binwidth,prob_input_name,"logg",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool) # 0.005
            # feh_fit = prob_distributions_1D(feh,prob_input,"[Fe/H] [dex]",feh_binwidth,prob_input_name,"feh",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool) # 0.05
            # feh_spec_fit = prob_distributions_1D(feh_spec,prob_input,"[Fe/H] spec [dex]",feh_spec_binwidth,prob_input_name,"feh_spec",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool)    
            # age_fit = prob_distributions_1D(age,prob_input,"Age [Gyrs]",age_binwidth,prob_input_name,"age",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool) # 0.5
            # mass_fit = prob_distributions_1D(mass,prob_input,"Mass [M$_\odot$]",mass_binwidth,prob_input_name,"mass",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool) # 0.02
            # radius_fit = prob_distributions_1D(radius,prob_input,"Radius [R$_\odot$]",radius_binwidth,prob_input_name,"radius",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool)
                
            ### Even with age_step prior there tends to be a probability of age in the PMS
            ### we purposely select the second probability as that is what we are looking for.
            
            mu_1_age = age_fit[0]
            sigma_1_age = age_fit[1]
            mu_2_age = age_fit[2]
            sigma_2_age = age_fit[3]
                
            mu_age = mu_2_age
            sigma_age = sigma_2_age
                
            age_fit = [mu_age,sigma_age]
            
            ### best params and errors to save for Teff, logg, [Fe/H], abundances, Mass, Age, Radius, Luminosity ###
            
            best_params_combined = [teff_fit[0],\
                                    logg_fit[0],\
                                    feh_fit[0],\
                                    feh_spec_fit[0],\
                                    best_spec_params[3],\
                                    best_spec_params[4],\
                                    best_spec_params[5],\
                                    best_spec_params[6],\
                                    best_spec_params[7],\
                                    mass_fit[0],\
                                    age_fit[0],\
                                    radius_fit[0],\
                                    luminosity_fit[0]]
                
            best_params_combined = np.array(best_params_combined)
            
            
            best_params_combined_err = [(teff_fit[1] ** 2 + (best_spec_params_err[0]*1000)**2 + teff_binwidth_prob_input**2)**0.5,\
                                        (logg_fit[1] ** 2 + best_spec_params_err[1]**2 + logg_binwidth_prob_input**2)**0.5,\
                                        (feh_fit[1] ** 2 + best_spec_params_err[2]**2 + feh_binwidth_prob_input**2)**0.5,\
                                        (feh_spec_fit[1] ** 2 + best_spec_params_err[2]**2 + best_spec_params_err[5]**2 + feh_spec_binwidth_prob_input**2)**0.5,\
                                        best_spec_params_err[3],\
                                        best_spec_params_err[4],\
                                        best_spec_params_err[5],\
                                        best_spec_params_err[6],\
                                        best_spec_params_err[7],\
                                        (mass_fit[1]**2 + mass_binwidth_prob_input**2)**0.5,\
                                        (age_fit[1]**2 + age_binwidth_prob_input**2)**0.5,\
                                        (radius_fit[1]**2 + radius_binwidth_prob_input**2)**0.5,\
                                        (luminosity_fit[1]**2 + lumin_binwidth_prob_input**2)**0.5]
            
            best_params_combined_err = np.array(best_params_combined_err)   
            
            ### best params and errors to save for Teff, logg, [Fe/H] and abundances ###
            
            # best_params_combined = [teff_fit[0],\
            #                         logg_fit[0],\
            #                         feh_fit[0],\
            #                         feh_spec_fit[0],\
            #                         best_spec_params[3],\
            #                         best_spec_params[4],\
            #                         best_spec_params[5],\
            #                         best_spec_params[6],\
            #                         best_spec_params[7]]
                
            # best_params_combined = np.array(best_params_combined)
            
            # # ref_params_combined = [Ref_data]
            
            # best_params_combined_err = [(teff_fit[1] ** 2 + (best_spec_params_err[0]*1000)**2 + teff_binwidth**2)**0.5,\
            #                             (logg_fit[1] ** 2 + best_spec_params_err[1]**2 + logg_binwidth**2)**0.5,\
            #                             (feh_fit[1] ** 2 + best_spec_params_err[2]**2 + feh_binwidth**2)**0.5,\
            #                             (feh_spec_fit[1] ** 2 + best_spec_params_err[2]**2 + best_spec_params_err[5]**2 + feh_spec_binwidth**2)**0.5,\
            #                             best_spec_params_err[3],\
            #                             best_spec_params_err[4],\
            #                             best_spec_params_err[5],\
            #                             best_spec_params_err[6],\
            #                             best_spec_params_err[7]]
            
            # best_params_combined_err = np.array(best_params_combined_err)   
            
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0]/2.**30
            
            print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))

            
            directory_output = stellar_filename # name of directory is the name of the star
            directory_check = os.path.exists(f"../Output_data/Stars_Lhood_combined_spec_phot/{directory_output}")
            
            if  directory_check == True:
            
                print(f"../Output_data/Stars_Lhood_combined_spec_phot/{directory_output} directory exists")
                
            else:
                
                print(f"../Output_data/Stars_Lhood_combined_spec_phot/{directory_output} directory does not exist")
                
                os.makedirs(f"../Output_data/Stars_Lhood_combined_spec_phot/{directory_output}")
                
                print(f"../Output_data/Stars_Lhood_combined_spec_phot/{directory_output} directory has been created")
        
            
            np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_{directory_output}{spec_extension}{extra_save_string}.txt",best_params_combined,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")
            np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_err_{directory_output}{spec_extension}{extra_save_string}.txt",best_params_combined_err,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")
            ''
    print(f"Total time elapsed for star {stellar_names[inp_index]} --- {(time.time()-start_time)/60} minutes ---")


def normpdf(x, mean, sd):
    var = sd**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(x-mean)**2/(2*var))
    return num/denom

def prob_distributions_1D(Parameter,Probability,Param_name,Param_binwidth,prob_input_name,param_filename,makefig_bool,savefig_bool,stellar_id,chisq_red_bool):
       
    # ax_1d.plot(Parameter,Probability,marker='o',markerfacecolor='tomato',markeredgecolor='k',linestyle='none')
        
    # DataSetBinCount = 1 + np.log2(len(Parameter)) 
    
    # DataSetBinWidth = (max(Parameter) - min(Parameter)) / DataSetBinCount
    
    # print("NEW BINWIDTH",int(DataSetBinWidth))
    
    # Param_binwidth = int(DataSetBinWidth)
    
    """
    Need to cut the data down and get rid of any zero regions which waste time
    """
    
    Parameter_zero_clean = Parameter[np.where(Probability!=float(0))[0]]
    Probability_zero_clean = Probability[np.where(Probability!=float(0))[0]]
    # print(len(Parameter),len(Parameter_zero_clean))
    
    # Parameter = Parameter_zero_clean
    # Probability = Probability_zero_clean
    
    # Param_binwidth = bin_width_optimal(Parameter)
    
    # print("NEW BINWIDTH",Param_binwidth)
            
    data_entries, bins = np.histogram(Parameter, bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability)

    # data_entries, bins = np.histogram(Parameter,weights=Probability)
        
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    
    PMS_age_limit = 1 # Gyrs, could be less
    
    if param_filename == "age":
    
        data_entries_PMS = data_entries[binscenters<PMS_age_limit]    
        binscenters_PMS = binscenters[binscenters<PMS_age_limit]
    
        data_entries_MS = data_entries[binscenters>PMS_age_limit]        
        binscenters_MS = binscenters[binscenters>PMS_age_limit]
    
        peak_data_PMS = max(data_entries_PMS)
        parameter_peak_loc_PMS = binscenters_PMS[np.where(data_entries_PMS==peak_data_PMS)[0]][0]
    
        peak_data_MS = max(data_entries_MS)
        parameter_peak_loc_MS = binscenters_MS[np.where(data_entries_MS==peak_data_MS)[0]][0]

    
    # """
    # Interpolate the bincenters first
    # """
    
    # hist_interp = INTERP.interp1d(binscenters, data_entries)
    # xnew = np.linspace(min(binscenters),max(binscenters),5)
    # ynew = hist_interp(xnew)

    peak_data = max(data_entries)            
    parameter_peak_loc = binscenters[np.where(data_entries==peak_data)[0]][0]
    
    # print("PEAK",peak_data)
    
    """
    Calculate the mean and std of data
    """

    # mu_data = parameter_peak_loc
    
    # variance = 0
    
    # for data_loop_index in range(len(data_entries)):
        
    #     variance += ((binscenters[data_loop_index]-mu_data)**2)/len(data_entries)
        
    # sigma_data = variance ** 0.5
    
    # print(sigma_data)
    
    # print(bins)
    # print(data_entries)
    
    # print(min(bins))
    
    # print(peak_data)
    # print(max(ynew))
    
    '''
    number_suc_true = 0
    
    while number_suc_true < 100:
        
        print("Binwidth = ",Param_binwidth)
        
        try:
    
            popt, pcov = curve_fit(fit_function_normal, xdata=binscenters, ydata=data_entries, p0=[peak_data, parameter_peak_loc, Param_binwidth])
            # popt, pcov = curve_fit(fit_function_normal, xdata=xnew, ydata=ynew, p0=[peak_data, parameter_peak_loc, Param_binwidth])
            
            suc=True
                                    
            if number_suc_true + 1 >= 100:
                
                break
            
            else:
            
                number_suc_true += 1    
            
                Param_binwidth = Param_binwidth * 1.25     
            
                suc = False
            
        except RuntimeError:
            
            print("Error - curve_fit failed ")#stars[ind_spec])
            
            Param_binwidth = Param_binwidth * 0.75
            
            popt=np.ones([3])
            popt*=np.nan
            
            suc=False
    '''
    
    try:

            if param_filename == "age":        
            
                popt, pcov = curve_fit(bimodal, xdata=binscenters, ydata=data_entries, p0=[peak_data_PMS, parameter_peak_loc_PMS, Param_binwidth,peak_data_MS,parameter_peak_loc_MS,Param_binwidth],bounds = ((0, min(Parameter), 1e-5*Param_binwidth, 0, min(Parameter), 1e-5*Param_binwidth), (1, max(Parameter), np.inf, 1, max(Parameter), np.inf)))

            else:
                
                popt, pcov = curve_fit(fit_function_normal, xdata=binscenters, ydata=data_entries, p0=[peak_data, parameter_peak_loc, Param_binwidth])
            
            # popt, pcov = curve_fit(fit_function_normal, xdata=xnew, ydata=ynew, p0=[peak_data, parameter_peak_loc, Param_binwidth])
            
            suc=True
                                            
    except RuntimeError:
            
            print("Error - curve_fit failed ")#stars[ind_spec])
                        
            
            if param_filename == "age":
                
                popt=np.ones([6])
                
            else: 
                
                popt=np.ones([3])

            popt*=np.nan
            
            suc=False


    if param_filename == "age":
    
        mu_fit = popt[1]
        sigma_fit = popt[2]
        
        mu_fit_2 = popt[4]
        sigma_fit_2 = popt[5]
        
        # print([peak_data_PMS, parameter_peak_loc_PMS, Param_binwidth,peak_data_MS,parameter_peak_loc_MS,Param_binwidth])
        # print([popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]])
        
        fitted_params = [mu_fit,sigma_fit,mu_fit_2,sigma_fit_2]
    
    else:
        
        mu_fit = popt[1]
        sigma_fit = popt[2]

        fitted_params = [mu_fit,sigma_fit]

    
    if makefig_bool == True:
    
        xspace = np.linspace(min(Parameter),max(Parameter),1000)
                
        # xspace = np.linspace(min(bins),max(bins),1000)
        
        fig_prob_1d = plt.figure(figsize=(12,12))
        
        ax_1d = fig_prob_1d.add_subplot(111)
        
        ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability,density=False,label=f'Bin width = {Param_binwidth:.2f}',alpha=0.5,histtype='stepfilled',color='tomato',edgecolor='k')
        # ax_1d.hist(Parameter,weights=Probability,density=False,label=f'Bin width = {Param_binwidth}',alpha=0.5,histtype='stepfilled',color='tomato',edgecolor='k')
        
        # ax_1d.plot(xnew,ynew,'g-',label='interpolation')
        
        # ax_1d.plot(binscenters,data_entries,'yo')
        
        # lab_input = [peak_data_PMS, parameter_peak_loc_PMS, Param_binwidth,peak_data_MS,parameter_peak_loc_MS,Param_binwidth]
        
        # ax_1d.plot(xspace,bimodal(xspace,*lab_input),'b-')
        
        if suc == True:
        
            if param_filename == "temperature":
            
                ax_1d.plot(xspace, fit_function_normal(xspace, *popt), color='blue', linewidth=2.5, label=r'$\mu$ = ' + f'{mu_fit:.0f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.0f}\n')
            
            elif param_filename == "age":
                
                ax_1d.plot(xspace, bimodal(xspace, *popt), color='blue', linewidth=2.5, label=r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n\n' + r'$\mu_2$ = ' + f'{mu_fit_2:.3f}\n' + r'$\sigma_2$ = ' + f'{sigma_fit_2:.3f}\n')        
                
            else:
                
                ax_1d.plot(xspace, fit_function_normal(xspace, *popt), color='blue', linewidth=2.5, label=r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n')        
        
        if suc == False:
            
            print("Fitting failed, change bin size")
        
        ax_1d.set_ylabel('Probability')
        # ax_1d.set_ylabel('Counts')
        ax_1d.set_xlabel(f'{Param_name}')
        
        # hist, bins = np.histogram(Parameter,weights=Probability)
        
        # Gauss_prob = normpdf(Parameter,np.average(Parameter),Param_binwidth)
                
        # ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),density=False,label=f'Bin width = {Param_binwidth}',alpha=1,histtype='stepfilled')
    
        # ax_1d.bar(Parameter, Probability, width=0.08, bottom=None)
    
        ax_1d.legend(loc='upper right',fontsize=20)
        
        ax_1d.set_xlim([min(Parameter),max(Parameter)])
        
        ax_1d.set_yscale("log")
                
        # ax_1d.set_ylim(ymin=0)
        
        prob_title_name = prob_input_name.replace("_"," ").replace("w","x")
        
        plt.title(f"{prob_title_name}")
        
        plt.show()
        
        if savefig_bool == True:
            
            directory_figures= stellar_id # name of directory is the name of the star
            directory_check = os.path.exists(f"../Output_figures/1d_distributions/{directory_figures}")
            
            if  directory_check == True:
                
                pass
                # print(f"../Output_figures/1d_distributions/{directory_figures} directory exists")
                
            else:
                
                print(f"../Output_figures/1d_distributions/{directory_figures} directory does not exist")
                
                os.makedirs(f"../Output_figures/1d_distributions/{directory_figures}")
                
                print(f"../Output_figures/1d_distributions/{directory_figures} directory has been created")
        
            # fig_prob_1d.savefig(f"../Output_figures/test_folder/1d_distributions_sun/{prob_input_name}_vs_{param_filename}.png")
            
            if chisq_red_bool == False:
            
                fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_no_norm.png")

            elif chisq_red_bool == True:
            
                fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_norm.png")
        
        # plt.close('all')
        
    return fitted_params
    
    return [mu_fit,sigma_fit]

def Log_prob_distributions_1D(Parameter,Probability,Param_name,Param_binwidth,prob_input_name,param_filename,makefig_bool,savefig_bool,stellar_id,chisq_red_bool,savefig_extra_name):
       
    
    """
    Need to cut the data down and get rid of any zero regions which waste time
    """
    
    Parameter_zero_clean = Parameter[np.where(Probability!=0.0)[0]]
    Probability_zero_clean = Probability[np.where(Probability!=0.0)[0]]
            
    # Parameter = Parameter_zero_clean
    # Probability = Probability_zero_clean
    
    data_entries, bins = np.histogram(Parameter, bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability)
    
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    
    binscenters_zero_clean = binscenters[np.where(data_entries!=0.0)[0]]    
    data_entries_zero_clean = data_entries[np.where(data_entries!=0.0)[0]]
        
    data_entries = data_entries_zero_clean
    binscenters = binscenters_zero_clean
    
    peak_data = max(data_entries)            
    parameter_peak_loc = binscenters[np.where(data_entries==peak_data)[0]][0]
    
    buffer = 1e-5
    
    if parameter_peak_loc>0:
        
        peak_loc_min = (1-buffer) * parameter_peak_loc
        
        peak_loc_max = (1+buffer) * parameter_peak_loc
        
    elif parameter_peak_loc < 0:

        peak_loc_min = (1+buffer) * parameter_peak_loc
        
        peak_loc_max = (1-buffer) * parameter_peak_loc
        
    log_peak_data = np.log(peak_data)
    
    log_peak_data_min = np.log((1-buffer) * peak_data)
    
    log_data_entries = np.log(data_entries)
    
    
    if param_filename == "age":
        
        ### need to take care of PMS ages that pop up
        
        PMS_break = 1 #Gyr
        
        data_entries_PMS = data_entries[binscenters<PMS_break]
        data_entries_MS = data_entries[binscenters>=PMS_break]        
        
        binscenters_PMS = binscenters[binscenters<PMS_break]
        binscenters_MS = binscenters[binscenters>=PMS_break]
        
        if len(data_entries_PMS) == 0:
            
            peak_data_PMS = data_entries_MS[0]
            parameter_peak_loc_PMS = min(binscenters_MS)
            peak_loc_PMS_min = (1-buffer) * parameter_peak_loc_PMS
            peak_loc_PMS_max = (1+buffer) * parameter_peak_loc_PMS
            
        else:
        
            peak_data_PMS = max(data_entries_PMS)
            parameter_peak_loc_PMS = binscenters_PMS[np.where(data_entries_PMS==peak_data_PMS)[0]][0]
            peak_loc_PMS_min = (1-buffer) * parameter_peak_loc_PMS
            peak_loc_PMS_max = (1+buffer) * parameter_peak_loc_PMS
            
        if len(data_entries_MS) == 0:
            
            peak_data_MS = data_entries_PMS[len(data_entries_PMS)-1]
            parameter_peak_loc_MS = max(binscenters_PMS)
            peak_loc_MS_min = (1-buffer) * parameter_peak_loc_MS
            peak_loc_MS_max = (1+buffer) * parameter_peak_loc_MS
            
        else:

            peak_data_MS = max(data_entries_MS)
            parameter_peak_loc_MS = binscenters_MS[np.where(data_entries_MS==peak_data_MS)[0]][0]
            peak_loc_MS_min = (1-buffer) * parameter_peak_loc_MS
            peak_loc_MS_max = (1+buffer) * parameter_peak_loc_MS
        
        log_peak_data_PMS = np.log(peak_data_PMS)        
        log_peak_data_PMS_min = np.log((1-buffer)*peak_data_PMS)
        
        log_peak_data_MS = np.log(peak_data_MS)        
        log_peak_data_MS_min = np.log((1-buffer)*peak_data_MS)
                
                    
    try:
        
            if param_filename == "age":   
                
                popt, pcov = curve_fit(fit_function_quad_bimodal, 
                                       xdata=binscenters, 
                                       ydata=log_data_entries, 
                                       p0=[log_peak_data_PMS, 
                                           parameter_peak_loc_PMS, 
                                           Param_binwidth, 
                                           log_peak_data_MS, 
                                           parameter_peak_loc_MS, 
                                           Param_binwidth],
                                       bounds=((log_peak_data_PMS_min,
                                                 peak_loc_PMS_min,
                                                 1e-5*Param_binwidth,
                                                 log_peak_data_MS_min,
                                                 peak_loc_MS_min,
                                                 1e-5*Param_binwidth),       
                                                (log_peak_data_PMS,
                                                 peak_loc_PMS_max,
                                                 1e1*Param_binwidth,
                                                 log_peak_data_MS,
                                                 peak_loc_MS_max,
                                                 1e1*Param_binwidth)))
   

                # giving only the 2nd peak, what about fitting the first? 

                # well we always pick the 2nd one anyways, couldn't come up with anything otherwise

                # means the bimodal stuff was a waste in time

                # I can't seem to get it working at all!

                # hwo do I stop the computer from rounding down?                                                                                                                                                                               

                # popt, pcov = curve_fit(fit_function_quad, xdata=binscenters, ydata=log_data_entries, p0=[log_peak_data_MS, parameter_peak_loc_MS, Param_binwidth],bounds=((log_peak_data_MS_min,peak_loc_MS_min,1e-5*Param_binwidth),(log_peak_data_MS,peak_loc_MS_max,1e2*Param_binwidth)))

                
            else: 
        
            # popt, pcov = curve_fit(fit_function_quad, xdata=binscenters, ydata=log_data_entries, p0=[log_peak_data, parameter_peak_loc, Param_binwidth],bounds=((min(np.log(data_entries)),min(Parameter),1e-5*Param_binwidth),(log_peak_data,max(Parameter),1e2*Param_binwidth)))
            
                popt, pcov = curve_fit(fit_function_quad, xdata=binscenters, ydata=log_data_entries, p0=[log_peak_data, parameter_peak_loc, Param_binwidth],bounds=((log_peak_data_min,peak_loc_min,1e-5*Param_binwidth),(log_peak_data,peak_loc_max,1e2*Param_binwidth)))
                        
            suc=True
                                            
    except RuntimeError:
            
            if param_filename == "age":
                
                popt=np.ones([6])
                
            else: 
                
                popt=np.ones([3])

            popt*=np.nan
            
            suc=False


        
    if param_filename == "age":
    
        mu_fit = popt[1]
        sigma_fit = popt[2]
        
        mu_fit_2 = popt[4]
        sigma_fit_2 = popt[5]
        
        # print([peak_data_PMS, parameter_peak_loc_PMS, Param_binwidth,peak_data_MS,parameter_peak_loc_MS,Param_binwidth])
        # print([popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]])
        
        fitted_params = [mu_fit,sigma_fit,mu_fit_2,sigma_fit_2]
    
    else:
        
        mu_fit = popt[1]
        sigma_fit = popt[2]

        fitted_params = [mu_fit,sigma_fit]
    
    if makefig_bool == True:
    
        xspace = np.linspace(min(Parameter),max(Parameter),1000)
                        
        fig_prob_1d = plt.figure(figsize=(12,12))
        
        ax_1d = fig_prob_1d.add_subplot(111)
        
        ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability,density=False,label=f'Bin width = {Param_binwidth:.2f}',alpha=0.5,histtype='stepfilled',color='tomato',edgecolor='k')
        
        if suc == True:
        
            if param_filename == "temperature":
            
                ax_1d.plot(xspace, np.exp(fit_function_quad(xspace, *popt)), color='blue', linewidth=2.5, label=r'$\mu$ = ' + f'{mu_fit:.0f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.0f}\n')

            elif param_filename == "age":
                
                ax_1d.plot(xspace, np.exp(fit_function_quad_bimodal(xspace, *popt)), color='blue', linewidth=2.5, label=r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n\n' + r'$\mu_2$ = ' + f'{mu_fit_2:.3f}\n' + r'$\sigma_2$ = ' + f'{sigma_fit_2:.3f}\n')        
            
            else:
                
                ax_1d.plot(xspace, np.exp(fit_function_quad(xspace, *popt)), color='blue', linewidth=2.5, label=r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n')        
        
        if suc == False:
            
            print("Fitting failed, change bin size")
        
        ax_1d.set_ylabel('Probability')
        # ax_1d.set_ylabel('Counts')
        ax_1d.set_xlabel(f'{Param_name}')
        
        # hist, bins = np.histogram(Parameter,weights=Probability)
        
        # Gauss_prob = normpdf(Parameter,np.average(Parameter),Param_binwidth)
                
        # ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),density=False,label=f'Bin width = {Param_binwidth}',alpha=1,histtype='stepfilled')
    
        # ax_1d.bar(Parameter, Probability, width=0.08, bottom=None)
    
        ax_1d.legend(loc='upper right',fontsize=20)
        
        ax_1d.set_xlim([min(Parameter),max(Parameter)])
        
        # ax_1d.set_yscale("log")
                
        ax_1d.set_ylim(ymin=0)
        
        prob_title_name = prob_input_name.replace("_"," ").replace("w","x")
        
        plt.title(f"{prob_title_name}")
        
        plt.show()
        
        if savefig_bool == True:
            
            directory_figures= stellar_id # name of directory is the name of the star
            directory_check = os.path.exists(f"../Output_figures/1d_distributions/{directory_figures}")
            
            if  directory_check == True:
                
                pass
                
            else:
                
                print(f"../Output_figures/1d_distributions/{directory_figures} directory does not exist")
                
                os.makedirs(f"../Output_figures/1d_distributions/{directory_figures}")
                
                print(f"../Output_figures/1d_distributions/{directory_figures} directory has been created")
                    
            if chisq_red_bool == False:
            
                fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_no_norm_covariance_log_fit{savefig_extra_name}.png")

            elif chisq_red_bool == True:
            
                fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_norm_covariance_log_fit{savefig_extra_name}.png")
        
        plt.close('all')
        
    return fitted_params
    
    return [mu_fit,sigma_fit]

def Log_multiple_prob_distributions_1D(Parameter,Probability_arr,Param_name,Param_binwidth,prob_input_name_arr,param_filename,makefig_bool,savefig_bool,stellar_id,chisq_red_bool,col_arr,extra_name):
       
    
    """
    Need to cut the data down and get rid of any zero regions which waste time
    """
    
    fig_prob_1d = plt.figure(figsize=(12,12))
            
    ax_1d = fig_prob_1d.add_subplot(111)
        
    for Prob_index in range(len(Probability_arr)):
        
        Probability = Probability_arr[Prob_index]
        prob_input_name = prob_input_name_arr[Prob_index]
    
        Parameter_zero_clean = Parameter[np.where(Probability!=0.0)[0]]
        Probability_zero_clean = Probability[np.where(Probability!=0.0)[0]]
                
        # Parameter = Parameter_zero_clean
        # Probability = Probability_zero_clean
        
        data_entries, bins = np.histogram(Parameter, bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability)
        
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        
        binscenters_zero_clean = binscenters[np.where(data_entries!=0.0)[0]]    
        data_entries_zero_clean = data_entries[np.where(data_entries!=0.0)[0]]
        
        data_entries = data_entries_zero_clean
        binscenters = binscenters_zero_clean
        
        peak_data = max(data_entries)            
        parameter_peak_loc = binscenters[np.where(data_entries==peak_data)[0]][0]
                
        buffer = 1e-5
        
        if parameter_peak_loc>0:
            
            peak_loc_min = (1-buffer) * parameter_peak_loc
            
            peak_loc_max = (1+buffer) * parameter_peak_loc
            
        elif parameter_peak_loc < 0:
    
            peak_loc_min = (1+buffer) * parameter_peak_loc
            
            peak_loc_max = (1-buffer) * parameter_peak_loc
            
        log_peak_data = np.log(peak_data)
        
        log_peak_data_min = np.log((1-buffer) * peak_data)
        
        log_data_entries = np.log(data_entries)
        
        
        if param_filename == "age":
            
            ### need to take care of PMS ages that pop up
            
            PMS_break = 1 #Gyr
            
            data_entries_PMS = data_entries[binscenters<PMS_break]
            data_entries_MS = data_entries[binscenters>=PMS_break]        
            
            binscenters_PMS = binscenters[binscenters<PMS_break]
            binscenters_MS = binscenters[binscenters>=PMS_break]
            
            peak_data_PMS = max(data_entries_PMS)
            parameter_peak_loc_PMS = binscenters_PMS[np.where(data_entries_PMS==peak_data_PMS)[0]][0]
            peak_loc_PMS_min = (1-buffer) * parameter_peak_loc_PMS
            peak_loc_PMS_max = (1+buffer) * parameter_peak_loc_PMS
    
            peak_data_MS = max(data_entries_MS)
            parameter_peak_loc_MS = binscenters_MS[np.where(data_entries_MS==peak_data_MS)[0]][0]
            peak_loc_MS_min = (1-buffer) * parameter_peak_loc_MS
            peak_loc_MS_max = (1+buffer) * parameter_peak_loc_MS
            
            log_peak_data_PMS = np.log(peak_data_PMS)        
            log_peak_data_PMS_min = np.log((1-buffer)*peak_data_PMS)
            
            log_peak_data_MS = np.log(peak_data_MS)        
            log_peak_data_MS_min = np.log((1-buffer)*peak_data_MS)
                    
                        
        try:
            
                if param_filename == "age":   
                    
                    popt, pcov = curve_fit(fit_function_quad_bimodal, 
                                           xdata=binscenters, 
                                           ydata=log_data_entries, 
                                           p0=[log_peak_data_PMS, 
                                               parameter_peak_loc_PMS, 
                                               Param_binwidth, 
                                               log_peak_data_MS, 
                                               parameter_peak_loc_MS, 
                                               Param_binwidth],
                                           bounds=((log_peak_data_PMS_min,
                                                    peak_loc_PMS_min,
                                                    1e-5*Param_binwidth,
                                                    log_peak_data_MS_min,
                                                    peak_loc_MS_min,
                                                    1e-5*Param_binwidth),       
                                                    (log_peak_data_PMS,
                                                    peak_loc_PMS_max,
                                                    1e1*Param_binwidth,
                                                    log_peak_data_MS,
                                                    peak_loc_MS_max,
                                                    1e1*Param_binwidth)))
                                                    
                    
                else: 
                            
                    popt, pcov = curve_fit(fit_function_quad, xdata=binscenters, ydata=log_data_entries, p0=[log_peak_data, parameter_peak_loc, Param_binwidth],bounds=((log_peak_data_min,peak_loc_min,1e-5*Param_binwidth),(log_peak_data,peak_loc_max,1e2*Param_binwidth)))
                            
                suc=True
                                                
        except RuntimeError:
                
                if param_filename == "age":
                    
                    popt=np.ones([6])
                    
                else: 
                    
                    popt=np.ones([3])
    
                popt*=np.nan
                
                suc=False
            
            
        if param_filename == "age":
        
            mu_fit = popt[1]
            sigma_fit = popt[2]
            
            mu_fit_2 = popt[4]
            sigma_fit_2 = popt[5]
                        
            fitted_params = [mu_fit,sigma_fit,mu_fit_2,sigma_fit_2]
        
        else:
            
            mu_fit = popt[1]
            sigma_fit = popt[2]
    
            fitted_params = [mu_fit,sigma_fit]
        
        if makefig_bool == True:
        
            xspace = np.linspace(min(Parameter),max(Parameter),1000)
                                                    
            if suc == True:
                
                
                ### NEED TO RENORMALISE THESE DISTRIBUTIONS ###
            
                if param_filename == "temperature":
                    
                    
                    y_unnorm = np.exp(fit_function_quad(xspace, *popt))
                    
                    # N_factor = np.sum(binscenters * data_entries)
                    N_factor = np.sum(data_entries)
                    
                    y_norm = y_unnorm/N_factor
                
                    ax_1d.plot(xspace, y_norm, color=col_arr[Prob_index], linewidth=2.5, label=f'{prob_input_name}\n' + r'$\mu$ = ' + f'{mu_fit:.0f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.0f}\n')
    
                elif param_filename == "age":

                    y_unnorm = np.exp(fit_function_quad_bimodal(xspace, *popt))
                    
                    # N_factor = np.sum(binscenters * data_entries)
                    N_factor = np.sum(data_entries)
                    
                    y_norm = y_unnorm/N_factor
                    
                    ax_1d.plot(xspace, y_norm, color=col_arr[Prob_index], linewidth=2.5, label=f'{prob_input_name}\n' + r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n\n' + r'$\mu_2$ = ' + f'{mu_fit_2:.3f}\n' + r'$\sigma_2$ = ' + f'{sigma_fit_2:.3f}\n')        
                
                else:

                    y_unnorm = np.exp(fit_function_quad(xspace, *popt))
                    
                    # N_factor = np.sum(binscenters * data_entries)
                    N_factor = np.sum(data_entries)
                    
                    y_norm = y_unnorm/N_factor
                    
                    ax_1d.plot(xspace, y_norm, color=col_arr[Prob_index], linewidth=2.5, label=f'{prob_input_name}\n' + r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n')        
            
            if suc == False:
                
                print("Fitting failed, change bin size")
                            
    ax_1d.set_ylabel('Probability',fontsize=35)
    ax_1d.set_xlabel(f'{Param_name}',fontsize=35)
    
    # hist, bins = np.histogram(Parameter,weights=Probability)
    
    # Gauss_prob = normpdf(Parameter,np.average(Parameter),Param_binwidth)
            
    # ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),density=False,label=f'Bin width = {Param_binwidth}',alpha=1,histtype='stepfilled')

    # ax_1d.bar(Parameter, Probability, width=0.08, bottom=None)

    ax_1d.legend(loc='upper right',fontsize=20)
    
    ax_1d.set_xlim([min(Parameter),max(Parameter)])
    
    # ax_1d.set_yscale("log")
            
    ax_1d.set_ylim(ymin=0)
    
    prob_title_name = prob_input_name.replace("_"," ").replace("w","x")
    
    
    ax_1d.tick_params(axis='both',which='major',labelsize=35)
    
    # plt.title(f"{prob_title_name}")
    
    plt.show()
    
    if savefig_bool == True:
        
        directory_figures= stellar_id # name of directory is the name of the star
        directory_check = os.path.exists(f"../Output_figures/1d_distributions/{directory_figures}")
        
        if  directory_check == True:
            
            pass
            # print(f"../Output_figures/1d_distributions/{directory_figures} directory exists")
            
        else:
            
            print(f"../Output_figures/1d_distributions/{directory_figures} directory does not exist")
            
            os.makedirs(f"../Output_figures/1d_distributions/{directory_figures}")
            
            print(f"../Output_figures/1d_distributions/{directory_figures} directory has been created")
    
        # fig_prob_1d.savefig(f"../Output_figures/test_folder/1d_distributions_sun/{prob_input_name}_vs_{param_filename}.png")
        
        if chisq_red_bool == False:
        
            fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/Prob_1d_fits_vs_{param_filename}_{stellar_id}_no_norm_covariance_log_fit{extra_name}.png")

        elif chisq_red_bool == True:
        
            fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/Prob_1d_fits_vs_{param_filename}_{stellar_id}_norm_covariance_log_fit{extra_name}.png")
    
    plt.close('all')
            
def fit_function_quad(x,*labels_inp):
    
    A_log = labels_inp[0] 
    mu = labels_inp[1]
    sigma = labels_inp[2]
    
    # print(labels_inp)
    
    return -0.5 * (x - mu) ** 2 / sigma ** 2 + A_log

def fit_function_quad_bimodal(x,*labels_inp):
    
    A_1 = labels_inp[0]
    mu_1 = labels_inp[1]
    sigma_1 = labels_inp[2]

    A_2 = labels_inp[3]
    mu_2 = labels_inp[4]
    sigma_2 = labels_inp[5]
                        
    bi_gauss = np.exp(A_1 -1.0 * (x - mu_1)**2 / (2 * sigma_1**2)) + np.exp(A_2 -1.0 * (x - mu_2)**2 / (2 * sigma_2**2))
    
    # go through and replace zeros with 1e-323, smallest possible number before logging
    
    for zero_check in range(len(bi_gauss)):
        
        if bi_gauss[zero_check] == 0.0:
            
            bi_gauss[zero_check] = 1e-323
            
        elif  bi_gauss[zero_check] != 0.0:
            
            continue
    
    log_bi_gauss = np.log(bi_gauss)
    
    return log_bi_gauss


def fit_function_normal(x,*labels_inp):
    
    A = labels_inp[0] 
    mu = labels_inp[1]
    sigma = labels_inp[2]
    
    return (A * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))

def bimodal(x,*labels_inp):

    A_1 = labels_inp[0] 
    mu_1 = labels_inp[1]
    sigma_1 = labels_inp[2]

    A_2 = labels_inp[3] 
    mu_2 = labels_inp[4]
    sigma_2 = labels_inp[5]
        
    bi_gauss = A_1 * np.exp(-1.0 * (x - mu_1)**2 / (2 * sigma_1**2)) + A_2 * np.exp(-1.0 * (x - mu_2)**2 / (2 * sigma_2**2))
        
    return bi_gauss


### INPUT DATA ###

Input_phot_data = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/PLATO_bmk_phot_data/PLATO_photometry.csv',delimiter = ',',dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; G
2 ; eG
3 ; Bp
4 ; eBp
5 ; Rp
6 ; eRp
7 ; Gaia Parallax (mas)
8 ; eGaia Parallax (mas)
9 ; AG [mag], Gaia Extinction
10 ; Gaia DR2 ids
11 ; B (Johnson filter) [mag]
12 ; eB (Johnson filter) [mag]
13 ; V (Johnson filter) [mag]
14 ; eV (Johnson filter) [mag]
15 ; H (2MASS filter) [mag]
16 ; eH (2MASS filter) [mag]
17 ; J (2MASS filter) [mag]
18 ; eJ (2MASS filter) [mag]
19 ; K (2MASS filter) [mag]
20 ; eK (2MASS filter) [mag]
21 ; SIMBAD Parallax (mas)
22 ; eSIMBAD Parallax (mas)
23 ; E(B-V), reddening value from stilism 	
24 ; E(B-V)_upp, upper reddening uncertainty from stilism 	
25 ; E(B-V)_low, lower reddening uncertainty from stilism 	
"""

Input_ast_data = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/PLATO_stars_seism.txt',delimiter = ',',dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; nu_max, maximum frequency [microHz]
2 ; err nu_max pos, upper uncertainty
3 ; err nu_max neg, lower uncertainty
4 ; d_nu, large separation frequency [microHz]
5 ; err d_nu pos, upper uncertainty [microHz]
6 ; err d_nu neg, lower uncertainty [microHz]
"""

Input_spec_data = np.loadtxt('../Input_data/spectroscopy_observation_data/spectra_list_PLATO.txt',delimiter=',',dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; spectra type (e.g. HARPS or UVES_cont_norm_convolved etc)
2 ; spectra path
"""

Ref_data = np.loadtxt("../Input_data/Reference_data/PLATO_stars_lit_params.txt",delimiter=",",dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; Teff [K]
2 ; err Teff [K]
3 ; Logg [dex]
4 ; err Logg [dex]
5 ; [Fe/H] [dex]
6 ; err [Fe/H] [dex]
"""

Ref_data_other = np.loadtxt("../Input_data/Reference_data/PLATO_stars_lit_params.txt",delimiter=",",dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; Vmic
2 ; err Vmic
3 ; Vbrd
4 ; err Vbrd
5 ; [Mg/Fe]
6 ; err [Mg/Fe]
7 ; [Ti/Fe]
8 ; err [Ti/Fe]
9 ; [Mn/Fe]
10 ; err [Mn/Fe]
"""

correlation_arr = correlation_table(Input_spec_data,Input_phot_data)

stellar_names = Input_phot_data[:,0]

error_mask_recreate_bool = False # if False, then teff varying mask is assumed
error_map_use_bool = True
cont_norm_bool = False
rv_shift_recalc = [False,-100,100,0.05]
conv_instrument_bool = False
input_spec_resolution = 20000
numax_iter_bool = False
niter_numax_MAX = 5
recalc_metals_bool = False
feh_recalc_fix_bool = False
logg_fix_bool = False

spec_fix_values = [5770,4.44,0]

### photometry grid limits

Teff_resolution = 250 # K
logg_resolution = 0.3 # dex
feh_resolution = 0.3 # dex

phot_ast_central_values_bool = False # if set to True, values below will be used
phot_ast_central_values_arr = [5900,4,0] # these values are at the centre of the photometry grid

chi_2_red_bool = False

best_spec_bool = True 
phot_ast_track_bool = False
spec_track_bool = False
bayes_scheme_bool = False


### naming conventions for the settings above

# mode_kw = "LITE_logg"
# mode_kw = "LITE_numax"
# mode_kw = "MASK"
mode_kw = "BLIND"
# mode_kw = "FULL"
# mode_kw = "BLIND_numax"



spec_covariance_bool = True # To use spectroscopic covariance in creation of PDF

### string that is attached to the filename, can be useful for specifying a certain way you ran the code

# extra_save_string = "_no_evo_prior"
# extra_save_string = ""
# extra_save_string = "_solar_emask"
extra_save_string = "_teff_emask"

if __name__ == '__main__':
    
    inp_index = 15
    
    main_func_multi_obs(inp_index)
    

'''
num_processor = 8

inp_index_list = np.arange(len(stellar_names))

if __name__ == '__main__':
#    
    p = mp.Pool(num_processor) # Pool distributes tasks to available processors using a FIFO schedulling (First In, First Out)
    p.map(main_func_multi_obs,inp_index_list) # You pass it a list of the inputs and it'll iterate through them
    p.terminate() # Terminates process 
'''
