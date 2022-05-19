#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:26:18 2020

@author: gent
"""

### import python packages

from astropy.io import fits as pyfits

import numpy as np
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.stats import iqr as IQR_SCIPY
import traceback
import SAPP_spectroscopy.Payne.astro_constants_cgs as astroc
### import SAPP scripts


import os
#import psutil

from SAPP_stellar_evolution_scripts.photometry_asteroseismology_lhood_TEST import photometry_asteroseismic_lhood_stellar_list as photometry_ast_pdf
from SAPP_stellar_evolution_scripts.asteroseismology_lhood_TEST import asteroseismic_lhood_stellar_list as ast_pdf
import SAPP_spectroscopy.Payne.SAPP_best_spec_payne_v1p1 as mspec_new

from SAPP_spectroscopy.Payne.SAPP_best_spec_payne_v1p1 import cheb

np.set_printoptions(suppress=True,formatter={'float': '{: 0.3f}'.format})

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
        
    # correlation_arr = []
    # for star in phot_ast_IDs: # going through each ro
    #     # print("star =",star)
    #     star_spec_list = []
    #     for spec_index in range(len(spec_IDs)):
    #         if spec_IDs[spec_index] == star:
    #             star_spec_list.append(spec_index)
    #         else: 
    #             continue
    #             # star_spec_list.append(np.nan)
    #     star_spec_list = np.array(star_spec_list)
    #     correlation_arr.append(star_spec_list)
    # correlation_arr = np.array(correlation_arr)
    
    correlation_arr = []
    for i in range(len(phot_ast_IDs)):
        spec_id_where = np.where(spec_IDs == phot_ast_IDs[i])[0]
        if len(spec_id_where) > 0:
            correlation_arr.append(spec_id_where)
        else:
            continue
    correlation_arr = np.array(correlation_arr)
    
    return correlation_arr

def logg_numax(numax,nu_max_err,teff,teff_err):
    """
    Takes input numax value and teff to calculate logg using scaling relation
    """
    
    
    logg_numax = np.log10((numax/astroc.nu_max_sol) * (teff/astroc.teff_sol) ** 0.5 * astroc.surf_grav_sol)
    # logg_numax = np.log10((numax/astroc.nu_max_sol) * (teff/h5f_PLATO_consts['CGS/solar/Teff'][()]) ** 0.5 * astroc.surf_grav_sol)

    # one way of calculating errors
    # nu_max_iter_arr = [numax-nu_max_err,numax,numax+nu_max_err]
    # logg_numax_lower = np.log10((nu_max_iter_arr[0]/astroc.nu_max_sol) * (teff/astroc.teff_sol) ** 0.5 * astroc.surf_grav_sol)
    # logg_numax_upper = np.log10((nu_max_iter_arr[2]/astroc.nu_max_sol) * (teff/astroc.teff_sol) ** 0.5 * astroc.surf_grav_sol)
    # logg_numax_upper_err = logg_numax_upper - logg_numax
    # logg_numax_lower_err = logg_numax - logg_numax_lower
    # print("linear errors",logg_numax,logg_numax_upper_err,logg_numax_lower_err)
    
    dlogg_dnumax = 1/(np.log(10) * numax)
    dlogg_dteff = 0.5/(np.log(10) * teff)
    
    # teff_err=0
    
    logg_numax_err = np.sqrt(abs(dlogg_dnumax)**2*(nu_max_err)**2 + abs(dlogg_dteff)**2*(teff_err)**2)
    logg_numax_upper_err = logg_numax_err
    logg_numax_lower_err = logg_numax_err
    
    # print("proper errors",logg_numax,logg_numax_lower_err,logg_numax_upper_err)
    
    return logg_numax,logg_numax_lower_err,logg_numax_upper_err

def spectroscopy_stellar_track_collect(inp_arr,phot_param_space):
    
    """
    This function creates spectroscopy PDF based on the PDF created for Photometry  
    """
    
    # print(inp_arr)
    
    inp_index = inp_arr[0] # index in ref. to star list
    
    best_fit_spectroscopy = inp_arr[1] # best fit spec params initial guess
    
    star_field = inp_arr[2] # if True, reduce the chi2, if False, do other normalisation
    
    spec_save_extension = inp_arr[3]
    
    spec_obs_number = spec_save_extension.split("_")[3]
    
    save_spec_phot_param_space_bool = inp_arr[4]
        
    # import_path = "../Input_data/spectroscopy_model_data/Payne_input_data/"
    
    # name="NN_results_RrelsigN20.npz"#NLTE NN trained spectra grid results
    
    # temp=np.load(import_path+name)
    
    # x_min = temp["x_min"] # minimum limits of grid
    # x_max = temp["x_max"] # maximum limits of grid
    
    # print(f"stellar_names[inp_index]")
    
    stellar_filename = stellar_names[inp_index].replace(" ","_")
    
    ### load photometry, asteroseismology stellar tracks
    # param_space_phot = np.loadtxt(f"../Output_data/Stars_Lhood_photometry/{stellar_filename}/stellar_track_collection_{stellar_filename}_test_3.txt",delimiter=",")

    # param_space_phot = np.loadtxt(f"../Output_data/Stars_Lhood_photometry/{star_field}/stellar_track_collection_{stellar_filename}_OBS_NUM_{spec_obs_number}_test_4.txt",delimiter=",")
    
    param_space_phot = phot_param_space
    
    print("PARAM SPACE PHOT COL LENGTH",len(param_space_phot.T))
    
    ### load photometry, asteroseismology stellar tracks with no stellar evo influence
    # i.e. photometry we care about is two colours: Bp-Rp, V-K and asteroseismology is nu_max, magnitudes and delta_nu require some form of stellar evolution information.
    # param_space_test = np.loadtxt(f"../Output_data/Stars_Lhood_photometry/{stellar_filename}/stellar_track_collection_{stellar_filename}_test_3_non_stellar_evo_priors.txt",delimiter=",")
    # param_space_phot = np.loadtxt(f"../Output_data/Stars_Lhood_photometry/{stellar_filename}/stellar_track_collection_{stellar_filename}_test_3_ast_NaN.txt",delimiter=",")



    teff_phot = param_space_phot[:,0]
    logg_phot = param_space_phot[:,1]
    feh_phot = param_space_phot[:,2]
        
    best_params = best_fit_spectroscopy[0]
    
    best_params_err = best_fit_spectroscopy[1]

    cov_matrix = best_fit_spectroscopy[2]

    
    # ch2_best = best_fit_spectroscopy[4] # chi-square of best fit results
    # wvl_con = best_fit_spectroscopy[5] # wavelength of spectra post processing
    # wvl_obs_input = best_fit_spectroscopy[9] # original wavelength of obs spectra
    obs_norm = best_fit_spectroscopy[3] # observation flux normalised
    err_norm = best_fit_spectroscopy[4] # err of observation flux normalised
    wvl_obs = best_fit_spectroscopy[5]
    wvl_obs_input = best_fit_spectroscopy[6]
    cheb = best_fit_spectroscopy[7]
    
    # cov_matrix = best_fit_spectroscopy[11] # covariance matrix
    
    # cov_matrix_new = cov_matrix[1:,1:]

    if cheb == 0:
        cov_matrix_new = cov_matrix[1:,1:]
    else:
        cov_matrix_new = cov_matrix[1:-cheb,1:-cheb]
        
    corr_matrix = correlation_from_covariance(cov_matrix_new) # correlation matrix
    
    # print("corr matrix",corr_matrix)
        
    """
    Plot correlation matrix 
    """
    
    '''
    spectral_parameters_names = ["T$_{eff}$","log(g)","[Fe/H]","Vmic","Vbrd","[Mg/Fe]","[Ti/Fe]","[Mn/Fe]"]
    
    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(spectral_parameters_names)))
    ax.set_yticks(np.arange(len(spectral_parameters_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(spectral_parameters_names)
    ax.set_yticklabels(spectral_parameters_names)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor") 
    
    for i in range(len(spectral_parameters_names)):
        for j in range(len(spectral_parameters_names)):
            text = ax.text(j, i, corr_matrix[i, j],
                       ha="center", va="center", color="w")
    
    fig.tight_layout()
    plt.show()
        
    '''

    print("Spectroscopic temperature error",best_params_err[0])    
    # print("Spectroscopic temperature error",best_params_err[0]*1000)
    print("Spectroscopic logg error",best_params_err[1])
    print("Spectroscopic feh error",best_params_err[2])
        
    # if best_params_err[0]*1000 < Teff_thresh:
    #     teff_spec_err = best_params_err[0]*1000 + Teff_thresh + Teff_sys[1]
    # else:
    #     teff_spec_err = best_params_err[0]*1000 + Teff_sys[1]

    # 4MOST NN
    if best_params_err[0] < Teff_thresh:
        teff_spec_err = best_params_err[0] + Teff_thresh + Teff_sys[1]
    else:
        teff_spec_err = best_params_err[0] + Teff_sys[1]
        
    if best_params_err[1] < Logg_thresh:
        logg_spec_err = best_params_err[1] + Logg_thresh + Logg_sys[1]
    else:
        logg_spec_err = best_params_err[1] + Logg_sys[1]
        
    if best_params_err[2] < FeH_thresh:
        feh_spec_err = best_params_err[2] + FeH_thresh + FeH_spec_sys[1]
    else:
        feh_spec_err = best_params_err[2] + FeH_spec_sys[1]

    sigma_arr = [teff_spec_err,logg_spec_err,feh_spec_err]

    print("sigma arr",sigma_arr)
    
    
    cov_matrix_norm = np.zeros([3,3])     
    for i in range(len(sigma_arr)):        
        for j in range(len(sigma_arr)):                        
            cov_matrix_norm[i][j] = corr_matrix[i][j] * sigma_arr[i] * sigma_arr[j]        
    cov_matrix_norm_inv = np.linalg.inv(cov_matrix_norm) # this is the inverse of the normalised covariance matrix
    
    
    params_fix = best_params[3:] # this still applies with new grid, however Magnesium's position changes as grid changes
    
    # MgFe = params_fix[2] # magnesium abundance hr10 grid
    # MgFe = params_fix[5] # magnesium abundance RVS grid 
    MgFe = params_fix[1] # magnesium abundance 4MOST NN, converted when spec results saved 
        
    print("NUMBER OF POINTS",len(teff_phot))
    
    # this is taking into account covariance between the primary parameters
    
    chi2_spec_third_arr = []

    start_time_spec_tracks = time.time()

    feh_de_enhance = feh_phot - (0.22/0.4 * (MgFe - MgFe_sys[0])) # this has [Mg/Fe] systematic applied
    # chi2_teff_i = (((best_params[0]*1000 - Teff_sys[0])-teff_phot)/(sigma_arr[0]))**2
    chi2_teff_i = (((best_params[0] - Teff_sys[0])-teff_phot)/(sigma_arr[0]))**2 # 4MOST NN
    chi2_logg_i = (((best_params[1] - Logg_sys[0])-logg_phot)/(sigma_arr[1]))**2
    chi2_feh_i = (((best_params[2] - FeH_spec_sys[0])-feh_de_enhance)/(sigma_arr[2]))**2
    chi2_spec_second_arr = chi2_teff_i + chi2_logg_i + chi2_feh_i

    for test_index in range(len(feh_phot)):
        ### third type of chi2
        ### creates chi2 grid sampling photometry space with best fit spec values while including co-variance

        # diff_vector = np.array([teff_phot[test_index]-(best_params[0]*1000),logg_phot[test_index]-best_params[1],feh_de_enhance[test_index] - 0.22/0.4 * MgFe_sys[0] -best_params[2]])
        # diff_vector = np.array([teff_phot[test_index]-(best_params[0]*1000 - Teff_sys[0]),logg_phot[test_index]-(best_params[1]-Logg_sys[0]),feh_de_enhance[test_index] - (best_params[2] - FeH_spec_sys[0]) ])
        diff_vector = np.array([teff_phot[test_index]-(best_params[0] - Teff_sys[0]),logg_phot[test_index]-(best_params[1]-Logg_sys[0]),feh_de_enhance[test_index] - (best_params[2] - FeH_spec_sys[0]) ]) # 4MOST NN
        product_1 = np.matmul(diff_vector,cov_matrix_norm_inv)
        chi2_spec_third = np.matmul(product_1,diff_vector.T)
        chi2_spec_third_arr.append(chi2_spec_third)

    # ch2_spec_arr = np.array(ch2_spec_arr)
    chi2_spec_third_arr = np.array(chi2_spec_third_arr)

    
    ### create regular spaced teff, logg, feh arrays ###
    '''
    N_reg = 15
    teff_reg = np.linspace(min(teff_phot),max(teff_phot),N_reg)
    # chi2_teff_reg_i = (((best_params[0]*1000 - Teff_sys[0])-teff_reg)/(sigma_arr[0]))**2
    logg_reg = np.linspace(min(logg_phot),max(logg_phot),N_reg)
    # chi2_logg_reg_i = (((best_params[1] - Logg_sys[0])-logg_reg)/(sigma_arr[1]))**2
    feh_reg = np.linspace(min(feh_phot),max(feh_phot),N_reg)
    # chi2_feh_reg_i = (((best_params[2] - FeH_spec_sys[0])-feh_reg)/(sigma_arr[2]))**2
    teff_reg_grid = []
    logg_reg_grid = []
    feh_reg_grid = []
    for teff_reg_i in range(N_reg):
        for logg_reg_i in range(N_reg):
            for feh_reg_i in range(N_reg):
                teff_reg_grid.append(teff_reg[teff_reg_i])
                logg_reg_grid.append(logg_reg[logg_reg_i])
                feh_reg_grid.append(feh_reg[feh_reg_i])
    teff_reg_grid = np.array(teff_reg_grid)            
    logg_reg_grid = np.array(logg_reg_grid)            
    feh_reg_grid = np.array(feh_reg_grid)            
                
    print("regular grid created!")
    
    # collect 3D ch2 no cov info 
            
    ch2_check_direct_collect = []
    ch2_check_gaus_collect = []
    ch2_check_gaus_cov_collect = []

    for check_i in range(len(teff_reg_grid)):
        
        # direct space #
        
        labels_check = np.hstack((teff_reg_grid[check_i]/1000,logg_reg_grid[check_i],feh_reg_grid[check_i],best_params[3:len(best_params)-1]))
        labels_norm = (labels_check-mspec_new.x_min)/(mspec_new.x_max-mspec_new.x_min)-0.5
        labels_norm = np.hstack(([0],labels_norm,np.zeros([cheb])))
        fit = mspec_new.restore(np.array([0,0,0,[]]),*labels_norm) # model spectrum
        w0 = mspec_new.w0.copy()
        fit = np.interp(wvl_obs,w0,fit)
        ch2_check=np.sum(((obs_norm-fit)/err_norm)**2)/2.15e4#/(len(obs_norm)-len(labels_norm))
        # ch2_check=np.sum(((obs_norm-fit)/err_norm)**2)/2.15e3#/(len(obs_norm)-len(labels_norm))
        ch2_check_direct_collect.append(ch2_check)
        
        # no cov gaussian space #
        
        chi2_teff_reg_i = (((best_params[0]*1000 - Teff_sys[0])-teff_reg_grid[check_i])/(sigma_arr[0]))**2
        chi2_logg_reg_i = (((best_params[1] - Logg_sys[0])-logg_reg_grid[check_i])/(sigma_arr[1]))**2
        chi2_feh_reg_i = (((best_params[2] - FeH_spec_sys[0])-feh_reg_grid[check_i])/(sigma_arr[2]))**2
        chi2_spec_second_reg_arr = chi2_teff_reg_i + chi2_logg_reg_i + chi2_feh_reg_i
        ch2_check_gaus_collect.append(chi2_spec_second_reg_arr)
        
        # cov gaussian space #
        
        diff_vector = np.array([teff_reg_grid[check_i]-(best_params[0]*1000 - Teff_sys[0]),logg_reg_grid[check_i]-(best_params[1]-Logg_sys[0]),feh_reg_grid[check_i] - (best_params[2] - FeH_spec_sys[0]) ])
        product_1 = np.matmul(diff_vector,cov_matrix_norm_inv)
        chi2_spec_third = np.matmul(product_1,diff_vector.T)
        ch2_check_gaus_cov_collect.append(chi2_spec_third)
        
    ch2_check_direct_collect = np.array(ch2_check_direct_collect)
    ch2_check_direct_collect -= min(ch2_check_direct_collect)    
    L_check_direct_collect = np.exp(-ch2_check_direct_collect/2)
    ch2_check_gaus_collect = np.array(ch2_check_gaus_collect)
    ch2_check_gaus_collect -= min(ch2_check_gaus_collect)    
    L_check_gaus_collect = np.exp(-ch2_check_gaus_collect/2)
    ch2_check_gaus_cov_collect = np.array(ch2_check_gaus_cov_collect)
    ch2_check_gaus_cov_collect -= min(ch2_check_gaus_cov_collect)   
    L_check_gaus_cov_collect = np.exp(-ch2_check_gaus_cov_collect/2)
    
    print("chi2 calculated!")
    
    # plot 3D results
    from mpl_toolkits import mplot3d
    fig3D = plt.figure()
    ax3D = fig3D.add_subplot(111, projection='3d')
    
    # c1 = ax3D.scatter3D(teff_reg_grid, logg_reg_grid, feh_reg_grid, c=np.log10(L_check_gaus_cov_collect), cmap='plasma',vmin=-20,marker='o')
    # c1 = ax3D.scatter3D(teff_reg_grid, logg_reg_grid, feh_reg_grid, c=np.log10(L_check_gaus_collect), cmap='plasma',vmin=-20,marker='o')
    feh_mask = (feh_reg_grid <= 0.4)&(feh_reg_grid >= -0.1)
    # feh_mask = (feh_reg_grid <= max(feh_reg_grid))&(feh_reg_grid >= min(feh_reg_grid))

    c1 = ax3D.scatter3D(teff_reg_grid[feh_mask], logg_reg_grid[feh_mask], feh_reg_grid[feh_mask], c=np.log10(L_check_gaus_collect[feh_mask]), cmap='plasma',vmin=-2,marker='o',alpha=0.3)    
    # c1 = ax3D.scatter3D(teff_reg_grid[feh_mask], logg_reg_grid[feh_mask], feh_reg_grid[feh_mask], c=np.log10(L_check_gaus_cov_collect[feh_mask]), cmap='plasma',vmin=-5,marker='o',alpha=0.3)    
    # c1 = ax3D.scatter3D(teff_reg_grid[feh_mask], logg_reg_grid[feh_mask], feh_reg_grid[feh_mask], c=np.log10(L_check_direct_collect[feh_mask]), cmap='plasma',vmin=-5,marker='o',alpha=0.3)
    
    teff_buffer = 20
    logg_buffer = 0.01
    feh_buffer = 0.01
    
    # x, y = np.meshgrid(teff_reg_grid, logg_reg_grid)
    # z = np.meshgrid(feh_reg_grid)
    
    # cset = ax3D.contour(x, y, z, zdir='x', offset=max(teff_reg_grid)-teff_buffer, cmap='plasma')
    # cset = ax3D.contour(x, y, z, zdir='y', offset=min(logg_reg_grid)+logg_buffer, cmap='plasma')
    # cset = ax3D.contour(x, y, z, zdir='z', offset=min(feh_reg_grid)-feh_buffer, cmap='plasma')
    
    # ax3D.set_xlim(min(teff_reg_grid)-teff_buffer, max(teff_reg_grid))
    # ax3D.set_ylim(min(logg_reg_grid), max(logg_reg_grid)+logg_buffer)
    # ax3D.set_zlim(min(feh_reg_grid)-feh_buffer, max(feh_reg_grid))
    
    plt.colorbar(c1,label='Log10(Prob)')
    
    ax3D.view_init(elev=10., azim=90)
    # for angle in np.arange(0,360,30):
    #     ax3D.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)

    plt.title('$\chi^2$ Gaus No Cov')
    # plt.title('$\chi^2$ Gaus Cov')    
    # plt.title('$\chi^2$ Direct')
    
    plt.show()
    '''
    ###

    # plt.plot(teff_reg,np.exp(-ch2_teff_reg_i_cov/2),'ro',linestyle='-',label='gaussian cov')
    # plt.plot(teff_reg,np.exp(-chi2_teff_reg_i/2),'b+',linestyle='-',label='gaussian no cov')
    # plt.plot(teff_reg,np.exp(-ch2_check_collect/2),'g^',linestyle='-',label='direct space')
    # plt.legend(loc='lower left')
    # plt.show()


    
    ### normalise these chisq
    
    L_spec_second = np.exp(-(chi2_spec_second_arr-min(chi2_spec_second_arr))/2)

    L_spec_third = np.exp(-(chi2_spec_third_arr-min(chi2_spec_third_arr))/2)

    L_spec_w_phot = np.vstack([param_space_phot.T,L_spec_second,L_spec_third]).T
                                
    print(f"time elapsed --- {(time.time() - start_time_spec_tracks)/60} minutes ---")
    
    # pid = os.getpid()
    # py = psutil.Process(pid)
    # memoryUse = py.memory_info()[0]/2.**30
    
    # print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))


    if save_spec_phot_param_space_bool:

        directory_combined = star_field # name of directory is the name of the star
        
        directory_check = os.path.exists(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined}")
    
        if  directory_check == True:
        
            print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined} directory exists")
            
        else:
            
            print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined} directory does not exist")
            
            os.makedirs(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined}")
            
            print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined} directory has been created")
            
        # np.savetxt(f'../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{stellar_filename}/stellar_track_collection_w_spec_and_prob_{stellar_filename}_test_3{spec_save_extension}{extra_save_string}.txt',chi2_spec_w_phot,delimiter=",")
    
        np.savetxt(f'../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_combined}/stellar_track_collection_w_spec_and_prob_{stellar_filename}_test_4{spec_save_extension}{extra_save_string}.txt',L_spec_w_phot,delimiter=",")
                
    # return best_params
    return L_spec_w_phot


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

def spectroscopy_regular_grid_collect(stellar_filename,spec_obs_number,mode_kw,extra_save_string):
        
    best_spec_params = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt")
    best_spec_params_err = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt")    
    cov_matrix = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{stellar_filename}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt')
        
    # phot_limits = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/phot_interp_space_limits_{stellar_filename}.txt")
    phot_limits = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/phot_interp_space_limits_{stellar_filename}_EXTRA_wide.txt")

    temp_max_limit = phot_limits[0][0]
    temp_min_limit = phot_limits[0][1]
    logg_max_limit = phot_limits[1][0]
    logg_min_limit = phot_limits[1][1]
    feh_max_limit = phot_limits[2][1]
    feh_min_limit = phot_limits[2][0]
    
    fixed_params = best_spec_params[3:] # 8 parameters found from best fit, 5 others are kept fixed i.e. vmic,vbrd,Mg,Ti,Mn
       
    # Ngrid_phot = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/phot_interp_space_grid_size_{stellar_filename}.txt")
    Ngrid_phot = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/phot_interp_space_grid_size_{stellar_filename}_EXTRA_wide.txt")
        
    # Ngrid_FeH,Ngrid_Temp,Ngrid_logg
    
    Ngrid_Temp = int(Ngrid_phot[0])
    Ngrid_logg = int(Ngrid_phot[1])
    Ngrid_FeH = int(Ngrid_phot[2])
    
    T_range = np.linspace(temp_max_limit,temp_min_limit,Ngrid_Temp,endpoint=True)
    logg_range = np.linspace(logg_max_limit,logg_min_limit,Ngrid_logg,endpoint=True)
    FeH_range = np.linspace(feh_min_limit,feh_max_limit,Ngrid_FeH,endpoint=True)
    
    Ngrid = [Ngrid_FeH,Ngrid_logg,Ngrid_Temp]

    MgFe = fixed_params[2] # MgFe for hr10 model and hr21 model (Mikhail's label)
    # MgFe = fixed_params[5] # MgFe for RVS model (Jeff's label)
    
    cov_matrix_new = cov_matrix[1,1]
    
    # cov_matrix_new = np.vstack((cov_matrix_new[:,1],\
    #                             cov_matrix_new[:,2],\
    #                             cov_matrix_new[:,3],\
    #                             cov_matrix_new[:,4],\
    #                             cov_matrix_new[:,5],\
    #                             cov_matrix_new[:,6],\
    #                             cov_matrix_new[:,7],\
    #                             cov_matrix_new[:,8])).T
        
    corr_matrix = correlation_from_covariance(cov_matrix_new) # correlation matrix    
    
    sigma_arr = [best_spec_params_err[0]*1000,best_spec_params_err[1],best_spec_params_err[2]] #Teff, logg, [Fe/H] error
        
    cov_matrix_norm = np.zeros([3,3]) 
    
    for x in range(len(sigma_arr)):
        
        for y in range(len(sigma_arr)):
                        
            cov_matrix_norm[x][y] = corr_matrix[x][y] * sigma_arr[x] * sigma_arr[y]
        
    cov_matrix_norm_inv = np.linalg.inv(cov_matrix_norm) # this is the inverse of the normalised covariance matrix

    print("beginning spectroscopy grid building")
    
    param_space = []
    
    # we are giving the chisq function the FeH coordinate which has to exist in order for
    # the FeH enhanced point to be valid 
    # i.e. [Fe/H]st = [Fe/H]_spec + 0.22/0.4 * [Mg/Fe]_spec, need to find [Fe/H]_spec
    # then save metalicity point as [Fe/H]st which is part of FeH_range
        
    for i in range(0,Ngrid[0]):
#        FeH_alpha_enhanced = FeH_range[i] + 0.22/0.4 * MgFe # Aldo suggested that I convert Mikhails [Fe/H] point here in order to compare to points in parameter space
        FeH_alpha_enhanced = FeH_range[i]       
        for j in range(0,Ngrid[1]):
            for k in range(0,Ngrid[2]):
                
                # chi2_teff_i = (((best_spec_params[0]*1000)-T_range[k])/(best_spec_params_err[0]*1000))**2
                # chi2_logg_i = (((best_spec_params[1])-logg_range[j])/(best_spec_params_err[1]))**2
                # chi2_feh_i = (((best_spec_params[2])-FeH_alpha_enhanced)/(best_spec_params_err[2]))**2
                # chi2_spec = chi2_teff_i + chi2_logg_i + chi2_feh_i
                                
                diff_vector = np.array([T_range[k]-(best_spec_params[0]*1000),logg_range[j]-best_spec_params[1],FeH_alpha_enhanced-best_spec_params[2]])
                
                # print(diff_vector)
                
                product_1 = np.matmul(diff_vector,cov_matrix_norm_inv)
                
                chi2_spec = np.matmul(product_1,diff_vector.T)

                # print("chi2_spec",chi2_spec)                

                param_space.append([T_range[k],logg_range[j],FeH_alpha_enhanced,chi2_spec])
                
        # print(f"Point {i+1} in parameter space [Fe/H] = {FeH_alpha_enhanced} for Star ID {stellar_filename}")             
        
    param_space = np.array(param_space)
    
    # np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/spectroscopy_regular_grid_1D_{stellar_filename}.txt",param_space)
    np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/spectroscopy_regular_grid_1D_{stellar_filename}_EXTRA_wide.txt",param_space)

    print("spectroscopy grid saved")

    return param_space

def grid_regularisation(inp_arr):
    
    
    inp_index = inp_arr[0]
    spec_obs_number = inp_arr[1]
    mode_kw = inp_arr[2]
    extra_save_string = inp_arr[3]
    grid_resolution = inp_arr[4]
    
    
    Teff_resolution = grid_resolution[0]
    logg_resolution = grid_resolution[1]
    feh_resolution = grid_resolution[2]
    
    asteroseismolgy_logg_cut = 5 # dex
    
    photometry_non_gaia_colour_prob_cut = 1e-100
    
    stellar_inp_phot_cut = [stellar_names[inp_index],\
                            [Teff_resolution,\
                             logg_resolution,\
                             feh_resolution],
                             asteroseismolgy_logg_cut,\
                             photometry_non_gaia_colour_prob_cut]
        
    stellar_inp_interp = stellar_names[inp_index]
    spec_type = "hr10"
    stellar_inp_decomp = [stellar_names[inp_index],spec_type]

    stellar_filename = stellar_names[inp_index].replace(" ","_")    

    # post_cut_limits(stellar_inp_phot_cut)

    # spectroscopy_regular_grid_collect(stellar_filename,spec_obs_number,mode_kw,extra_save_string)
    
    # lhood_interpolation_stellar_evo(stellar_inp_interp)

    # interp_decomp(stellar_inp_decomp)
    
    # param_space_reg_spec = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/spectroscopy_regular_grid_1D_{stellar_filename}.txt")
    # param_space_reg_spec = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/spectroscopy_regular_grid_1D_{stellar_filename}_EXTRA_wide.txt")

    ##param_space_reg_all  = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/full_1d_param_space_StarID{stellar_filename}_{spec_type}.txt")
    param_space_reg_all  = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/full_1d_param_space_StarID{stellar_filename}_{spec_type}_EXTRA_wide.txt")

    param_space_reg_colour = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/regularisation/{stellar_filename}/COLOUR_1d_param_space_StarID{stellar_filename}_{spec_type}_EXTRA_wide.txt")

    ### NORMALISATION ###
    
    teff = param_space_reg_all[:,0]
    logg = param_space_reg_all[:,1]
    feh = param_space_reg_all[:,2]
    phot_non_gaia =  param_space_reg_all[:,3]
    phot_gaia = param_space_reg_all[:,4]
    phot_ast = param_space_reg_all[:,5]
    spec = param_space_reg_all[:,6]
    phot_combined_no_spec = param_space_reg_all[:,7]
    
    phot_non_gaia_col = param_space_reg_colour[:,3]
    
    # print(param_space_reg_all[:,0]/param_space_reg_spec[:,0])
    # print(param_space_reg_all[:,1]/param_space_reg_spec[:,1])
    # print(param_space_reg_all[:,2]/param_space_reg_spec[:,2])
        
    chi2_prob_spec = spec
    chi2_prob_spec_3_norm = chi2_prob_spec - np.nanmin(chi2_prob_spec)
    prob_spec = np.exp(-chi2_prob_spec_3_norm/2) 

    chi2_prob_phot_non_gaia = phot_non_gaia
    chi2_prob_phot_non_gaia_3_norm = chi2_prob_phot_non_gaia - np.nanmin(chi2_prob_phot_non_gaia)
    prob_phot_non_gaia = np.exp(-chi2_prob_phot_non_gaia_3_norm/2) 
    
    chi2_prob_phot_col_non_gaia = phot_non_gaia_col
    chi2_prob_phot_col_non_gaia_3_norm = chi2_prob_phot_col_non_gaia - np.nanmin(chi2_prob_phot_col_non_gaia)
    prob_phot_non_col_gaia = np.exp(-chi2_prob_phot_col_non_gaia_3_norm/2) 

    
    chi2_prob_phot_gaia = phot_gaia
    chi2_prob_phot_gaia_3_norm = chi2_prob_phot_gaia - np.nanmin(chi2_prob_phot_gaia)
    prob_phot_gaia = np.exp(-chi2_prob_phot_gaia_3_norm/2) 

    chi2_prob_phot_ast = phot_ast
    chi2_prob_phot_ast_3_norm = chi2_prob_phot_ast - np.nanmin(chi2_prob_phot_ast)
    prob_phot_ast = np.exp(-chi2_prob_phot_ast_3_norm/2) 

    chi2_prob_phot_combined = phot_combined_no_spec + spec
    chi2_prob_phot_combined_3_norm = chi2_prob_phot_combined - np.nanmin(chi2_prob_phot_combined)
    prob_phot_combined = np.exp(-chi2_prob_phot_combined_3_norm/2) 
    
    param_space_reg_all_new = np.vstack((teff,logg,feh,prob_spec,prob_phot_non_gaia,prob_phot_gaia,prob_phot_ast,prob_phot_combined,prob_phot_non_col_gaia)).T
        
    return param_space_reg_all_new

def main_func_multi_obs(inp_index):
    
    start_time = time.time()
    ### photometry, asteroseismology limit input
            
    phot_ast_limits = [Teff_resolution,logg_resolution,feh_resolution]
                                
    ### spectroscopy INITIALISATION ###
        
    # need to develop a system for multiple observations w.r.t. Bayesian scheme
    
    stellar_filename = stellar_names[inp_index].replace(" ","_")
    
    for spec_obs_number in range(len(correlation_arr[inp_index])):
    # for spec_obs_number in range(1):
    # for spec_obs_number in range(len(Input_spec_data[:,0])):
        
        if  spec_obs_number !=3: ### TEMPORARY LINE
            continue

        # if  spec_obs_number !=2: ### TEMPORARY LINE
            # continue
        
        # if spec_obs_number !=0: ### TEMPORARY LINE
                # continue

        # if spec_obs_number ==0: ### TEMPORARY LINE
                
        #         continue

        # print(Input_spec_data[:,2][spec_obs_number])

        # if Input_spec_data[:,2][spec_obs_number] != alphe_1d_red_spec_list[0]: # checking to see if its in the good list
        # if Input_spec_data[:,2][spec_obs_number] not in alphe_1d_red_spec_list: # checking to see if its in the good list
            
            # continue
        
        # if spec_obs_number == 0:
        #     rv_shift_direct = -0.438750027
        # elif spec_obs_number == 1:
        #     rv_shift_direct = -0.458422036
        # elif spec_obs_number == 2:
        #     rv_shift_direct = -0.386287778
        # elif spec_obs_number == 3:
        #     rv_shift_direct = -176.9163404
        # elif spec_obs_number == 4:
        #     rv_shift_direct = -1.101655928

        
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Observation number {spec_obs_number + 1} for star {stellar_names[inp_index]} in spec mode: {mode_kw}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        
        
        
        # spec_type = Input_spec_data[:,1][correlation_arr[inp_index][spec_obs_number]]\
                    # + "_" + str(spec_obs_number)
        # spec_path = Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]        
        spec_path = "../Input_data/spectroscopy_observation_data/PLATO_spectra/PLATO_spec/" + Input_spec_data[:,1][correlation_arr[inp_index][spec_obs_number]]+ "/" +Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]        
        # spec_path = "SAPP_spectroscopy/Payne/hr10_ges_spec_analysis/fit_files_ges_idr6/"+Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]        
        error_map_spec_path = Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
        error_mask_index = inp_index
        nu_max = Input_ast_data[inp_index][1]
        nu_max_err = Input_ast_data[inp_index][2]
        # nu_max = np.nan
        # nu_max_err = np.nan
        numax_input_arr = [float(nu_max),float(nu_max_err),niter_numax_MAX]
        recalc_metals_inp = [spec_fix_values[0],spec_fix_values[1],spec_fix_values[2],feh_recalc_fix_bool]
        # spec_type = "GES_HR10"
        # # spec_type = Input_spec_data[:,1][inp_index]\
        # #             + "_" + str(spec_obs_number)
        # spec_path = Input_spec_path#Input_spec_data[inp_index]
        # error_map_spec_path = Input_spec_path#Input_spec_data[inp_index]
        # error_mask_index = inp_index
        # nu_max = np.nan#Input_ast_data[inp_index][1]
        # nu_max_err = np.nan#Input_ast_data[inp_index][2]
        # numax_input_arr = [float(nu_max),float(nu_max_err),niter_numax_MAX]
        # recalc_metals_inp = [spec_fix_values[0],spec_fix_values[1],spec_fix_values[2],feh_recalc_fix_bool]
        # # logg_fix_load = np.loadtxt(f"../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/seismology_lhood_results/{stellar_filename}_seismic_logg.txt",dtype=str)
        # logg_fix_input_arr = [np.nan,np.nan,np.nan]

        # # spec_type = "GES_HR10"
        # spec_type = Input_spec_data[:,1][correlation_arr[inp_index][spec_obs_number]]
        # spec_path = "../Input_data/spectroscopy_observation_data/AlPhe/" + Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
        # error_map_spec_path = "../Input_data/spectroscopy_observation_data/AlPhe/" + Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
        # error_mask_index = inp_index
        # nu_max = Input_ast_data[inp_index][1]
        # nu_max_err = Input_ast_data[inp_index][2]
        # numax_input_arr = [float(nu_max),float(nu_max_err),niter_numax_MAX]
        # recalc_metals_inp = [spec_fix_values[0],spec_fix_values[1],spec_fix_values[2],feh_recalc_fix_bool]
        # # logg_fix_load = np.loadtxt(f"../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/seismology_lhood_results/{stellar_filename}_seismic_logg.txt",dtype=str)
        # logg_fix_input_arr = [np.nan,np.nan,np.nan]

        # spec_path = "../Input_data/spectroscopy_observation_data/PLATO_spectra/PLATO_spec/" + Input_spec_data[:,1][correlation_arr[inp_index][spec_obs_number]]+ "/" +Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]        
        # error_map_spec_path = "../Input_data/spectroscopy_observation_data/PLATO_spectra/PLATO_spec/" + Input_spec_data[:,1][correlation_arr[inp_index][spec_obs_number]]+ "/" +Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
        # error_mask_index = inp_index
        # nu_max = Input_ast_data[inp_index][1]
        # nu_max_err = Input_ast_data[inp_index][2]
        # numax_input_arr = [float(nu_max),float(nu_max_err),niter_numax_MAX]
        # recalc_metals_inp = [spec_fix_values[0],spec_fix_values[1],spec_fix_values[2],feh_recalc_fix_bool]
        
        
        """
        Testing diff numax modes, right now these do not exist, so presume solar
        """        
        # if spec_obs_number == 1:
        #     spec_path = "blahblah"
        
        spec_init_run = [spec_path,\
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
                    logg_fix_input_arr,\
                    [unique_emask_params_bool,unique_emask_params],\
                    Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]].replace(".txt",""),\
                    rv_shift_direct,\
                    rv_shift_direct_err,\
                    numax_first_step_logg,\
                    extra_save_string] # use to just be stellar names 

        try:
        # if spec_obs_number >= 0: # remember to  put this back!
            
            if MSAP2_01_bool == True:
                
                """
                --------
                MSAP2-01
                --------
                
                Derive asteroseismic Logg using numax and or d_nu if seismic/granular logg isn't available
                
                """
                
                # Input_ast_data[inp_index][4:] = ['nan','nan','nan']
                
                # check for logg input 
                # if logg given whilst logg_ast_bool = True, then it means logg from MSAP3 was given
                
                logg_ast_fix_flag = False
                logg_msap2_01_mode = "none"
                
                if ~np.isnan(logg_fix_input_arr[0]):
                    # then granular/seismic logg is given from MSAP3
                    logg_ast_fix = logg_fix_input_arr[0]
                    logg_ast_fix_upper_err = logg_fix_input_arr[1]
                    logg_ast_fix_lower_err = logg_fix_input_arr[2]
                    
                    logg_msap2_01_mode = "msap4"
                    
                else:
                    # Let SAPP calc seismic logg 
                
                    numax_val = float(Input_ast_data[inp_index][1])
                    numax_val_err = float(Input_ast_data[inp_index][2])
                    dnu_val = float(Input_ast_data[inp_index][4])
                    
                    
                    
                    if np.isnan(numax_val) and np.isnan(dnu_val):
                        # i.e. there is no asteroseismic information and logg from MSAP3 was not given
                        # this section shouldn't have run
                        
                        print("DP3_128_DELTA_NU_AV, DP3_128_NU_MAX from MSAP3 and/or seismic IDP_123_LOGG_VARLC from MSAP4 not given")
                        logg_ast_fix = np.nan
                        logg_ast_fix_upper_err = np.nan
                        logg_ast_fix_lower_err = np.nan
                        
                        logg_ast_fix_flag = True
                        
                    elif ~np.isnan(numax_val) and ~np.isnan(dnu_val):
                        # we can calc ast_PDF
                        
                        # make [Fe/H] and Teff radii really small
                        
                        phot_ast_limits[0] = 100 # K
                        phot_ast_limits[1] = 0.5 # dex
                        phot_ast_limits[2] = 0.1 # dex
                        
                        if phot_ast_central_values_bool == False:
                            
                            ## Teff_SAPP, Logg_SAPP, FeH_SAPP from MSteSci1 would be used here.
                            # for now, solar is picked 
                
                            phot_ast_central_values = [6583,3.99,0.04]
                                    
                        elif phot_ast_central_values_bool == True:
                            
                            phot_ast_central_values = phot_ast_central_values_arr
                
                        stellar_inp_ast = [stellar_names[inp_index],phot_ast_central_values,phot_ast_limits,star_field_names[inp_index],spec_obs_number,save_ast_space_bool,Input_ast_data[inp_index],extra_save_string]
                                                            
                        start_ast = time.time()    
                        ast_param_space,ast_flag = ast_pdf(stellar_inp_ast)
                        print(f"Asteroseismology calculation for star {stellar_names[inp_index]} time elapsed --- {(time.time()-start_ast)/60} minutes --- ")    
                        
                        prob_ast_pdf = ast_param_space[:,3]
                        logg_ast_pdf = ast_param_space[:,1]
                                                
                        if ~ast_flag:
                            logg_ast_fix = np.average(logg_ast_pdf,weights=prob_ast_pdf)
                            logg_ast_fix_upper_err = np.sqrt(np.average((logg_ast_pdf-np.average(logg_ast_pdf,weights=prob_ast_pdf))**2,weights=prob_ast_pdf))
                            logg_ast_fix_lower_err = np.sqrt(np.average((logg_ast_pdf-np.average(logg_ast_pdf,weights=prob_ast_pdf))**2,weights=prob_ast_pdf))
                            logg_msap2_01_mode = "ast_pdf"
                                                        
                        else:
                            print("Ast PDF space empty, IDP_122_LOGG_SAPP_TMP set to NaN")
                            logg_ast_fix = np.nan
                            logg_ast_fix_upper_err = np.nan
                            logg_ast_fix_lower_err = np.nan
                            logg_ast_fix_flag = True
                            
                    elif ~np.isnan(numax_val) and np.isnan(dnu_val):
                        
                        # i.e. numax exists but delta nu does not
                        # here we do start of iterative method
                        
                        # we would use Teff_SAPP here from MSteSci1
                        # for now, setting to solar with arbitrary error
                    
                        logg_numax_arr = logg_numax(numax_val,numax_val_err,6583,47)
                        logg_ast_fix = logg_numax_arr[0]
                        logg_ast_fix_lower_err = logg_numax_arr[1]
                        logg_ast_fix_upper_err = logg_numax_arr[2]
                        logg_msap2_01_mode = "numax_iter"
                        
                    else: # this would be if only delta nu existed,wihtout numax? unlikely
                        print("Only DP3_128_DELTA_NU_AV available, IDP_122_LOGG_SAPP_TMP set to NaN")
                        logg_ast_fix = np.nan
                        logg_ast_fix_upper_err = np.nan
                        logg_ast_fix_lower_err = np.nan
                        logg_ast_fix_flag = True
                
                if ~logg_ast_fix_flag:
                    
                    logg_ast_fix_save = np.array([logg_msap2_01_mode,logg_ast_fix,logg_ast_fix_lower_err,logg_ast_fix_upper_err],dtype=str)
                    
                    print("IDP_122_LOGG_SAPP_TMP",logg_ast_fix_save)
                    
                    stellar_filename = stellar_names[inp_index].replace(" ","_")
                            
                    directory_asteroseismology = star_field_names[inp_index] # name of directory is the name of the star
                    directory_check = os.path.exists(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology}")                    
                    if  directory_check == True:                    
                        print(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology} directory exists")                        
                    else:                        
                        print(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology} directory does not exist")                        
                        os.makedirs(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology}")                        
                        print(f"../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology} directory has been created")
                        
                    np.savetxt(f'../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology}/MSAP2_01_logg_fix_{logg_msap2_01_mode}_{stellar_filename}_{extra_save_string}.txt',logg_ast_fix_save,fmt='%s',delimiter=",",\
                               header='Logg_ast_fix_mode, IDP_122_LOGG_SAPP_TMP/dex, IDP_122_LOGG_SAPP_TMP_lower_err/dex, IDP_122_LOGG_SAPP_TMP_upper_err/dex')

            # print(best_spec_bool)

            if best_spec_bool == True:
                    
                # if spec_obs_number + 1 != 1:
                    
                #     continue
            
                if MSAP2_02_bool:
                    """
                    MSAP2-02 is just the spectroscopic module specialsed 
                    It is run differently depending on how MSAP2-01 was ran
                    If MSAP2-01 didn't run, this module will not run
                    """
                    
                    stellar_filename = stellar_names[inp_index].replace(" ","_")
                    directory_asteroseismology = star_field_names[inp_index] # name of directory is the name of the star                    
                    
                    try:
                        logg_ast_fix_load = np.loadtxt(f'../Output_data/Stars_Lhood_asteroseismology/{directory_asteroseismology}/MSAP2_01_logg_fix_{MSAP2_01_test_case_kw}_{stellar_filename}_{extra_save_string}.txt',dtype=str,delimiter=",")
                    except:
                        raise FileNotFoundError("Could not load IDP_122_LOGG_SAPP_TMP from MSAP2-01")
                        
                    logg_ast_fix_mode = logg_ast_fix_load[0]
                    
                    
                    if logg_ast_fix_mode == "msap4" or logg_ast_fix_mode == "ast_pdf":
                                                
                        # fix logg mode with this logg
                        spec_init_run[9] = False
                        spec_init_run[13] = True
                        spec_init_run[14] = [np.array(logg_ast_fix_load[1],dtype=float),\
                                             np.array(logg_ast_fix_load[2],dtype=float),\
                                             np.array(logg_ast_fix_load[3],dtype=float)]
                        
                        best_fit_spectroscopy = mspec_new.find_best_val(spec_init_run)
                        
                        
                    elif logg_ast_fix_mode =="numax_iter":
                        # numax iter method, but use this logg as the first step somehow
                        # this is a very very very obtuse mode in numax
                        # numax_first_step_logg = 
                        spec_init_run[19] = [True,\
                                             np.array(logg_ast_fix_load[1],dtype=float),\
                                             np.array(logg_ast_fix_load[2],dtype=float),\
                                             np.array(logg_ast_fix_load[3],dtype=float)]
                        spec_init_run[9] = True
                        spec_init_run[13] = False
                        
                        best_fit_spectroscopy = mspec_new.find_best_val(spec_init_run)
                    
                    else:
                        # i.e logg_ast_fix_mode =="none"
                        # if none, then that means MSAP2-01 failed somewhere
                        raise ValueError("IDP_122_LOGG_SAPP_TMP from MSAP2-01 failed at some point. Re-do and check MSAP2-01 step")
                
                else:
                    
                    # if MSAP2_02 bool false, then run normal spec by default
   
                    best_fit_spectroscopy = mspec_new.find_best_val(spec_init_run)
                
                
                # print("check results",best_fit_spectroscopy)
                                
                best_spec_params = best_fit_spectroscopy[0]   
                best_spec_params_err = best_fit_spectroscopy[1]   
                
                cov_matrix = best_fit_spectroscopy[11]   
                
                cov_matrix_new = cov_matrix[1:,1:]
                            
                corr_matrix = correlation_from_covariance(cov_matrix_new) # correlation matrix
                
                wvl_corrected = best_fit_spectroscopy[5]
                
                obs_norm = best_fit_spectroscopy[6]
                
                usert_norm = best_fit_spectroscopy[10]
                
                fit_norm = best_fit_spectroscopy[7]
                
                wvl_obs_input = best_fit_spectroscopy[9]
                
                spectra_save = np.vstack((wvl_corrected,obs_norm,usert_norm,fit_norm)).T
                
                rv_shift_spec = best_fit_spectroscopy[3]
                
                """
                Plot correlation matrix 
                """
                
                '''
                
                ### finding the lower triangle ###
                
                low_tri_bool = np.tril(np.ones(np.shape(corr_matrix))).astype(np.bool)[0:8,0:8]
                
                # print(low_tri_bool)
                
                spectral_parameters_names = ["T$_{eff}$","log(g)","[Fe/H]","Vmic","Vbrd","[Mg/Fe]","[Ti/Fe]","[Mn/Fe]"]
                
                for i in range(len(spectral_parameters_names)):
                    for j in range(len(spectral_parameters_names)):
                                            
                        if low_tri_bool[i][j] == False:
                            
                            corr_matrix[i,j] = np.nan
    
                            
                fig, ax = plt.subplots()
                im = ax.imshow(corr_matrix, vmin=-1, vmax=1)
                
                # We want to show all ticks...
                ax.set_xticks(np.arange(len(spectral_parameters_names)))
                ax.set_yticks(np.arange(len(spectral_parameters_names)))
                # ... and label them with the respective list entries
                ax.set_xticklabels(spectral_parameters_names)
                ax.set_yticklabels(spectral_parameters_names)
                
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor") 
                
                for i in range(len(spectral_parameters_names)):
                    for j in range(len(spectral_parameters_names)):
                        
                        if np.isnan(corr_matrix[i, j]) == True:
                            
                            continue
                        
                        else:
                            
                            text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                                   ha="center", va="center", color="k",fontsize=16)
                            
                # ax.set_title("$\delta$ Eri: Correlation Coefficient Table",fontsize=20)
                ax.set_title("HD49933: Correlation Coefficient Table",fontsize=20)
                            
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                        
                cbar = ax.figure.colorbar(im, ax=ax,label="p",cax=cax)
                cbar.ax.tick_params(labelsize=16)
    
                plt.setp(ax.spines.values(), linewidth=0)
                
            
                fig.tight_layout()
                plt.show()
                '''
                                        
                N_pixels = len(best_fit_spectroscopy[6])
                N_params = len(best_spec_params)
                spec_dof = abs(N_pixels - N_params) # degrees of freedom
    
                stellar_filename = stellar_names[inp_index].replace(" ","_")
        
                # directory_spectroscopy = stellar_filename # name of directory is the name of the star
                
                directory_spectroscopy = star_field_names[inp_index] # name of directory is the name of the star
                directory_check = os.path.exists(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")
                
                if  directory_check == True:
                
                    print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory exists")
                    
                else:
                    
                    print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory does not exist")
                    
                    os.makedirs(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")
                    
                    print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory has been created")
                        
                ## 4MOST abundances are [X/H] rather than [X/Fe]
                # changes in the NN need to be changed more easily
            
                spec_abundances = best_spec_params[3:]
                
                # convert abundances to [X/Fe] = [X/H] - [Fe/H]
                spec_abundances = spec_abundances - best_spec_params[2]
            
                # best_spec_params = np.hstack((best_spec_params,spec_dof,rv_shift_spec)) # HR10
                
                # convert errors -> d[X/Fe] = (abs(-feh)**2 * dxh**2 + abs(xh)**2 * dfeh**2) ** 0.5
                spec_abundances_errs = (abs(best_spec_params[2]) ** 2 * best_spec_params_err[2] ** 2 +\
                                       abs(best_spec_params[3:]) ** 2 * best_spec_params_err[3:] ** 2) ** 0.5 
                best_spec_params = np.hstack((best_spec_params[:3],spec_abundances,spec_dof,rv_shift_spec))    
                best_spec_params_err = np.hstack((best_spec_params_err[:3],spec_abundances_errs))    
                        
                ### Multiobs save
    
        
                ''
                
                if MSAP2_02_bool:
                    np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',best_spec_params,\
                               header="IDP_122_TEFF_SPECTROSCOPY/K, IDP_122_LOGG_SPECTROSCOPY/dex, [M/H]/dex, [K/Fe]/dex, [Mg/Fe]/dex, [Ba/Fe]/dex, [Al/Fe]/dex, [Ti/Fe]/dex, [Na/Fe]/dex, [Ca/Fe]/dex, [Ni/Fe]/dex, [Cu/Fe]/dex, [Co/Fe]/dex, [Cr/Fe]/dex, [Nd/Fe]/dex, [O/Fe]/dex, [Y/Fe]/dex, [Eu/Fe]/dex, [V/Fe]/dex, [Mo/Fe]/dex, [Mn/Fe]/dex, [Sc/Fe]/dex, [Si/Fe]/dex, [La/Fe]/dex, [Zn/Fe]/dex, [Sr/Fe]/dex, [Rb/Fe]/dex, [Zr/Fe]/dex, [Li/Fe]/dex, [Sm/Fe]/dex, [Ru/Fe]/dex, [Ce/Fe]/dex, [C/Fe]/dex, degrees_of_freedom, RV/Km/s")
                    np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',best_spec_params_err,\
                               header="IDP_122_TEFF_SPECTROSCOPY/K, IDP_122_LOGG_SPECTROSCOPY/dex, [M/H]/dex, [K/Fe]/dex, [Mg/Fe]/dex, [Ba/Fe]/dex, [Al/Fe]/dex, [Ti/Fe]/dex, [Na/Fe]/dex, [Ca/Fe]/dex, [Ni/Fe]/dex, [Cu/Fe]/dex, [Co/Fe]/dex, [Cr/Fe]/dex, [Nd/Fe]/dex, [O/Fe]/dex, [Y/Fe]/dex, [Eu/Fe]/dex, [V/Fe]/dex, [Mo/Fe]/dex, [Mn/Fe]/dex, [Sc/Fe]/dex, [Si/Fe]/dex, [La/Fe]/dex, [Zn/Fe]/dex, [Sr/Fe]/dex, [Rb/Fe]/dex, [Zr/Fe]/dex, [Li/Fe]/dex, [Sm/Fe]/dex, [Ru/Fe]/dex, [Ce/Fe]/dex, [C/Fe]/dex")

                    np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',cov_matrix)
    
                    if spectra_save_bool:
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_save_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',spectra_save,header="wavelength,flux,error")
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_wvl_obs_input_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',wvl_obs_input,header="wavelength,flux,error")
                else:
                    
                    if np.isnan(float(nu_max)):
                    
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',best_spec_params)
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',best_spec_params_err)    
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',cov_matrix)
        
                        if spectra_save_bool:
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_save_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',spectra_save,header="wavelength,flux,error")
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_wvl_obs_input_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt',wvl_obs_input,header="wavelength,flux,error")
                    else:
                        
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',best_spec_params)
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',best_spec_params_err)    
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',cov_matrix)
        
                        if spectra_save_bool:
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_save_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',spectra_save,header="wavelength,flux,error")
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_wvl_obs_input_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt',wvl_obs_input,header="wavelength,flux,error")
                ''
                # pid = os.getpid()
                # py = psutil.Process(pid)
                # memoryUse = py.memory_info()[0]/2.**30
                
                # print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
            ### SPEC BEST VALUES MULTIOBS ###
            
            # These have assumed to be calculated IF the first spec mode has ran
            
            # if spec_obs_number + 1 >= 10: # temporary 
                # if spec_obs_number >= 13: # temporary 
                    
                    # continue
                
                
            if phot_ast_track_bool or spec_track_bool or bayes_scheme_bool or recalc_metals_scheme_bool:
            
                directory_spectroscopy = star_field_names[inp_index] # name of directory is the name of the star
                directory_check = os.path.exists(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")
        
                if np.isnan(float(nu_max)):
        
                    best_spec_params = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace('_numax','')}{extra_save_string}.txt")
                    best_spec_params_err = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace('_numax','')}{extra_save_string}.txt")
        
                else:
        
                    best_spec_params = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt")
                    best_spec_params_err = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt")


            if MSAP2_03_bool:
               """
               MSAP2-03 combines information from MSteSci1 and MSAP2 to derive a final set of SAPP parameters, this includes IRFM, SBCR, Interferometry etc
               
               For now the main algorithm is TBA due to the ocmplexity of shared inputs into a Bayesian scheme. 
               
               Thus, the temporary solution is simply a combination of results from MSAP2-02 and Teff_phot from MSteSci1. For testing we are choosing procyons parameters. 
               
               If MSAP2-03 is being run, then the extra_save_string exists specifically with _MSAP2 inside it. So, the spectroscopy loaded here by definition IS the one run from MSAP2-02.
               """
               directory_spectroscopy = star_field_names[inp_index] # name of directory is the name of the star
               directory_check = os.path.exists(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")

               
               best_spec_params = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw.replace('_numax','')}{extra_save_string}.txt")
               best_spec_params_err = np.loadtxt(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw.replace('_numax','')}{extra_save_string}.txt")

               teff_MSAP2 = best_spec_params[0]
               teff_MSAP2_err = best_spec_params_err[0]
               logg_MSAP2 = best_spec_params[1]
               logg_MSAP2_err = best_spec_params_err[1]
               teff_phot = 6583
               teff_phot_err = 47
               
               teff_range = np.linspace(6000,7000,1000)
               
               L_teff_phot  = np.exp(-(teff_phot-teff_range)**2/(2*(teff_phot_err)**2))
               L_teff_spec = np.exp(-(teff_MSAP2-teff_range)**2/(2*(teff_MSAP2_err)**2))
               
               L_teff_MSAP3 = L_teff_phot * L_teff_spec
               
               teff_MSAP3 = np.average(teff_range,weights=L_teff_MSAP3)
               teff_MSAP3_err = np.sqrt(np.average((teff_range-np.average(teff_range,weights=L_teff_MSAP3))**2,weights=L_teff_MSAP3))
               
               # now we have a combined teff from photometry and the new spectroscopy
               # we have the new logg from spectroscopy
               # last step is to re-calculate abundances
               
               # make sure both logg fix and numax are false, and that the first step numax wouldn't activate
               # now, spectroscopy should be 'back' to normal
               
               spec_init_run[9] = False
               spec_init_run[13] = False
               spec_init_run[14] = [np.nan,np.nan,np.nan]
               spec_init_run[19] = [False,[logg_MSAP2,logg_MSAP2_err,logg_MSAP2_err]]
               spec_init_run[11] = True # this is recalc metals
               spec_init_run[12] = [teff_MSAP3,logg_MSAP2,np.nan,False] # letting [Fe/H] be re-calculated
               
               spec_init_run[17] = best_spec_params[-1]
               spec_init_run[18] = 0.05
                
                # still keep error masks however right?
                
                # need to pass on prev. RV information to save time... 
                
               extra_save_string_new = extra_save_string + "_recalc_met"
                                
               spec_init_run[20] = extra_save_string_new
                
                # print("spec_path",spec_path)
                
               best_fit_spectroscopy_recalc = mspec_new.find_best_val(spec_init_run)
               best_spec_params_recalc = best_fit_spectroscopy_recalc[0]   
               best_spec_params_recalc_err = best_fit_spectroscopy_recalc[1]  
               
               # add error from teff and logg in this re-calc
               best_spec_params_recalc_err[0] = np.sqrt(best_spec_params_recalc_err[0]**2 + teff_MSAP3_err**2)
               best_spec_params_recalc_err[1] = np.sqrt(best_spec_params_recalc_err[1]**2 + logg_MSAP2_err**2)
               
               cov_matrix_recalc = best_fit_spectroscopy_recalc[11] # how would one convert the cov matrix values?        
               rv_shift_spec = best_fit_spectroscopy_recalc[3]
               N_pixels = len(best_fit_spectroscopy_recalc[6])
               N_params = len(best_spec_params_recalc)
               spec_dof = abs(N_pixels - (N_params-2)) # degrees of freedom                        

               ## metals have been re-calculated with new results.
               
               # for now, save these results in spectroscopy, they are the same as spec, but now they are new
               # this immediately doubles the results.
               
               spec_abundances = best_spec_params_recalc[3:]
                
                # convert abundances to [X/Fe] = [X/H] - [Fe/H]
               spec_abundances = spec_abundances - best_spec_params_recalc[2]
            
                # best_spec_params = np.hstack((best_spec_params,spec_dof,rv_shift_spec)) # HR10
                
                # convert errors -> d[X/Fe] = (abs(-feh)**2 * dxh**2 + abs(xh)**2 * dfeh**2) ** 0.5
               spec_abundances_errs = (abs(best_spec_params_recalc[2]) ** 2 * best_spec_params_recalc_err[2] ** 2 +\
                                       abs(best_spec_params_recalc[3:]) ** 2 * best_spec_params_recalc_err[3:] ** 2) ** 0.5 
               best_spec_params = np.hstack((best_spec_params_recalc[:3],spec_abundances,spec_dof,rv_shift_spec))    
               best_spec_params_err = np.hstack((best_spec_params_recalc_err[:3],spec_abundances_errs))    
                               
               directory_spectroscopy = star_field_names[inp_index] # name of directory is the name of the star
               directory_check = os.path.exists(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")                
               if  directory_check == True:                
                    print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory exists")                    
               else:                    
                    print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory does not exist")                    
                    os.makedirs(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")                   
                    print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory has been created")



               
               np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw}{extra_save_string_new}.txt',best_spec_params_recalc,\
                      header="IDP_122_TEFF_SAPP/K, IDP_122_LOGG_SAPP/dex, [M/H]_SAPP/dex, [K/Fe]/dex, [Mg/Fe]/dex, [Ba/Fe]/dex, [Al/Fe]/dex, [Ti/Fe]/dex, [Na/Fe]/dex, [Ca/Fe]/dex, [Ni/Fe]/dex, [Cu/Fe]/dex, [Co/Fe]/dex, [Cr/Fe]/dex, [Nd/Fe]/dex, [O/Fe]/dex, [Y/Fe]/dex, [Eu/Fe]/dex, [V/Fe]/dex, [Mo/Fe]/dex, [Mn/Fe]/dex, [Sc/Fe]/dex, [Si/Fe]/dex, [La/Fe]/dex, [Zn/Fe]/dex, [Sr/Fe]/dex, [Rb/Fe]/dex, [Zr/Fe]/dex, [Li/Fe]/dex, [Sm/Fe]/dex, [Ru/Fe]/dex, [Ce/Fe]/dex, [C/Fe]/dex, degrees_of_freedom, RV/Km/s")
               np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw}{extra_save_string_new}.txt',best_spec_params_recalc_err,\
                      header="IDP_122_TEFF_SAPP/K, IDP_122_LOGG_SAPP/dex, [M/H]_SAPP/dex, [K/Fe]/dex, [Mg/Fe]/dex, [Ba/Fe]/dex, [Al/Fe]/dex, [Ti/Fe]/dex, [Na/Fe]/dex, [Ca/Fe]/dex, [Ni/Fe]/dex, [Cu/Fe]/dex, [Co/Fe]/dex, [Cr/Fe]/dex, [Nd/Fe]/dex, [O/Fe]/dex, [Y/Fe]/dex, [Eu/Fe]/dex, [V/Fe]/dex, [Mo/Fe]/dex, [Mn/Fe]/dex, [Sc/Fe]/dex, [Si/Fe]/dex, [La/Fe]/dex, [Zn/Fe]/dex, [Sr/Fe]/dex, [Rb/Fe]/dex, [Zr/Fe]/dex, [Li/Fe]/dex, [Sm/Fe]/dex, [Ru/Fe]/dex, [Ce/Fe]/dex, [C/Fe]/dex")
               np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{MSAP2_01_test_case_kw}_{mode_kw}{extra_save_string_new}.txt',cov_matrix_recalc,\
                      header='IDP_122_COVMAT_SAPP')

            ### photometry input    
            
            if phot_ast_track_bool == True:
            
                if phot_ast_central_values_bool == False:
                
                    # phot_ast_central_values = [best_spec_params[0]*1000,best_spec_params[1],best_spec_params[2]]
                    phot_ast_central_values = [best_spec_params[0],best_spec_params[1],best_spec_params[2]]
                                    
                elif phot_ast_central_values_bool == True:
                    
                    phot_ast_central_values = phot_ast_central_values_arr#[phot_ast_central_values_arr[0],phot_ast_central_values_arr[1],phot_ast_central_values_arr[2]]
        
                stellar_inp_phot = [stellar_names[inp_index],phot_ast_central_values,phot_ast_limits,star_field_names[inp_index],spec_obs_number,phot_magnitude_set,save_phot_space_bool,Input_phot_data[inp_index],Input_ast_data[inp_index],extra_save_string]
                        
                # if spec_obs_number == 0: # only need to do photometry once, multiple spec obs likely close enough, no need to repeat 
                
                    # start_time_phot = time.time()
                    
                start_phot = time.time()    
                phot_param_space,phot_flag = photometry_ast_pdf(stellar_inp_phot)
                print(f"Photometry and Asteroseismology calculation for star {stellar_names[inp_index]} time elapsed --- {(time.time()-start_phot)/60} minutes --- ")    
            
                # print(f"photometry time = {time.time()-start_time_phot}")
            
                # if spec_obs_number == 0:
                    
                    # this is because there is no point creating a photometry space for multiple spectra
                    # the spaces will be very similar in extent
                    # the only time this fails is if we have a bad spectra
                    # so in blind mode this makes sense as it can be very inconsistent
                    # but because the search range in photometry space is so large
                    # just create it once for the first observation
                    # you only need one set of central limits decided by spectroscopy.
                    
                if phot_flag == False:
                    
                    # spec_track_bool = False
                    # bayes_scheme_bool = False
                    
                    print("Photometry failed due to empty parameter space")
                    print("-- due to central Teff, Logg, [Fe/H] values not existing in evo tracks")
                    
                    # for now, just fail it entirely
                    
                    # how do we save the flag?
                    
                    # because right now it means it'll just be missing from the final folder
                    
                    # np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_{stellar_filename}{spec_extension}{extra_save_string}{cov_save_string}.txt",best_params_combined,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")
                    # np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_err_{stellar_filename}{spec_extension}{extra_save_string}{cov_save_string}.txt",best_params_combined_err,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")
                    
                    continue
                    
                    
                    
                    
                    # break                                                                                      
    
            if spec_track_bool == True:
    
                if np.isnan(float(nu_max)):
                
                    cov_matrix = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt')
    
                    if spectra_save_bool:
                        spectra_save = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_save_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt')
                        obs_norm = spectra_save[:,1]
                        usert_norm = spectra_save[:,2]                
                        wvl_obs = spectra_save[:,0]
                        wvl_obs_input = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_wvl_obs_input_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string}.txt')
                    else:
                        obs_norm = []
                        usert_norm = []
                        wvl_obs = []
                        wvl_obs_input = []
    
                else:
                    
                    cov_matrix = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt')
    
                
                    if spectra_save_bool:
                        spectra_save = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_save_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt')
                        obs_norm = spectra_save[:,1]
                        usert_norm = spectra_save[:,2]     
                        wvl_obs = spectra_save[:,0]
                        wvl_obs_input = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}/spectra_wvl_obs_input_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string}.txt')
                    else:
                        obs_norm = []
                        usert_norm = []
                        wvl_obs = []
                        wvl_obs_input = []
    
                # best_fit_spectroscopy_input = [best_spec_params,best_spec_params_err,cov_matrix]
                best_fit_spectroscopy_input = [best_spec_params,best_spec_params_err,cov_matrix,obs_norm,usert_norm,wvl_obs,wvl_obs_input,cheb]
     
                spec_extension = f"_OBS_NUM_{spec_obs_number+1}_SPEC_MODE_{mode_kw}"
                stellar_inp_spec = [inp_index,best_fit_spectroscopy_input,star_field_names[inp_index],spec_extension,save_spec_phot_space_bool]
        
                start_spec = time.time()
                
                if save_phot_space_bool == True:
                    
                    phot_param_space = np.loadtxt(f'../Output_data/Stars_Lhood_photometry/{directory_spectroscopy}/stellar_track_collection_{stellar_filename}_OBS_NUM_{spec_obs_number + 1}_test_4_{extra_save_string}.txt',delimiter=",") 
                
                spec_phot_param_space = spectroscopy_stellar_track_collect(stellar_inp_spec,phot_param_space)
                
                print(f"Spectroscopy calculation for star {stellar_names[inp_index]} time elapsed --- {(time.time()-start_spec)/60} minutes --- ")
        
        
            if bayes_scheme_bool == True:
                        
                ### load parameter space i.e. photometry, asteroseismology, spectroscopy, chi2 values
                
                if np.isnan(float(nu_max)):
    
                    spec_extension = f"_OBS_NUM_{spec_obs_number+1}_SPEC_MODE_{mode_kw.replace('_numax','')}"
                    
                else:
                    
                    spec_extension = f"_OBS_NUM_{spec_obs_number+1}_SPEC_MODE_{mode_kw}"
                    
                
                directory_photometry = star_field_names[inp_index]
                
                if save_spec_phot_space_bool == True:
                    
                    spec_phot_param_space = np.loadtxt(f'../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_photometry}/stellar_track_collection_w_spec_and_prob_{stellar_filename}_test_4{spec_extension}{extra_save_string}.txt',delimiter=",") 
    
                param_space = spec_phot_param_space
    
                teff  = param_space[:,0]
                logg = param_space[:,1]
                feh = param_space[:,2]
                # feh_spec = feh - (0.22/0.4 * (best_spec_params[5]-MgFe_sys[0])) # HR10 NN
                feh_spec = feh - (0.22/0.4 * (best_spec_params[4]-MgFe_sys[0])) # 4MOST NN
                L_prob_mag = param_space[:,3]
                L_prob_ast = param_space[:,4]
                age = param_space[:,5]
                mass = param_space[:,6]
                radius = param_space[:,7]
                age_step = param_space[:,8]
                Luminosity = param_space[:,9]
                L_prob_spec_2 = param_space[:,10] # without covariance
                L_prob_spec_3 = param_space[:,11] # with covariance
                
                prob_mag_comb = L_prob_mag
                
                if spec_covariance_bool == True:
                    prob_spec_use = L_prob_spec_3                
                else:                
                    prob_spec_use = L_prob_spec_2    
                
                # prob_ast = np.ones([len(prob_spec_use)])#L_prob_ast
                prob_ast = L_prob_ast
                
                IMF_prior = mass ** (-2.35)
                Age_prior = age_step
                
                prior_PDF = IMF_prior * Age_prior
                
    
                prob_input = prob_spec_use * prob_mag_comb * prior_PDF * prob_ast                                   
                # prob_input = prob_spec_use * prob_ast * prob_mag_comb * prior_PDF
                prob_input_name = "prob_comb_3_w_prob_priors"
    
                # print("PROB_COMB_MAX",max(prob_input))
                # print("PROB_SPEC_MAX",max(prob_spec_use))
                # print("PROB_PHOT_MAX",max(prob_mag_comb))
                # print("PROB_PRIOR_MAX",max(prior_PDF))
    
                #saving complete PDF folder
                
                if save_final_space_bool:
                
                    PDF_tot_save = np.vstack((teff,\
                                            logg,\
                                            feh,\
                                            feh_spec,\
                                            L_prob_mag,\
                                            L_prob_ast,\
                                            age ,\
                                            mass ,\
                                            radius ,\
                                            age_step ,\
                                            Luminosity ,\
                                            L_prob_spec_2,\
                                            L_prob_spec_3,\
                                            prior_PDF)).T
                    
                    np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_photometry}/{stellar_filename}_OBS_NUM_{spec_obs_number + 1}_PDF_tot_test_{extra_save_string}.txt",PDF_tot_save,delimiter=",",fmt="%.5e",\
                                    header="Teff, Logg, feh, feh_spec, L_prob_phot, L_prob_ast, age/Myr, mass, radius, age_step/Myr, luminosity, chi2_prob_spec_no_cov, chi2_prob_spec_cov, prior_PDF")
                
                            
                savefig_bool = False
                makefig_bool = False 
                            
                ### bins for distributions
                
                # teff_binwidth_prob_input = 25 # K
                # logg_binwidth_prob_input = 0.025 # dex
                # feh_binwidth_prob_input = 0.05 # dex
                # feh_spec_binwidth_prob_input = 0.05 # dex
                # age_binwidth_prob_input = 0.5#0.25 # Gyrs
                # mass_binwidth_prob_input = 0.02 # Msol
                # radius_binwidth_prob_input = 0.05 # Rsol
                # lumin_binwidth_prob_input = 0.15 # Lsol
    
                teff_binwidth_prob_input = 25 # K
                logg_binwidth_prob_input = 0.01 # dex
                feh_binwidth_prob_input = 0.05 # dex
                feh_spec_binwidth_prob_input = 0.05 # dex
                age_binwidth_prob_input = 0.25#0.25 # Gyrs
                mass_binwidth_prob_input = 0.02 # Msol
                radius_binwidth_prob_input = 0.025 # Rsol
                lumin_binwidth_prob_input = 0.25 # Lsol
                
                ### probability array inputs to graph distributions
    
                # prob_input_name_arr = ["Spec"]
                # prob_input_arr = [prob_spec_use]
                # prob_col_arr = ["tomato"]
    
            
                # prob_input_name_arr = ["Spectroscopy","Photometry+Parallax","ast"]
                # prob_input_arr = [prob_spec_use,prob_mag_comb,prob_ast]
                # prob_col_arr = ["deepskyblue","tomato","k","g"]
                
                # prob_input_name_arr = ["Combined_no_Spec","Combined","Spectroscopy"]
                # prob_input_arr = [prob_input_without_spec,prob_input,prob_spec_use]
                # prob_col_arr = ["deepskyblue","tomato","k"]
                
                prob_input_name_arr = ["Photometry+Parallax","Combined","Spectroscopy"]
                prob_input_arr = [prob_mag_comb,prob_input,prob_spec_use]
                prob_col_arr = ["k","deepskyblue","tomato"]
                
                # prob_input_name_arr_phillip = ["Spectroscopy (_so)","Photometry+Parallax","final (_wa)","Prior"]
                # prob_input_arr_phillip = [prob_spec_use,prob_mag_comb,prob_input,prior_PDF]
                prob_input_name_arr_phillip = ["Spectroscopy (_so)","Photometry Bp-Rp","final (_wa)","Ast"]
                prob_input_arr_phillip = [prob_spec_use,prob_mag_comb,prob_input,prob_ast]
                prob_col_arr_phillip = ["deepskyblue","tomato","k","g"]
                path_phillip = f"../Output_figures/1d_distributions_no_histogram/{stellar_filename}.png"
                param_binwidth_arr = [teff_binwidth_prob_input,\
                                      logg_binwidth_prob_input,\
                                      feh_spec_binwidth_prob_input,\
                                      age_binwidth_prob_input,\
                                      mass_binwidth_prob_input,\
                                      radius_binwidth_prob_input,\
                                      lumin_binwidth_prob_input]                
                if makefig_bool:
                    
                    plot_pdf_all(prob_input_arr_phillip,prob_input_name_arr_phillip,prob_col_arr_phillip,teff,logg,feh_spec,age,mass,radius,Luminosity,path_phillip,param_binwidth_arr,directory_photometry)
     
                # Log_multiple_prob_distributions_1D(teff,prob_input_arr,"T$_{eff}$ [K]",teff_binwidth_prob_input,prob_input_name_arr,"temperature",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")#,ref_param = float(Ref_data[:,1][inp_index]),ref_param_err = float(Ref_data[:,2][inp_index])) # 50
                # Log_multiple_prob_distributions_1D(logg,prob_input_arr,"Logg [dex]",logg_binwidth_prob_input,prob_input_name_arr,"logg",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")#,ref_param = float(Ref_data[:,3][inp_index]),ref_param_err = float(Ref_data[:,4][inp_index])) # 0.005
                # Log_multiple_prob_distributions_1D(feh,prob_input_arr,"[Fe/H] [dex]",feh_binwidth_prob_input,prob_input_name_arr,"feh",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")#,ref_param = float(Ref_data[:,5][inp_index]),ref_param_err = float(Ref_data[:,6][inp_index])) # 0.05
                # Log_multiple_prob_distributions_1D(feh_spec,prob_input_arr,"[Fe/H] spec [dex]",feh_spec_binwidth_prob_input,prob_input_name_arr,"feh_spec",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")#,ref_param = float(Ref_data[:,5][inp_index]),ref_param_err = float(Ref_data[:,6][inp_index]))    
                # Log_multiple_prob_distributions_1D(age,prob_input_arr,"Age [Gyrs]",age_binwidth_prob_input,prob_input_name_arr,"age",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison") # 0.5
                # Log_multiple_prob_distributions_1D(mass,prob_input_arr,"Mass [M$_\odot$]",mass_binwidth_prob_input,prob_input_name_arr,"mass",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison") # 0.02
                # Log_multiple_prob_distributions_1D(radius,prob_input_arr,"Radius [R$_\odot$]",radius_binwidth_prob_input,prob_input_name_arr,"radius",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")
                # Log_multiple_prob_distributions_1D(Luminosity,prob_input_arr,"Luminosity [L$_\odot$]",lumin_binwidth_prob_input,prob_input_name_arr,"lumin",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,col_arr = prob_col_arr,extra_name = "spec_comparison")
    
            
                ### BEST FIT PARAMETER ESTIMATION via fitting Log(Gaussian) of PDF from prob_input
                ''
                                        
                # teff_fit = Log_prob_distributions_1D(teff,prob_input,"T$_{eff}$ [K]",teff_binwidth_prob_input,prob_input_name,"temperature",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 50
                # logg_fit = Log_prob_distributions_1D(logg,prob_input,"Logg [dex]",logg_binwidth_prob_input,prob_input_name,"logg",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.005
                # feh_fit = Log_prob_distributions_1D(feh,prob_input,"[Fe/H] [dex]",feh_binwidth_prob_input,prob_input_name,"feh",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.05
                # feh_spec_fit = Log_prob_distributions_1D(feh_spec,prob_input,"[Fe/H] spec [dex]",feh_spec_binwidth_prob_input,prob_input_name,"feh_spec",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension)    
                # age_fit = Log_prob_distributions_1D(age,prob_input,"Age [Gyrs]",age_binwidth_prob_input,prob_input_name,"age",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.5
                # mass_fit = Log_prob_distributions_1D(mass,prob_input,"Mass [M$_\odot$]",mass_binwidth_prob_input,prob_input_name,"mass",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension) # 0.02
                # radius_fit = Log_prob_distributions_1D(radius,prob_input,"Radius [R$_\odot$]",radius_binwidth_prob_input,prob_input_name,"radius",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension)
                # luminosity_fit = Log_prob_distributions_1D(Luminosity,prob_input,"Luminosity [L$_\odot$]",lumin_binwidth_prob_input,prob_input_name,"lumin",makefig_bool=makefig_bool,savefig_bool=savefig_bool,stellar_id = stellar_filename,chisq_red_bool=chi_2_red_bool,savefig_extra_name = spec_extension)
                            
                # print(teff_fit)
                # print(logg_fit)
                # print(feh_spec_fit)
                
                # print("prob input",max(prob_input),"prob_spec",max(prob_spec_use),"prob_phot",max(prob_mag_comb),"prior",max(prior_PDF),"ast",max(prob_ast))
                
                
                if max(prob_input) == 0:
                                    
                    ## because this means that none of the distributions lead to a good final solution
                    
                    ## at this point we would inspect all of the distributions to see whats the problem
                    
                    ## however at this time, a quality control procedure doesn't exist
                    
                    ## likely the photometry is fine if the quality is good and reddening and distance is known
                    
                    ## so spectroscopy is probably messing up
                    
                    ## analyse the photometry solution only?
                    
                    ## eck it just depends
                    
                    ## spectra and phot could be bad for several reasons, as with asteroseismology
                    
                    ## for now, return NaNs
                    
                    teff_fit_wa = [np.nan,np.nan]
                    logg_fit_wa = [np.nan,np.nan]
                    feh_fit_wa = [np.nan,np.nan]
                    feh_spec_fit_wa = [np.nan,np.nan]
                    age_fit_wa = [np.nan,np.nan]
                    mass_fit_wa = [np.nan,np.nan]
                    radius_fit_wa = [np.nan,np.nan]
                    luminosity_fit_wa = [np.nan,np.nan]
                    
                else:
                
                    teff_fit_wa = [np.average(teff,weights=prob_input),np.sqrt(np.average((teff-np.average(teff,weights=prob_input))**2,weights=prob_input))]
                    logg_fit_wa = [np.average(logg,weights=prob_input),np.sqrt(np.average((logg-np.average(logg,weights=prob_input))**2,weights=prob_input))]
                    feh_fit_wa = [np.average(feh,weights=prob_input),np.sqrt(np.average((feh-np.average(feh,weights=prob_input))**2,weights=prob_input))]
                    feh_spec_fit_wa =[np.average(feh_spec,weights=prob_input),np.sqrt(np.average((feh_spec-np.average(feh_spec,weights=prob_input))**2,weights=prob_input))]        
                    
                    # PMS_split = 1 # 1 Gyr
                    # age_MS = age[age >= PMS_split]
                    # age_PMS = age[age<PMS_split]
        
                    # if len(age_MS) == 0:
                        
                    #     age_fit_MS_wa = [np.nan,np.nan]
                        
                    # else:
        
                    #     age_fit_MS_wa = [np.average(age_MS,weights=prob_input[age >= PMS_split]),np.sqrt(np.sum((age_MS - np.average(age_MS,weights=prob_input[age >= PMS_split]))**2)/(len(age_MS)-1))]
        
                    
                    # if len(age_PMS) == 0:
                        
                    #     age_fit_PMS_wa = [np.nan,np.nan]
                        
                    # else:
        
                    #     age_fit_PMS_wa = [np.average(age_PMS,weights=prob_input[age<PMS_split]),np.sqrt(np.sum((age_PMS - np.average(age_PMS,weights=prob_input[age<PMS_split]))**2)/(len(age_PMS)-1))]
        
                    
                    # age_fit_wa = np.hstack((age_fit_PMS_wa,age_fit_MS_wa))
                    age_fit_wa = [np.average(age,weights=prob_input),np.sqrt(np.average((age-np.average(age,weights=prob_input))**2,weights=prob_input))]
                    mass_fit_wa = [np.average(mass,weights=prob_input),np.sqrt(np.average((mass-np.average(mass,weights=prob_input))**2,weights=prob_input))]
                    radius_fit_wa = [np.average(radius,weights=prob_input),np.sqrt(np.average((radius-np.average(radius,weights=prob_input))**2,weights=prob_input))]
                    luminosity_fit_wa = [np.average(Luminosity,weights=prob_input),np.sqrt(np.average((Luminosity-np.average(Luminosity,weights=prob_input))**2,weights=prob_input))]
        
                    # print("hist fit",age_fit)
                    # print("whole wa fit",age_fit_whole_wa)
                    # print("wa fit",age_fit_wa)
    
                ''
                # pid = os.getpid()
                # py = psutil.Process(pid)
                # memoryUse = py.memory_info()[0]/2.**30
                
                # print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
    
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
                
                # mu_1_age = age_fit[0]
                # sigma_1_age = age_fit[1]
                # mu_2_age = age_fit[2]
                # sigma_2_age = age_fit[3]
                    
                # mu_age = mu_2_age
                # sigma_age = sigma_2_age
                    
                # age_fit = [mu_age,sigma_age]
                
                ### best params and errors to save for Teff, logg, [Fe/H], abundances, Mass, Age, Radius, Luminosity  -- HR10 grid ###
                
                # best_params_combined = [teff_fit[0],\
                #                         logg_fit[0],\
                #                         feh_fit[0],\
                #                         feh_spec_fit[0],\
                #                         best_spec_params[3],\
                #                         best_spec_params[4],\
                #                         best_spec_params[5]-MgFe_sys[0],\
                #                         best_spec_params[6],\
                #                         best_spec_params[7],\
                #                         mass_fit[0],\
                #                         age_fit[0],\
                #                         radius_fit[0],\
                #                         luminosity_fit[0]]
    
                # 4MOST NN caused change of MgFe statement 
    
                best_params_combined = [teff_fit_wa[0],\
                                        logg_fit_wa[0],\
                                        feh_fit_wa[0],\
                                        feh_spec_fit_wa[0],\
                                        1,\
                                        1,\
    #                                    best_spec_params[5]-MgFe_sys[0],\
                                        best_spec_params[4]-MgFe_sys[0],\
                                        best_spec_params[7],\
                                        best_spec_params[20],\
                                        mass_fit_wa[0],\
                                        age_fit_wa[0],\
                                        radius_fit_wa[0],\
                                        luminosity_fit_wa[0]]
    
                    
                best_params_combined = np.array(best_params_combined)
                
                print("best_params_combined",best_params_combined)
                
                # best_params_combined_err = [teff_fit[1],\
                #                             logg_fit[1],\
                #                             feh_fit[1],\
                #                             feh_spec_fit[1],\
                #                             best_spec_params_err[3],\
                #                             best_spec_params_err[4],\
                #                             best_spec_params_err[5]+MgFe_sys[1],\
                #                             best_spec_params_err[6],\
                #                             best_spec_params_err[7],\
                #                             mass_fit[1],\
                #                             age_fit[1],\
                #                             radius_fit[1],\
                #                             luminosity_fit[1]]
    
                best_params_combined_err = [teff_fit_wa[1],\
                                            logg_fit_wa[1],\
                                            feh_fit_wa[1],\
                                            feh_spec_fit_wa[1],\
                                            0.05,\
                                            0.05,\
                                            # best_spec_params_err[5]+MgFe_sys[1],\
                                            best_spec_params_err[4]+MgFe_sys[1],\
                                            best_spec_params_err[7],\
                                            best_spec_params_err[20],\
                                            mass_fit_wa[1],\
                                            age_fit_wa[1],\
                                            radius_fit_wa[1],\
                                            luminosity_fit_wa[1]]
    
                
                best_params_combined_err = np.array(best_params_combined_err)   
                
                print("best_params_combined_err",best_params_combined_err)
                
                ''
                ### best params and errors to save for Teff, logg, [Fe/H], abundances, Mass, Age, Radius, Luminosity  -- RVS grid ###
                ''
                # best_params_combined = [teff_fit[0],\
                #                         logg_fit[0],\
                #                         feh_fit[0],\
                #                         feh_spec_fit[0],\
                #                         best_spec_params[13],\
                #                         best_spec_params[15],\
                #                         best_spec_params[8],\
                #                         best_spec_params[11],\
                #                         best_spec_params[12],\
                #                         mass_fit[0],\
                #                         age_fit[0],\
                #                         radius_fit[0],\
                #                         luminosity_fit[0]]
                    
                # best_params_combined = np.array(best_params_combined)
                
                
                # best_params_combined_err = [(teff_fit[1] ** 2 + (best_spec_params_err[0]*1000)**2 + teff_binwidth_prob_input**2)**0.5,\
                #                             (logg_fit[1] ** 2 + best_spec_params_err[1]**2 + logg_binwidth_prob_input**2)**0.5,\
                #                             (feh_fit[1] ** 2 + best_spec_params_err[2]**2 + feh_binwidth_prob_input**2)**0.5,\
                #                             (feh_spec_fit[1] ** 2 + best_spec_params_err[2]**2 + best_spec_params_err[5]**2 + feh_spec_binwidth_prob_input**2)**0.5,\
                #                             best_spec_params_err[13],\
                #                             best_spec_params_err[15],\
                #                             best_spec_params_err[8],\
                #                             best_spec_params_err[11],\
                #                             best_spec_params_err[12],\
                #                             (mass_fit[1]**2 + mass_binwidth_prob_input**2)**0.5,\
                #                             (age_fit[1]**2 + age_binwidth_prob_input**2)**0.5,\
                #                             (radius_fit[1]**2 + radius_binwidth_prob_input**2)**0.5,\
                #                             (luminosity_fit[1]**2 + lumin_binwidth_prob_input**2)**0.5]
                
                # best_params_combined_err = np.array(best_params_combined_err)   
    
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
                
                # pid = os.getpid()
                # py = psutil.Process(pid)
                # memoryUse = py.memory_info()[0]/2.**30
                
                # print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))
    
                
                # directory_output = stellar_filename # name of directory is the name of the star
                
                directory_output = star_field_names[inp_index] # name of directory is the name of the star
                directory_check = os.path.exists(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}")
                
                if  directory_check == True:
                
                    print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output} directory exists")
                    
                else:
                    
                    print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output} directory does not exist")
                    
                    os.makedirs(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}")
                    
                    print(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output} directory has been created")
            
            
                if spec_covariance_bool == False: 
                    
                    # the initial spec calculation doesn't change if this bool is true or false, the PDF created changes
                    
                    # since we're just passing all the information, then the only thing which matters is save statements at the end
                    
                    # we calculate covariance and no covariance anyways, this just specifies at the final results
    
                    cov_save_string = "_no_spec_Cov"
                
                elif spec_covariance_bool == True:
                    
                    cov_save_string = ""
    
    
                np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_{stellar_filename}{spec_extension}{extra_save_string}{cov_save_string}.txt",best_params_combined,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")
                np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_err_{stellar_filename}{spec_extension}{extra_save_string}{cov_save_string}.txt",best_params_combined_err,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")
                
                if recalc_metals_scheme_bool:
                    
                    print("re-calculating metals based on bayesian scheme")
                                                    
                    rv_shift = best_spec_params[-1]
                    rv_shift_err = 0.05
                    
                    # still keep error masks however right?
                    
                    # need to pass on prev. RV information to save time... 
                    
                    extra_save_string_new = extra_save_string + "_recalc_met"
                                                        
                    spec_init_run = [spec_path,\
                                error_map_spec_path,\
                                error_mask_index,\
                                error_mask_recreate_bool,\
                                error_map_use_bool,\
                                cont_norm_bool,\
                                rv_shift_recalc,\
                                conv_instrument_bool,\
                                input_spec_resolution,\
                                False,\
                                numax_input_arr,\
                                True,\
                                [best_params_combined[0],\
                                 best_params_combined[1],\
                                 best_params_combined[3],\
                                 False],\
                                logg_fix_bool,\
                                logg_fix_input_arr,\
                                [unique_emask_params_bool,unique_emask_params],\
                                Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]].replace(".txt",""),\
                                rv_shift,\
                                rv_shift_err,\
                                [False,[np.nan,np.nan,np.nan]]] # use to just be stellar names 
    
                    best_fit_spectroscopy_recalc = mspec_new.find_best_val(spec_init_run)
                    best_spec_params_recalc = best_fit_spectroscopy_recalc[0]   
                    best_spec_params_recalc_err = best_fit_spectroscopy_recalc[1]   
                    
                    rv_shift_spec = best_fit_spectroscopy_recalc[3]
                    N_pixels = len(best_fit_spectroscopy_recalc[6])
                    N_params = len(best_spec_params_recalc)
                    spec_dof = abs(N_pixels - (N_params-2)) # degrees of freedom                        
    
                    spec_abundances = best_spec_params_recalc[3:]
                    # convert abundances to [X/Fe] = [X/H] - [Fe/H]
                    spec_abundances = spec_abundances - best_spec_params_recalc[2]          
                    # best_spec_params = np.hstack((best_spec_params,spec_dof,rv_shift_spec)) # HR10              
                    # convert errors -> d[X/Fe] = (abs(-feh)**2 * dxh**2 + abs(xh)**2 * dfeh**2) ** 0.5
                    spec_abundances_errs = (abs(best_spec_params_recalc[2]) ** 2 * best_spec_params_recalc_err[2] ** 2 +\
                                           abs(best_spec_params_recalc[3:]) ** 2 * best_spec_params_recalc_err[3:] ** 2) ** 0.5 
                    best_spec_params_recalc = np.hstack((best_spec_params_recalc[:3],spec_abundances,spec_dof,rv_shift_spec))    
                    best_spec_params_recalc_err = np.hstack((best_spec_params_recalc_err[:3],spec_abundances_errs))    
    
                    cov_matrix_recalc = best_fit_spectroscopy_recalc[11] # how would one convert the cov matrix values?        
                    wvl_corrected_recalc = best_fit_spectroscopy_recalc[5]            
                    obs_norm_recalc = best_fit_spectroscopy_recalc[6]                
                    usert_norm_recalc = best_fit_spectroscopy_recalc[10]                
                    fit_norm_recalc = best_fit_spectroscopy_recalc[7]                
                    wvl_obs_input_recalc = best_fit_spectroscopy_recalc[9]                
                    spectra_save_recalc = np.vstack((wvl_corrected_recalc,obs_norm_recalc,usert_norm_recalc,fit_norm_recalc)).T
    
                    # The cov matrix used in the cov step is using MgFe converted but the covariance values for teff, logg, feh --> wait, only those... doesn't use the other dimesnions right?
    
                    if numax_iter_bool:
                    
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string_new}.txt',best_spec_params_recalc)
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string_new}.txt',best_spec_params_recalc_err)    
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string_new}.txt',cov_matrix_recalc)
        
                        if spectra_save_bool:
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectra_save_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string_new}.txt',spectra_save_recalc,header="wavelength,flux,error")
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectra_wvl_obs_input_{stellar_filename}_{spec_obs_number+1}_{mode_kw.replace("_numax","")}{extra_save_string_new}.txt',wvl_obs_input_recalc,header="wavelength,flux,error")
                    else:
                        
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectroscopy_best_params_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string_new}.txt',best_spec_params_recalc)
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectroscopy_best_params_error_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string_new}.txt',best_spec_params_recalc_err)    
                        np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/covariance_matrix_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string_new}.txt',cov_matrix_recalc)
        
                        if spectra_save_bool:
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectra_save_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string_new}.txt',spectra_save_recalc,header="wavelength,flux,error")
                            np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_output}/spectra_wvl_obs_input_{stellar_filename}_{spec_obs_number+1}_{mode_kw}{extra_save_string_new}.txt',wvl_obs_input_recalc,header="wavelength,flux,error")
                    
                    # now update the bayesian results with the new [Fe/H] and [Mg/Fe]
    
                    best_params_combined_recalc = [teff_fit_wa[0],\
                                                    logg_fit_wa[0],\
                                                    feh_fit_wa[0],\
                                                    best_spec_params_recalc[2]-FeH_spec_sys[0],\
                                                    1,\
                                                    1,\
                #                                    best_spec_params[5]-MgFe_sys[0],\
                                                    best_spec_params_recalc[4]-MgFe_sys[0],\
                                                    best_spec_params_recalc[7],\
                                                    best_spec_params_recalc[20],\
                                                    mass_fit_wa[0],\
                                                    age_fit_wa[0],\
                                                    radius_fit_wa[0],\
                                                        luminosity_fit_wa[0]]
                    best_params_combined_recalc = np.array(best_params_combined_recalc)
                    print("best_params_combined_recalc \n",best_params_combined_recalc)
                        
                    
                    best_params_combined_recalc_err = [teff_fit_wa[1],\
                                                        logg_fit_wa[1],\
                                                        feh_fit_wa[1],\
                                                        best_spec_params_recalc_err[2]+FeH_spec_sys[1],\
                                                        0.05,\
                                                        0.05,\
                                                        # best_spec_params_err[5]+MgFe_sys[1],\
                                                        best_spec_params_err[4]+MgFe_sys[1],\
                                                        best_spec_params_err[7],\
                                                        best_spec_params_err[20],\
                                                        mass_fit_wa[1],\
                                                        age_fit_wa[1],\
                                                        radius_fit_wa[1],\
                                                        luminosity_fit_wa[1]]
                                            
                    best_params_combined_recalc_err = np.array(best_params_combined_recalc_err)
                    print("best_params_combined_recalc_err \n",best_params_combined_recalc_err)
                    
                    
                    np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_{stellar_filename}{spec_extension}{extra_save_string_new}{cov_save_string}.txt",best_params_combined_recalc,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")
                    np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/multiobs/{directory_output}/best_fit_params_err_{stellar_filename}{spec_extension}{extra_save_string_new}{cov_save_string}.txt",best_params_combined_recalc_err,fmt='%.5f', delimiter=',',header = "Teff [K] \t Logg \t [Fe/H] \t Vmic \t Vsini \t [Mg/Fe] \t [Ti/Fe] \t [Mn/Fe] \t Mass \t Age \t Radius \t Luminosity")

        except:
              spec_extension = f"_OBS_NUM_{spec_obs_number+1}_SPEC_MODE_{mode_kw}"
              print(f"star {stellar_filename} spec {spec_extension} : {Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]} failed")       
              flag_fails[correlation_arr[inp_index][spec_obs_number]] = 1
              
              print(traceback.print_stack())
              
              continue
   
    directory_spectroscopy = star_field_names[inp_index] # name of directory is the name of the star
    directory_check = os.path.exists(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")                
    if  directory_check == True:                
         print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory exists")                    
    else:                    
         print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory does not exist")                    
         os.makedirs(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy}")                    
         print(f"../Output_data/Stars_Lhood_spectroscopy/multiobs/{directory_spectroscopy} directory has been created")
    np.savetxt("../Output_data/Stars_Lhood_spectroscopy/multiobs/4MOST_OpR2.5/flag_fails.txt",flag_fails,fmt='%i')
    np.savetxt("../Output_data/Stars_Lhood_spectroscopy/multiobs/4MOST_OpR2.5/flag_no_rv.txt",flag_no_rv,fmt='%i')
    
    print(f"Total time elapsed for star {stellar_names[inp_index]} --- {(time.time()-start_time)/60} minutes ---")

def plot_pdf_all(prob_input_arr,prob_input_name_arr,prob_col_arr,teff,logg,feh_spec,age,mass,radius,lum,path,Param_binwidth_arr,plt_title):
  f,ax = plt.subplots(2,4,sharey=True,figsize=(19,8))
  plt.subplots_adjust(hspace=0.3,wspace=0)
  ax  = np.reshape(ax,-1)
  paras = [teff,logg,feh_spec,age,mass,radius,lum]
  para_names = ['Teff','logg','[Fe/H]','age','mass','radius','lum']
  for i,para in enumerate(paras):
    axi = ax[i]
    axi.set_xlabel(para_names[i],fontsize='large')
    for j in range(len(prob_input_arr)):
        
      ### calc the weighted mean and std
      # prob_w_ave_i = np.average(paras[i],weights=prob_input_arr[j])
      # prob_std_dev_i = np.sqrt(np.average((paras[i]-prob_w_ave_i)**2, weights=prob_input_arr[j]))
      
      # print(para_names[i],prob_input_name_arr[j],prob_w_ave_i,prob_std_dev_i)
        
      x,y = paras[i],prob_input_arr[j]
      mask = np.argsort(x)
      x,y = x[mask],y[mask]/np.max(y)
      # axi.hist(x,bins=int(abs((max(x)-min(x))/Param_binwidth_arr[i])),density=True,alpha=0.1,histtype='stepfilled',color='lightgrey',edgecolor='k')
      # if prob_input_name_arr[j] == "final (_wa)":
      #     axi.axvline(prob_w_ave_i,color='r',linestyle='--',linewidth=4)
      axi.plot(x,y,color=prob_col_arr[j],label=prob_input_name_arr[j],alpha=0.6)
  ax[-2].legend(loc='upper left',fontsize='small',bbox_to_anchor=(1.,1.01))
  ax[-1].axis('off')
  
  f.suptitle(plt_title,fontsize=35)
  
  plt.show()
  f.savefig(path,bbox_inches='tight')

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
        
        # ax_1d.set_yscale("log")
                
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
    
def chkList(lst):
    if len(lst) < 0 :
        res = True
    res = all(ele == lst[0] for ele in lst)
      
    if(res):
        eql_check = True
    else:
        eql_check = False
        
    return eql_check 

def Log_prob_distributions_1D(Parameter,Probability,Param_name,Param_binwidth,prob_input_name,param_filename,makefig_bool,savefig_bool,stellar_id,chisq_red_bool,savefig_extra_name):
       
    """
    Need to cut the data down and get rid of any zero regions which waste time
    """
    
    Parameter_zero_clean = Parameter[np.where(Probability!=0.0)[0]]
    Probability_zero_clean = Probability[np.where(Probability!=0.0)[0]]
            
    Parameter = Parameter_zero_clean
    Probability = Probability_zero_clean
    
    ###
    # what if the probability given is all 1's?
    # should I just never use it therefore no requirement?
    # but this could happen with photometry e.g. NaN bands
    # so, what do I do?
    # return np.nan? 
    # okay yes do that
    ###
    
    eql_check = chkList(Probability)
    
    if eql_check:
                
        if param_filename == "age":
        
            return [np.nan,np.nan,np.nan,np.nan]
    
        else:
            
            return [np.nan,np.nan]
        
    else:
    
        data_entries, bins = np.histogram(Parameter, bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability)
        # data_entries, bins = np.histogram(Parameter, bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)))
              
        # plt.hist(data_entries,bins)
        # plt.hist(Parameter, bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)))
        
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        
        binscenters_zero_clean = binscenters[np.where(data_entries!=0.0)[0]]    
        data_entries_zero_clean = data_entries[np.where(data_entries!=0.0)[0]]
            
        data_entries = data_entries_zero_clean
        binscenters = binscenters_zero_clean
        
        peak_data = max(data_entries)            
        parameter_peak_loc = binscenters[np.where(data_entries==peak_data)[0]][0]
        
        # print("DATA",data_entries)
        # print("PEAK DATA",peak_data)
        
        buffer = 1e-3
        
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
                
                    popt, pcov = curve_fit(fit_function_quad,
                                           xdata=binscenters,
                                           ydata=log_data_entries,
                                           p0=[log_peak_data, parameter_peak_loc, Param_binwidth],
                                           bounds=((log_peak_data_min,
                                                    peak_loc_min,
                                                    1e-5*Param_binwidth),
                                                   (log_peak_data,
                                                    peak_loc_max,
                                                    1e2*Param_binwidth)))
                            
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
            
            # plt.title(f"{stellar_id} \n {prob_title_name}")
            plt.title(f"{stellar_id} \n {extra_save_string.replace('_',' ').replace('p','.')}")
            
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
                
                    # fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_no_norm_covariance_log_fit{savefig_extra_name}.png")
                    fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_no_norm_covariance_log_fit{savefig_extra_name}{extra_save_string}.png")
    
                elif chisq_red_bool == True:
                
                    # fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_norm_covariance_log_fit{savefig_extra_name}.png")
                    fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/{prob_input_name}_vs_{param_filename}_{stellar_id}_norm_covariance_log_fit{savefig_extra_name}{extra_save_string}.png")
            
            plt.close('all')
            
        return fitted_params
    
    # return [mu_fit,sigma_fit]

def Log_multiple_prob_distributions_1D(Parameter,Probability_arr,Param_name,Param_binwidth,prob_input_name_arr,param_filename,makefig_bool,savefig_bool,stellar_id,chisq_red_bool,col_arr,extra_name):#,ref_param,ref_param_err):
       
    
    """
    Need to cut the data down and get rid of any zero regions which waste time
    """
    
    fig_prob_1d = plt.figure(figsize=(12,12))
            
    ax_1d = fig_prob_1d.add_subplot(111)
        
    for Prob_index in range(len(Probability_arr)):
        
        Probability = Probability_arr[Prob_index]
        prob_input_name = prob_input_name_arr[Prob_index]
        
        eql_check = chkList(Probability)
        
        if eql_check:
            
            continue
                
        # Parameter_zero_clean = Parameter[np.where(Probability!=0.0)[0]]
        # Probability_zero_clean = Probability[np.where(Probability!=0.0)[0]]

        # print(np.where(np.isnan(Probability)==False)[0])

        # Parameter_nan_clean = Parameter[np.where(np.isnan(Probability)==False)[0]]
        # Probability_nan_clean = Probability[np.where(np.isnan(Probability)==False)[0]]

                
        # Parameter = Parameter_zero_clean
        # Probability = Probability_zero_clean
        
        # Parameter = Parameter_nan_clean
        # Probability = Probability_nan_clean
        
        
        data_entries, bins = np.histogram(Parameter, bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability)
        
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        
        binscenters_zero_clean = binscenters[np.where(data_entries!=0.0)[0]]    
        data_entries_zero_clean = data_entries[np.where(data_entries!=0.0)[0]]
    
        data_entries = data_entries_zero_clean
        binscenters = binscenters_zero_clean
                
        peak_data = np.nanmax(data_entries)            
        parameter_peak_loc = binscenters[np.where(data_entries==peak_data)[0]][0]
                        
        buffer = 1e-3
        
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
            
            peak_data_PMS = np.nanmax(data_entries_PMS)
            parameter_peak_loc_PMS = binscenters_PMS[np.where(data_entries_PMS==peak_data_PMS)[0]][0]
            peak_loc_PMS_min = (1-buffer) * parameter_peak_loc_PMS
            peak_loc_PMS_max = (1+buffer) * parameter_peak_loc_PMS
    
            peak_data_MS = np.nanmax(data_entries_MS)
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
                    
                    # print(f"{prob_input_name},{param_filename},{peak_data_MS}")
                                                                        
                else: 
                    
                    # print(f"{prob_input_name},{param_filename},{peak_data}")
                            
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
                
                # "Photometry+Parallax","Combined","Spectroscopy"

                norm_sep = prob_norm_factors[np.intersect1d(np.where(prob_norm_input_names == prob_input_name)[0],np.where(parameter_norm_input_names == param_filename)[0])[0]]
                
                # if prob_input_name == "Photometry+Parallax":
                                                        
                #    norm_sep = 200
                   
                # elif prob_input_name == "Combined":
                    
                #    norm_sep = 5e-4
                   
                   
                # elif prob_input_name == "Spectroscopy":
                    
                #    norm_sep = 5
                   
                # else:
                    
                #    norm_sep = 1
                  
                
                
                ### NEED TO RENORMALISE THESE DISTRIBUTIONS ###
                
                ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability/np.sum(Probability),density=False,alpha=0.25,histtype='stepfilled',color=col_arr[Prob_index],edgecolor='k')
                # ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),weights=Probability/norm_sep,density=False,alpha=0.25,histtype='stepfilled',color=col_arr[Prob_index],edgecolor='k')
            
                if param_filename == "temperature":
                    
                    y_unnorm = np.exp(fit_function_quad(xspace, *popt))
                    
                    # N_factor = np.sum(binscenters * data_entries)
                    N_factor = np.sum(data_entries)
                    
                    y_norm = y_unnorm/N_factor
                    # y_norm = y_unnorm/norm_sep
                
                    ax_1d.plot(xspace, y_norm, color=col_arr[Prob_index], linewidth=5, label=f'{prob_input_name}\n' + r'$\mu$ = ' + f'{mu_fit:.0f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.0f}\n')
    
                elif param_filename == "age":

                    y_unnorm = np.exp(fit_function_quad_bimodal(xspace, *popt))
                    
                    # N_factor = np.sum(binscenters * data_entries)
                    N_factor = np.sum(data_entries)
                    
                    y_norm = y_unnorm/N_factor
                    # y_norm = y_unnorm/norm_sep
                    
                    ax_1d.plot(xspace, y_norm, color=col_arr[Prob_index], linewidth=2.5, label=f'{prob_input_name}\n' + r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n\n' + r'$\mu_2$ = ' + f'{mu_fit_2:.3f}\n' + r'$\sigma_2$ = ' + f'{sigma_fit_2:.3f}\n')        
                
                else:

                    y_unnorm = np.exp(fit_function_quad(xspace, *popt))
                    
                    # N_factor = np.sum(binscenters * data_entries)
                    N_factor = np.sum(data_entries)
                    
                    y_norm = y_unnorm/N_factor
                    # y_norm = y_unnorm/norm_sep
                    
                    ax_1d.plot(xspace, y_norm, color=col_arr[Prob_index], linewidth=2.5, label=f'{prob_input_name}\n' + r'$\mu$ = ' + f'{mu_fit:.3f}\n' + r'$\sigma$ = ' + f'{sigma_fit:.3f}\n')        
            
            if suc == False:
                
                print("Fitting failed, change bin size")
                            
    ax_1d.set_ylabel('Probability',fontsize=35)
    ax_1d.set_xlabel(f'{Param_name}',fontsize=35)
    
    # hist, bins = np.histogram(Parameter,weights=Probability)
    
    # Gauss_prob = normpdf(Parameter,np.average(Parameter),Param_binwidth)
            
    # ax_1d.hist(Parameter,bins=int(abs((max(Parameter)-min(Parameter))/Param_binwidth)),density=False,label=f'Bin width = {Param_binwidth}',alpha=1,histtype='stepfilled')

    # ax_1d.bar(Parameter, Probability, width=0.08, bottom=None)
        
    # if param_filename == "temperature" or param_filename == "feh" or param_filename == "feh_spec":
    
    #     ax_1d.set_xlim([min(Parameter),max(Parameter)])
        
    #     ax_1d.axvline(x = ref_param,linestyle='-',color='grey',linewidth=2,label='Ref.')
    #     ax_1d.axvspan(ref_param - ref_param_err, ref_param + ref_param_err, alpha=0.15, color='lightgrey')
        
    # ax_1d.axvline(x = ref_param - ref_param_err,linestyle='-',color='grey',linewidth=2)
    # ax_1d.axvline(x = ref_param + ref_param_err,linestyle='-',color='grey',linewidth=2)
    
    # ax_1d.set_yscale("log")

    ax_1d.legend(loc='upper right',fontsize=20)

            
    ax_1d.set_ylim(ymin=0,ymax=1)

    ax_1d.set_xlim([min(xspace),max(xspace)])

    
    prob_title_name = prob_input_name.replace("_"," ").replace("w","x")
    
    
    ax_1d.tick_params(axis='both',which='major',labelsize=35)
    
    # plt.title(f"{prob_title_name}")
    plt.title(f"{stellar_id} \n {extra_save_string.replace('_',' ').replace('p','.')}",fontsize=30)
    
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
        
            fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/Prob_1d_multifits_vs_{param_filename}_{stellar_id}_no_norm_covariance_log_fit{extra_name}{extra_save_string}.png")

        elif chisq_red_bool == True:
        
            fig_prob_1d.savefig(f"../Output_figures/1d_distributions/{stellar_id}/Prob_1d_multifits_vs_{param_filename}_{stellar_id}_norm_covariance_log_fit{extra_name}{extra_save_string}.png")
    
    plt.close('all')
            
def fit_function_quad(x,*labels_inp):
    
    A_log = labels_inp[0] 
    mu = labels_inp[1]
    sigma = labels_inp[2]
        
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


Input_spec_data = np.loadtxt('../Input_data/spectroscopy_observation_data/spectra_list_PLATO_new.txt',delimiter=',',dtype=str)
# Input_spec_data = np.loadtxt(w_giants_path+'Input_data/spectroscopy_observation_data/spectra_list_PLATO_new_w_giants.txt',delimiter=',',dtype=str)


"""
0 ; source_id (Literature name of the star)
1 ; spectra type (e.g. HARPS or UVES_cont_norm_convolved etc)
2 ; spectra path
"""

Input_phot_data = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/PLATO_bmk_phot_data/PLATO_photometry_eDR3.csv',delimiter = ',',dtype=str)
# Input_phot_data = np.loadtxt(w_giants_path+'Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/PLATO_bmk_phot_data/PLATO_photometry_eDR3_w_giants.csv',delimiter = ',',dtype=str)
star_field_names = Input_phot_data[:,0]

"""
0 ; source_id (Literature name of the star)
1 ; G
2 ; eG
3 ; Bp
4 ; eBp
5 ; Rp
6 ; eRp
7 ; distance BJ 21, photogeometric distance in pc from Bailer-Jones 2021 eDR3 data 
8 ; distance err up, upper error in photogeometric distance from BJ21
9 ; distance err low, lower error in photogeometric distance from BJ21
10 ; AG [mag], Gaia Extinction
11 ; Gaia DR2 ids
12 ; B (Johnson filter) [mag]
13 ; eB (Johnson filter) [mag]
14 ; V (Johnson filter) [mag]
15 ; eV (Johnson filter) [mag]
16 ; H (2MASS filter) [mag]
17 ; eH (2MASS filter) [mag]
18 ; J (2MASS filter) [mag]
19 ; eJ (2MASS filter) [mag]
20 ; K (2MASS filter) [mag]
21 ; eK (2MASS filter) [mag]
22 ; E(B-V), reddening value from stilism 	
23 ; E(B-V)_upp, upper reddening uncertainty from stilism tool or VHS
24 ; E(B-V)_low, lower reddening uncertainty from stilism tool or VHS
25 ; Gaia Parallax (mas)
26 ; eGaia Parallax (mas)
27 ; SIMBAD Parallax (mas)
28 ; eSIMBAD Parallax (mas)	
"""

Input_ast_data = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/PLATO_stars_seism.txt',delimiter = ',',dtype=str)
# Input_ast_data = np.loadtxt(w_giants_path+'Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/PLATO_stars_seism_w_giants.txt',delimiter = ',',dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; nu_max, maximum frequency [microHz]
2 ; err nu_max pos, upper uncertainty
3 ; err nu_max neg, lower uncertainty
4 ; d_nu, large separation frequency [microHz]
5 ; err d_nu pos, upper uncertainty [microHz]
6 ; err d_nu neg, lower uncertainty [microHz]
"""

Ref_data = np.loadtxt("../Input_data/Reference_data/PLATO_stars_lit_params.txt",delimiter=",",dtype=str)
# Ref_data = np.loadtxt(w_giants_path+'Input_data/Reference_data/PLATO_stars_lit_params_w_giants.txt',delimiter=",",dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; Teff [K]
2 ; err Teff [K]
3 ; Logg [dex]
4 ; err Logg [dex]
5 ; [Fe/H] [dex]
6 ; err [Fe/H] [dex]
"""

Ref_data_other = np.loadtxt("../Input_data/Reference_data/PLATO_stars_lit_other_params.txt",delimiter=",",dtype=str)
# Ref_data_other = np.loadtxt(w_giants_path+'Input_data/Reference_data/PLATO_stars_lit_other_params_w_giants.txt',delimiter=",",dtype=str)

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

flag_fails = np.zeros([len(Input_spec_data)])
flag_no_rv = np.zeros([len(Input_spec_data)])

correlation_arr = correlation_table(Input_spec_data,Input_phot_data)

stellar_names = Input_phot_data[:,0]

error_mask_recreate_bool =False # if False, then teff varying mask is assumed
error_map_use_bool = True
cont_norm_bool = True
rv_shift_recalc = [False,-400,400,0.5]
# rv_shift_direct = np.nan
rv_shift_direct = -30
rv_shift_direct_err = np.nan
# gives option to use rv_shift straight away
# if not nan, then ignore rv_shift_recalc above
conv_instrument_bool = True
input_spec_resolution = 20000
numax_iter_bool = False
niter_numax_MAX = 2
recalc_metals_bool = False
feh_recalc_fix_bool = False
logg_fix_bool = False
# logg_fix_input_arr = [3.99,0.01,0.01]
logg_fix_input_arr = [np.nan,np.nan,np.nan]
spec_fix_values = [5777,4.44,0]
unique_emask_params_bool = False
unique_emask_params = [5777,4.44,0,1,1.6,0,0,0] # solar example 
numax_first_step_logg = [False,np.nan,np.nan,np.nan]

### photometry grid limits

Teff_resolution = 500 # K 800
logg_resolution = 0.3 # dex 0.6
feh_resolution = 0.6 # dex 0.6

phot_ast_central_values_bool = False # if set to True, values below will be used
phot_ast_central_values_arr = [5900,4,0] # these values are at the centre of the photometry grid

phot_magnitude_set = "all" # you can have all, gaia, non-gaia, gaia_col and gaia_bprp_col as strings to indicate filters 

save_ast_space_bool = False
spectra_save_bool = False 
save_phot_space_bool = False # if this was set true here, it must still be true for the next modules, same for next bool variable 
save_spec_phot_space_bool = False
save_final_space_bool = False

chi_2_red_bool = False

best_spec_bool = False    
phot_ast_track_bool = False
spec_track_bool = False
bayes_scheme_bool = False
recalc_metals_scheme_bool = False

## PLATO MSAP2 modules

MSAP2_01_bool = False 
MSAP2_02_bool = False
MSAP2_03_bool = False

# MSAP2_01_test_case_kw = "msap4"
# MSAP2_01_test_case_kw = "ast_pdf"
MSAP2_01_test_case_kw = "numax_iter"

### naming conventions for the settings above

# mode_kw = "LITE_logg"
# mode_kw = "LITE_numax"
mode_kw = "MASK"
# mode_kw = "BLIND"
# mode_kw = "FULL"
# mode_kw = "BLIND_numax"

spec_covariance_bool = True # To use spectroscopic covariance in creation of PDF

Teff_thresh = 0
Logg_thresh = 0
FeH_thresh = 0

# systematics from benchmarking Gent et al (2021)    
# be free to change/improve

if spec_covariance_bool:

    # Teff_sys = [0,0]#[21.165,34.86]
    # Logg_sys = [0,0]#[0.007,0.018]
    # # Teff_sys = [0,34.86]
    # # Logg_sys = [0,0.018]
    # FeH_spec_sys = [0,0]#[0.029,0.029]
    # MgFe_sys = [0,0]#[0.019,0.046]

    Teff_sys = [21.165,34.86]
    Logg_sys = [0.007,0.018]
    FeH_spec_sys = [0.029,0.029]
    MgFe_sys = [0.019,0.046]
    
else:
    
    Teff_sys = [21.165,34.86]
    Logg_sys = [0.007,0.018]
    FeH_spec_sys = [0.029,0.029]
    MgFe_sys = [0.019,0.046]

    # Teff_sys = [0,0]#[21.165,34.86]
    # Logg_sys = [0,0]#[0.007,0.018]
    # # Teff_sys = [0,34.86]
    # # Logg_sys = [0,0.018]
    # FeH_spec_sys = [0,0]#[0.029,0.029]
    # MgFe_sys = [0,0]#[0.019,0.046]


### string that is attached to the filename, can be useful for specifying a certain way you ran the code


extra_save_string = "_4MOST_NN_PLATO_test"


if MSAP2_01_bool or MSAP2_02_bool or MSAP2_03_bool:
    extra_save_string += "_MSAP2"


if __name__ == '__main__':

    inp_index = 8
    # main_func_multi_obs(inp_index)
            
'''
num_processor = mp.cpu_count()
inp_index_list = np.arange(len(stellar_names))

if __name__ == '__main__':
#    
    # Limit the threads from numpy and scipy (important for multiprocessing)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    p = mp.Pool(num_processor) # Pool distributes tasks to available processors using a FIFO schedulling (First In, First Out)
    p.map(main_func_multi_obs,inp_index_list) # You pass it a list of the inputs and it'll iterate through them
    p.terminate() # Terminates process 

'''
