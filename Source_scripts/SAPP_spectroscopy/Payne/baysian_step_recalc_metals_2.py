#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:38:17 2020

@author: gent
"""

import numpy as np
#from multiprocessing import Pool
#import matplotlib.pyplot as plt
#from PyAstronomy import pyasl
import time

import SAPP_spectroscopy.Payne.SAPP_best_spec_payne_v1p1 as mspec_new

np.set_printoptions(suppress=True,formatter={'float': '{: 0.2f}'.format})

# import_path = "Payne/"
import_path = "../Input_data/spectroscopy_model_data/Payne_input_data/"

#name="NN_results_RrelsigL20.npz"#LTE
name="NN_results_RrelsigN20.npz"#NLTE

temp=np.load(import_path+name)

w_array_0 = temp["w_array_0"]
w_array_1 = temp["w_array_1"]
w_array_2 = temp["w_array_2"]
b_array_0 = temp["b_array_0"]
b_array_1 = temp["b_array_1"]
b_array_2 = temp["b_array_2"]
x_min = temp["x_min"]
x_max = temp["x_max"]

#print(x_min,x_max)

#number of parameters in Payne model
num_labels=w_array_0.shape[1]

#wavelength scale of the models
w0=np.linspace(5329.,5616,11480)[40:-40]
w0=w0[::2]


def recalc_metals_given_grid(best_fit_params,best_fit_params_errors,spec_best_fit_params_1st,spec_input_recalc):
    
    """
    This needs to take in arrays which are best fit parameters, errors, and spectroscopy for a given grid
    Then it needs to return the new parameters, errors for a given grid, these need to be added
    """
    
    teff_best = float(best_fit_params[1])
    logg_best = float(best_fit_params[2])
    feh_best = float(best_fit_params[3])
    
    feh_best_err = [float(best_fit_params_errors[4]),float(best_fit_params_errors[5])]
        
    feh_spec_1st = spec_best_fit_params_1st[2]
    mg_spec_1st = spec_best_fit_params_1st[5]
    ti_spec_1st = spec_best_fit_params_1st[6]
    mn_spec_1st = spec_best_fit_params_1st[7]
        
    feh_spec_1st_en = feh_spec_1st + 0.22/0.4 * mg_spec_1st
    
    feh_best_unen = feh_best - 0.22/0.4 * mg_spec_1st # feh best is metallicity values which are enhanced by Mg/Fe, need to 'de-enhance' them, re-run and enhance with new value
    
    # unless Mg/Fe is really large, there doesn't seem to be much difference between the above and below...
    
#    feh_best_unen = feh_best # feh best is metallicity values which are enhanced by Mg/Fe, need to 'de-enhance' them, re-run and enhance with new value
        
    params_fix = [teff_best,logg_best,feh_best_unen]
    
    # For grids other than spec, we give the normal [Fe/H]
    
    # However, these are assumed "enhanced" by alpha elements
    
    # [Fe/H] in the code is spetrocsopic
    
    # So surely, we need to give "de-enhanced" values
    
    # In order to do this, we do the correction for Mg by minusing it off
    
    # Problem then is that our new [Fe/H] results are related by the first [Mg/Fe results]
    
    # apararently doesn't matter...
    
    # What about the new ones, do we present them as enhanced?
    
    # No, show [Fe/H] dehanced since [Mg/Fe] is presented with them
    
    # Same goes for the legend values?
    
    # Hmmmm, we present [Fe/H]_legend as ones from PDF, these are assumed enhanced
    
    # To compare to spetroscopic, spetroscopic needs to be enhanced
    
    # Is dehancing them by some [Mg/Fe] value valid? 
    
    # Check e-mails!
    
    ### This is the array of Teff, logg, Feh to fix and a bool as to fix Feh 
    ### or not
    
    recalc_metals_inp = spec_input_recalc[12]
    
    ### This tells the spec code whether to use the array above
    
    recalc_metals_bool = spec_input_recalc[11]
    
    ### We told the code to fix [Fe/H] or not, need to save that answer
    
    fix_feh_bool = recalc_metals_inp[3]
    
    ### regardless if we fixed values for spec before, now we're 
    ### fixing them to the ones from PDF
    ### combine together to make a new recalc_metals input array    
    
    recalc_metals_inp_new = np.hstack((params_fix,fix_feh_bool))

    ### set the recalc metals to True
       
    recalc_metals_bool_new = True
    
    ### changing the spec input we define in main.py with the updated info
    
    spec_input_recalc[12] = recalc_metals_inp_new
        
    spec_input_recalc[11] = recalc_metals_bool_new
    
    refit_inp = spec_input_recalc
    
    ### call the spectroscopy module
                
    new_fit = mspec_new.find_best_val(refit_inp)
        
    spec_best_fit_params_2nd = new_fit[0] 
    feh_spec_2nd = spec_best_fit_params_2nd[2]    
    mg_spec_2nd = spec_best_fit_params_2nd[5]
    ti_spec_2nd = spec_best_fit_params_2nd[6]
    mn_spec_2nd = spec_best_fit_params_2nd[7]
    
#    print(teff_best,logg_best,feh_spec_2nd,feh_spec_2nd,mg_spec_2nd)
#    print(teff_best,logg_best,mn_spec_1st,mg_spec_2nd)
#    print("PDF Best Teff and Logg",teff_best,logg_best)
#    print("Spectroscopy 1st Best Teff and Logg", spec_best_fit_params_1st[0]*1000,spec_best_fit_params_1st[1])    
    
    spec_best_fit_params_2nd_errors = new_fit[2]
    
    feh_spec_2nd_en = feh_spec_2nd + 0.22/0.4 * mg_spec_2nd 
        
#    feh_err = abs(feh_spec_2nd_en-feh_spec_1st)
#    mg_err = abs(mg_spec_2nd-mg_spec_1st)
#    ti_err = abs(ti_spec_2nd-ti_spec_1st)
#    mn_err = abs(mn_spec_2nd-mn_spec_1st)
    
    feh_err = spec_best_fit_params_2nd_errors[2]
    mg_err = spec_best_fit_params_2nd_errors[3]
    ti_err = spec_best_fit_params_2nd_errors[6]
    mn_err = spec_best_fit_params_2nd_errors[7]
    
    if fix_feh_bool == True:
    
        feh_new_err = (np.array(feh_err) ** 2 + np.array(feh_best_err) ** 2)**0.5 # errors calculated in quadrature

        feh_new_err_en = (feh_new_err ** 2 + (0.22/0.4 * mg_err) ** 2)**0.5 
    
    elif fix_feh_bool == False: # if feh is being refit, then the previous feh uncertainty isn't accounted for.
        
        feh_new_err = np.ones([2]) * np.array(feh_err) 
        
        feh_new_err_en = np.ones([2]) * (feh_new_err ** 2 + (0.22/0.4 * mg_err) ** 2)**0.5 
    
    """
    Need to double check this Enhanced stuff with aldo
    
    I believe we agreed that we save the raw spectroscopic results
    
    Enhanced was only for combining PDFs
    
    Double check!!!
    """
    
    
    best_fit_errors_add = np.array([feh_new_err[0],feh_new_err[1],mg_err,mg_err,ti_err,ti_err,mn_err,mn_err],dtype=float) # these are tacked onto the original errors
#    best_fit_errors_add = np.array([feh_new_err_en[0],feh_new_err_en[1],mg_err,mg_err,ti_err,ti_err,mn_err,mn_err],dtype=float) # these are tacked onto the original errors

    best_fit_params_add = [np.array([np.format_float_scientific(feh_spec_2nd,3)] ,dtype=float),np.array([np.format_float_scientific(mg_spec_2nd,3)] ,dtype=float),np.array([np.format_float_scientific(ti_spec_2nd,3)],dtype=float),np.array([np.format_float_scientific(mn_spec_2nd,3)] ,dtype=float)] # these are tacked onto the original params
    # best_fit_params_add = [np.array([np.format_float_scientific(feh_spec_2nd_en,3)] ,dtype=float),np.array([np.format_float_scientific(mg_spec_2nd,3)] ,dtype=float),np.array([np.format_float_scientific(ti_spec_2nd,3)],dtype=float),np.array([np.format_float_scientific(mn_spec_2nd,3)] ,dtype=float)] # these are tacked onto the original params
    
    return best_fit_params_add,best_fit_errors_add
    

def recalc_metals(stellar_inp):
    
    stellar_id = stellar_inp[0].replace(" ","_")
    stellar_index = int(float(stellar_inp[1]))
    fix_feh_bool = stellar_inp[2]
    
    best_fit_params = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/{stellar_id}/best_fit_params_{stellar_id}.txt",dtype=str) # Order #  Grid Type 	 P_best 	 T_eff/K 	 logg/dex 	 [Fe/H]/dex 	 Mass/Msol 	 Age/Gyr 	 Radius/Rsol
    best_fit_params_errors = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/{stellar_id}/best_fit_params_errors_{stellar_id}.txt",dtype=str) # Order #  Grid Type 	 neg_sigma_T_eff/K 	 pos_sigma_T_eff/K 	 neg_sigma_logg/dex 	 pos_sigma_logg/dex 	 neg_sigma_[Fe/H]/dex 	 pos_sigma_[Fe/H]/dex 	 neg_sigma_Mass/Msol 	 pos_sigma_Mass/Msol 	 neg_sigma_Age/Gyr 	 pos_sigma_Age/Gyr 	 neg_sigma_Radius/Rsol 	 pos_sigma_Radius/Rsol
    
    N_grid_types = np.shape(best_fit_params)[0]
    
    best_fit_params_metals = []
    best_fit_params_errors_metals = []
    
    for grid_type in range(N_grid_types):
    
        teff_best = float(best_fit_params[grid_type][2])
        logg_best = float(best_fit_params[grid_type][3])
        feh_best = float(best_fit_params[grid_type][4])
    
        feh_best_err = [float(best_fit_params_errors[grid_type][5]),float(best_fit_params_errors[grid_type][6])]
    
        spec_best_fit_params_1st = np.loadtxt(f'../Output_data/Stars_Lhood_spectroscopy/{stellar_id}/best_fit_spec_params_starID{stellar_id}') # Order # Teff/1000, logg, Fe/H, vmic, vsini, Mg/Fe, Ti/Fe, Mn/Fe
    
        feh_spec_1st = spec_best_fit_params_1st[2]
        mg_spec_1st = spec_best_fit_params_1st[5]
        ti_spec_1st = spec_best_fit_params_1st[6]
        mn_spec_1st = spec_best_fit_params_1st[7]
        
        feh_spec_1st_en = feh_spec_1st + 0.22/0.4 * mg_spec_1st
        
        feh_best_unen = feh_best # feh best is metallicity values which are enhanced by Mg/Fe, need to 'de-enhance' them, re-run and enhance with new value

#        feh_best_unen = feh_best - 0.22/0.4 * mg_spec_1st # feh best is metallicity values which are enhanced by Mg/Fe, need to 'de-enhance' them, re-run and enhance with new value

        
        params_fix = [teff_best,logg_best,feh_best_unen]
        
        refit_inp = [stellar_index,params_fix,fix_feh_bool]
                
        new_fit = mspec_new.find_best_val(refit_inp)
        
        spec_best_fit_params_2nd = new_fit[0] 
        feh_spec_2nd = spec_best_fit_params_2nd[2]
        mg_spec_2nd = spec_best_fit_params_2nd[5]
        ti_spec_2nd = spec_best_fit_params_2nd[6]
        mn_spec_2nd = spec_best_fit_params_2nd[7]
        
        feh_spec_2nd_en = feh_spec_2nd + 0.22/0.4 * mg_spec_2nd 
        
        feh_err = abs(feh_spec_2nd_en-feh_spec_1st)
        mg_err = abs(mg_spec_2nd-mg_spec_1st)
        ti_err = abs(ti_spec_2nd-ti_spec_1st)
        mn_err = abs(mn_spec_2nd-mn_spec_1st)
                
        feh_new_err = (np.array(feh_err) ** 2 + np.array(feh_best_err) ** 2)**0.5 # errors calculated in quadrature
        
        best_fit_params_add = [np.format_float_scientific(feh_spec_2nd,3),np.format_float_scientific(mg_spec_2nd,3),np.format_float_scientific(ti_spec_2nd,3),np.format_float_scientific(mn_spec_2nd,3)] # these are tacked onto the original params
        best_fit_errors_add = [feh_new_err[0],feh_new_err[1],mg_err,mg_err,ti_err,ti_err,mn_err,mn_err] # these are tacked onto the original errors
            
        best_fit_params_metals.append(np.hstack([best_fit_params[grid_type][:5],best_fit_params_add,best_fit_params[grid_type][5:]])) # Trying to add abundances and errors at the end!
        best_fit_params_errors_metals.append(np.hstack([best_fit_params_errors[grid_type][:7],best_fit_errors_add,best_fit_params_errors[grid_type][7:]]))

#    np.savetxt(f"Stars_Lhood_combined_spec_phot/Gaia_benchmark/{stellar_id}/best_fit_params_{stellar_id}_metals.txt",best_fit_params_metals,fmt='%s',header='Grid Type 	 \t P_best 	 \t T_eff/K 	 \t logg/dex 	 \t [Fe/H]/dex 	 \t Mass/Msol 	 \t Age/Gyr 	 \t Radius/Rsol 	 \t [Fe/H]_new/dex 	 \t [Mg/Fe]/dex 	 \t [Ti/Fe]/dex 	 \t [Mn/Fe]/dex') # Order #  Grid Type 	 P_best 	 T_eff/K 	 logg/dex 	 [Fe/H]/dex 	 Mass/Msol 	 Age/Gyr 	 Radius/Rsol 	 [Fe/H]_new/dex 	 [Mg/Fe]/dex 	 [Ti/Fe]/dex 	 [Mn/Fe]/dex 
#    np.savetxt(f"Stars_Lhood_combined_spec_phot/Gaia_benchmark/{stellar_id}/best_fit_params_errors_{stellar_id}_metals.txt",best_fit_params_errors_metals,fmt='%s',header='Grid Type 	\t neg_sigma_T_eff/K 	\t pos_sigma_T_eff/K 	\t neg_sigma_logg/dex 	\t pos_sigma_logg/dex 	\t neg_sigma_[Fe/H]/dex 	\t pos_sigma_[Fe/H]/dex 	\t neg_sigma_Mass/Msol 	\t pos_sigma_Mass/Msol 	\t neg_sigma_Age/Gyr 	\t pos_sigma_Age/Gyr 	\t neg_sigma_Radius/Rsol 	\t pos_sigma_Radius/Rsol 	\t neg_sigma_FeH_new/dex 	\t pos_sigma_FeH_new/dex 	\t neg_sigma_MgFe_new/dex 	\t pos_sigma_MgFe_new/dex 	 \t neg_sigma_TiFe_new/dex 	 \t pos_sigma_TiFe_new/dex 	 \t neg_sigma_MnFe_new/dex 	 \t pos_sigma_MnFe_new/dex') # Order #  Grid Type 	\t neg_sigma_T_eff/K 	\t pos_sigma_T_eff/K 	\t neg_sigma_logg/dex 	\t pos_sigma_logg/dex 	\t neg_sigma_[Fe/H]/dex 	\t pos_sigma_[Fe/H]/dex 	\t neg_sigma_Mass/Msol 	\t pos_sigma_Mass/Msol 	\t neg_sigma_Age/Gyr 	\t pos_sigma_Age/Gyr 	\t neg_sigma_Radius/Rsol 	\t pos_sigma_Radius/Rsol 	\t neg_sigma_FeH_new/dex 	\t pos_sigma_FeH_new/dex 	\t neg_sigma_MgFe_new/dex 	\t pos_sigma_MgFe_new/dex 	 \t neg_sigma_TiFe_new/dex 	 \t pos_sigma_TiFe_new/dex 	 \t neg_sigma_MnFe_new/dex 	 \t pos_sigma_MnFe_new/dex
    np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/{stellar_id}/best_fit_params_{stellar_id}_metals.txt",best_fit_params_metals,fmt='%s',header='Grid Type 	 \t P_best 	 \t T_eff/K 	 \t logg/dex 	 \t [Fe/H]/dex \t [Fe/H]_new/dex 	 \t [Mg/Fe]/dex 	 \t [Ti/Fe]/dex 	 \t [Mn/Fe]/dex 	 \t Mass/Msol 	 \t Age/Gyr 	 \t Radius/Rsol') # Order #  Grid Type 	 P_best 	 T_eff/K 	 logg/dex 	 [Fe/H]/dex 	 Mass/Msol 	 Age/Gyr 	 Radius/Rsol 	 [Fe/H]_new/dex 	 [Mg/Fe]/dex 	 [Ti/Fe]/dex 	 [Mn/Fe]/dex 
    np.savetxt(f"../Output_data/Stars_Lhood_combined_spec_phot/{stellar_id}/best_fit_params_errors_{stellar_id}_metals.txt",best_fit_params_errors_metals,fmt='%s',header='Grid Type 	\t neg_sigma_T_eff/K 	\t pos_sigma_T_eff/K 	\t neg_sigma_logg/dex 	\t pos_sigma_logg/dex 	\t neg_sigma_[Fe/H]/dex 	\t pos_sigma_[Fe/H]/dex \t neg_sigma_FeH_new/dex 	\t pos_sigma_FeH_new/dex 	\t neg_sigma_MgFe_new/dex 	\t pos_sigma_MgFe_new/dex 	 \t neg_sigma_TiFe_new/dex 	 \t pos_sigma_TiFe_new/dex 	 \t neg_sigma_MnFe_new/dex 	 \t pos_sigma_MnFe_new/dex 	\t neg_sigma_Mass/Msol 	\t pos_sigma_Mass/Msol 	\t neg_sigma_Age/Gyr 	\t pos_sigma_Age/Gyr 	\t neg_sigma_Radius/Rsol 	\t pos_sigma_Radius/Rsol') # Order #  Grid Type 	\t neg_sigma_T_eff/K 	\t pos_sigma_T_eff/K 	\t neg_sigma_logg/dex 	\t pos_sigma_logg/dex 	\t neg_sigma_[Fe/H]/dex 	\t pos_sigma_[Fe/H]/dex 	\t neg_sigma_Mass/Msol 	\t pos_sigma_Mass/Msol 	\t neg_sigma_Age/Gyr 	\t pos_sigma_Age/Gyr 	\t neg_sigma_Radius/Rsol 	\t pos_sigma_Radius/Rsol 	\t neg_sigma_FeH_new/dex 	\t pos_sigma_FeH_new/dex 	\t neg_sigma_MgFe_new/dex 	\t pos_sigma_MgFe_new/dex 	 \t neg_sigma_TiFe_new/dex 	 \t pos_sigma_TiFe_new/dex 	 \t neg_sigma_MnFe_new/dex 	 \t pos_sigma_MnFe_new/dex
    
        
#        if grid_type >= 5:
#            print(best_fit_params[grid_type][0],-0.04,feh_spec_1st_en,feh_best,feh_spec_2nd_en,feh_new_err) # All metallicities compared to eachother
    
#find_best_val_fix(['55058', [5886.525, 4.403, 0.047], False])     
#find_best_val_fix(['55229', [5079.87339229293, 4.581391555555556, -0.2399931536509208], False])  5.03  3.69 -0.00  
#find_best_val_fix(['55229', [5032.359017866728, 3.6865300193581643, -0.0030059361232899207], False])  

#start_time = time.time()  

#param_fix = [5890,4,0.05]
#stellar_inp = [55218,param_fix,False]
#best_values = find_best_val_fix(stellar_inp)
#print(best_values)

#Gaia_obs_data = np.loadtxt('../photometry_observation_data/Gaia_benchmark_stars_data_list.txt',dtype=str)
#
#stellar_ids = Gaia_obs_data[:,1] # Target IDs for stars
#stellar_ids = Gaia_obs_data[:,0] # Names of stars
#stellar_indexes = Gaia_obs_data[:,2] # indexes associated with ids
#
#fix_feh_bool = True
#
#stellar_inp = []
#
#for inp_index in range(len(Gaia_obs_data)):
#    
#    stellar_inp.append([stellar_ids[inp_index],stellar_ids[inp_index],stellar_indexes[inp_index],fix_feh_bool])
#
#stellar_inp = stellar_inp[4:]

#recalc_metals(stellar_inp[0])

#num_processor=4 # number of processors to be used for multiprocessing
#
#if __name__ == '__main__':
#    p = mp.Pool(num_processor) # Pool distributes tasks to available processors using a FIFO schedulling (First In, First Out)
#    p.map(recalc_metals,stellar_inp) # You pass it a list of the inputs and it'll iterate through them
#    p.terminate() # Terminates process 

#print(f"Time Elapsed -- {time.time()-start_time} --- seconds")




