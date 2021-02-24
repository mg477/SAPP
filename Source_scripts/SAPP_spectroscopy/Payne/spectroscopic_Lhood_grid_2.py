#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:25:30 2019

@author: gent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:18:20 2019

@author: gent
"""

import numpy as np
# import Payne.test2obs6runclean_matt as mspec_hr10
# import Payne_hr21.test2obs6runclean21_hr21original as mspec_hr21

import SAPP_spectroscopy.Payne.SAPP_best_spec_payne_v1p1 as mspec_new
# import SAPP_best_spec_payne_v1p1 as mspec_new

# import SAPP_spectroscopy.Payne.astro_constants_cgs as astroc

# import astro_constants_cgs as astroc
import time
import os
import psutil
#from Payne_hr21.convolve import conv_res
import matplotlib.pyplot as plt

# APPLYING THIS RESIDUAL RULE TO EVERY SINGLE SPECTRA MIGHT NOT BE CORRECT
# E.G. FOR BAD MODELS I.E. ONES WHERE THE INPUT PARAMETERS ARE FAR FROM BEST FIT VALUES, OF COURSE THE RESIDUAL WILL BE LARGER THAN
# THE THRESHOLD, THUS THE FURTHER AWAY YOU GET FROM THE OBSERVATIONS, THE MORE POINTS IN THE LHOOD SPACE WILL HAVE THEIR ERRORS INCREASED
# I THINK THAT MAYBE IT SHOULD ONLY BE APPLIED TO THE BEST FIT CHISQ
# FIND THE ERRORS WHERE THE MINIMUM CHISQ DEVIATES TOO MUCH (IT SHOULD MATCH VERY WELL AS IT IS BEST FIT)
# THEN FIX AND USE THESE NEW ERRORS FOR ALL OTHER POINTS IN PARAMETER SPACE

np.set_printoptions(suppress=True,formatter={'float': '{: 0.2f}'.format})


def Lhood_norm_red_chi(obs_norm,err_norm,params_inp,wavelength_input):
    
    """
    obs_norm: normalised observational parameters from results function
    err_norm: normalised errors for observational parameters from results function
    params_inp: input model parameters in 'human format'
    return: Likelihood value based on reduced chi-squared of parameter space
    purpose: this calculates a likelihood value by calculating the reduced
    chi-squared of the parameter space between the normalised observational values
    and the normalised model values.
    """
    
    params_norm = (params_inp-x_min)/(x_max-x_min)-0.5 # normalises parameters to 'Payne' values
    
    # model_norm = mspec_hr21.restore([],*params_norm) # normalised model values to be compared to observational
    model_norm = mspec_new.restore(wavelength_input,*params_norm) # normalised model values to be compared to observational
    
    """
    What makes this different to fixed code?
    We're essentially fixing the parameters here aren't we?
    Fixed code refits with some fixed parameters
    here, no refitting occurs, we just want the model spec from given params
    """
    
    nu = len(obs_norm)-len(params_norm) # degrees of freedom
    
    chi_2_red = np.sum(((obs_norm-model_norm)/err_norm)**2/nu) # reduced chisq for given point in parameter space
    
    #    print("reduced chisq after reduction ",chi_2_red)

    #    Lhood = np.exp((1-chi_2_red)/2) # Likelihood normalised to value of 1 when chisq_red = 1 i.e. best fit
    Lhood = np.exp((-chi_2_red)/2) # Likelihood normalised to value of 1 when chisq_red = 1 i.e. best fit
    
#    if Lhood > 0.76:
#    
#        print("GRID PARAMS =",params_inp)
#        print("GRID CHISQ =",chi_2_red)
#        print("GRID LHOOD =",Lhood)
    
    return Lhood

#@jit
def grid_sampling(Ngrid,T_range,logg_range,FeH_range,fixed_params,obs_norm,err_norm,stellar_id,wavelength_input):
    
    """
    Ngrid: This is an array of the grid dimesions being created in order of [Fe/H], logg, Teff e.g. Ngrid = [50,50,50]
    """
    
    MgFe = fixed_params[2] # MgFe for hr10 model and hr21 model (Mikhail's label)
#    MgFe = fixed_params[5] # MgFe for hr21 model (Jeff's label)
        
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
                
                params_inp = np.hstack(([T_range[k],logg_range[j],FeH_range[i]-(0.22/0.4 * MgFe)],fixed_params))
                
                # print(params_inp)
                
                Lhood = Lhood_norm_red_chi(obs_norm,err_norm,params_inp,wavelength_input) 
                
                param_space.append([T_range[k],logg_range[j],FeH_alpha_enhanced,Lhood]) # This builds up 1D arrays
        
        print(f"Point {i+1} in parameter space [Fe/H] = {FeH_alpha_enhanced} for Star ID{stellar_id}") # rearrange such that FeH iterates through i
    
    param_space = np.array(param_space)
    
    return param_space


# def spec_lhood(stellar_id_arr):
def spec_lhood(ind_spec_arr):
    
    # stellar_id = stellar_id_arr[0] # Literature name of the star
    # spec_id = int(float(stellar_id_arr[1])) # Spectroscopic id index
    
    # spec_path = "../Input_data/spectroscopy_observation_data/" + ind_spec_arr[0]

    """
    Below is where the spectral information is fed in, this function should be tailored to the specific type of file
    and so can be easily changed. This information is all standard.
    """
    
    ### Gaia-ESO fit files
    
    # wavelength,\
    # flux_norm,\
    # error_med,\
    # rv_shift,\
    # rv_shift_err,\
    # snr_star,\
    # flux_norm_sigma_ave,\
    # flux_sigma_ave = read_fits_GES_GIR_spectra(spec_path) 
    
    ### Text files

    # wavelength,\
    # flux_norm,\
    # error_med,\
    # rv_shift,\
    # rv_shift_err,\
    # snr_star = mspec_new.read_txt_spectra(spec_path,rv_calc=True)
    
    ### HARPS/UVES fit files
    
    # wavelength,\
    # flux_norm,\
    # error_med,\
    # rv_shift,\
    # rv_shift_err,\
    # snr_star = read_fits_spectra(spec_path)    
    
    # spec_residual_multiplier = float(ind_spec_arr[11])
    # spec_type_string = ind_spec_arr[12]
    
    # stellar_id = ind_spec_arr[13] 
    
    
    # best_fit_spec_input = ind_spec_arr[:11]

    best_fit_spec_input = ind_spec_arr[0]
    
#    stellar_name = stellar_id_arr[3] # Literature name of the star (not RA+DEC)
    
    # stellar_id = stellar_id.replace(" ","_")
    
    stellar_id = ind_spec_arr[1]
    
    spec_type = ind_spec_arr[2]
    
    phot_limits = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/{stellar_id}/phot_interp_space_limits_{stellar_id}.txt")

    temp_max_limit = phot_limits[0][0]
    temp_min_limit = phot_limits[0][1]
    logg_max_limit = phot_limits[1][0]
    logg_min_limit = phot_limits[1][1]
    feh_max_limit = phot_limits[2][1]
    feh_min_limit = phot_limits[2][0]
    
    best_fit_spectroscopy = mspec_new.find_best_val(best_fit_spec_input)     
    
# [final,efin_upper,efin_lower,rv_shift,ch2_save,wvl_corrected,obs,fit,snr_star,wvl_obs_input,usert]
    
    # best_fit_spectroscopy = mspec_hr21.find_best_val(spec_id) 
    # result=best_fit_spectroscopy[0] # results for best fit 
    params = best_fit_spectroscopy[0]
    ch2_best = best_fit_spectroscopy[4] # chi-square of best fit results
    wvl_con = best_fit_spectroscopy[5]
    wvl_obs_input = best_fit_spectroscopy[9]
    
    # params=result[:num_labels] # these are the best fit params

                
    directory = stellar_id
    directory_check = os.path.exists(f"../Output_data/Stars_Lhood_spectroscopy/{stellar_id}")
    
    if directory_check == True:
        
        print(f"../Output_data/Stars_Lhood_spectroscopy/{stellar_id} directory exists")
        
    else:

        print(f"../Output_data/Stars_Lhood_spectroscopy/{stellar_id} directory does not exist")
        
        os.makedirs(f"../Output_data/Stars_Lhood_spectroscopy/{directory}")
        
        print(f"../Output_data/Stars_Lhood_spectroscopy/{stellar_id} directory has been created")        


    np.savetxt(f'../Output_data/Stars_Lhood_spectroscopy/{stellar_id}/best_fit_spec_params_starID{stellar_id}_{spec_type}',params)
        
    # obs_norm=result[num_labels:-len(w0)] # is w0 suppose to be how long the flag is?

    obs_norm = best_fit_spectroscopy[6]
    
    # err_norm=result[-len(w0):]

    err_norm = best_fit_spectroscopy[10]
    
    params_norm = (params-x_min)/(x_max-x_min)-0.5 # normalises parameters to 'Payne' values

    model_norm = mspec_new.restore(wvl_obs_input,*params_norm)
    
    # model_norm = mspec_hr10.restore([[],wvl_con],*params_norm) # normalised model values to be compared to observational
    # model_norm = mspec_hr21.restore([],*params_norm) # normalised model values to be compared to observational
                
    residual = abs(obs_norm-model_norm)
    
    # Check where the errors for the best fit spectra are bad, then apply this to the observed errors
    
    # if ch2_best > 1.5: # If the reduced chi2 is above 1.5, then inspect the residual for bad lines
    
    #     for res_index in range(len(err_norm)):
        
    #         if residual[res_index] > 0.01:
    
    #             err_norm[res_index] = err_norm[res_index] * spec_residual_multiplier
    
    ch2_best_new = np.sum(((obs_norm-model_norm)/err_norm)**2/(len(obs_norm)-len(params_norm)))
    
    print("BEST PARAMS =",params)
    print("BEST PARAMS RED CHISQ =",ch2_best_new)
    print("BEST PARAMS LHOOD =",np.exp((-ch2_best_new)/2))
    
#    a=b
        
    ### End of calculation ###

    fixed_params = params[3:] # 8 parameters found from best fit, 5 others are kept fixed i.e. vmic,vbrd,Mg,Ti,Mn
       
    Ngrid_phot = np.loadtxt(f"../Output_data/Stars_Lhood_combined_spec_phot/{stellar_id}/phot_interp_space_grid_size_{stellar_id}.txt")
        
    # Ngrid_FeH,Ngrid_Temp,Ngrid_logg
    
    Ngrid_Temp = int(Ngrid_phot[0])
    Ngrid_logg = int(Ngrid_phot[1])
    Ngrid_FeH = int(Ngrid_phot[2])
    
    T_range = np.linspace(temp_max_limit/1000,temp_min_limit/1000,Ngrid_Temp,endpoint=True)
    logg_range = np.linspace(logg_max_limit,logg_min_limit,Ngrid_logg,endpoint=True)
    FeH_range = np.linspace(feh_min_limit,feh_max_limit,Ngrid_FeH,endpoint=True)
    
    Ngrid = [Ngrid_FeH,Ngrid_logg,Ngrid_Temp]

    param_space = grid_sampling(Ngrid,T_range,logg_range,FeH_range,fixed_params,obs_norm,err_norm,stellar_id,wvl_obs_input)
        
    np.save(f'../Output_data/Stars_Lhood_spectroscopy/{stellar_id}/Lhood_space_spectroscopy_tot_starID{stellar_id}_{spec_type}',param_space)
    
    print(f"Star {stellar_id} saved")
    
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    
    print("Memory = {} GB, CPU usage = {} %".format(memoryUse,psutil.cpu_percent()))

    
    return param_space

def plot_lhood_feh_slices(param_space_spec):

    teff_example = param_space_spec[:,0]*1000
    logg_example = param_space_spec[:,1]
    feh_example = param_space_spec[:,2]
    prob_example = param_space_spec[:,3]
    
    max_prob = max(prob_example)
    
    feh_best = feh_example[np.where(prob_example == max_prob)[0]]
    
    feh_example_unique = np.unique(feh_example)
    
    for i in range(len(feh_example_unique)):
        
        feh_slice = feh_example_unique[i]
    
        teff_cut = teff_example[feh_example == feh_slice]
        logg_cut = logg_example[feh_example == feh_slice]
        prob_cut = prob_example[feh_example == feh_slice]
        # feh_cut = feh_example[feh_example == feh_best]

        fig_test = plt.figure()
        ax_test = fig_test.add_subplot(111)
        
        if feh_slice == feh_best:
            
            plt.title(f"Best [Fe/H] = {feh_best[0]}")
            
        else:
            
            plt.title(f"[Fe/H] = {feh_slice:.2f}")
            
        # c1 = ax_test.scatter(teff_example,logg_example,c=np.log10(prob_example),cmap='jet',s=100,marker='s')
        
        c1 = ax_test.scatter(teff_cut,logg_cut,c=np.log10(prob_cut),cmap='jet',s=100,marker='s',vmin=-15,vmax=0)
        # c1 = ax_test.scatter(teff_cut,logg_cut,c=prob_cut,cmap='jet',s=100,marker='s',vmin=1e-15,vmax=0)
        
        plt.colorbar(c1,label='Log10(Prob)')
        # plt.colorbar(c1,label='Prob')
        
        plt.show()


# start_time = time.time()

### for single use

# import_path = ""
import_path = "../Input_data/spectroscopy_model_data/Payne_input_data/"

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

'''
### HARPS 18sco spectra, normalised and convolved already (pre-process)

spec_path = '../../../Input_data/test_spectra/18_sco/ADP_18sco_snr396_HARPS_17.707g_error_synth_flag_True_cont_norm_convolved_hr10_.txt'
error_map_spec_path = '../../../Input_data/test_spectra/18_sco/ADP_18sco_snr396_HARPS_17.707g_error_synth_flag_True_cont_norm_convolved_hr10_.txt'
error_mask_index = 0

error_mask_recreate_bool = True
error_map_use_bool = True
cont_norm_bool = False
rv_shift_recalc = False
conv_instrument_bool = False
input_spec_resolution = 21000
numax_iter_bool = True
niter_numax_MAX = 5
numax_input_arr = [3170,159,niter_numax_MAX]
recalc_metals_bool = False
feh_recalc_fix_bool = False
recalc_metals_inp = [5770,4.44,0,feh_recalc_fix_bool] 

ind_spec_arr = [spec_path,\
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
                recalc_metals_inp]
    
    
param_space_spec = spec_lhood(ind_spec_arr)

plot_lhood_feh_slices(param_space_spec)
'''

 
#stellar_id_list = np.loadtxt('../photometry_observation_data/stellar_id_combined_photometry_INDEXES',dtype='str') # This is the list of stellar id's for cluster NGC2420
#stellar_index_correlation_spec_list = np.loadtxt('../selection_star_indexes_NGC2420',usecols=0).astype(int) # astype makes sure that the indexes are integers
#stellar_id_correlation_spec_list = stellar_id_list[stellar_index_correlation_spec_list]
#stellar_spec_index_correlation_spec_list = np.loadtxt('../selection_star_indexes_NGC2420',usecols=1).astype(int)

#stellar_Gaia_list = np.loadtxt('../photometry_observation_data/Gaia_benchmark_stars_data_list.txt',dtype=str)
#stellar_spec_index_correlation_spec_list =  stellar_Gaia_list[:,2] # indexes which correspond with Mikhails stars.txt list
#stellar_id_correlation_spec_list = stellar_Gaia_list[:,1] # Target ID RA+DEC with respect to the indexes
#stellar_names_list = stellar_Gaia_list[:,0] # Names of the stars
#spec_residual_multiplier_arr = [3.45,1.4,1.5,3.4,3.3]
#
#stellar_id_inp = []
#
#for i in range(0,len(stellar_Gaia_list)): # creates a 2d matrix giving columns of target IDs and star indexes
#    
#    stellar_id_inp.append([stellar_id_correlation_spec_list[i],stellar_spec_index_correlation_spec_list[i],spec_residual_multiplier_arr[i],stellar_names_list[i]])
    

############################### REQUIRED FROM MIKHAILS CODE ####################    


#name="NN_results_RrelsigL20.npz"#LTE
#name="NN_results_RrelsigN20.npz"#NLTE

### THIS IS FOR HR21 OKAY
'''
import_path = "Payne_hr21/"
#import_path = ""

name = import_path+ 'NN_results_1.npz' # Mikhail hr21 grid 

temp=np.load(name)

w_array_0 = temp["w_array_0"]
w_array_1 = temp["w_array_1"]
w_array_2 = temp["w_array_2"]
b_array_0 = temp["b_array_0"]
b_array_1 = temp["b_array_1"]
b_array_2 = temp["b_array_2"]
x_min = temp["x_min"]
x_max = temp["x_max"]

#number of parameters in Payne model
num_labels=w_array_0.shape[1]

#wavelength scale of the models
w0=temp["wvl"]

fn='test-idr6all123norm.npz'
#temp=np.load(path+fn)
temp=np.load(import_path+fn)

wvl=temp["w_lam"][12877:]

#wvl = wvl[wvl<=8800]
snr_rvs = np.loadtxt(import_path+"snr_rvs_hr21.txt")
snr = snr_rvs[0]
rvs = snr_rvs[1]
hdfile =import_path+ "GaiaESO_Giraffe_hr21.h5"
#hdfile = "GaiaESO_Giraffe_RVS.h5"
stars = np.loadtxt(import_path+"stars_hr21.txt",dtype=str)
#to mask smth
masking=False
#lines to be masked
wgd = [8498,8542.1,8727.38,8662.15,8736.1,8752.08,8764.08]
'''

### THIS IS FOR Gaia RVS OKAY
'''
import_path = "Payne_hr21/"
#import_path = ""

name = import_path+ 'NN_results_Gaia_RVS_grid_high_sample.npz' # Mikhail hr21 grid 

temp=np.load(name)

w_array_0 = temp["w_array_0"]
w_array_1 = temp["w_array_1"]
w_array_2 = temp["w_array_2"]
b_array_0 = temp["b_array_0"]
b_array_1 = temp["b_array_1"]
b_array_2 = temp["b_array_2"]
x_min = temp["x_min"]
x_max = temp["x_max"]

#number of parameters in Payne model
num_labels=w_array_0.shape[1]

#wavelength scale of the models
w0=temp["wvl"]

fn='test-idr6all123norm.npz'
#temp=np.load(path+fn)
temp=np.load(import_path+fn)

wvl=temp["w_lam"][12877:]

wvl = wvl[wvl<=8800]

snr_rvs = np.loadtxt(import_path+"snr_rvs_hr21.txt")
snr = snr_rvs[0]
rvs = snr_rvs[1]
#hdfile =import_path+ "GaiaESO_Giraffe_hr21.h5"
hdfile = import_path+ "GaiaESO_Giraffe_RVS.h5"
stars = np.loadtxt(import_path+"stars_hr21.txt",dtype=str)
#to mask smth
masking=False
#lines to be masked
wgd = [8498,8542.1,8727.38,8662.15,8736.1,8752.08,8764.08]
'''


#hdfile=import_path+"idr6hr101521all.h5"
#
##file with names snr, rv from crossvalidation non 
##names are in python 2.7 string format !!!!! will not work with python 3
#fn=import_path+"test-idr6all1norm.npz"
##temp=np.load(path+fn)
#temp=np.load(fn)
#
#stars_1=temp["stars"]
#np.savetxt("stars.txt",stars_1,fmt='%s')
#
#snr=temp["snr"][0]#[:1422]
#rvs=temp['rvs'][0]
#wvl=temp["w_lam"]
#
#fn=import_path+'test-idr6all123norm.npz'
##temp=np.load(path+fn)
#temp=np.load(fn)
#
##Y_u_all=temp['Y_u_test']#.T
##Y_u_alle=temp['Y_u_test_err']#.T
##Y_u_alle[4853:4866]=100
#snr=temp["snr"][2]#[:1422]
#rvs=temp['rvs'][2]#*0
#wvl=temp["w_lam"][12877:]#[pixs]
##Vsini estimates
#rots=temp["labels"][:,[3,7,11]]
#rots=rots.T
#rots=rots[-1]
#
#idxes=np.arange(len(snr))
##to select spectra of good quality
#i21=snr>30
#idxes=idxes[i21]
#stars=temp["stars"]#[i21]
#stars_2 = stars
#np.savetxt("stars_2.txt",stars_2,fmt='%s')

'''
# THIS IS FOR HR10 OKAY

import_path = "Payne/"
#import_path = ""


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

#file with observations
#testDR3public3rest.hdf5 testDR3public2.hdf5
#hdfile="testDR3public3rest.hdf5"
#hdfile="testDR3public3rest1.hdf5"#not normalised 
hdfile="idr6hr10all.h5"
hdfile = import_path + hdfile 

#file with names snr, rv from crossvalidation non 
#names are in python 2.7 string format !!!!! will not work with python 3
fn="test-idr6all1norm.npz"
temp=np.load(import_path + fn)
stars=temp["stars"]
#to run with python3
stars=np.array([it.decode("utf8") for it in stars ])
snr=temp["snr"][0]#[:1422]
labels=temp["labels"][:,:4]
rvs=temp['rvs'][0]
wvl=temp["w_lam"]
'''
