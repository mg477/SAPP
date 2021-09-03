#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:26:13 2020

@author: gent
"""

from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
from scipy import ndimage
# import Payne.astro_constants_cgs as astroc
# import astro_constants_cgs as astroc
from astropy.io import fits as pyfits
import scipy.interpolate as sci
# from scipy import stats
import SAPP_spectroscopy.Payne.astro_constants_cgs as astroc

from SAPP_spectroscopy.Payne.continuum_norm_spectra import continuum_normalise_spectra
# from continuum_norm_spectra import continuum_normalise_spectra

from matplotlib.pyplot import rc
import time

rc('text', usetex=False)
plt.rcParams["font.family"] = "Times New Roman"

np.set_printoptions(suppress=True,formatter={'float': '{: 0.5f}'.format})

def correlation_from_covariance(covariance):
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    
    return correlation

# define sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def lrlu(z):
    z[z<0]*=1e-2
    return z

def relu(z):
    z[z<0.0]=0.0
    return z

def mad(arr):
    """Calculate median absolute deviation"""
    arr = np.array(arr)
    med = np.median(arr)
    return np.median(abs(arr-med))


def sigclip(data, sig):
    """Sigma clip the data"""
    n = len(data)

    start = 0
    step = 500
    end = start + step

    dataout = np.zeros(n)
    while end < n:
        data2 = data[start:end]
        m = np.median(data2)
        s = mad(data2)

        idx = data[start:end] > m+sig*s # how does this line work?
        dataout[start:end][~idx] = data[start:end][~idx] # what does ~ operator do?
        for i in range(start, end):
            if data[i] > m+sig*s:
                dataout[i] = dataout[i-1]
        start = end
        end = start+step

    data2 = data[start:n]
    m = np.median(data2)
    s = mad(data2)
    for i in range(start,n):
        if data[i] > m+sig*s:
            dataout[i] = data[i-1]

    idx = data[start:n] > m+sig*s
    dataout[start:n][~idx] = data[start:n][~idx]
    return dataout


#function that restore payne model spectrum apply doppler shift and normalisation
# def restore1(wvl_arr, *labels):
#     #part of spectrum    
    
#     wvl_min_bool = wvl_arr[0]
#     wvl_max_bool = wvl_arr[1]
#     wvl_obs = wvl_arr[2:] # this needs to be used to cut pred flux, it is already been interpolated and shifted to rest frame
        
#     #1st layer
#     if cheb == 0:
#         first=relu(np.dot(w_array_0,labels[1:])+ b_array_0)        
#     elif cheb >= 1:
#         first=relu(np.dot(w_array_0,labels[1:-cheb])+ b_array_0)

#     #2nd layer
#     snd=sigmoid(np.dot(w_array_1,first) + b_array_1)
    
#     #3nd layer
#     trd=(np.dot(w_array_2,snd) + b_array_2)
#     predict_flux = trd

#     # print("spec restore step : time elapsed ----",time.time()-start_time,"----- seconds ----")

#     if cheb == 0:
        
#         params_norm = np.array(labels[1:])
        
#         params_un_norm = (params_norm + 0.5)*(np.array(x_max)-np.array(x_min)) + np.array(x_min)
        
#         # print(params_un_norm)
        
#         # with open("../Output_data/Stars_Lhood_spectroscopy/HD49933/HD49933_curvefit_collect_diff_start.txt",'a') as the_file:
            
#             # the_file.write(np.array(params_un_norm,dtype=str) +'\n')
#             # the_file.write(str(params_un_norm).replace("[","").replace("]","\n"))
        
        
#     else:
        
#         params_norm = np.array(labels[1:-cheb])
        
#         cfc=np.array(labels[-cheb:])
#         cfc[0]+=1.
#         #multiply with normalisation polynomials
#         cnt=np.dot(gks,cfc)
#         predict_flux*=cnt
            
# #    predict_flux = sigclip(predict_flux, sig=3.0) # sigma clipping model (for now)

#     # print("un-normalise parameters : time elapsed ----",time.time()-start_time,"----- seconds ----")

#     w0_temp = w0

#     if len(wvl_obs) > 0: # then we can do stuff with it
    
#         if int(wvl_min_bool) == 1: # cut model min to obs min
            
#             predict_flux = predict_flux[w0_temp>=min(wvl_obs)]
#             w0_temp = w0_temp[w0_temp>=min(wvl_obs)]
            
#         if int(wvl_max_bool) == 1: # cut model max to obs max
        
#             predict_flux = predict_flux[w0_temp<=max(wvl_obs)]
#             w0_temp = w0_temp[w0_temp<=max(wvl_obs)]

#     # print("model limit edit : time elapsed ----",time.time()-start_time,"----- seconds ----")
                                        
#     return predict_flux

def restore1(wvl_arr, *labels):
    #part of spectrum    
    
    wvl_min_bool = wvl_arr[0]
    wvl_max_bool = wvl_arr[1]
    wvl_obs = wvl_arr[2:] # this needs to be used to cut pred flux, it is already been interpolated and shifted to rest frame
        
    
    # dummy = list(labels[1:])
        
    # fix=1#logg index
    # logg_fix_norm = (4.44-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
    
    # dummy[1] = logg_fix_norm

    # labels_new = dummy[0],logg_fix_norm,dummy[2],dummy[3],dummy[4],dummy[5],dummy[6],dummy[7]
    # labels_new = dummy[0],dummy[1],dummy[2],dummy[3],dummy[4],dummy[5],dummy[6],dummy[7]
    
    # print(labels_new)
    
    labels_new = labels[1:]
    
    #1st layer
    
    # first=relu(np.dot(w_array_0,labels[1:])+ b_array_0)    
    
    first=relu(np.dot(w_array_0,labels_new)+ b_array_0)        


    #2nd layer
    snd=sigmoid(np.dot(w_array_1,first) + b_array_1)
    
    #3nd layer
    trd=(np.dot(w_array_2,snd) + b_array_2)
    predict_flux = trd

    w0_temp = w0

    if len(wvl_obs) > 0: # then we can do stuff with it
    
        if int(wvl_min_bool) == 1: # cut model min to obs min
            
            predict_flux = predict_flux[w0_temp>=min(wvl_obs)]
            w0_temp = w0_temp[w0_temp>=min(wvl_obs)]
            
        if int(wvl_max_bool) == 1: # cut model max to obs max
        
            predict_flux = predict_flux[w0_temp<=max(wvl_obs)]
            w0_temp = w0_temp[w0_temp<=max(wvl_obs)]


    # with open("../../../Output_data/curvefit_iteration_results/curvefit_collect_numax_start.txt",'a') as the_file:
    # with open("../../../Output_data/curvefit_iteration_results/curvefit_collect_central_start.txt",'a') as the_file:
    
        # the_file.write(str(params_un_norm).replace("[","").replace("]","\n"))

    
    # print("1")
    
    # num_print = str([1])
    
    # print(num_print +'\n')
    
    # with open("../../../Output_data/curvefit_iterations.txt",'a') as the_file:
        
        # the_file.write(np.array(num_print,dtype=str) +'\n')
        # the_file.write(str(num_print).replace("[","").replace("]","\n"))

        # the_file.write(num_print +'\n')
        # the_file.write(num_print.replace("[","").replace("]","\n"))

    params_norm = np.array(labels_new)
        
    params_un_norm = (params_norm + 0.5)*(np.array(x_max)-np.array(x_min)) + np.array(x_min)

    # print(params_un_norm)

        
    return predict_flux

def restore1_test(input_dict, *labels):
    #part of spectrum    
    
    wvl_min_bool = input_dict["wvl_min_bool"]
    wvl_max_bool = input_dict["wvl_max_bool"]
    wvl_obs = input_dict["wvl_to_cut"] # this needs to be used to cut pred flux, it is already been interpolated and shifted to rest frame
    
    teff_fix = input_dict["Teff_fix"]
    logg_fix = input_dict["logg_fix"]
    feh_fix = input_dict["feh_fix"]
    vmic_fix = input_dict["vmic_fix"]
    vbrd_fix = input_dict["vbrd_fix"]
    mgfe_fix = input_dict["mgfe_fix"]
    tife_fix = input_dict["tife_fix"]
    mnfe_fix = input_dict["mnfe_fix"]
        
    dummy = list(labels[1:])
    
    if np.isnan(teff_fix) == False:
        
        # fix this paramter
        
        fix=0 #teff index
        # dummy[fix] = (teff_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     

        dummy_add = (teff_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     
        
        dummy = dummy_add,dummy[0],dummy[1],dummy[2],dummy[3],dummy[4],dummy[5],dummy[6]

    if np.isnan(logg_fix) == False:
        
        # fix this paramter
        
        fix=1 #logg index
        
        # dummy[fix] = (logg_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     

        dummy_add = (logg_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     
        
        dummy = dummy[0],dummy_add,dummy[1],dummy[2],dummy[3],dummy[4],dummy[5],dummy[6]

    if np.isnan(feh_fix) == False:
        
        # fix this paramter
        
        fix=2 #feh index
        dummy[fix] = (feh_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     
    
    if np.isnan(vmic_fix) == False:
        
        # fix this paramter
        
        fix=3 #vmic index
        dummy[fix] = (vmic_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     

    if np.isnan(vbrd_fix) == False:
        
        # fix this paramter
        
        fix=4 #vbrd index
        dummy[fix] = (vbrd_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     

    if np.isnan(mgfe_fix) == False:
        
        # fix this paramter
        
        fix=5 #mgfe index
        dummy[fix] = (mgfe_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     

    if np.isnan(tife_fix) == False:
        
        # fix this paramter
        
        fix=6 #tife index
        dummy[fix] = (tife_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     

    if np.isnan(mnfe_fix) == False:
        
        # fix this paramter
        
        fix=7 #mnfe index
        dummy[fix] = (mnfe_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5     



    labels_new = dummy[0],dummy[1],dummy[2],dummy[3],dummy[4],dummy[5],dummy[6],dummy[7]
    
    # print(labels_new)
    
    # labels_new = dummy
    
    #1st layer
    
    # first=relu(np.dot(w_array_0,labels[1:])+ b_array_0)    
    
    first=relu(np.dot(w_array_0,labels_new)+ b_array_0)        


    #2nd layer
    snd=sigmoid(np.dot(w_array_1,first) + b_array_1)
    
    #3nd layer
    trd=(np.dot(w_array_2,snd) + b_array_2)
    predict_flux = trd

    w0_temp = w0

    if len(wvl_obs) > 0: # then we can do stuff with it
    
        if int(wvl_min_bool) == 1: # cut model min to obs min
            
            predict_flux = predict_flux[w0_temp>=min(wvl_obs)]
            w0_temp = w0_temp[w0_temp>=min(wvl_obs)]
            
        if int(wvl_max_bool) == 1: # cut model max to obs max
        
            predict_flux = predict_flux[w0_temp<=max(wvl_obs)]
            w0_temp = w0_temp[w0_temp<=max(wvl_obs)]

    params_norm = np.array(labels_new)
        
    params_un_norm = (params_norm + 0.5)*(np.array(x_max)-np.array(x_min)) + np.array(x_min)

    # with open("../../../Output_data/curvefit_iteration_results/curvefit_collect_numax_start.txt",'a') as the_file:
    with open("../../../Output_data/curvefit_iteration_results/curvefit_collect_central_start.txt",'a') as the_file:
    
        the_file.write(str(params_un_norm).replace("[","").replace("]","\n"))

    
    # print("1")
    # 
    # num_print = str([1])
    
    # print(num_print +'\n')
    
    # with open("../../../Output_data/curvefit_iteration_results/curvefit_iterations.txt",'a') as the_file:
        
        # the_file.write(np.array(num_print,dtype=str) +'\n')
        # the_file.write(str(num_print).replace("[","").replace("]","\n"))

        # the_file.write(num_print +'\n')
        # the_file.write(num_print.replace("[","").replace("]","\n"))


    # print(params_un_norm)

        
    return predict_flux


#function that restore payne model spectrum apply doppler shift and normalisation
def restore(wvl_arr, *labels):
    #part of spectrum

    wvl_min_bool = wvl_arr[0]
    wvl_max_bool = wvl_arr[1]
    wvl_obs = wvl_arr[2:] # this needs to be used to cut pred flux, it is already been interpolated and shifted to rest frame

    #1t layer
    # first=relu(np.dot(w_array_0,labels[1:])+ b_array_0) # there wasn't a 1 here before
    first=relu(np.dot(w_array_0,labels[:])+ b_array_0) # there wasn't a 1 here before
    #2d layer
    snd=sigmoid(np.dot(w_array_1,first) + b_array_1)
    #3d layer
    trd=(np.dot(w_array_2,snd) + b_array_2)
    predict_flux = trd
    
    w0_temp = w0

    if len(wvl_obs) > 0: # then we can do stuff with it
    
        if int(wvl_min_bool) == 1: # cut model min to obs min
            
            predict_flux = predict_flux[w0_temp>=min(wvl_obs)]
            w0_temp = w0_temp[w0_temp>=min(wvl_obs)]
            
        if int(wvl_max_bool) == 1: # cut model max to obs max
        
            predict_flux = predict_flux[w0_temp<=max(wvl_obs)]
            w0_temp = w0_temp[w0_temp<=max(wvl_obs)]
                                        
    return predict_flux

def load_TS_spec(path):
    """loads spectra from TS"""
    
    spec = np.loadtxt(path)
    wavelength = spec[:,0] # angstrom
    flux = spec[:,1] # normalised
    
    return wavelength,flux

def convolve_python(wavelength_input,flux_input,Resolution):

    smpl=(wavelength_input[1]-wavelength_input[0]) # wavelength range must have regular spacing
    fwhm=np.mean(wavelength_input)/Resolution # Resolution is the intended resolution
    flux_convolve=ndimage.filters.gaussian_filter1d(flux_input,fwhm*0.4246609/smpl)
    
    return wavelength_input,flux_convolve

def instrument_conv_res(wave, flux, Resolution):
        
    fwhm = (np.mean(wave)/Resolution) * 1000 # to get into correct units 
    
    #create a file to feed to ./faltbon of spectrum that needs convolving
    
    f = open('original_instrument.txt', 'w')
    spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
    for j in range(len(wave)):
        f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
    f.close()
    os.system("{ echo original_instrument.txt; echo convolve_instrument.txt; echo %f; echo 2; } | ./faltbon" % fwhm)
    
    wave_conv, flux_conv, spud = np.loadtxt("convolve_instrument.txt", unpack='True') #read in our new convolved spectrum
    return wave_conv, flux_conv


def RV_correction_vel_to_wvl_non_rel(waveobs,xRV): 
    
    """ 
    
    This applies RV correction assuming it is known e.g. gravitational redshift
    for the Sun is 0.6 km/s
    
    N.B. xRV is RV correction in km/s 
    
    
    """
    
    waveobs_corr = waveobs.copy()
    
    CCC  = 299792.458 # SPOL km/s
    
    for pixel in range(len(waveobs_corr)):
    
        waveobs_corr[pixel] = waveobs_corr[pixel]/((xRV)/CCC + 1.)
        
    return waveobs_corr # teying to make an array or tuple here

def rv_cross_corelation_no_fft(spec_obs,spec_template,rv_min,rv_max,drv,title,savefig_bool,Resolution,Resolution_convolve):
    
    """
    Calculate RV shift using CC method
    
    Based on PyAstronomy rv_correction documentation
    
    Method does not use FFT method
    
    returns: rv shift, arrays of rv, CC function
    """
    
    wvl = spec_obs[0]
    obs = spec_obs[1]
    usert = spec_obs[2]
    
    spec_template_wvl = spec_template[0] # template needs to have much larger wavelength coverage than obs!!!
    spec_template_flux = spec_template[1]
        
    # spec_template_wvl,spec_template_flux = instrument_conv_res(spec_template_wvl, spec_template_flux, 18000)
    
    if Resolution_convolve == True:
    
        spec_template_wvl,spec_template_flux = convolve_python(spec_template_wvl, spec_template_flux, Resolution)
                    
    drvs = np.arange(rv_min,rv_max+drv,drv)
    
    cc = np.zeros(len(drvs))
    
    ch2_cc = np.zeros(len(drvs))
    
    # Speed of light in km/s
    c = 299792.458
    
    # obs_cut = obs.copy()
    # wvl_cut = wvl.copy()
    # usert_cut = usert.copy()
        
    for i,rv in enumerate(drvs):

        obs_cut = obs.copy()
        wvl_cut = wvl.copy()
        usert_cut = usert.copy()

        spec_template_wvl_doppler_shift = spec_template_wvl*(1.0 + rv/c)
        
        fi = sci.interp1d(spec_template_wvl_doppler_shift, spec_template_flux)
        
        ## annoying to check every time, but this will vary per calculation!
                
        # if max(wvl) > max(spec_template_wvl_doppler_shift):
                
                # cut to max
                
        obs_cut = obs_cut[wvl_cut <=  max(spec_template_wvl_doppler_shift)]
        usert_cut = usert_cut[wvl_cut <=  max(spec_template_wvl_doppler_shift)]
        wvl_cut = wvl_cut[wvl_cut <=  max(spec_template_wvl_doppler_shift)]
                
        # if min(wvl) <  min(spec_template_wvl_doppler_shift):
                
                # cut to model min

        obs_cut = obs_cut[wvl_cut >= min(spec_template_wvl_doppler_shift)]
        usert_cut = usert_cut[wvl_cut >= min(spec_template_wvl_doppler_shift)]
        wvl_cut = wvl_cut[wvl_cut >= min(spec_template_wvl_doppler_shift)]
        
        # model is interpolated onto observed spectrum... this is strange        
        cc[i] = np.sum(obs_cut * fi(wvl_cut)) # is this the literal definition of CC, where's the fourier transform?
        
        
        chi2_rv_arr_i = (fi(wvl_cut) - obs_cut)**2/usert_cut ** 2

        no_inf_index_i = np.isfinite(chi2_rv_arr_i)
        
        chi2_rv_arr_clean_i = chi2_rv_arr_i[no_inf_index_i]
        
        ch2_cc[i] = np.nansum(chi2_rv_arr_clean_i/(len(chi2_rv_arr_clean_i)-8)) # 8 dimensions, duh
        
    # Find the index of maximum cross-correlation function
    max_ind = np.argmax(cc)
        
    ### calc chi2 i.e. how good is the fit?
        
    spec_template_wvl_doppler_shift = spec_template_wvl*(1.0 + drvs[max_ind]/c)
    
    fi_chi = sci.interp1d(spec_template_wvl_doppler_shift, spec_template_flux)
    
    obs_chi = obs.copy()
    usert_chi = usert.copy()
    wvl_chi = wvl.copy()
    
    obs_chi = obs_chi[wvl_chi <=  max(spec_template_wvl_doppler_shift)]
    usert_chi = usert_chi[wvl_chi <=  max(spec_template_wvl_doppler_shift)]
    wvl_chi = wvl_chi[wvl_chi <=  max(spec_template_wvl_doppler_shift)]
            
    obs_chi = obs_chi[wvl_chi >= min(spec_template_wvl_doppler_shift)]
    usert_chi = usert_chi[wvl_chi >= min(spec_template_wvl_doppler_shift)]
    wvl_chi = wvl_chi[wvl_chi >= min(spec_template_wvl_doppler_shift)]
    
    model_flux = fi_chi(wvl_chi)
    
    chi2_rv_arr = (model_flux-obs_chi) ** 2 / usert_chi ** 2
    
    ## clean through inf values
    
    no_inf_index = np.isfinite(chi2_rv_arr)
    
    chi2_rv_arr_clean = chi2_rv_arr[no_inf_index]
    
    chi2_rv = np.nansum(chi2_rv_arr_clean/(len(chi2_rv_arr_clean)-8)) # 8 dimensions, duh
     
    
    
    ### RV plot ###
                
    # title_filename = star_ids_bmk[ind_spec].split("_snr")[0] + "_snr" + star_ids_bmk[ind_spec].split("_snr")[1].split("_")[0]
        
    if savefig_bool == True:

        # print(drvs[max_ind],chi2_rv_arr_clean,chi2_rv)
        
        print("RV due to min chi2",drvs[np.argmin(ch2_cc)],ch2_cc[np.argmin(ch2_cc)])
        print("RV due to max CC",drvs[max_ind],ch2_cc[max_ind])

            
        fig = plt.figure(figsize=(16,8))
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.plot(drvs,cc,'ko',markersize=0.5,label='$\Delta$ RV$_{max}$ ' + f'= {drvs[max_ind]:0.2f} Km/s')
        ax1.set_ylabel("CC ($\Delta$RV)",fontsize=30)
        ax1.set_xlabel("$\Delta$RV [Km/s]",fontsize=30)
        
        ax1.axvline(x=drvs[np.argmin(ch2_cc)],color='y',linestyle=':')
        
        if drvs[max_ind] > 0.0:
            
            ax1.axvline(x=drvs[max_ind],color='tomato',linestyle=':')
    
        else:
            
            ax1.axvline(x=drvs[max_ind],color='b',linestyle=':')
        
        ax1.axvline(x=0,color='g',linestyle=':')
        
        ax1.legend(loc='lower right',fontsize=25)
        
        # convolve model down to hr10
                
        ax2.plot(spec_template_wvl,spec_template_flux,color='gray',linestyle='-',label='Spectral Template')
        ax2.plot(wvl,obs,color='k',linestyle='-',label = f'Obs')
        
        wvl_shift = RV_correction_vel_to_wvl_non_rel(wvl,drvs[max_ind])
    
        ax2.plot(wvl_shift,obs,color='k',linestyle='--',label='Obs RV shifted')
        
        ax2.set_xlim([np.mean(wvl)+100-5,np.mean(wvl)+100+7])
        
        ax2.set_ylabel("Flux",fontsize=30)
        ax2.set_xlabel("$\lambda$ [$\AA$]",fontsize=30)
        
        ax2.legend(loc='lower right',fontsize=25)
        
        # ax2.locator_params(axis="x", nbins=5)
        
        fig.tight_layout()
        
        plt.setp(ax1.spines.values(), linewidth=3)
        plt.setp(ax2.spines.values(), linewidth=3)
        
        plt.title(f'{title}')
        
        # plt.savefig(f"PLATO/rv_correction_figures/{star_spec_name}_{title_filename}_cross_correlation_rv_array.pdf",dpi=250) 
    
        # plt.close("all")
        
        plt.show()
                        
    # return drvs[max_ind],drvs,cc,chi2_rv
    return drvs[np.argmin(ch2_cc)],drvs,cc,chi2_rv

def star_model(wavelength,labels,rv_shift):
    
    """
    takes wavelength and wanted stellar parameters, it spits out model you want
    """
    
    wvl_min_bool = False#min(wavelength)
    wvl_max_bool = False#max(wavelength)
    wvl_to_cut = [] # keep empty
    
    # convert parameters to normalised space 

    labels_norm = (labels-x_min)/(x_max-x_min)-0.5 
    
    # labels_inp = np.hstack(([0],labels_norm))
    # labels_inp = np.hstack((np.zeros([1+cheb]),labels_norm))
    labels_inp = np.hstack((np.zeros([1]),labels_norm,np.zeros([cheb])))
    
    wvl_obs_input = np.hstack((wvl_min_bool,wvl_max_bool,rv_shift,wvl_to_cut))

    model = restore1(wvl_obs_input,*labels_inp) # in this case, the model is convolved from hr21 to RVS, this is good? 

    return w0, model 



def RV_multi_template(wvl,
                      obs,
                      usert,
                      rv_min,
                      rv_max,
                      drv,
                      spec_resolution):
    
    """
    This takes in a hr10 observation and rv initial limits to find the most likely rv
    
    based on four templates 
    
    
    
    """

    # print("=====================")
    # print("SOLAR RV")
                
    model_wvl,model_flux = star_model(wvl,[5.777,4.44,0.00,1,1.6,0,0,0],0)             # this is for hr10 NN trained
    # model_wvl,model_flux = star_model(wvl,[5.777,4.44,0.00,0,0,0,0,0,0,0,0,0,0,1,1,1.6])             # this is for RVS NN trained        
    # model_wvl,model_flux = star_model(wvl,[5.777,4.44,0.00,0,0,0,0,0,0,0,0,0,0])             # this is for 2ND RVS NN trained        

    # plt.plot(model_wvl,model_flux)   
    # plt.show()
    # plt.close('all')


    ## in this case the model wvl range will not be always larger than the observed
    ## this means to interpolate the obs to the model wvl range, you need to make sure
    ## that wvl obs is within the range of the model
    ## as the model will not exist outside its own range, you should NOT extrapolate the model
    ## therefore, you must cut the obs in both ends just in case, only the obs, do not worry about the model
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min,\
                                                          rv_max,\
                                                          drv,\
                                                          title="SOLAR",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)            

        
    # fab, now grab that rough RV value, redo this and choose a range around that
    
    rv_min_focus = rv_shift - 2 # km/s
    rv_max_focus = rv_shift + 2 # km/s
    drv_focus = 0.05 # km/s
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min_focus,\
                                                          rv_max_focus,\
                                                          drv_focus,\
                                                          title="SOLAR",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)
    
    solar_chi2 = chi2_rv
    solar_rv = rv_shift
    solar_rv_err = drv_focus
        
    # now we have to do this again with two other templates 

    ### Solar star METAL POOR RV TEMPLATE ###
    # check the abundances with Maria

    model_wvl,model_flux = star_model(wvl,[5.777,4.44,-2.00,1,1.6,-0.2,-0.2,-0.8],0)             # this is for hr10 NN trained
    # model_wvl,model_flux = star_model(wvl,[5.777,4.44,0.00,0,0,0,0,0,0,0,0,0,0,1,1,1.6])             # this is for RVS NN trained        
    # model_wvl,model_flux = star_model(wvl,[5.777,4.44,0.00,0,0,0,0,0,0,0,0,0,0])             # this is for 2ND RVS NN trained        

    # plt.plot(model_wvl,model_flux)   
    # plt.show()
    # plt.close('all')


    ## in this case the model wvl range will not be always larger than the observed
    ## this means to interpolate the obs to the model wvl range, you need to make sure
    ## that wvl obs is within the range of the model
    ## as the model will not exist outside its own range, you should NOT extrapolate the model
    ## therefore, you must cut the obs in both ends just in case, only the obs, do not worry about the model
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min,\
                                                          rv_max,\
                                                          drv,\
                                                          title="SOLAR MP",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)            

        
    # fab, now grab that rough RV value, redo this and choose a range around that
    
    rv_min_focus = rv_shift - 2 # km/s
    rv_max_focus = rv_shift + 2 # km/s
    drv_focus = 0.05 # km/s
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min_focus,\
                                                          rv_max_focus,\
                                                          drv_focus,\
                                                          title="SOLAR MP",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)
    
    solar_poor_chi2 = chi2_rv
    solar_poor_rv = rv_shift
    solar_poor_rv_err = drv_focus
        


    ### RGB solar metallicty RV TEMPLATE ###

    # print("=====================")
    # print("RGB solar [Fe/H] RV")

                
    model_wvl,model_flux = star_model(wvl,[4.400,1.5,0.00,1,1,0,0,0],0)                  # this is for hr10 NN trained 
    # model_wvl,model_flux = star_model(wvl,[4400,1.5,0.00,0,0,0,0,0,0,0,0,0,0,1,1,1])                  # this is for RVS NN trained 
    # model_wvl,model_flux = star_model(wvl,[4400,1.5,0.00,0,0,0,0,0,0,0,0,0,0])                  # this is for 2nd RVS NN trained 
    
    ## in this case the model wvl range will not be always larger than the observed
    ## this means to interpolate the obs to the model wvl range, you need to make sure
    ## that wvl obs is within the range of the model
    ## as the model will not exist outside its own range, you should NOT extrapolate the model
    ## therefore, you must cut the obs in both ends just in case, only the obs, do not worry about the model
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min,\
                                                          rv_max,\
                                                          drv,\
                                                          title="RGB",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)            

        
    # fab, now grab that rough RV value, redo this and choose a range around that
    
    rv_min_focus = rv_shift - 2 # km/s
    rv_max_focus = rv_shift + 2 # km/s
    drv_focus = 0.05 # km/s
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min_focus,\
                                                          rv_max_focus,\
                                                          drv_focus,\
                                                          title="RGB",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)

    rgb_chi2 = chi2_rv
    rgb_rv = rv_shift
    rgb_rv_err = drv_focus


    ### RGB poor metallicty RV TEMPLATE ###

    # print("=====================")
    # print("RGB poor [Fe/H] RV")
                
    model_wvl,model_flux = star_model(wvl,[4.400,1.5,-2,1,1,-0.2,-0.2,-0.8],0)          # this is for hr10 NN trained         
    # model_wvl,model_flux = star_model(wvl,[4400,1.5,-2,0,0,0,0,0,-0.2,0,0,-0.2,-0.8,1,1,1])          # this is for RVS NN trained      
    # model_wvl,model_flux = star_model(wvl,[4400,1.5,-2,0,0,0,0,0,-0.2,0,0,-0.2,-0.8])          # this is for 2nd RVS NN trained      
    
    ## in this case the model wvl range will not be always larger than the observed
    ## this means to interpolate the obs to the model wvl range, you need to make sure
    ## that wvl obs is within the range of the model
    ## as the model will not exist outside its own range, you should NOT extrapolate the model
    ## therefore, you must cut the obs in both ends just in case, only the obs, do not worry about the model
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min,\
                                                          rv_max,\
                                                          drv,\
                                                          title="RGB MP",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)            

        
    # fab, now grab that rough RV value, redo this and choose a range around that
    
    rv_min_focus = rv_shift - 2 # km/s
    rv_max_focus = rv_shift + 2 # km/s
    drv_focus = 0.05 # km/s
    
    rv_shift,rv_array,cross_corr,chi2_rv = rv_cross_corelation_no_fft([wvl,obs,usert],\
                                                          [model_wvl,model_flux],\
                                                          rv_min_focus,\
                                                          rv_max_focus,\
                                                          drv_focus,\
                                                          title="RGB MP",\
                                                          savefig_bool=False,\
                                                          Resolution=spec_resolution,
                                                          Resolution_convolve = False)

    rgb_mp_chi2 = chi2_rv
    rgb_mp_rv = rv_shift
    rgb_mp_rv_err = drv_focus

    chi2_results = [solar_chi2,solar_poor_chi2,rgb_chi2,rgb_mp_chi2]
    rv_results = [solar_rv,solar_poor_rv,rgb_rv,rgb_mp_rv]
    rv_err_results = [solar_rv_err,solar_poor_rv,rgb_rv_err,rgb_mp_rv_err]

    chi2_best = min(chi2_results)

    rv_best = rv_results[np.argsort(chi2_results)[0]]

    rv_err_best = rv_err_results[np.argsort(chi2_results)[0]]    
    
    return rv_best,rv_err_best


def read_fits_GES_GIR_spectra(path):
        
    hdulist = pyfits.open(path)
        
    flux_norm = hdulist[2].data # this is the normalised spectrum
    
    flux_norm_inverse_variance = hdulist[3].data
    flux_inverse_variance = hdulist[1].data
    
    wav_zero_point = hdulist[0].header['CRVAL1'] # wavelength zero point
    
    wav_increment = hdulist[0].header['CD1_1'] # wavelength increment
    
    wavelength = wavelength = np.arange(0,len(flux_norm))*wav_increment + wav_zero_point
    
    rv_shift = hdulist[5].data['VEL'][0] # 5 is the velocity column, documentation says to use this
    
    rv_shift_err = hdulist[5].data['EVEL'][0] # km/s
    
    SNR_med = hdulist[5].data['S_N'][0] 
    
    error_med = flux_norm/SNR_med # can't seem to find error from fits
    
    return wavelength,flux_norm,error_med,rv_shift,rv_shift_err,SNR_med,np.average(1/flux_norm_inverse_variance**0.5),np.average(1/flux_inverse_variance**0.5)


### error mask for solar doesn't necessarily have same wvl as other spectra
### for some reason...
### shouldn't they all have the same wvl?
### originally they do
### however they are rv corrected by different amounts
### you need to account for this
### error mask in v1 is created from residual between obs and fit both in mod scale and rest frame

### you need to rv correct the spectra other wise fit will not match and thus e-mask will only
### take into account rv difference
### why did the stars not have a problem before hand?
### because they were interpolated onto the same scale as the model but not cut correctly

### need to interpolate/cut error mask to the observations, if it deosn't exist in a region then place 0
### i.e. no weight on that pixel

def read_txt_spectra(path):
    
    """
    Reads simple text file where first column is wavelength
    and second column is flux.
    
    There is no RV information so its calculated here
    
    The SNR information is in the name of the spectra
    """
    
    spectra = np.loadtxt(path)
    
    wavelength = spectra[:,0]
    flux = spectra[:,1]
    
    SNR_med = int(float(path.split("snr")[1].split("_")[0]))
    
    error_med = flux/SNR_med 
           
    rv_shift = np.nan
    rv_shift_err = np.nan
        
    return wavelength,flux,error_med,rv_shift,rv_shift_err,SNR_med
    
def read_fits_spectra(path):
        
    hdulist = pyfits.open(path)
        
    scidata = hdulist[1].data
            
    wavelength = scidata[0][0]
    flux = scidata[0][1]
    error = scidata[0][2]
    
    ### need to do same thing with SNR and RV shift here!!!
    
    ### maybe the fits file has RV info within it...
    
    SNR_med = int(float(path.split("snr")[1].split("_")[0]))
    
    error_med = flux/SNR_med 
    
    # print("No RV correction found, calculating...")
    
    ### rv correction must be done post continuum normalisation
    
    rv_shift = np.nan
    rv_shift_err = np.nan
    
    return wavelength,flux,error_med,rv_shift,rv_shift_err,SNR_med    


#fit the spectrum
def find_best_val(ind_spec_arr):
    
    """
    
    ind_spec_arr: array of inputs that determine how best fit variables will
    be found. Descriptions are below.
    
    return: final; best fit parameters in all dimensions of NN module
            efin_upper; the upper uncertainty of best parameters
            efin_lower; the lower uncertainty of best parameters
            rv_shift; the radial velocity shift of the spectra collected or calculated
            ch2_save; the reduced chi-squared value for the best fit spectra
            wvl_corrected; the wavelength of the best fit model spectra
            obs; the flux of the observed spectra post process (rv corrected, normalised, convolved etc)
            fit; the flux of the best fit model spectra in the rest frame
            snr_star; the SNR of the star
    
    purpose: This function takes in any spectra with a resolution the same as 
             or above the NN trained grid and can calculate the RV correction,
             continuum normalise the spectra, convolve the spectra down. 
             
             The spectra however processed is then used to find the best fit
             model spectra and so the best fit parameters associated with the 
             Payne NN trained module. 
             
             Currently you can apply an "error mask" which is a way of 
             including model uncertainties by adding the residual between 
             observed spectra and reference model in quadrature. You can pick 
             any spectra you want to create an error mask with.
             
             To further improve the best fit parameters, a procedure which uses
             asteroseismology data (specifically nu_max) to refine the logg
             estimate via an iterative process.
             
             If you would like to add another process or another way of
             treating the spectra, then you would have to add to the inputs,
             edit the part of the code which concerns curvefitting as there
             parameters can be fixed to a specific value or left free to vary.
             
             ###
             
             N.B. To read in a spectra, you have to create your own function,
             simply because fit files for different spectrographs are
             different. 
             
             There are 3 already here which read a text file, Gaia-ESO fit file
             and HARPS/UVES fit file.
             
             Note that if the rv correction does not come with the fit file,
             you must still output an rv correction, set it to NaN.
             
             Or like in the "read_txt_spectra()" function you can write a bool
             to use SAPP's rv correction calculation function to find it for 
             you.
             
             Alternatively if you already know the rv correction for the
             spectra, load it in with your spectra load function.
    
    """
    
                
    spec_path = "../Input_data/spectroscopy_observation_data/" + ind_spec_arr[0] # running in main.py
    # spec_path = "../../../Input_data/spectroscopy_observation_data/" + ind_spec_arr[0] # running directly here
    
    # print("SPECTRAL PATH",spec_path)
    
    # observation spec we want to look at now
    
    error_map_spec_path = "../Input_data/spectroscopy_observation_data/" + ind_spec_arr[1] # running in main.py
    # error_map_spec_path = "../../../Input_data/spectroscopy_observation_data/" + ind_spec_arr[1] # running directly here
    
    # observation spec we want to use for the error map
    
    error_mask_index = ind_spec_arr[2] 
    
    # Refers to list of reference values, Sun is index number 10
    
    error_mask_recreate_arr = ind_spec_arr[3]

    error_mask_recreate_bool = error_mask_recreate_arr[0]
    
    emask_kw_instrument = error_mask_recreate_arr[1]
    
    emask_kw_teff = error_mask_recreate_arr[2]
        
    # If the error mask needs to be re-made or doesn't exist for the spectra, then this is True
    
    # otherwise, this is False
    
    error_map_use_bool = ind_spec_arr[4]
    
    # True: Use the error map (loaded or created), False: Do not use error map at all
    
    cont_norm_bool = ind_spec_arr[5]
    
    # True: Continuum normalise the spectra with SAPP's continuum procedure, False: Do not normalise
    
    rv_shift_recalc_arr = ind_spec_arr[6]
    
    rv_shift_recalc = rv_shift_recalc_arr[0]
    rv_min = rv_shift_recalc_arr[1]
    rv_max = rv_shift_recalc_arr[2]
    drv  = rv_shift_recalc_arr[3]
    
    # If rv shift from spectra needs to be re-calculated/doesn't exist, this is True
    
    # otherwise, this is False
    
    conv_instrument_bool = ind_spec_arr[7] 
    
    # True: convolve observation spectra to input resolution, False: Do not convolve spectra
    
    # Resolution_obs > input resolution, you cannot increase the resolution above what it already is.
    
    input_spec_resolution = ind_spec_arr[8]
    
    # Input resolution, this needs to match the Payne NN resolution file
    
    numax_iter_bool = ind_spec_arr[9]
    
    # True: Use nu_max input to improve logg estimation via iterative scaling process, False: do not use this info
    
    nu_max = ind_spec_arr[10][0]
    
    if np.isnan(nu_max) == True:
        
        numax_iter_bool = False
    
    # nu_max decided by user, if it is negative or NaN then there will be issues
    
    nu_max_err = ind_spec_arr[10][1]
    
    # uncertainty of nu_max, currently not used but might be useful
    
    niter_MAX = ind_spec_arr[10][2]
    
    # maximum number of iterations for this process, 5 is typically good enough
    
    recalc_metals_bool = ind_spec_arr[11]
    
    # True: re-calculate all parameters with Teff, logg, [Fe/H] fixed at some value, False: do not do this
    
    recalc_metals_inp = ind_spec_arr[12]
    
    # array [Teff,logg,[Fe/H]]
    
    logg_fix_bool = ind_spec_arr[13]
    
    logg_fix_inp_arr = ind_spec_arr[14]
    
    logg_fix = logg_fix_inp_arr[0]
    
    logg_fix_err_up = logg_fix_inp_arr[1]
    
    logg_fix_err_low = logg_fix_inp_arr[2]
    
    if np.isnan(logg_fix) == True:
        
        logg_fix_bool = False
        
    unique_emask_params_arr = ind_spec_arr[15]
    
    """
    Below is where the spectral information is fed in, this function should be tailored to the specific type of file
    and so can be easily changed. This information is all standard.
    """
    
    # print(nu_max,nu_max_err,numax_iter_bool)
    
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

    wavelength,\
    flux_norm,\
    error_med,\
    rv_shift,\
    rv_shift_err,\
    snr_star = read_txt_spectra(spec_path)
        
    ### HARPS/UVES fit files
    
    # wavelength,\
    # flux_norm,\
    # error_med,\
    # rv_shift,\
    # rv_shift_err,\
    # snr_star = read_fits_spectra(spec_path)
    
    print("SNR",snr_star)
    print("RV ",rv_shift,"+-",rv_shift_err,"Km/s")
    
    if snr_star <= 0: 
        
        # this can happen, for now the function returns zero
        
        # ideally SAPP would have a function that can calculate SNR incase you cannot
        
        # Note a good guess for HARPS is 300, and UVES is 200.
        
        return 0
    
    else:        
        
        if cont_norm_bool == True:
            
            # sigma clip the flux and error due to cosmic rays
            
            # do not go below 2, otherwise you're removing too much information
                
            flux_clipped = sigclip(flux_norm,sig=2.5)
            error_clipped = sigclip(error_med,sig=2.5)    
            
            ### before everything, continuum normalise
            
            # these are the zones which split up the spectra to normalise
            
            # further detail can be found in the continuum_normalise_spectra() script
            
            geslines_synth_loaded = np.loadtxt("sapp_seg_v1_hr10.txt")
            
            spec_norm_results = continuum_normalise_spectra(wavelength = wavelength,\
                                        flux = flux_clipped,\
                                        error_flux = error_clipped,\
                                        SNR_star = snr_star,\
                                        continuum_buffer=0,
                                        secondary_continuum_buffer=2.5,
                                        geslines_synth=geslines_synth_loaded,\
                                        med_bool=True)
            
            wavelength_normalised_stitch = spec_norm_results[0]
            flux_normalised_stitch = spec_norm_results[1]
            error_flux_normalised_stitch = spec_norm_results[2]
            # continuum_stitch = spec_norm_results[3]
            # geslines_synth = spec_norm_results[4]    
            
            wvl = wavelength_normalised_stitch
            obs = flux_normalised_stitch
            usert = error_flux_normalised_stitch
                    
        elif cont_norm_bool == False:
    
            wvl = wavelength
            obs = flux_norm
            usert = error_med
            
            
        if conv_instrument_bool == True: # convolve observation spectra to lower resolution

            wvl,obs = convolve_python(wvl,obs,input_spec_resolution)
            
            # Should we convolve the error as well?
            
            # wvl_new,obs_new = convolve_python(wvl,obs,input_spec_resolution)
            # wvl_new,usert_new = convolve_python(usert,usert,input_spec_resolution)
            
            # overwrites the spectra with the convolved information 
            
        """
        Does the RV value exist? If not, calculate it.
        """
                
        # spec_template_wvl = wvl_solar_model
        # spec_template_flux = flux_solar_model
        
        # the values below should be edited outside of the function
        
        # there must be a more dynamic way of choosing drv and the limits
        
        # as it can take a bit of time
                
        spec_resolution = input_spec_resolution
        
        # print("Pre-RV correction : time elapsed ----",time.time()-start_time,"----- seconds ----")
        
        if np.isnan(rv_shift):
            
            # if the rv_shift doesn't exist, NaN is given. 
            
            print("Re-calculating RV correction...")
            
            
            rv_shift, rv_shift_err = RV_multi_template(wvl,
                                                      obs,
                                                      usert,
                                                      rv_min,
                                                      rv_max,
                                                      drv,
                                                      spec_resolution)
                
            rv_shift_err = drv
        
        if rv_shift_recalc == True:
            
            # if you want to re-calculate it, then it will
            
            print("Re-calculating RV correction...")
            
            
            rv_shift, rv_shift_err = RV_multi_template(wvl,
                                                      obs,
                                                      usert,
                                                      rv_min,
                                                      rv_max,
                                                      drv,
                                                      spec_resolution)


            rv_shift_err = drv
                
            print("Recalc RV ",rv_shift,"+-",drv,"Km/s")

            
        # RV correcting the observed spectra's wavelength
        
        wvl_corrected = RV_correction_vel_to_wvl_non_rel(wvl,rv_shift) 
                
        
        """
        LIMIT TREATMENT
        
        Here the limits of the observation spectra are compared to the model.
        
        There's no guarantee they match, this could be a pre-process, but it 
        is annoying.
        
        The following lines ensure that the observed spectra matches the model
        spectra's wavelength limits. 
        
        """
                
        wvl_to_cut = [] # this will stay empty if model isn't being cut
    
        w0_new = w0
                    
        if min(wvl_corrected) > min(w0_new):
        
            # cut model to obs minimum
            
            wvl_min_bool = True # this will affect restore1
            
            w0_new = w0_new[w0_new >= min(wvl_corrected)]        
    
            wvl_to_cut = wvl_corrected
            
        else:
            
            wvl_min_bool = False # this will not affect restore1
                    
        if max(wvl_corrected) < max(w0_new): # basically your model should be cut like w0_new
        
            # cut model to obs minimum
            
            wvl_max_bool = True # this will affect restore1
    
            w0_new = w0_new[w0_new <= max(wvl_corrected)]        
            
            wvl_to_cut = wvl_corrected
            
        else:
            
            wvl_max_bool = False # this will not affect restore1
    
        # great, now obs is cut to model if it was bigger. If mod is bigger, then it'll pass on
    
        wvl_obs_input = np.hstack((wvl_min_bool,wvl_max_bool,wvl_to_cut))
    
        obs = np.interp(w0_new,wvl_corrected,obs) # this will cut obs to model if model is smaller
        usert = np.interp(w0_new,wvl_corrected,usert)
        wvl_corrected = np.interp(w0_new,wvl_corrected,wvl_corrected)
        
        ## process zeros in error ##
        
        # if zero, make them the nominal error i.e.
        # err = flux/SNR
        
        usert[usert==0] = obs[usert==0]/snr_star

        
        if error_map_use_bool == True:
            
            if error_mask_recreate_bool == True:

                print("USING STELLAR ERROR MASK")
                
                residual_error_map = create_error_mask(error_map_spec_path,error_mask_index,cont_norm_bool,rv_shift,rv_shift_err,conv_instrument_bool,input_spec_resolution,unique_emask_params_arr)
                
                residual_error_map_arr = [residual_error_map]
    
            elif error_mask_recreate_bool == False:
                
                # emask_kw_instrument = error_mask_recreate_arr[1]
    
                # emask_kw_teff = error_mask_recreate_arr[2]
    
    
                if emask_kw_teff == "solar":

                    print("USING SOLAR ERROR MASK")
                    
                    if emask_kw_instrument == "HARPS":
                        
                        residual_error_map_1 = np.loadtxt("../Output_data/test_spectra_emasks/sun/HARPS_Sun-1_Ceres_snr300_error_synth_flag_True_cont_norm_convolved_hr10_.txt")
                        residual_error_map_2 = np.loadtxt("../Output_data/test_spectra_emasks/sun/HARPS_Sun-2_Ganymede_snr300_error_synth_flag_True_cont_norm_convolved_hr10_.txt")
                        residual_error_map_3 = np.loadtxt("../Output_data/test_spectra_emasks/sun/HARPS_Sun-3_Vesta_snr300_error_synth_flag_True_cont_norm_convolved_hr10_error_mask.txt")

                        residual_error_map_arr = [residual_error_map_1,
                                                  residual_error_map_2,
                                                  residual_error_map_3]

                    
                    elif emask_kw_instrument == "UVES":
                        

                        residual_error_map_1 = np.loadtxt("../Output_data/test_spectra_emasks/sun/UVES_Sun_snr200_error_synth_flag_True_cont_norm_convolved_hr10_.txt")                    
                        
                        residual_error_map_arr = [residual_error_map_1]
                    
                if emask_kw_teff == "teff_varying":
                
                    print("USING TEFF VARYING ERROR MASK")
                    
                    if float(PLATO_bmk_lit[:,1][error_mask_index]) < 5500:
                        
                        ### load low temp error mask :  del eri
                        
                        ### need to load up and create a HARPS equivelent for low temp stars
                        
                        if emask_kw_instrument == "UVES" or "HARPS":
    
                             residual_error_map_1 = np.loadtxt("../Output_data/test_spectra_emasks/deleri/ADP_deleri_snr246_UVES_52.774g_error_synth_flag_False_cont_norm_convolved_hr10_.txt")
                             residual_error_map_2 = np.loadtxt("../Output_data/test_spectra_emasks/deleri/ADP_deleri_snr262_UVES_52.794g_error_synth_flag_False_cont_norm_convolved_hr10_.txt")
                             residual_error_map_3 = np.loadtxt("../Output_data/test_spectra_emasks/deleri/UVES_delEri_snr200_error_synth_flag_True_cont_norm_convolved_hr10_.txt")         
                        
                             residual_error_map_arr = [residual_error_map_1,
                                                       residual_error_map_2,
                                                       residual_error_map_3]
                         
                    elif 5500 <= float(PLATO_bmk_lit[:,1][error_mask_index]) < 6000:
                        
                        if emask_kw_instrument == "HARPS":
                            
                            residual_error_map_1 = np.loadtxt("../Output_data/test_spectra_emasks/sun/HARPS_Sun-1_Ceres_snr300_error_synth_flag_True_cont_norm_convolved_hr10_.txt")
                            residual_error_map_2 = np.loadtxt("../Output_data/test_spectra_emasks/sun/HARPS_Sun-2_Ganymede_snr300_error_synth_flag_True_cont_norm_convolved_hr10_.txt")
                            residual_error_map_3 = np.loadtxt("../Output_data/test_spectra_emasks/sun/HARPS_Sun-3_Vesta_snr300_error_synth_flag_True_cont_norm_convolved_hr10_error_mask.txt")
    
                            residual_error_map_arr = [residual_error_map_1,
                                                      residual_error_map_2,
                                                      residual_error_map_3]
    
                        
                        elif emask_kw_instrument == "UVES":
                            
    
                            residual_error_map_1 = np.loadtxt("../Output_data/test_spectra_emasks/sun/UVES_Sun_snr200_error_synth_flag_True_cont_norm_convolved_hr10_.txt")                    
                            
                            residual_error_map_arr = [residual_error_map_1]
                         
                    
                    elif float(PLATO_bmk_lit[:,1][error_mask_index]) >= 6000:
                        
                        ### load high temp error mask : Procyon
                        
                        if emask_kw_instrument == "HARPS":

                            residual_error_map_1 = np.loadtxt("../Output_data/test_spectra_emasks/Procyon/HARPS_Procyon_snr300_error_synth_flag_True_cont_norm_convolved_hr10_.txt")

                            residual_error_map_arr = [residual_error_map_1]

                            
                        if emask_kw_instrument == "UVES":
    
                            residual_error_map_1 = np.loadtxt("../Output_data/test_spectra_emasks/Procyon/ADP_procyon_snr493_UVES_21.007g_error_synth_flag_False_cont_norm_convolved_hr10_.txt")
                            residual_error_map_2 = np.loadtxt("../Output_data/test_spectra_emasks/Procyon/ADP_procyon_snr544_UVES_21.033g_error_synth_flag_False_cont_norm_convolved_hr10_.txt")
                            residual_error_map_3 = np.loadtxt("../Output_data/test_spectra_emasks/Procyon/ADP_procyon_snr549_UVES_48.418g_error_synth_flag_False_cont_norm_convolved_hr10_.txt")
                            residual_error_map_4 = np.loadtxt("../Output_data/test_spectra_emasks/Procyon/UVES_Procyon_snr200_error_synth_flag_True_cont_norm_convolved_hr10_.txt")                    
    
    
                            residual_error_map_arr = [residual_error_map_1,
                                                       residual_error_map_2,
                                                       residual_error_map_3,
                                                       residual_error_map_4]

            # this function compares the error masks to the spectra it is being combined with
            # processes it such that the wavelength scale and limits match the observations
            # any points missing are designated to be zero
            # any points extra are simply cut out
            
            wvl_err_mask_collect = []
            err_mask_collect = []
            
            for emask_number in range(len(residual_error_map_arr)):

                wvl_err_mask_dummy , err_mask_dummy = error_mask_trim_process(residual_error_map_arr[emask_number],wvl_corrected,obs)

                wvl_err_mask_collect.append(wvl_err_mask_dummy)
                err_mask_collect.append(err_mask_dummy)
                
            wvl_err_mask_ave = np.mean(np.array(wvl_err_mask_collect),axis=0)
            err_mask_collect_ave = np.mean(np.array(err_mask_collect),axis=0)
                        
            # sanity check, these should all be the same #
            # print(wvl_err_mask_collect[0])
            # print(wvl_corrected)
            # print(wvl_err_mask_ave)
            
            # if (wvl_err_mask_ave == wvl_corrected).all() == False:
                
            #     print("something went wrong! Error mask wavelength doesn't match observed wavelength")
                
            err_mask = err_mask_collect_ave 
            wvl_err_mask = wvl_err_mask_ave
            
            """
            IF error mask wvl is larger than current wvl obs, simply interpolate.
            cutting of edges is done for you.
            
            Else (error mask wvl is smaller than current wvl obs), interpolate, the
            wavlength values will repeat, change those repeats to zeros.
            """
                
            usert = (usert**2 + err_mask**2)**0.5
    
        popt=[0]
        
        
        """ 
        LINE MASKING
        
        This is for regions of the code which have bad lines, so they get
        'Noised-Up'.
        
        Not the best method, however it works for the fitting routine.
        
        Reconmended for regions which show signs of Tellurics.
        """
        
        #masking=False
        #masking=True
        #set error_spectrum=1000 around masked lines
        if masking:
    #        wav=wvl/(1+(rv+popt[0])/299792.458)
            #mask experiment
            windx=np.array([])
            for i in range(len(wgd)):
                windx=np.append(windx,np.where(abs(wvl_corrected-wgd[i])<1.751)[0])
            windx=[int(it) for it in windx]
            windx=np.unique(windx)
            #print(len(windx))
            spix=wvl_corrected<0
            spix[windx]=True
            usert[spix]=1000
            
        # print("Line Masking : time elapsed ----",time.time()-start_time,"----- seconds ----")
        
        """
        An overall estimate of Teff, logg, [Fe/H] may already exist
        and the only thing left to do is based on those parameters, 
        recalculate the others using this module.
        
        If the user would like to use the iterative mode/just find a full 
        solution then set this to False.
        """
        
        popt_init= np.zeros(num_labels+1+cheb) # start from centre of grid
        
        # popt_init[1] = (5777/1000-x_min[0])/(x_max[0]-x_min[0])-0.5 # start from any position you want
        # popt_init[2] = (4.20068-x_min[1])/(x_max[1]-x_min[1])-0.5
        # popt_init[3] = (0.3-x_min[2])/(x_max[2]-x_min[2])-0.5
        
        if recalc_metals_bool == True: 
            
            temp_fix = recalc_metals_inp[0]
            logg_fix = recalc_metals_inp[1]
            feh_fix = recalc_metals_inp[2]
            fix_feh_bool = recalc_metals_inp[3] 
            
            # if this is True, then fix to value above
            # if this is False, then doesn't matter, it'll be recalculated
            
            #flag for success
            suc=True
            #argiments for curve_fit
            kwargs={'loss':'linear',"max_nfev":1e8,'xtol':1e-4}
        
            try:
                init= np.zeros(num_labels+1+cheb)
                lb=init-0.49 # was 0.51 before
                hb=init+0.49
                
                fix=0#Teff index
                init[fix+1]=(temp_fix/1000-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4
                
                fix=1#logg index
                init[fix+1]=(logg_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4
        
                if fix_feh_bool == True: # if this is true then feh is fixed to best fit value from grids
                        fix=2#feh index
                        init[fix+1]=(feh_fix-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                        lb[fix+1]=init[fix+1]-1e-4
                        hb[fix+1]=init[fix+1]+1e-4
        
                popt, pcov = curve_fit(restore1,wvl_obs_input,\
                       obs,\
                       p0 = init,\
                       #p0 = np.zeros(num_labels+1+cheb),\
                       sigma=usert,\
                       #sigma=obs*0+1,\
                       absolute_sigma=True,\
                       bounds=(lb,hb),**kwargs)
                
                suc=True
            except RuntimeError:
                print("Error - curve_fit failed ")#stars[ind_spec])
                popt=np.ones(num_labels+1+cheb)
                pcov=np.ones((num_labels+1+cheb,num_labels+1+cheb))
                popt*=np.nan
                suc=False
        
            
            #only spectral parameters 
            if cheb == 0:
                
                labels_fit = popt[1:]
                #error from curve_fit
                efin=np.sqrt(pcov.diagonal()[1:])
                                
            elif cheb >=1:
                                
                labels_fit = popt[1:-cheb]
                #error from curve_fit
                efin=np.sqrt(pcov.diagonal()[1:-cheb])

        
            #renormalise spectral parameters
            final = ((labels_fit.T + 0.5)*(x_max-x_min) + x_min).T

            # print("BEST fit parameters =",final)
                            
            #renormalise spectral parameters errors
            efin=efin*(x_max-x_min)

            efin_upper = efin
            efin_lower = efin

            # print("parameters ERROR upper =",efin_upper)
            # print("parameters ERROR lower =",efin_lower)
            
        elif recalc_metals_bool == False:
        
            """
            Finding all the best fit parameters and so model.
            """
                
            #flag for success
            suc=True
            #argiments for curve_fit
            kwargs={'loss':'linear',"max_nfev":1e8,'xtol':1e-4}
            
            # print("Before parameter estimation : time elapsed ----",time.time()-start_time,"----- seconds ----")

            # input_dict = {
            #             "wvl_min_bool": wvl_min_bool,
            #             "wvl_max_bool": wvl_max_bool,
            #             "wvl_to_cut": wvl_to_cut,
            #             "Teff_fix": np.nan,
            #             "logg_fix": 4.44,
            #             "feh_fix": np.nan,
            #             "vmic_fix": np.nan,
            #             "vbrd_fix": np.nan,
            #             "mgfe_fix": np.nan,
            #             "tife_fix": np.nan,
            #             "mnfe_fix": np.nan
            #             }

            # popt_init_fix = np.zeros(num_labels+1+cheb-1)
            
            # logg_fix_norm = (4.44-x_min[1])/(x_max[1]-x_min[1])-0.5
            # popt_init_logg = [0,0,logg_fix_norm,0,0,0,0,0,0]

            try:
                
                popt, pcov = curve_fit(restore1,wvl_obs_input,\
                                      obs,\
                                      p0 = popt_init,#np.zeros(num_labels+1+cheb),\
                                      sigma=usert,\
                                      absolute_sigma=True,\
                                      bounds=(-.49,.49),**kwargs) # was 0.59 before
                
                
                
                ### NEW WAY OF FIXING VARIABLES
                
                # popt, pcov = curve_fit(restore1_test,input_dict,\
                #                       obs,\
                #                       p0 = popt_init_fix,#np.zeros(num_labels+1+cheb),\
                #                       sigma=usert,\
                #                       absolute_sigma=True,\
                #                       bounds=(-.49,.49),**kwargs) # was 0.59 before
                                        

                ### OLD WAY OF FIXING VARIABLES ###

                # init= np.zeros(num_labels+1+cheb)
                # lb=init-0.49
                # hb=init+0.49
                    
                # fix=1#logg index
                # init[fix+1]=(4.44-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                # # print((4.44-x_min[fix])/(x_max[fix]-x_min[fix])-0.5)
                # lb[fix+1]=init[fix+1]-1e-4
                # hb[fix+1]=init[fix+1]+1e-4
   
                # # print("Initial parameter estimation FIX : time elapsed ----",time.time()-start_time,"----- seconds ----")  
                
                # popt, pcov = curve_fit(restore1,wvl_obs_input,\
                #     obs,\
                #     p0 = init,\
                #     #p0 = np.zeros(num_labels+1+cheb),\
                #     sigma=usert,\
                #     #sigma=obs*0+1,\
                #     absolute_sigma=True,\
                #     bounds=(lb,hb),**kwargs)

                suc=True
                
                # print("Initial parameter estimation : time elapsed ----",time.time()-start_time,"----- seconds ----")
                
            except RuntimeError:
                       # print("Error - curve_fit failed 4:")
                       popt=np.ones(num_labels+1+cheb)
                       pcov=np.ones((num_labels+1+cheb,num_labels+1+cheb))
                       popt*=np.nan
                       suc=False
        
                     #only spectral parameters 
            # if cheb == 0:
            #                 labels_fit = popt[1:]
            #              #error from curve_fit
            #                 efin=np.sqrt(pcov.diagonal()[1:])
                        
            # elif cheb >=1:
                        
            #                 labels_fit = popt[1:-cheb]
            #              #error from curve_fit
            #                 efin=np.sqrt(pcov.diagonal()[1:-cheb])
                                            
                     #renormalise spectral parameters

            ### add fixed logg value
            
            # convert the fixed value, to normalised space 

            # logg_fix_norm = (4.44-x_min[1])/(x_max[1]-x_min[1])-0.5
            # teff_fix_norm = (6.990-x_min[0])/(x_max[0]-x_min[0])-0.5

            # popt =  np.hstack((popt[0],popt[1],logg_fix_norm,labels_fit[2:]))
            
            labels_fit = popt[1:]
            
            # labels_fit = np.hstack((labels_fit[0],logg_fix_norm,labels_fit[1:]))
            # labels_fit = np.hstack((teff_fix_norm,labels_fit[0:]))
            
            efin=np.sqrt(pcov.diagonal()[1:])
            
            # efin = np.hstack((efin[0],0.00,efin[1:]))
            # efin = np.hstack((0.00,efin[0:]))
                        
            final = ((labels_fit.T + 0.5)*(x_max-x_min) + x_min).T
                    
            # print("BEST fit parameters =",final)
            
            #renormalise spectral parameters errors
            efin=efin*(x_max-x_min)
                
            efin_upper = efin
            efin_lower = efin
             
            # print("parameters ERROR upper =",efin_upper)
            # print("parameters ERROR lower =",efin_lower)
             
            # print("Before parameter estimation nu max : time elapsed ----",time.time()-start_time,"----- seconds ----")
            
            if numax_iter_bool == True:
                
                final_orig = final # the initial run 
                efin_orig = efin
                popt_orig = popt
                
                """
                The estimates above are a first gues. 
                
                Now calculate the logg from Teff and nu_max via scaling relation.
                
                Rinse and repeat until maximum iteration.
                
                Within iterations, find the "best" parameters by determining when
                the Teff estimate doesn't change within 10 K. 
                
                N.B. DO THIS USING NU MAX +- NU MAX ERROR, THEREFORE NEED TO TRIPLE THE CALCULATIONS
                
                """
                
                nu_max_iter_arr = [nu_max-nu_max_err,nu_max,nu_max+nu_max_err]
                
                final_nu_max_collect = []
                efin_nu_max_collect = []
                popt_nu_max_collect = []
                pcov_nu_max_collect = []
                
                for nu_max_loop in range(len(nu_max_iter_arr)):
                    
                    if nu_max_loop == 0:
                        
                        print("====== nu_max lower ======")
                        # continue
                        
                    if nu_max_loop == 1:
                        
                        print("====== nu_max central ======")
                        
                    if nu_max_loop == 2:
                        
                        print("====== nu_max upper ======")
                        # continue
                
                    ncount = 0        
                    
                    final_collect = [final_orig]
                    efin_collect = [efin_orig]
                    popt_collect = [popt_orig]
                    
                    while ncount <= niter_MAX:
                        
                        # print(type(nu_max),astroc.nu_max_sol,final[0],astroc.teff_sol,astroc.surf_grav_sol)
                        
                        if ncount == 0: # use the initial guess
                            
                            logg_numax = np.log10((nu_max_iter_arr[nu_max_loop]/astroc.nu_max_sol) * (final_orig[0]*1000/astroc.teff_sol) ** 0.5 * astroc.surf_grav_sol)
                            
                        else: # use the parameter "final" which is overwritten 
                            
                            logg_numax = np.log10((nu_max_iter_arr[nu_max_loop]/astroc.nu_max_sol) * (final[0]*1000/astroc.teff_sol) ** 0.5 * astroc.surf_grav_sol)
            
                        # print(nu_max_loop,ncount,final[0]*1000,logg_numax)
                        
                        # print(nu_max_loop,ncount,logg_numax,":",nu_max_iter_arr[nu_max_loop],astroc.nu_max_sol,final[0]*1000,astroc.teff_sol,astroc.surf_grav_sol)
            
                        try:
                                                         
                                init= np.zeros(num_labels+1+cheb)
                                lb=init-0.49
                                hb=init+0.49
                                    
                                fix=1#logg index
                                init[fix+1]=(logg_numax-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                                lb[fix+1]=init[fix+1]-1e-4
                                hb[fix+1]=init[fix+1]+1e-4
           
                                # print("Initial parameter estimation FIX : time elapsed ----",time.time()-start_time,"----- seconds ----")  
                                
                                popt, pcov = curve_fit(restore1,wvl_obs_input,\
                                   obs,\
                                   p0 = init,\
                                   #p0 = np.zeros(num_labels+1+cheb),\
                                   sigma=usert,\
                                   #sigma=obs*0+1,\
                                   absolute_sigma=True,\
                                   bounds=(lb,hb),**kwargs)
                    
                                suc=True
                                
                                # print("Post parameter estimation FIX : time elapsed ----",time.time()-start_time,"----- seconds ----")                            

                        except RuntimeError:
                                print("Error - curve_fit failed")#stars[ind_spec])
                                popt=np.ones(num_labels+1+cheb)
                                pcov=np.ones((num_labels+1+cheb,num_labels+1+cheb))
                                popt*=np.nan
                                suc=False
                            
                                #only spectral parameters 
                        if cheb == 0:
                                labels_fit = popt[1:]
                                    #error from curve_fit
                                efin=np.sqrt(pcov.diagonal()[1:])
                                    
                        elif cheb >=1:
                                    
                                labels_fit = popt[1:-cheb]
                                    #error from curve_fit
                                efin=np.sqrt(pcov.diagonal()[1:-cheb])
                                    
                                #renormalise spectral parameters
                        final = ((labels_fit.T + 0.5)*(x_max-x_min) + x_min).T
                            
                            #renormalise spectral parameters errors
                        efin=efin*(x_max-x_min)
                        
                        efin_upper = efin
                        efin_lower = efin
                        
                        print("===== niter_loop",nu_max_loop,"===== niter",ncount+1,"============================================")
                        print("BEST fit parameters =",final)
                        print("parameters ERROR upper =",efin_upper)
                        print("parameters ERROR lower =",efin_lower)
                        print("======================================================")
                        
                        ncount += 1
                        
                        final_collect.append(final)
                        efin_collect.append(efin)
                        popt_collect.append(popt)
                    
                    
                    # need to search through and find where Tdiff = 10
                    for tdiff_search_loop in range(1,len(final_collect)):
                        
                           tdiff_dummy = (final_collect[tdiff_search_loop][0]-final_collect[tdiff_search_loop-1][0])*1000
                        
                           if tdiff_dummy <= 10:
                               
                                if tdiff_search_loop == len(final_collect)-1:
            
                                    tdiff_index = tdiff_search_loop # this temperature is the one we want as it only changes by 10 K
                                    
            
                                else:                   
                            
                                    tdiff_index = tdiff_search_loop+1 # this temperature is the one we want as it only changes by 10 K
                            
                                break
            
                    tdiff_dummy_collect = [] 
            
                    # need to search through and find where Tdiff = 10
                    for tdiff_search_loop in range(1,len(final_collect)):
                        
                           tdiff_dummy = (final_collect[tdiff_search_loop][0]-final_collect[tdiff_search_loop-1][0])*1000
                        
                           tdiff_dummy_collect.append(tdiff_dummy)
            
                    final = final_collect[tdiff_index]
                    popt = popt_collect[tdiff_index]        
                    efin_spec = efin_collect[tdiff_index]    
                    
                    
                    ###
                        
                    """
                    Error is calculated by collecting all the values from the "oscillatory"
                    section of the iterations. As the solution will vary within the bounds
                    of this region.
                    """
                    
                    # efin = np.array([\
                    #         (np.std(np.array(final_collect)[:,0][tdiff_index:])**2 + efin_spec[0]**2)**0.5,\
                    #         (np.std(np.array(final_collect)[:,1][tdiff_index:])**2 + efin_spec[1]**2)**0.5,\
                    #         (np.std(np.array(final_collect)[:,2][tdiff_index:])**2 + efin_spec[2]**2)**0.5,\
                    #         (np.std(np.array(final_collect)[:,3][tdiff_index:])**2 + efin_spec[3]**2)**0.5,\
                    #         (np.std(np.array(final_collect)[:,4][tdiff_index:])**2 + efin_spec[4]**2)**0.5,\
                    #         (np.std(np.array(final_collect)[:,5][tdiff_index:])**2 + efin_spec[5]**2)**0.5,\
                    #         (np.std(np.array(final_collect)[:,6][tdiff_index:])**2 + efin_spec[6]**2)**0.5,\
                    #         (np.std(np.array(final_collect)[:,7][tdiff_index:])**2 + efin_spec[7]**2)**0.5])


                    efin = []
                    
                    for efin_i in range(len(efin_spec)):
                        
                        efin.append((np.std(np.array(final_collect)[:,efin_i][tdiff_index:])**2 + efin_spec[efin_i]**2)**0.5)
                        
                    efin = np.array(efin)

                    
                    final_nu_max_collect.append(final)
                    efin_nu_max_collect.append(efin)
                    popt_nu_max_collect.append(popt)
                    pcov_nu_max_collect.append(pcov)
                    
                    
                d_final_upper = final_nu_max_collect[2] - final_nu_max_collect[1]
                d_final_lower = final_nu_max_collect[1] - final_nu_max_collect[0]
                
                print("lower syst error", d_final_lower)
                print("upper syst error",d_final_upper)
                
                efin_upper =  (efin_nu_max_collect[2] ** 2 + d_final_upper ** 2)**0.5
                efin_lower = (efin_nu_max_collect[0] ** 2 + d_final_lower ** 2)**0.5
                
                final = final_nu_max_collect[1]
                popt = popt_nu_max_collect[1]
                pcov = pcov_nu_max_collect[1]
                
                # print("Post numax parameter estimation : time elapsed ----",time.time()-start_time,"----- seconds ----")
                
                
            elif logg_fix_bool == True:
                
                
                logg_fix_arr = [logg_fix-logg_fix_err_low,logg_fix,logg_fix + logg_fix_err_up]
                
                final_logg_collect = []
                efin_logg_collect = []
                popt_logg_collect = []
                pcov_logg_collect = []
                
                
                for logg_loop in logg_fix_arr:
                    
                    
                
                    try:
                                    init= np.zeros(num_labels+1+cheb)
                                    lb=init-0.51
                                    hb=init+0.51
                                        
                                    fix=1#logg index
                                    init[fix+1]=(logg_loop-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                                    lb[fix+1]=init[fix+1]-1e-4
                                    hb[fix+1]=init[fix+1]+1e-4
                        
                                    popt, pcov = curve_fit(restore1,wvl_obs_input,\
                                       obs,\
                                       p0 = init,\
                                       #p0 = np.zeros(num_labels+1+cheb),\
                                       sigma=usert,\
                                       #sigma=obs*0+1,\
                                       absolute_sigma=True,\
                                       bounds=(lb,hb),**kwargs)
                        
                                    suc=True
                    except RuntimeError:
                                    print("Error - curve_fit failed")#stars[ind_spec])
                                    popt=np.ones(num_labels+1+cheb)
                                    pcov=np.ones((num_labels+1+cheb,num_labels+1+cheb))
                                    popt*=np.nan
                                    suc=False
                                
                                    #only spectral parameters 
                    if cheb == 0:
                                    labels_fit = popt[1:]
                                        #error from curve_fit
                                    efin=np.sqrt(pcov.diagonal()[1:])
                                        
                    elif cheb >=1:
                                        
                                    labels_fit = popt[1:-cheb]
                                        #error from curve_fit
                                    efin=np.sqrt(pcov.diagonal()[1:-cheb])
                                        
                                    #renormalise spectral parameters
                    final = ((labels_fit.T + 0.5)*(x_max-x_min) + x_min).T
                                
                                #renormalise spectral parameters errors
                    efin=efin*(x_max-x_min)
                                                        
                    print("===== logg_loop",logg_loop,"===============================================")
                    print("BEST fit parameters =",final)
                    print("parameters ERROR =",efin)
                    print("======================================================")

                    final_logg_collect.append(final)
                    efin_logg_collect.append(efin)
                    popt_logg_collect.append(popt)
                    pcov_logg_collect.append(pcov)

                d_final_upper = final_logg_collect[2] - final_logg_collect[1]
                d_final_lower = final_logg_collect[1] - final_logg_collect[0]
                
                print("lower syst error", d_final_lower)
                print("upper syst error",d_final_upper)
                
                efin_upper =  (efin_logg_collect[2] ** 2 + d_final_upper ** 2)**0.5
                efin_lower = (efin_logg_collect[0] ** 2 + d_final_lower ** 2)**0.5
                
                final = final_logg_collect[1]
                popt = popt_logg_collect[1]
                pcov = pcov_logg_collect[1]
                
                
                
        # print(pcov[1][1]* (x_max[0]-x_min[0]) *1000)
        
        # pcorr = correlation_from_covariance(pcov)
        
        # print(pcorr)
        
        ### normalize the covariance matrix, how? 

        print("BEST fit parameters =",final)
        print("parameters ERROR upper =",efin_upper)
        print("parameters ERROR lower =",efin_lower)
        
        # input_dict = {
        #                 "wvl_min_bool": wvl_min_bool,
        #                 "wvl_max_bool": wvl_max_bool,
        #                 "wvl_to_cut": wvl_to_cut,
        #                 "Teff_fix": np.nan,
        #                 "logg_fix": 4.44,
        #                 "feh_fix": np.nan,
        #                 "vmic_fix": np.nan,
        #                 "vbrd_fix": np.nan,
        #                 "mgfe_fix": np.nan,
        #                 "tife_fix": np.nan,
        #                 "mnfe_fix": np.nan
        #                 }
        
         #chi^2
        # fit= restore1_test(input_dict,*popt)
        fit= restore1(wvl_obs_input,*popt)
        ch2=np.sum(((obs-fit)/usert)**2)/(len(obs)-len(popt))
             
        ch2_save = np.round(ch2,decimals=2)
         
        return [final,efin_upper,efin_lower,rv_shift,ch2_save,wvl_corrected,obs,fit,snr_star,wvl_obs_input,usert,pcov]

def find_best_val_fix_all(ind_spec_arr): # used for creating error maps from reference values

    spec_path = ind_spec_arr[0]
    
    labels_inp = ind_spec_arr[1]
    
    cont_norm_bool = ind_spec_arr[2]
    
    rv_shift_recalc = ind_spec_arr[3] # by this point it should have already been calculated
        
    rv_shift_err_recalc = ind_spec_arr[4]
    
    conv_instrument_bool = ind_spec_arr[5]
    
    input_spec_resolution = ind_spec_arr[6]
    
    if np.isnan(rv_shift_recalc) == True:
        
        rv_calc = True
        
    else:
        
        rv_calc = False
    
    ### rv correction calculation ###
    
    # wavelength,\
    # flux_norm,\
    # error_med,\
    # rv_shift,\
    # rv_shift_err,\
    # snr_star,\
    # flux_norm_sigma_ave,\
    # flux_sigma_ave = read_fits_GES_GIR_spectra(spec_path) 
    
    wavelength,\
    flux_norm,\
    error_med,\
    rv_shift,\
    rv_shift_err,\
    snr_star = read_txt_spectra(spec_path,rv_calc)
    
    # wavelength,\
    # flux_norm,\
    # error_med,\
    # rv_shift,\
    # rv_shift_err,\
    # snr_star = read_fits_spectra(spec_path)

    if cont_norm_bool == True:
            
        flux_clipped = sigclip(flux_norm,sig=2.5)
        error_clipped = sigclip(error_med,sig=2.5)    
        
        ### before everything, continuum normalise
        
        geslines_synth_loaded = np.loadtxt("sapp_seg_v1_hr10.txt")
        
        spec_norm_results = continuum_normalise_spectra(wavelength = wavelength,\
                                    flux = flux_clipped,\
                                    error_flux = error_clipped,\
                                    SNR_star = snr_star,\
                                    continuum_buffer=0,
                                    secondary_continuum_buffer=2.5,
                                    geslines_synth=geslines_synth_loaded,\
                                    med_bool=True)
        
        wavelength_normalised_stitch = spec_norm_results[0]
        flux_normalised_stitch = spec_norm_results[1]
        error_flux_normalised_stitch = spec_norm_results[2]
        # continuum_stitch = spec_norm_results[3]
        # geslines_synth = spec_norm_results[4]    
        
        wvl = wavelength_normalised_stitch
        obs = flux_normalised_stitch
        usert = error_flux_normalised_stitch
    
    elif cont_norm_bool == False:

        wvl = wavelength
        obs = flux_norm
        usert = error_med
        
    if conv_instrument_bool == True: # convolve observation spectra to lower resolution
        
        wvl,obs = convolve_python(wvl,obs,input_spec_resolution)
        # wvl,usert = convolve_python(usert,obs,input_spec_resolution)
            
        # overwrites the spectra with the convolved information 

     
    
    wvl_corrected = RV_correction_vel_to_wvl_non_rel(wvl,rv_shift_recalc)

#     print(f"RV correction = {rv_shift} km/s")
    
    wvl_to_cut = [] # this will stay empty if model isn't being cut

    w0_new = w0
    
    # cutting for v1
        
    if min(wvl_corrected) > min(w0_new):
    
        # cut model to obs minimum
        
        wvl_min_bool = True # this will affect restore1
        
        w0_new = w0_new[w0_new >= min(wvl_corrected)]        

        wvl_to_cut = wvl_corrected
        
    else:
        
        wvl_min_bool = False # this will not affect restore1
                
    if max(wvl_corrected) < max(w0_new): # basically your model should be cut like w0_new
    
        # cut model to obs minimum
        
        wvl_max_bool = True # this will affect restore1

        w0_new = w0_new[w0_new <= max(wvl_corrected)]        
        
        wvl_to_cut = wvl_corrected
        
    else:
        
        wvl_max_bool = False # this will not affect restore1

    # great, now obs is cut to model if it was bigger. If mod is bigger, then it'll pass on

    wvl_obs_input = np.hstack((wvl_min_bool,wvl_max_bool,wvl_to_cut))
        
    obs = np.interp(w0_new,wvl_corrected,obs) # this will cut obs to model if model is smaller
    usert = np.interp(w0_new,wvl_corrected,usert)
    wvl_corrected = np.interp(w0_new,wvl_corrected,wvl_corrected)
        
    popt=[0]
    
    '''
    #masking=False
    #masking=True
    #set error_spectrum=1000 around masked lines
    if masking:
#        wav=wvl/(1+(rv+popt[0])/299792.458)
        #mask experiment
        windx=np.array([])
        for i in range(len(wgd)):
            windx=np.append(windx,np.where(abs(wav-wgd[i])<0.751)[0])
        windx=[int(it) for it in windx]
        windx=np.unique(windx)
        #print(len(windx))
        spix=wav<0
        spix[windx]=True
        usert[spix]=1000
    ''' 
    
    #some list
    set=[]#np.arange(5700)
    
    #flag for success
    suc=True
    #argiments for curve_fit
    kwargs={'loss':'linear',"max_nfev":1e8,'xtol':1e-4}
    
        
    try:
                init= np.zeros(num_labels+1+cheb)
                lb=init-0.51
                hb=init+0.51
                    
                fix=0#Teff index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4

                fix=1#logg index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4

                fix=2#feh index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4

                fix=3#vmic index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4

                fix=4#vbrd index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4

                fix=5#mgfe index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4

                fix=6#tife index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4

                fix=7#mnfe index
                init[fix+1]=(labels_inp[fix]-x_min[fix])/(x_max[fix]-x_min[fix])-0.5
                lb[fix+1]=init[fix+1]-1e-4
                hb[fix+1]=init[fix+1]+1e-4
    
                popt, pcov = curve_fit(restore1,wvl_obs_input,\
                   obs,\
                   p0 = init,\
                   #p0 = np.zeros(num_labels+1+cheb),\
                   sigma=usert,\
                   #sigma=obs*0+1,\
                   absolute_sigma=True,\
                   bounds=(lb,hb),**kwargs)
    
                suc=True
    except RuntimeError:
#                 print("Error - curve_fit failed 4:")
                popt=np.ones(num_labels+1+cheb)
                pcov=np.ones((num_labels+1+cheb,num_labels+1+cheb))
                popt*=np.nan
                suc=False
                            
                #only spectral parameters 
    if cheb == 0:
                labels_fit = popt[1:]
                    #error from curve_fit
                efin=np.sqrt(pcov.diagonal()[1:])
                    
    elif cheb >=1:
                    
                labels_fit = popt[1:-cheb]
                    #error from curve_fit
                efin=np.sqrt(pcov.diagonal()[1:-cheb])
                    
                #renormalise spectral parameters
    final = ((labels_fit.T + 0.5)*(x_max-x_min) + x_min).T
            
            #renormalise spectral parameters errors
    efin=efin*(x_max-x_min)
            
    efin_upper=efin
    efin_lower=efin
     
    #chi^2
    fit= restore1(wvl_obs_input,*popt)
    ch2=np.sum(((obs-fit)/usert)**2)/(len(obs)-len(popt))
                
    return [final,efin_upper,efin_lower,rv_shift,ch2,wvl_corrected,obs,fit,w0_new]

def create_error_mask(spec_path,error_mask_index,cont_norm_bool,rv_shift,rv_shift_err,conv_instrument_bool,input_spec_resolution,unique_emask_params_arr):
                    
    ### star_id_use plugs into PLATO_bmk_lit and PLATO_bmk_lit_other_params to grab literature values :) 
    
    if unique_emask_params_arr[0] == True:
        
        # print(unique_params_arr[2],unique_params_arr[1])
        
        unique_emask_params = unique_emask_params_arr[1]
        
        star_plot_temp =  unique_emask_params[0]
        star_plot_logg =  unique_emask_params[1]
        star_plot_feh =  unique_emask_params[2]      
        star_plot_vmic =  unique_emask_params[3]
        star_plot_vsini =  unique_emask_params[4]
        star_plot_mgfe =  unique_emask_params[5]
        star_plot_tife =  unique_emask_params[6]
        star_plot_mnfe =  unique_emask_params[7]
        
    else:
        
        # print(error_mask_index)
        
        star_plot_lit_values = PLATO_bmk_lit[error_mask_index] #name, teff, logg, feh (inc errors)
            
        # print(star_conv_list_HARPS[HARPS_loop_index],star_ids_bmk[star_id_use_HARPS])
        
        # print(star_plot_lit_values)
        
        star_plot_temp = float(star_plot_lit_values[1])
        star_plot_logg = float(star_plot_lit_values[3])
        star_plot_feh = float(star_plot_lit_values[5])
            
        star_plot_other_lit_values = PLATO_bmk_lit_other_params[error_mask_index] #name, vmic, vsini, mgfe, tife, mnfe (inc errors)
            
        star_plot_vmic = float(star_plot_other_lit_values[1])
        star_plot_vsini = float(star_plot_other_lit_values[3])
        star_plot_mgfe = float(star_plot_other_lit_values[5])
        star_plot_tife = float(star_plot_other_lit_values[7])
        star_plot_mnfe = float(star_plot_other_lit_values[9])
        
    labels_inp_star_plot = np.array([star_plot_temp/1000,\
                              star_plot_logg,\
                              star_plot_feh,\
                              star_plot_vmic,\
                              star_plot_vsini,\
                              star_plot_mgfe,\
                              star_plot_tife,\
                              star_plot_mnfe])
            
    spec_results = find_best_val_fix_all([spec_path,labels_inp_star_plot,cont_norm_bool,rv_shift,rv_shift_err,conv_instrument_bool,input_spec_resolution])
        
    # final,efin_upper,efin_lower,rv_shift,ch2,wvl_corrected,obs,fit
        
    wavelength = spec_results[5]
    observation = spec_results[6] # should be the same as obs
    model = spec_results[7]
    
    w0_new = spec_results[8]
    
    # check they are the same
    # print(len(wavelength),wavelength) 
    # print(len(w0_new),w0_new)
            
    residual = observation - model

    spec_res_save = np.vstack((wavelength,abs(residual),residual)).T
    
    return spec_res_save

def stitch_regions(path_list,import_path,filename,savebool):
    
    """stitches a set of spectra together ordered by a list"""
            
    wavelength_stitch,flux_stitch = load_TS_spec(import_path + path_list[0])
    
    # need to check if the beginning and end of each section are the same!
    
    stitch_edge_value = max(wavelength_stitch)
        
    for stitch_index in range(1,len(path_list)):
                
        wavelength_stitch_dummy, flux_stitch_dummy = load_TS_spec(import_path + path_list[stitch_index])
        
        min_wavelength = min(wavelength_stitch_dummy)
        
        if min_wavelength == stitch_edge_value: # i.e. the first wavelength of the current region is the same as the last wavelength of the previous region

            flux_stitch_dummy = flux_stitch_dummy[:len(flux_stitch_dummy)-1] # i.e. the last wavelength is removed                        
            
            wavelength_stitch = wavelength_stitch[:len(wavelength_stitch)-1] # i.e. the last wavelength is removed                        
            
        wavelength_stitch = np.hstack((wavelength_stitch,wavelength_stitch_dummy))
        flux_stitch = np.hstack((flux_stitch,flux_stitch_dummy))
        
        stitch_edge_value = max(wavelength_stitch)
        
    if savebool == True:
        
        np.savetxt(f"{filename}" + ".txt",np.vstack((wavelength_stitch,flux_stitch)).T,header=' Wavelength/AA \t Flux')
    
    return wavelength_stitch,flux_stitch

def error_mask_trim_process(residual_error_map,wvl_corrected,obs):
    
    wvl_err_mask = residual_error_map[:,0]
    err_mask = residual_error_map[:,1]

    # print("wvl emask ",wvl_err_mask)
    # print("wvl obs ",wvl_corrected)

    # plt.plot(wvl_err_mask,err_mask,'g-',label='emask')

    # if error_mask_recreate_bool == False:

        ### interpolate the error mask onto the stellar wavelength scale
        
        # err_mask_interp = sci.UnivariateSpline(wvl_err_mask,err_mask,k=5)
        
        # err_mask = err_mask_interp(wvl_corrected)
        # wvl_err_mask = wvl_corrected
        
    # print("wvl_emask",wvl_err_mask)
    # print("emask",err_mask)
    
    # print("================================")

    # print("wvl_emask_new",wvl_corrected)
    # print("emask_new",err_mask_new)
    
    # plt.plot(wvl_corrected,obs,'b-',label='obs')
    # plt.plot(wvl_err_mask,err_mask,'r-',label='emask interp')
        
    # plt.legend(loc='upper right')
    # plt.show()
    
                        
    ### DEALLING WITH MINIMA ###

    # plt.plot(wvl_err_mask,err_mask,'b-',label='emask')            

    if min(wvl_err_mask) < min(wvl_corrected):
        # cut the error mask down to shape
        
        # print("wvl_err_mask",wvl_err_mask,len(wvl_err_mask))
        # print("wvl_corrected",wvl_corrected,len(wvl_corrected))
        
        err_mask = err_mask[wvl_err_mask >= min(wvl_corrected)]
        wvl_err_mask = wvl_err_mask[wvl_err_mask >= min(wvl_corrected)]
        
    elif min(wvl_err_mask) > min(wvl_corrected):
        
        # print("----------------------------------")
        # print(wvl_corrected,len(wvl_corrected),len(obs))
        # print(wvl_err_mask,len(wvl_err_mask),len(err_mask))
        
        

        # print(wvl_err_mask[:20],len(wvl_err_mask))
        # print(err_mask[:20],len(err_mask))

        # print("wow")

        
        # print(len(wvl_err_mask),len(err_mask))
        
        # little more tricky, cannot create data nor cut the observed spec down
        # solution, pad with zeroes, it will not effect the overall error
        
        wvl_obs_min = min(wvl_corrected)
        delta_obs = wvl_corrected[1] - wvl_corrected[0] # should be the same for all
        
        wvl_errmask_min = min(wvl_err_mask)
        delta_errmask = wvl_err_mask[1] - wvl_err_mask[0] # should be the same for all
                        
        # the difference in wavelength between minima
        
        wvl_range =  wvl_errmask_min - wvl_obs_min
        
        # print(wvl_range,wvl_errmask_min, wvl_obs_min)
        
        # the number of zeros we have to attach to the err mask y array
        
        # N_zeros = int(np.round(wvl_range/delta_errmask)) # this is assuming that del err mask is same as del obs
        # N_zeros = int(np.round(wvl_range/delta_obs)) # this is assuming that del err mask is same as del obs
                                    
        # print(N_zeros)
        
        # print(wvl_err_mask[:6],len(wvl_err_mask))
        # print(err_mask[:6],len(err_mask))
        # print(int(np.round(wvl_range/delta_obs)),int(np.round(wvl_range/delta_errmask)))
        
        # create array to tac on to the original
        
        # wvl_tac = np.arange(wvl_obs_min,wvl_errmask_min,delta_errmask)
        wvl_tac = np.arange(wvl_obs_min,wvl_errmask_min-delta_obs,delta_obs)
        
        if len(wvl_tac) == 0:
         
            # if the difference between emask and obs is the same as delta_obs itself
            # then tac on wvl_obs_min itself
         
            # i.e. wvl_errmask_min-delta_obs <= wvl_obs_min
            
            wvl_tac = np.arange(wvl_obs_min,wvl_errmask_min,delta_obs)

        # wvl_tac = np.arange(wvl_errmask_min,wvl_obs_min-delta_obs,delta_obs)

        N_zeros = len(wvl_tac)
                                    
        errmask_tac = np.zeros([N_zeros])
        
        wvl_err_mask = np.hstack((wvl_tac,wvl_err_mask))
        
        err_mask = np.hstack((errmask_tac,err_mask))
        
        # print(wvl_obs_min,wvl_errmask_min-delta_obs,delta_obs)
        # print(wvl_err_mask)
        # print(wvl_corrected)        


    ### DEALING WITH MAXIMA ###
    
    if max(wvl_err_mask) > max(wvl_corrected):
        
        # cut the error mask down to shape, simple
        
        # print(len(wvl_err_mask),len(err_mask))
        
        err_mask  = err_mask [wvl_err_mask  <= max(wvl_corrected)]
        wvl_err_mask  = wvl_err_mask [wvl_err_mask  <= max(wvl_corrected)]
        
    elif max(wvl_err_mask ) < max(wvl_corrected):
        
        
        # print(wvl_corrected,len(wvl_corrected),len(obs))
        # print(wvl_err_mask,len(wvl_err_mask),len(err_mask))
        
        # you need to pad with some zeroes
        
        wvl_obs_max = max(wvl_corrected)
        delta_obs = wvl_corrected[1] - wvl_corrected[0] # should be the same for all
        
        wvl_errmask_max = max(wvl_err_mask)
        delta_errmask = wvl_err_mask[1] - wvl_err_mask[0] # should be the same for all
        
        wvl_range =  wvl_obs_max - wvl_errmask_max
        
        # N_zeros = int(np.round(wvl_range/delta_errmask)) # this is assuming that del err mask is same as del obs
        
        wvl_tac = np.arange(wvl_errmask_max+delta_errmask,wvl_obs_max+delta_errmask,delta_errmask)
        
        # if len(wvl_tac) == 0:
        
        # print("-------")
        # print(wvl_err_mask)
        # print(wvl_tac)
        # print(wvl_corrected)
        
        
        N_zeros = len(wvl_tac)
            
        errmask_tac = np.zeros([N_zeros])

        wvl_err_mask = np.hstack((wvl_err_mask ,wvl_tac))
        
        err_mask = np.hstack((err_mask,errmask_tac))
        
    
    # print("wvl emask ",wvl_err_mask)
    # print("wvl obs ",wvl_corrected)
    
    # print(wvl_err_mask[:20],len(wvl_err_mask))
    # print(err_mask[:20],len(err_mask))
    
    ### now interpolate emask over to same wavelength scale as obs
    
    # if error_mask_recreate_bool == False:
    
    # print("---------------------------------")
    # print(wvl_corrected)
    # print(wvl_err_mask)
                
    emask_func = sci.interp1d(wvl_err_mask,err_mask)
    err_mask = emask_func(wvl_corrected)
    wvl_err_mask = wvl_corrected
            
    return wvl_err_mask,err_mask


Input_data_path_main = "../Input_data/" # running in main.py
                                       
# Input_data_path_main = "../../../Input_data/" # running directly here 

Input_data_path = Input_data_path_main

PLATO_bmk_lit = np.loadtxt(Input_data_path + "Reference_data/PLATO_stars_lit_params.txt",dtype=str,delimiter=',')
star_ids_bmk = PLATO_bmk_lit[:,0]
PLATO_bmk_lit_other_params = np.loadtxt(Input_data_path + "Reference_data/PLATO_stars_lit_other_params.txt",dtype=str,delimiter=',')

solar_model_list = np.loadtxt(Input_data_path + "spectroscopy_model_data/Sun_model_segments.txt",dtype=str)
wvl_solar_model, flux_solar_model = stitch_regions(solar_model_list,Input_data_path + "spectroscopy_model_data/Sun_model_R5e5_TS_hr10/","solar_model_stitched_hr10",savebool=False)

#name="NN_results_RrelsigL20.npz" #LTE
name="NN_results_RrelsigN20.npz" #NLTE

LTE_type = "NLTE"
#LTE_type = "LTE"

import_path = Input_data_path + "spectroscopy_model_data/Payne_input_data/"

temp=np.load(import_path+name)

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
w0=np.linspace(5329.,5616,11480)[40:-40]
w0=w0[::2]

#order of Chebyshev polynomials
### NEVER SET ABOVE ZERO

cheb=0

### MASKING ### 
 
#to mask
masking=False
#lines to be masked in interval 0.751 around them
wgd=np.array([5503.08,5577.34]) # bad lines example 


'''
spec_path = "18_sco/ADP_18sco_snr396_HARPS_17.707g_error_synth_flag_True_cont_norm_convolved_hr10_.txt"
error_map_spec_path = "18_sco/ADP_18sco_snr396_HARPS_17.707g_error_synth_flag_True_cont_norm_convolved_hr10_.txt"

error_mask_index = 0

emask_kw_instrument = "HARPS" # can be "HARPS" or "UVES"
emask_kw_teff = "teff_varying" # can be "solar","teff_varying", or "stellar"

error_mask_recreate_bool = False # if this is set to True, then emask_kw_teff defaults to "stellar"

error_mask_recreate_arr = [error_mask_recreate_bool,emask_kw_instrument,emask_kw_teff]

error_map_use_bool = False
cont_norm_bool = False
rv_shift_recalc = [False,-400,400,0.5]
conv_instrument_bool = False
input_spec_resolution = 20000
numax_iter_bool = False
niter_numax_MAX = 5
numax_input_arr = [3170,159,niter_numax_MAX]
recalc_metals_bool = False
feh_recalc_fix_bool = False
recalc_metals_inp = [5770,4.44,0,feh_recalc_fix_bool]
logg_fix_bool = False
logg_fix_load = np.loadtxt("../../../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/seismology_lhood_results/18sco_seismic_logg.txt",dtype=str)
logg_fix_input_arr = [float(logg_fix_load[1]),float(logg_fix_load[2]),float(logg_fix_load[3])]
unique_emask_params_bool = False
unique_emask_params = [5777,4.44,0,1,1.6,0,0,0] # solar example 



ind_spec_arr = [spec_path,\
                 error_map_spec_path,\
                 error_mask_index,\
                 error_mask_recreate_arr,\
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
                 [unique_emask_params_bool,unique_emask_params]]
find_best_val(ind_spec_arr)
'''
                                       
    
    
    
    
    
    
    
    
    
    
