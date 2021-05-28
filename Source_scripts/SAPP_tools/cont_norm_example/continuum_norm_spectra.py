#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:55:59 2020

@author: gent
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# use LaTex font
from matplotlib.pyplot import rc

font = {'family': 'serif',
       'serif': ['Computer Modern'],
       'size': 22}

rc('font', **font)
rc('text', usetex=True)


def continuum(wl, flux, error, snr, med_bool): # applied to each section right?
    
    wl_flux_df = pd.DataFrame(np.vstack((wl, flux, error)).T, columns = ['wl', 'flux1', 'error1']) 
    
    if med_bool == True:
        wl_flux_df['flux'] = wl_flux_df['flux1']/np.median(wl_flux_df['flux1'])  # Divide by the median
        wl_flux_df['error'] = wl_flux_df['error1']/np.median(wl_flux_df['flux1'])  # Divide by the median
    else:
        wl_flux_df['flux'] = wl_flux_df['flux1']  
        wl_flux_df['error'] = wl_flux_df['error1']  

    
    pol_fit = np.polyfit(wl_flux_df.wl, wl_flux_df.flux, 1) 
 
    cont_points_df_tmp = wl_flux_df.reset_index() 
    fit_line_new = np.poly1d(pol_fit) 
 
    for i in range(20): 
        
        condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr  > 0 
        cont_points_df_tmp = cont_points_df_tmp[condition] 
        pol_fit_new = np.polyfit(cont_points_df_tmp.wl, cont_points_df_tmp.flux, 1) 
        fit_line_new = np.poly1d(pol_fit_new) 
    
    return wl_flux_df.wl,wl_flux_df.flux/fit_line_new(wl_flux_df.wl),wl_flux_df.error,fit_line_new(wl_flux_df.wl)

def secondary_normalisation_given_zone(wavelength_norm,flux_norm,error_flux_norm,SNR,buffer,med_bool):
    
    ### Split wavelength ###

    flux_norm = np.array(flux_norm)
    wavelength_norm = np.array(wavelength_norm)
    error_flux_norm = np.array(error_flux_norm)

    wavelength_split_mid = int(len(wavelength_norm)/2)# int rounds down   
    
    # while 2*buffer >= (max(wavelength_norm)-min(wavelength_norm)): # i.e. if the last zone is really small, 
        
    #     buffer = 0.5 * buffer

    if 2*buffer >= (max(wavelength_norm)-min(wavelength_norm)):
        
        # if your buffer is on the order of the zone you are looking at
        # the zone itself is probably really small as the buffer tends to be small
        # if this is the case, don't continuum normalise it
        # stay with a first order continuum normalisation
        # if you still want to do it, make the buffer very small i.e. 10% range of wavelength zone
        
        buffer = 0.1 * (max(wavelength_norm)-min(wavelength_norm))
            
        
    flux_split_range = flux_norm[wavelength_norm<=wavelength_norm[wavelength_split_mid]+buffer]            
    error_split_range = error_flux_norm[wavelength_norm<=wavelength_norm[wavelength_split_mid]+buffer]
    wavelength_split_range = wavelength_norm[wavelength_norm<=wavelength_norm[wavelength_split_mid]+buffer]

    flux_split_range = flux_split_range[wavelength_split_range>wavelength_norm[wavelength_split_mid]-buffer]                
    error_split_range = error_split_range[wavelength_split_range>wavelength_norm[wavelength_split_mid]-buffer]   
    wavelength_split_range = wavelength_split_range[wavelength_split_range>wavelength_norm[wavelength_split_mid]-buffer]
    
    # buffer decides the size of the middle chosen
    
    given_value = 1 # finding flux value closest to 1 in the "middle range"
    a_list = flux_split_range
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closest_value = min(a_list, key=absolute_difference_function)
                                
    wavelength_split_mid_new = wavelength_norm[flux_norm==closest_value]
    
    # print("WVL MID POINT ARR",min(wavelength_norm),wavelength_norm[wavelength_split_mid],max(wavelength_norm))
    
    wavelength_first_half = wavelength_norm[wavelength_norm<=wavelength_split_mid_new]
    flux_norm_first_half = flux_norm[wavelength_norm<=wavelength_split_mid_new]
    error_flux_norm_first_half = error_flux_norm[wavelength_norm<=wavelength_split_mid_new]
    
    wavelength_second_half = wavelength_norm[wavelength_norm>wavelength_split_mid_new]
    flux_norm_second_half = flux_norm[wavelength_norm>wavelength_split_mid_new]
    error_flux_norm_second_half = error_flux_norm[wavelength_norm>wavelength_split_mid_new]
        
    ### second normalisation here ###

    wavelength_first_half,flux_norm_first_half,error_flux_norm_first_half,continuum_vardon_first_half = continuum(wavelength_first_half,flux_norm_first_half,error_flux_norm_first_half,SNR,med_bool)
    wavelength_second_half,flux_norm_second_half,error_flux_norm_second_half,continuum_vardon_second_half = continuum(wavelength_second_half,flux_norm_second_half,error_flux_norm_second_half,SNR,med_bool)
        
    wavelength_first_half = np.array(wavelength_first_half)
    flux_norm_first_half = np.array(flux_norm_first_half)
    error_flux_norm_first_half = np.array(error_flux_norm_first_half)
    wavelength_second_half = np.array(wavelength_second_half)
    flux_norm_second_half = np.array(flux_norm_second_half)
    error_flux_norm_second_half = np.array(error_flux_norm_second_half)
        
    flux_norm = np.append(flux_norm_first_half,flux_norm_second_half)
    wavelength_norm = np.append(wavelength_first_half,wavelength_second_half)
    continuum_vardon = np.append(continuum_vardon_first_half,continuum_vardon_second_half)
    error_flux_norm = np.append(error_flux_norm_first_half,error_flux_norm_second_half)
        
    return wavelength_norm,flux_norm,error_flux_norm,continuum_vardon

def continuum_normalise_spectra(wavelength,flux,error_flux,SNR_star,continuum_buffer,secondary_continuum_buffer,geslines_synth,med_bool):
    
    """
    wavelnength: observed spectral wavelength [Angstroms]
    
    flux: un-normalised flux
    
    error: error of flux, if it doesn't exist, give [] as an argument
    
    SNR_star: SNR of the star (assumed to be known/already calculated)
        
    continuum_buffer: User specified number of angstroms to add to the begining
    of the first zone and the and of the last zone.
    
    secondary_continuum_buffer: angstroms, serves same purpose as continuum buffer
    
    geslines_synth: list of segment zones which are used to split the spectra
    into appropriate "normalisable" sections.

    med_bool: True (if you want to divide by the median first), False if you don't.
    
    return: wavelength, flux, error (continuum normalised), final continuum, edited geslines_synth
    """
    
    ###
    window_max = geslines_synth[len(geslines_synth)-1][1] + continuum_buffer # wmax of the last window + buffer
    window_min = geslines_synth[0][0] - continuum_buffer # wmin of the last window
    
    geslines_synth[0][0] = window_min
    geslines_synth[len(geslines_synth)-1][1] = window_max
    
    
    # Beginning and End windows now match the model
    # Truncation is required because HARPS covers more than UVES
    
    ## hr10
    # geslines_synth_no_secondary_norm = [[4835.0,4895.0],[5118.2,5218.66],[5745.34,5755.0],[5871.95,5906.12],[6325.95,6372.25],[6520.5,6600.0]]

    ## hr21/Gaia RVS 
    geslines_synth_no_secondary_norm = [[8480,8510],[8530,8560],[8650,8685],[8692,8705],[8705,8719],[8740,8745],[8760,8770],[8770,8778],[8788.00,8800.00]]


    # The windows above are ones which should not go through secondary normalisation due to broad lines such as Halpha
    ###
    
    ### CHECK IF ERROR EXISTS ###
    
    if len(error_flux) > 0:
    
        ### NAN CHECK FOR ERROR ###
    
        N_nan = len(np.where(np.isnan(error_flux)==True)[0])
        
        if N_nan/len(error_flux) > 0.2:
            
            # print("More than 20\% of the errors are NaN values")
            # print("Estimating error via SNR")
            
            error_flux = 1/SNR_star * flux
    
    else:
        
        error_flux = 1/SNR_star * flux
        
    # basically an error will exist regardless 
        
    # what about spectra which have values less than zero?
    
    # same treatment?
    
    # if the snr is still larger than 0 then will still have to treat these
    
    # so yes, do it
    
    ### Search spectra for zeroes!!! ###
    zero_collect = []
    flux_without_zeroes = []
    wavelength_without_zeroes = []
    error_without_zeroes = []
            
    for zero_search in range(len(flux)):
        
        if flux[zero_search] <= 0: # was == before, now this catches negative fluxes
            
            zero_collect.append(zero_search)
            continue
        
        else:
            
            flux_without_zeroes.append(flux[zero_search])
            wavelength_without_zeroes.append(wavelength[zero_search])
            error_without_zeroes.append(error_flux[zero_search])
                        
    if len(zero_collect) > 0:
        
        flux = np.array(flux_without_zeroes)
        wavelength = np.array(wavelength_without_zeroes)
        error_flux = np.array(error_without_zeroes)
            
    ### Need to check for UVES/HARPS red-blue gaps ###
        
    wvl_RdBl_gap = []
    
    for wvl_index in range(len(wavelength)-1):
        
        if (wavelength[wvl_index+1]- wavelength[wvl_index]) > 5: 
            
            wvl_RdBl_gap.append([wavelength[wvl_index],wavelength[wvl_index+1]]) # there should only be one for UVES, two for HARPS
                                    
    flux_normalised_stitch = []
    error_flux_normalised_stitch = []
    wavelength_normalised_stitch = []
    continuum_stitch = []
    
    N_windows = len(geslines_synth)
            
    for zones in range(len(geslines_synth)):
        
        if len(geslines_synth) < N_windows: 
            
            # This means the array has changed i.e. a window has been removed.
            
            if zones == len(geslines_synth): # i.e. new end of the window + 1
                
                break
        
        wavelength_z_min = geslines_synth[zones][0]
        wavelength_z_max = geslines_synth[zones][1]
        
        # Need to run through gaps if there are multiple

        for gap_counter in range(0,len(wvl_RdBl_gap)): # if UVES, should only occur once.
            
            geslines_synth_new = []
            zone_gap_exits = []    
            
            # Need to check if the entire gap is within the next zone 
            if zones < len(geslines_synth) - 1:
                                
                wavelength_z_min_next = geslines_synth[zones+1][0]
                wavelength_z_max_next = geslines_synth[zones+1][1]
                                
                if wvl_RdBl_gap[gap_counter][0] > wavelength_z_min_next: 
                    
                    if wvl_RdBl_gap[gap_counter][1] < wavelength_z_max_next:
                                            
                        # If this zone contains the gap, adjust window to account for gap
                        # print("Significant gap in next zone of spectra")
                        # print("Gap occurs in windows =",wavelength_z_min_next,",",wavelength_z_max_next)
                        
                        # print("Creating wavelength window limits for zone")
                        
                        # print("New wavelength window for zone =",zones)
                        
                        geslines_synth[zones][1] = wvl_RdBl_gap[gap_counter][0] # current window end needs to be the window gaps begining 
                        wavelength_z_max = geslines_synth[zones][1]
                        
                        # print(wavelength_z_min,wavelength_z_max)
    
                        # Need to make the window next + 1 beginning to be the end of this gap
                        
                        # print("New wavelength window for zone =",zones+2)
                        
                        geslines_synth[zones+2][0] = wvl_RdBl_gap[gap_counter][1]
                        
                        # print(geslines_synth[zones+2][0],geslines_synth[zones+2][1])
                        
                        # Now need to get rid of element window in zone + 1
                                            
                        zone_gap_exits = [zones + 1]
                        
            if len(zone_gap_exits)>0:
                
                # geslines_synth have now changed slighly, need to take this into account for next part of loop
                        
                for zone_search in range(len(geslines_synth)):
                                                
                    if zone_search == zone_gap_exits[0]:
                                    
                        continue # do not append spectra
                    else:
                                    
                        geslines_synth_new.append(geslines_synth[zone_search])
                
                geslines_synth =  geslines_synth_new     
            
                gap_counter = gap_counter + 1
                
                # Now that an element has been removed, the loop will continue to 1 element beyond array size
                
        
        # now have zone limits decided, collect spectra information from the current zone
                
        flux_obs_zone = flux[wavelength <= wavelength_z_max]        
        error_flux_zone = error_flux[wavelength <= wavelength_z_max]
        wavelength_obs_zone = wavelength[wavelength <= wavelength_z_max]
        
        flux_obs_zone = flux_obs_zone[wavelength_obs_zone >= wavelength_z_min]
        error_flux_zone = error_flux_zone[wavelength_obs_zone >= wavelength_z_min]
        wavelength_obs_zone = wavelength_obs_zone[wavelength_obs_zone >= wavelength_z_min]    

        if len(flux_obs_zone) == 0:
            
            # print("Zone contains no data, continuum normalisation not possible")
            # print("Wavlength range =",wavelength_z_min,",",wavelength_z_max)
            # print("Skipping zone")
            
            continue
        
        # first stage continuum normalisation
                        
        wavelength_norm,flux_norm,error_flux_norm,continuum_vardon = continuum(wavelength_obs_zone,flux_obs_zone,error_flux_zone,SNR_star,med_bool)
        
        #secondary normalisation
        # need to check if zones are allowed to be normalised again
                            
        forbidden_secondary = False
        
        for zone_secondary_search in range(len(geslines_synth_no_secondary_norm)):
                
            if geslines_synth_no_secondary_norm[zone_secondary_search][0] == wavelength_z_min:

                forbidden_secondary = True
                
                break
            
            else:
                
                continue
            
        # Basically search through forbidden windows
        # If the current window is part of any the forbidden ones, forbidden_secondary becomes True
        # No secondary is performed
        # Otherwise it is still false
        # Secodnary normalisation is performed
        
        
        if forbidden_secondary != True:
            
            # this splits the zone in half, normalises each half, then joins back together
            
            wavelength_norm,flux_norm,error_flux_norm,continuum_vardon = secondary_normalisation_given_zone(wavelength_norm,flux_norm,error_flux_norm,SNR_star,secondary_continuum_buffer,med_bool)

            # print("Secondary normalisation performed on zone =",zones)
            
        else: # I.e. forbidden_secondary = True
            
            # print("Secondary normalisation not authorised for zone =",zones)
            
            pass
                    
        
        if zones == 0: # initialisation
                
            flux_normalised_stitch = flux_norm
            wavelength_normalised_stitch = wavelength_norm
            continuum_stitch = continuum_vardon
            error_flux_normalised_stitch = error_flux_norm
                    
        else:
            
            flux_normalised_stitch = np.append(flux_normalised_stitch, flux_norm) #  do I stitch with the buffer still there???
            wavelength_normalised_stitch = np.append(wavelength_normalised_stitch, wavelength_norm)
            continuum_stitch = np.append(continuum_stitch,continuum_vardon)
            error_flux_normalised_stitch = np.append(error_flux_normalised_stitch,error_flux_norm)
        

    # option to save edited segment list 

#    np.savetxt(f"sapp_seg_v1_"+ star_name.replace(".fits","") +".txt",geslines_synth,fmt='%10.2f',delimiter='\t')
    
    # print("Zone continuum fitting finished")
    
    ### Here I am noising up the gaps between spectra. To account for edge effects
        
    for wvl_pixel in range(len(flux_normalised_stitch)-1):
        
        wvl_diff = wavelength_normalised_stitch[wvl_pixel+1] - wavelength_normalised_stitch[wvl_pixel]
        
        if wvl_diff > 5: # Angstroms, some arbitrary difference, gap could be smaller! 
        
            # multiply error by 1000
                                    
            error_flux_normalised_stitch[wvl_pixel+1] = error_flux_normalised_stitch[wvl_pixel+1] * 1000
            error_flux_normalised_stitch[wvl_pixel] = error_flux_normalised_stitch[wvl_pixel] * 1000
                
    return [wavelength_normalised_stitch,flux_normalised_stitch,error_flux_normalised_stitch,continuum_stitch,geslines_synth]


# filename = "18_sco_HD146233.HARPS-N.dop.txt"

# star_example = np.loadtxt(filename)
# wavelength = star_example[:,0]
# flux = star_example[:,1]
# error_flux = []
# SNR_star = 300
# continuum_buffer = 0 # angstroms
# secondary_continuum_buffer = 2.5 # angstroms
# geslines_synth = np.loadtxt("sapp_seg_v1.txt")
# med_bool = True

# spec_norm_results = continuum_normalise_spectra(wavelength,\
#                             flux,\
#                             error_flux,\
#                             SNR_star,\
#                             continuum_buffer,\
#                             secondary_continuum_buffer,\
#                             geslines_synth,\
#                             med_bool)
    
# wavelength_normalised_stitch = spec_norm_results[0]
# flux_normalised_stitch = spec_norm_results[1]
# error_flux_normalised_stitch = spec_norm_results[2]
# continuum_stitch = spec_norm_results[3]
# geslines_synth = spec_norm_results[4]

### PLOTTING ### 
'''
fig = plt.figure(figsize=(18,12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(wavelength,flux,'r-',label='un-normalised flux')
ax2.plot(wavelength_normalised_stitch,flux_normalised_stitch,'k-',label='normalised flux')
ax2.plot(wavelength_normalised_stitch,continuum_stitch,'b-',label='continuum')
ax2.plot(wavelength_normalised_stitch,error_flux_normalised_stitch,'g-',label='error')

ax2.set_ylim([0,2])

ax1.set_ylabel("Flux")
ax2.set_ylabel("Flux")

ax2.set_xlabel("Wavelength [$\AA$]")

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")

title = filename.replace("_","").split(".txt")[0]
ax1.set_title(f"{title}")

plt.show()
'''
