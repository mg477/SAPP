#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:55:59 2020

@author: gent
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
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
        
        # print(flux)
        # print(wl)
        
        # print(wl_flux_df['flux1'])
        
        wl_flux_df['flux'] = wl_flux_df['flux1']/np.median(wl_flux_df['flux1'])  # Divide by the median
        wl_flux_df['error'] = wl_flux_df['error1']#/np.median(wl_flux_df['flux1'])  # Divide by the median
    else:
        wl_flux_df['flux'] = wl_flux_df['flux1']  
        wl_flux_df['error'] = wl_flux_df['error1']  

    
    pol_fit = np.polyfit(wl_flux_df.wl, wl_flux_df.flux, 1) 
 
    cont_points_df_tmp = wl_flux_df.reset_index() 
    fit_line_new = np.poly1d(pol_fit) 
     
    wl_range_zone_initial = max(cont_points_df_tmp.wl) - min(cont_points_df_tmp.wl)
    
    flux_zone_collect = []
    
    snr_thresh = 50
    
    for i in range(20): 
    # for i in range(100): 
        
        # if snr > snr_thresh:
            # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr > 0
        # elif snr <= snr_thresh:
            # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr**0.5 > 0
            
        # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr_thresh > 0                                 
        # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr**2 > 0                         
        condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr > 0                         
        # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr**0.5 > 0 
        # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr**0.333 > 0 
        # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr**0.111 > 0 
        # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1./snr**0.111 > 0 
        # condition = cont_points_df_tmp.flux - fit_line_new(cont_points_df_tmp.wl) + 1 > 0 

        cont_points_df_tmp = cont_points_df_tmp[condition] 
        
        if (max(cont_points_df_tmp.wl) - min(cont_points_df_tmp.wl)) <= (1/3) * wl_range_zone_initial:
            
            break
        
        pol_fit_new = np.polyfit(cont_points_df_tmp.wl, cont_points_df_tmp.flux, 1) 
        fit_line_new = np.poly1d(pol_fit_new) 
        
        '''
        # if min(wl_flux_df.wl) >=5548 and max(wl_flux_df.wl) <= 5565:
        # if min(wl_flux_df.wl) >=5583 and max(wl_flux_df.wl) <= 5607:
        if min(wl_flux_df.wl) >=5499.83 and max(wl_flux_df.wl) <= 5537.32:
                        
            # print(np.array(cont_points_df_tmp.wl))
            # print(np.array(cont_points_df_tmp.flux))
            # print(np.array(fit_line_new(cont_points_df_tmp.wl)))
            # print(np.array(condition))
            
            flux_zone_collect.append([np.array(wl_flux_df.wl),np.array(wl_flux_df.flux/fit_line_new(wl_flux_df.wl)),np.array(cont_points_df_tmp.wl),np.array(cont_points_df_tmp.flux)])
        '''
         
    # return wl_flux_df.wl,wl_flux_df.flux/fit_line_new(wl_flux_df.wl),wl_flux_df.error,fit_line_new(wl_flux_df.wl),flux_zone_collect

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
    
    # print(flux_norm[flux_norm==closest_value])
    
    # print("WVL MID POINT ARR",min(wavelength_norm),wavelength_norm[wavelength_split_mid],wavelength_split_mid_new,max(wavelength_norm))
    
    if len(wavelength_split_mid_new) > 1: # i.e. multiple points in the middle have the exact same flux value (e.g. 1)

        wavelength_split_mid_new = [np.average(wavelength_split_mid_new)]
        
    wavelength_first_half = wavelength_norm[wavelength_norm<=wavelength_split_mid_new]
    flux_norm_first_half = flux_norm[wavelength_norm<=wavelength_split_mid_new]
    error_flux_norm_first_half = error_flux_norm[wavelength_norm<=wavelength_split_mid_new]
    
    wavelength_second_half = wavelength_norm[wavelength_norm>wavelength_split_mid_new]
    flux_norm_second_half = flux_norm[wavelength_norm>wavelength_split_mid_new]
    error_flux_norm_second_half = error_flux_norm[wavelength_norm>wavelength_split_mid_new]
        
    ### second normalisation here ###
  

    # ax_cont.plot(wavelength_first_half,flux_norm_first_half,'r-')
    # ax_cont.plot(wavelength_second_half,flux_norm_second_half,'r-')
    

    # wavelength_first_half,flux_norm_first_half,error_flux_norm_first_half,continuum_vardon_first_half,flux_zone_collect_1 = continuum(wavelength_first_half,flux_norm_first_half,error_flux_norm_first_half,SNR,med_bool)
    # wavelength_second_half,flux_norm_second_half,error_flux_norm_second_half,continuum_vardon_second_half,flux_zone_collect_2 = continuum(wavelength_second_half,flux_norm_second_half,error_flux_norm_second_half,SNR,med_bool)

    wavelength_first_half,flux_norm_first_half,error_flux_norm_first_half,continuum_vardon_first_half = continuum(wavelength_first_half,flux_norm_first_half,error_flux_norm_first_half,SNR,med_bool)
    wavelength_second_half,flux_norm_second_half,error_flux_norm_second_half,continuum_vardon_second_half = continuum(wavelength_second_half,flux_norm_second_half,error_flux_norm_second_half,SNR,med_bool)


    '''
    # print(min(wavelength_first_half),max(wavelength_first_half),min(wavelength_second_half),max(wavelength_second_half))

    # if min(wavelength_second_half) >= 5548  and max(wavelength_second_half) <= 5565: 
    # if min(wavelength_second_half) >= 5552  and max(wavelength_second_half) <= 5565: 
    if min(wavelength_second_half) >= 5499.83  and max(wavelength_second_half) <= 5537.32: 

        fig = plt.figure()
        ax_cont = fig.add_subplot(111)  
        
        import matplotlib.cm as mplcm
        import matplotlib.colors as colors
        
        NUM_COLORS = len(flux_zone_collect_2)
        
        cm = plt.get_cmap('gist_rainbow')
        cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        ax_cont.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])    
        count = 0 
        
        # print(len(flux_zone_collect_2[99][2]))
        # print(len(flux_zone_collect_2[19][2]))
                
        if len(flux_zone_collect_2) > 0:
            
            # print(flux_zone_collect_2)
            
            for i in range(len(flux_zone_collect_2)):
                
                # print(len(flux_zone_collect_2[i][3]),",")
                
                print(max(flux_zone_collect_2[i][2]) - min(flux_zone_collect_2[i][2]),",")
                
                count += 1
                
                ax_cont.plot(flux_zone_collect_2[i][0],flux_zone_collect_2[i][1],linestyle='-',linewidth=1,label=f'{count}')
                ax_cont.plot(flux_zone_collect_2[i][2],flux_zone_collect_2[i][3],marker='o',markersize=5,markeredgecolor='k',markeredgewidth=0.5,linestyle='none')
                
                # print(count,flux_zone_collect_2[i][3])
            
        # ax_cont.legend(loc='upper right')
            
        plt.show()
    # ax_cont.plot(wavelength_first_half,flux_norm_first_half,'g-')
    # ax_cont.plot(wavelength_second_half,flux_norm_second_half,'g-')
    
    # ax_cont.set_xlim([min(wavelength_first_half),max(wavelength_second_half)])

    
    # print(min(wavelength_second_half),max(wavelength_second_half))
    '''
    
    
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

def continuum_normalise_spectra(wavelength,flux,error_flux,SNR_star,continuum_buffer,secondary_continuum_buffer,geslines_synth,med_bool,recalc_mg):
    
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
    
    # geslines_synth_no_secondary_norm = [[4835.0,4895.0],[5118.2,5218.66],[5745.34,5755.0],[5871.95,5906.12],[6325.95,6372.25],[6520.5,6600.0]]
    # geslines_synth_no_secondary_norm = [[6520.5,6600.0],[6543,6583]]
    # The windows above are ones which should not go through secondary normalisation due to broad lines such as Halpha
    ###
    
    # hr10
    geslines_synth_no_secondary_norm = [[4835.0,4895.0],[5118.2,5218.66],[5745.34,5755.0],[5871.95,5906.12],[6325.95,6372.25],[6520.5,6600.0]]
    # geslines_synth_no_secondary_norm = [[5499.83,5537.32],[5537.32,5564.46],[5564.46,5606.53]]

    ## hr21/Gaia RVS 
    # geslines_synth_no_secondary_norm = [[8480,8510],[8530,8560],[8650,8685],[8692,8705],[8705,8719],[8740,8745],[8760,8770],[8770,8778],[8788.00,8800.00]]

    
    ### CHECK IF ERROR EXISTS ###
    
    if len(error_flux) > 0:
    
        ### NAN CHECK FOR ERROR ###
        
        # print("error exists!")
    
        N_nan = len(np.where(np.isnan(error_flux)==True)[0])
        
        if N_nan/len(error_flux) > 0.2:
            
            # print("More than 20\% of the errors are NaN values")
            # print("Estimating error via SNR")
            
            error_flux = 1/SNR_star * flux
                        
            # print("Synthetic error made due to Nans")
            
        # else:
            # 
            # print("Original error all good")
                
    
    else:
        
        error_flux = 1/SNR_star * flux
                
        # print("Synthetic error made due to existance")
        
    # basically an error will exist regardless 
        
    # what about spectra which have values less than zero?
    
    # same treatment?
    
    # if the snr is still larger than 0 then will still have to treat these
    
    # so yes, do it
    
    # plt.plot(wavelength,error_flux)
    
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
        
    
    ### do the same thing with errors! ###
    
    zero_collect = []
    flux_without_zeroes = []
    wavelength_without_zeroes = []
    error_without_zeroes = []
        
    for zero_search in range(len(error_flux)):
        
        if error_flux[zero_search] <= 0: # was == before, now this catches negative fluxes
            
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
     
    # converting error to percent error
    
    error_flux = error_flux/flux
               
    ### Need to check for UVES/HARPS red-blue gaps ###
    
    # plt.plot(wavelength,error_flux)
        
    wvl_RdBl_gap = []
    
    for wvl_index in range(len(wavelength)-1):
        
        if (wavelength[wvl_index+1]- wavelength[wvl_index]) > 5: 
            
            wvl_RdBl_gap.append([wavelength[wvl_index],wavelength[wvl_index+1]]) # there should only be one for UVES, two for HARPS
                                    
    flux_normalised_stitch = []
    error_flux_normalised_stitch = []
    wavelength_normalised_stitch = []
    continuum_stitch = []
    
    ## HR10 continuum mask made from 4 different models
    
    cont_buffer = 0.01
    # cont_mask_array = np.loadtxt(f"average_model_cont_wvl_points_buffer_{cont_buffer}.txt",delimiter=",")
    cont_mask_array = np.loadtxt(f"SAPP_spectroscopy/Payne/average_model_cont_wvl_points_buffer_{cont_buffer}.txt",delimiter=",")
    cont_mask_wvl  = cont_mask_array[:,0]
    
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
                        
                        # print("test",geslines_synth[zones+1],wvl_RdBl_gap[gap_counter])

                                            
                        # If this zone contains the gap, adjust window to account for gap
                        # print("Significant gap in next zone of spectra")
                        # print("Gap occurs in windows =",wavelength_z_min_next,",",wavelength_z_max_next)
                        
                        # print("Creating wavelength window limits for zone")
                        
                        # print("New wavelength window for zone =",zones)
                        
                        geslines_synth[zones][1] = wvl_RdBl_gap[gap_counter][0] # current window end needs to be the window gaps begining 
                        wavelength_z_max = geslines_synth[zones][1]
                        
                        # print(wavelength_z_min,wavelength_z_max)
    
                        # Need to make the window next + 1 beginning to be the end of this gap
                        
                        # But what if the next zone is the last zone? I.e. the gap is in the last zone
                        
                        # then making this windows end being the gap beginning works, you need to make
                        # the next windows begginning, the gaps end
                        
                        if zones + 1 == len(geslines_synth) -1:
                            
                            geslines_synth[zones+1][0] = wvl_RdBl_gap[gap_counter][1]

                            # print("New wavelength window for zone =",zones+1)
                            
                            # print(geslines_synth[zones+1][0],geslines_synth[zones+1][1])
                            
                            # print(geslines_synth)
                            
                            zone_gap_exits = []
                                                        
                        else:
                            
                            # why is it if there are more zones, we do +2 and get rid of one?

                        
                            # print("New wavelength window for zone =",zones+2)
                            
                            geslines_synth[zones+2][0] = wvl_RdBl_gap[gap_counter][1]
                            
                            # print(geslines_synth[zones+2][0],geslines_synth[zones+2][1])
                            
                        # a=b
                        
                        # Now need to get rid of element window in zone + 1
                        
                        # how does the gap being in the end zone affect this exiting process?
                                            
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
        
        # print(wavelength,len(wavelength))
        # print(flux,len(flux))
        # print(wavelength_z_max)
        
        flux_obs_zone = flux[wavelength <= wavelength_z_max]        
        error_flux_zone = error_flux[wavelength <= wavelength_z_max]
        wavelength_obs_zone = wavelength[wavelength <= wavelength_z_max]
                
        flux_obs_zone = flux_obs_zone[wavelength_obs_zone >= wavelength_z_min]
        error_flux_zone = error_flux_zone[wavelength_obs_zone >= wavelength_z_min]
        wavelength_obs_zone = wavelength_obs_zone[wavelength_obs_zone >= wavelength_z_min]    
        
        cont_mask_wvl_zone = cont_mask_wvl[cont_mask_wvl <= wavelength_z_max]
        cont_mask_wvl_zone = cont_mask_wvl_zone[cont_mask_wvl_zone >= wavelength_z_min]
        
        
        if len(flux_obs_zone) == 0:
            
            # print("Zone contains no data, continuum normalisation not possible")
            # print("Wavlength range =",wavelength_z_min,",",wavelength_z_max)
            # print("Skipping zone")
            
            continue
        
        # first stage continuum normalisation
        
        # SNR_star = 150
                                
        # wavelength_norm,flux_norm,error_flux_norm,continuum_vardon,flux_zone_collect = continuum(wavelength_obs_zone,flux_obs_zone,error_flux_zone,SNR_star,med_bool)
        wavelength_norm,flux_norm,error_flux_norm,continuum_vardon = continuum(wavelength_obs_zone,flux_obs_zone,error_flux_zone,SNR_star,med_bool)
        
        '''
        if min(wavelength_norm) >= 5499.83  and max(wavelength_norm) <= 5537.32: 
            
    
            fig = plt.figure()
            ax_cont = fig.add_subplot(111)  
            
            import matplotlib.cm as mplcm
            import matplotlib.colors as colors
            
            NUM_COLORS = len(flux_zone_collect)
            
            cm = plt.get_cmap('gist_rainbow')
            cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
            scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
            ax_cont.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])    
            count = 0 
            
            # print(len(flux_zone_collect_2[99][2]))
            # print(len(flux_zone_collect_2[19][2]))
                                
            if len(flux_zone_collect) > 0:
                
                # print(flux_zone_collect_2)
                # plt.plot(wavelength_obs_zone,flux_obs_zone,'k-')
                # plt.plot(wavelength_norm,flux_norm,'k-')    
            
                for i in range(len(flux_zone_collect)):
                    
                    # print(len(flux_zone_collect[i][3]),",")
                    
                    # print(max(flux_zone_collect[i][2]) - min(flux_zone_collect[i][2]),",")
                    
                    count += 1
                    
                    ax_cont.plot(flux_zone_collect[i][0],flux_zone_collect[i][1],linestyle='-',linewidth=1,label=f'{count}')
                    # ax_cont.plot(flux_zone_collect[i][2],flux_zone_collect[i][3],marker='o',markersize=5,markeredgecolor='k',markeredgewidth=0.5,linestyle='none')
                    
                    # print(count,flux_zone_collect_2[i][3])
                
            # ax_cont.legend(loc='upper right')
                
            plt.show()

        '''
        
        # plt.plot(wavelength_obs_zone,error_flux_zone)
        # plt.plot(wavelength_norm,error_flux_norm)
        # plt.plot(wavelength_norm,continuum_vardon,'r-')
        
        # plt.plot(wavelength_norm,flux_norm,'r-')
        
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
            # pass
            # print("Secondary normalisation performed on zone =",zones)
            
        else: # I.e. forbidden_secondary = True
            
            # print("Secondary normalisation not authorised for zone =",zones)
            
            pass
                    
        
        # ## phillip normalisation ###
        
        # bins = np.linspace(wavelength_norm.min(),wavelength_norm.max(),100)
        # flx_median,lam_bins = binned_statistic(wavelength_norm,flux_norm,bins=bins,statistic='median')[:2]
        # flx_max,lam_bins = binned_statistic(wavelength_norm,flux_norm,bins=bins,statistic='max')[:2]
        # lam_bins = (lam_bins[1:] + lam_bins[:-1]) /2

        # lin_f = lambda x,a,b: a*x+b
        # popt,pcov = curve_fit(lin_f,lam_bins,flx_max)
        # lin_frozen = lambda x: lin_f(x,*popt)
        # flux_norm = flux_norm / lin_frozen(wavelength_norm)        
        
        ### Tertiary normalisation ###     
        
        if recalc_mg:
        
            if wavelength_z_min == 5499.83:
    
                # fi = sci.interp1d(spec_template_wvl_doppler_shift, spec_template_flux)
                
                # first step, interpolate full spectrum onto the wavelength points from the continuum mask
                
                # plt.plot(wavelength_norm,flux_norm,'ko',label='full spectrum')
        
                # flux_interp_cont = np.interp(wavelength_norm,wavelength_norm[flux_norm>1],flux_norm[flux_norm>1])                   
    
                ### find the Mg specific zone within this segment 
                
                # 5524,5532
                # mg_lower = 5524
                # mg_upper = 5532
                mg_lower = 5526.05
                mg_upper = 5531.05
                mask_mg = (wavelength_norm <= mg_upper) & (wavelength_norm >= mg_lower)
                # mask_left = (wavelength_norm < 5524) # 5526.2
                # mask_right = (wavelength_norm > 5532) # 5531.5
                mask_left = (wavelength_norm < mg_lower) # 
                mask_right = (wavelength_norm >  mg_upper) #
                
                wavelength_norm_mg = wavelength_norm[mask_mg]
                flux_norm_mg = flux_norm[mask_mg]
                wavelength_norm_left = wavelength_norm[mask_left]
                flux_norm_left = flux_norm[mask_left]
                wavelength_norm_right = wavelength_norm[mask_right]
                flux_norm_right = flux_norm[mask_right]
    
                flux_interp_cont = np.interp(cont_mask_wvl_zone,wavelength_norm_mg[flux_norm_mg>1],flux_norm_mg[flux_norm_mg>1]) 
                # flux_interp_cont = np.interp(cont_mask_wvl_zone,wavelength_norm,flux_norm) 
                
                # problem with this? 
                
                # if its a VERY low SNR spectra, then this grabs all of the noisy points above 1 and fits 
                # a line to those just because they match the continuum wavelengths
                # this also has the problem of spikes and what not
                # hmmm, its just vals above 1, therefore if they range from 1 to 1.2
                # then the line is around 1.1, any points above 1 would almost definitely be noise
                # 
        
                # plt.plot(cont_mask_wvl_zone,flux_interp_cont,'mo',label = 'interpolated cont mask')
        
                # second step, polyfit these points basically creating a line equation through regression
                
                poly_mask = np.polyfit(cont_mask_wvl_zone,flux_interp_cont,1)
                fit_line_mask = np.poly1d(poly_mask)
                        
                # third step, insert original wavelength of full spectrum into the fit 
        
                flux_cont_mask = fit_line_mask(wavelength_norm_mg)
                
                # flux_cont_mask = np.interp(wavelength_norm,cont_mask_wvl_zone,flux_interp_cont)
                
                # plt.plot(wavelength_norm,flux_cont_mask,'r-',label='linear fit')
        
                # fourth and final step, divide whole spectrum by this continuum fit        
                
                flux_norm_mg /= flux_cont_mask # divide by these to get mask
                
                flux_norm = np.hstack((flux_norm_left,flux_norm_mg,flux_norm_right))
                wavelength_norm = np.hstack((wavelength_norm_left,wavelength_norm_mg,wavelength_norm_right))
                
                # plt.plot(wavelength_norm,flux_norm,'go',label='re-normalised')
                
                # plt.plot(wavelength_norm,flux_norm,'g-')
                
                # plt.legend(loc='lower right')
                # plt.show()
                        
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
        
        
        
    


    # multiply percentage error carried through by the now normalised flux error
    
    ## phillip normalisation ###
    
    # bins = np.linspace(wavelength_normalised_stitch.min(),wavelength_normalised_stitch.max(),100)
    # flx_max,lam_bins = binned_statistic(wavelength_normalised_stitch,flux_normalised_stitch,bins=bins,statistic='median')[:2]
    # # flx_max,lam_bins = binned_statistic(wavelength_normalised_stitch,flux_normalised_stitch,bins=bins,statistic='max')[:2]
    # lam_bins = (lam_bins[1:] + lam_bins[:-1]) /2

    # lin_f = lambda x,a,b: a*x+b
    # popt,pcov = curve_fit(lin_f,lam_bins,flx_max)
    # lin_frozen = lambda x: lin_f(x,*popt)
    # flux_normalised_stitch = flux_normalised_stitch / lin_frozen(wavelength_normalised_stitch)        
    
    
    # this will give a "normalised" error
    
    # print(error_flux_normalised_stitch,flux_normalised_stitch)

    error_flux_normalised_stitch  = error_flux_normalised_stitch * flux_normalised_stitch

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
