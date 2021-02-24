#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:19:52 2020

@author: gent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:34:18 2020

@author: gent
"""

import numpy as np
from scipy import ndimage
from astropy.io import fits as pyfits
import pandas as pd
import os
import matplotlib.pyplot as plt

def convolve_python(wavelength_input,flux_input,Resolution):

    smpl=(wavelength_input[1]-wavelength_input[0]) # wavelength range must have regular spacing
    fwhm=np.mean(wavelength_input)/Resolution # Resolution is the intended resolution
    flux_convolve=ndimage.filters.gaussian_filter1d(flux_input,fwhm*0.4246609/smpl)
    
    return flux_convolve

#def instrument_conv_res(wave, flux, Resolution):
#        
#	wave = wave[:10000]
#	flux = flux[:10000]
#    
#	print(len(flux))
#    
##	print(flux)
#    
#	fwhm = (np.mean(wave)/Resolution) * 1000 # to get into correct units 
#    
#	print("FWHM = ",fwhm)
#            
#    #create a file to feed to ./faltbon of spectrum that needs convolving
#    
#	f = open('original_instrument.txt', 'w')
#	spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
#	for j in range(len(wave)):
#		f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
#	f.close()
#	os.system("{ echo original_instrument.txt; echo convolve_instrument.txt; echo %f; echo 2; } | ./faltbon" % fwhm)
#    
#	wave_conv, flux_conv, spud = np.loadtxt("convolve_instrument.txt", unpack='True') #read in our new convolved spectrum
#	return wave_conv, flux_conv


def read_fits_spectra(path):
        
    hdulist = pyfits.open(path)
    
#    print(hdulist.info())
    
    scidata = hdulist[1].data

#    wavelength = []
#    flux = []
#    error = []
    
#    for data_loop in range(len(scidata)):
#        
#        wavelength.append(scidata[data_loop][0])
#        flux.append(scidata[data_loop][1])
#        error.append(scidata[data_loop][2])
        
    
    wavelength = scidata[0][0]
    flux = scidata[0][1]
    error = scidata[0][2]
    
    return [wavelength,flux,error]

def load_txt_spec(path):
    """loads spectra from txt file"""
    
    # typically the txt files do not contain errors
    
    spec = np.loadtxt(path)
    wavelength = spec[:,0] # angstrom
    flux = spec[:,1] 
    
    if np.shape(spec)[1] == 3:
        
        error = spec[:,2]
        
        return [wavelength,flux,error]
    
    else:
    
        return [wavelength,flux]

def rms_scatter(flux,continuum_level): # can only be applied to normalised spectra
    
    if len(flux) == 0: # i.e. empty array
        
        return []
    
    else:
        
        rms = np.sqrt(((flux - continuum_level)**2).mean())
    
        return rms
    
    
def continuum(wl, flux, error, snr, med_bool): # applied to each section right?
    
    wl_flux_df = pd.DataFrame(np.vstack((wl, flux, error)).T, columns = ['wl', 'flux1', 'error1']) 
    
    if med_bool == True:
        wl_flux_df['flux'] = wl_flux_df['flux1']/np.median(wl_flux_df['flux1'])  # Divide by the median
        wl_flux_df['error'] = wl_flux_df['error1']/np.median(wl_flux_df['flux1'])  # Divide by the median
    else:
        wl_flux_df['flux'] = wl_flux_df['flux1']  
        wl_flux_df['error'] = wl_flux_df['error1']  

    
    pol_fit = np.polyfit(wl_flux_df.wl, wl_flux_df.flux, 1) 
    fit_line = np.poly1d(pol_fit) # what does this do?
 
    cont_points_df_tmp = wl_flux_df.reset_index() 
    fit_line_new = np.poly1d(pol_fit) 
 
    for i in range(20): # what is happening over here??
        
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


def continuum_normalise_spectra(wavelength,flux,error_flux,SNR_star,w0,continuum_buffer):
    
    ###
    w0_max = max(w0) + continuum_buffer
    w0_min = min(w0) - continuum_buffer
    
    geslines_synth = np.loadtxt("PLATO/sapp_seg_v1_hr10.txt")
    geslines_synth[0][0] = w0_min
    geslines_synth[len(geslines_synth)-1][1] = w0_max
    # Beginning and End windows now match the model
    # Truncation is required because HARPS covers more than UVES
    
    geslines_synth_no_secondary_norm = [[4835.0,4895.0],[5118.2,5218.66]]
    # The windows above are ones which should not go through secondary normalisation due to broad lines.
    ###
    
    ### CHECK IF ERROR EXISTS ###
    
    if len(error_flux) > 0:
    
        ### NAN CHECK FOR ERROR ###
    
        N_nan = len(np.where(np.isnan(error_flux)==True)[0])
        
        if N_nan/len(error_flux) > 0.2:
            
            print("More than 20\% of the errors are NaN values")
            print("Estimating error via SNR")
            
            error_flux = SNR_star/(10**5) * flux
    
    else:
        
        error_flux = SNR_star/(10**5) * flux
        
    # basically an error will exist regardless 
        
    ### Search for zeroes!!! ###
    zero_collect = []
    flux_without_zeroes = []
    wavelength_without_zeroes = []
    error_without_zeroes = []
            
    for zero_search in range(len(flux)):
        
        if flux[zero_search] == 0:
            
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
    
    # for the newest window, could just ignore zone 24
    
    # but windows change
    
    wvl_RdBl_gap = [] # With HARPS there might be two gaps...
    
    for wvl_index in range(len(wavelength)-1):
        
        if (wavelength[wvl_index+1]- wavelength[wvl_index]) > 5: 
            
            wvl_RdBl_gap.append([wavelength[wvl_index],wavelength[wvl_index+1]]) # there should only be one for UVES, two for HARPS
                                    
    flux_normalised_stitch = []
    error_flux_normalised_stitch = []
    wavelength_normalised_stitch = []
    continuum_stitch = []
    
    N_windows = len(geslines_synth)
        
    for zones in range(len(geslines_synth)):
#    for zones in range(31,32):
        
#        print(zones)
        
        if len(geslines_synth) < N_windows: 
            
            # This means the array has changed i.e. a window has been removed.
            
            if zones == len(geslines_synth): # i.e. new end of the window + 1
                
                break
        
        wavelength_z_min = geslines_synth[zones][0]
        wavelength_z_max = geslines_synth[zones][1]
        
#        gap_counter = 0 # Need to run through gaps if there are multiple

        for gap_counter in range(0,len(wvl_RdBl_gap)): # if UVES, should only occur once correct?
            
            geslines_synth_new = []
            zone_gap_exits = []    
            
            # Need to check if the entire gap is within the next zone 
            if zones < len(geslines_synth) - 1:
                
    #            print(gap_counter,zones)
                
                wavelength_z_min_next = geslines_synth[zones+1][0]
                wavelength_z_max_next = geslines_synth[zones+1][1]
                                
                if wvl_RdBl_gap[gap_counter][0] > wavelength_z_min_next: 
                    
                    if wvl_RdBl_gap[gap_counter][1] < wavelength_z_max_next:
                                            
                        # If this zone contains the gap, just skip for now
                        print("Significant gap in next zone of spectra")
                        print("Gap occurs in windows =",wavelength_z_min_next,",",wavelength_z_max_next)
    #                    print("Skipping zone =",zones)
                        
    #                    continue
                        
                        print("Creating wavelength window limits for zone")
                        
                        print("New wavelength window for zone =",zones)
                        
                        geslines_synth[zones][1] = wvl_RdBl_gap[gap_counter][0] # current window end needs to be the window gaps begining 
                        wavelength_z_max = geslines_synth[zones][1]
                        
                        print(wavelength_z_min,wavelength_z_max)
    
                        # Need to make the window next + 1 beginning to be the end of this gap
                        
                        print("New wavelength window for zone =",zones+2)
                        
                        geslines_synth[zones+2][0] = wvl_RdBl_gap[gap_counter][1]
                        
                        print(geslines_synth[zones+2][0],geslines_synth[zones+2][1])
                        
                        # Now need to get rid of element window in zone + 1
                                            
                        zone_gap_exits = [zones + 1]
                        
            if len(zone_gap_exits)>0:
                        
                for zone_search in range(len(geslines_synth)):
                                                
                    if zone_search == zone_gap_exits[0]:
                                    
                        continue # do not append spectra
                    else:
                                    
                        geslines_synth_new.append(geslines_synth[zone_search])
                
                geslines_synth =  geslines_synth_new     
            
                gap_counter = gap_counter + 1
                
                # Now that an element has been removed, the loop will continue to 1 element beyond array size
                
                # Need to fix this cleanly
                
        flux_obs_zone = flux[wavelength <= wavelength_z_max]        
        error_flux_zone = error_flux[wavelength <= wavelength_z_max]
        wavelength_obs_zone = wavelength[wavelength <= wavelength_z_max]
        
        flux_obs_zone = flux_obs_zone[wavelength_obs_zone >= wavelength_z_min]
        error_flux_zone = error_flux_zone[wavelength_obs_zone >= wavelength_z_min]
        wavelength_obs_zone = wavelength_obs_zone[wavelength_obs_zone >= wavelength_z_min]    
        

        if len(flux_obs_zone) == 0:
            
            print("Zone contains no data, continuum normalisation not possible")
            print("Wavlength range =",wavelength_z_min,",",wavelength_z_max)
            print("Skipping zone")
            
            continue
                        
        wavelength_norm,flux_norm,error_flux_norm,continuum_vardon = continuum(wavelength_obs_zone,flux_obs_zone,error_flux_zone,SNR_star,True)
        
        #secondary normalisation
        # need to check if zones are allowed to be normalised again
        
        ### NEED TO CHECK FOR ROGUE EMISSION LINES
                    
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
        # Secodnary is performed
        
        if forbidden_secondary != True:
            
#            flux_norm = sigclip(flux_norm,3)
            
            wavelength_norm,flux_norm,error_flux_norm,continuum_vardon = secondary_normalisation_given_zone(wavelength_norm,flux_norm,error_flux_norm,SNR_star,2.5,True)

            print("Secondary normalisation performed on zone =",zones)
            
        else: # I.e. forbidden_secondary = True
            
            print("Secondary normalisation not authorised for zone =",zones)
                
#        break
    
        
        if zones == 0:
                
            flux_normalised_stitch = flux_norm
            wavelength_normalised_stitch = wavelength_norm
            continuum_stitch = continuum_vardon
            error_flux_normalised_stitch = error_flux_norm
                    
        else:
            
            flux_normalised_stitch = np.append(flux_normalised_stitch, flux_norm) #  do I stitch with the buffer still there???
            wavelength_normalised_stitch = np.append(wavelength_normalised_stitch, wavelength_norm)
            continuum_stitch = np.append(continuum_stitch,continuum_vardon)
            error_flux_normalised_stitch = np.append(error_flux_normalised_stitch,error_flux_norm)
        

#    np.savetxt(f"sapp_seg_v1_"+ star_name.replace(".fits","") +".txt",geslines_synth,fmt='%10.2f',delimiter='\t')
    
    print("Zone continuum fitting finished")
    
    ### Here I am noising up the gaps between spectra. 
        
    for wvl_pixel in range(len(flux_normalised_stitch)-1):
        
        wvl_diff = wavelength_normalised_stitch[wvl_pixel+1] - wavelength_normalised_stitch[wvl_pixel]
        
        if wvl_diff > 5: # some arbitrary difference, gap could be smaller! 
                                    
            error_flux_normalised_stitch[wvl_pixel+1] = error_flux_normalised_stitch[wvl_pixel+1] * 1000
            error_flux_normalised_stitch[wvl_pixel] = error_flux_normalised_stitch[wvl_pixel] * 1000
                
    return [flux_normalised_stitch,error_flux_normalised_stitch,wavelength_normalised_stitch,continuum_stitch,geslines_synth]


PLATO_bmk_UVES_list = np.loadtxt("PLATO/PLATO_bmk_UVES_list.txt",dtype=str)

PLATO_bmk_HARPS_list = np.loadtxt("PLATO/PLATO_bmk_HARPS_list.txt",dtype=str)
#
stars_list = PLATO_bmk_UVES_list

#stars_list = PLATO_bmk_HARPS_list

star_spec_name =  "UVES"

#star_spec_name =  "HARPS"

name="NN_results_RrelsigN20.npz"#NLTE

#import_path = "Payne/"
import_path = ""

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

# THIS IS TO CONTINUUM NORMALISE #

''
for spectra in range(len(stars_list)): 
        
    spec_name_split = stars_list[spectra].split(".")
    
    identifier = spec_name_split[len(spec_name_split)-1]
    
    error_synth_flag = []
    
    if identifier == 'fits':
        
        spec = read_fits_spectra("PLATO/PLATO_spec/" + star_spec_name + "/" + stars_list[spectra])

        wavelength = spec[0]
        flux = spec[1]
        error = spec[2]
                
        SNR_star = int(float(stars_list[spectra].split("snr")[1].split("_"+star_spec_name)[0]))
        
        if star_spec_name == "HARPS":
            
            error_synth_flag = True
        
        else:
            
            error_synth_flag = False
                    
        ### Get rid of values which go below zero ###
        
        flux_no_zero = flux[flux>0]
        error_no_zero = error[flux>0]
        wavelength_no_zero = wavelength[flux>0]
        
        
        flux = flux_no_zero
        error = error_no_zero
        wavelength = wavelength_no_zero 
        ### sigma clip before normalisation for cosmic rays ###
        
#        print(stars_list[spectra])
        
#        if stars_list[spectra] == "ADP_alphacenb_snr939_UVES_52.640gm.fits":
#            
#            fig = plt.figure(figsize=(12,12))
#            ax1 = fig.add_subplot(111)
#            
#            ax1.plot(wavelength,flux,'r-')
                
        flux = sigclip(flux,sig=2.5)
        
#        if stars_list[spectra] == "ADP_alphacenb_snr939_UVES_52.640gm.fits":
#            
#            ax1.plot(wavelength,flux,'b-')
#            ax1.set_xlim([5616,5636])
#            plt.savefig("test.pdf",dpi=250)
#            plt.close()
#            break
            
        ### Here we need to continuum normalise ###
                        
        continuum_spec_data = continuum_normalise_spectra(wavelength,flux,error,SNR_star,w0,0)
        
        flux_normalised = continuum_spec_data[0]
        error_flux_normalised = continuum_spec_data[1]
        wavelength_normalised = continuum_spec_data[2]
        continuum_stitch = continuum_spec_data[3]
        geslines_synth = continuum_spec_data[4]
                
        if stars_list[spectra] == "ADP_alphacenb_snr939_UVES_52.640gm.fits":
                        
            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(111)
            
            ax1.axhline(y=1.0,color='k',linestyle='--')
            ax1.plot(wavelength_normalised,flux_normalised,'r-')
            plt.show()
            
            break
            
        
                            
#        spectrum_continuum_normalised_hr10 = np.vstack((wavelength_normalised,flux_normalised,error_flux_normalised,continuum_stitch)).T
        
#        np.savetxt("PLATO/PLATO_spec_cont_norm/" + star_spec_name + "/" + stars_list[spectra].replace(".fits","").replace(".txt","") + f"_error_synth_flag_{error_synth_flag}_cont_norm.txt",spectrum_continuum_normalised_hr10,fmt='%10.5f')    
        
#        break
    
#    if identifier == 'txt':
#        
#        spec = load_txt_spec("PLATO/PLATO_spec/" + star_spec_name + "/" + stars_list[spectra])
#        
#        wavelength = spec[0]
#        flux = spec[1]
#
#        ### Get rid of values which go below zero ###
#        
#        flux_no_zero = flux[flux>0]
#        
#        if len(spec) == 3: # only k2-138 has errors but it is good to put this here in case
#            
#            error = spec[2]
#
#            error_no_zero = error[flux>0]
#       
#            error_synth_flag = False
#
#            
#        else:
#            
#            error = []
#            
#            error_no_zero = error
#            
#            error_synth_flag = True
#        
#        wavelength_no_zero = wavelength[flux>0]
#
#        if star_spec_name == "UVES":
#            
#            SNR_star = 200
#            
#        if star_spec_name == "HARPS":
#            
#            SNR_star = 300
#                
#        flux = flux_no_zero
#        error = error_no_zero
#        wavelength = wavelength_no_zero
#        
#        ### sigma clip before normalisation for cosmic rays ###
#        
#        flux = sigclip(flux,sig=3)
#            
#        ### Here we need to continuum normalise ###
#                
#        continuum_spec_data = continuum_normalise_spectra(wavelength,flux,error,SNR_star,w0,0)
#                
#        flux_normalised = continuum_spec_data[0]
#        error_flux_normalised = continuum_spec_data[1]
#        wavelength_normalised = continuum_spec_data[2]
#        continuum_stitch = continuum_spec_data[3]
#        geslines_synth = continuum_spec_data[4]
#        
#        spectrum_continuum_normalised_hr10 = np.vstack((wavelength_normalised,flux_normalised,error_flux_normalised,continuum_stitch)).T
#        
#        np.savetxt("PLATO/PLATO_spec_cont_norm/" + star_spec_name + "/" + stars_list[spectra].replace(".fits","").replace(".txt","") + f"_snr{SNR_star}_error_synth_flag_{error_synth_flag}_cont_norm.txt",spectrum_continuum_normalised_hr10,fmt='%10.5f')    
    
#        break
    
    ### then we save spec ###
    
    ### load spec and convolve down ###
''

'''
UVES_cont_norm_spec = os.listdir("PLATO/PLATO_spec_cont_norm/UVES")    

HARPS_cont_norm_spec = os.listdir("PLATO/PLATO_spec_cont_norm/HARPS")    

#star_conv_list = UVES_cont_norm_spec

star_conv_list = HARPS_cont_norm_spec


if '.DS_Store' in star_conv_list:
    star_conv_list.remove('.DS_Store')

#star_spec_name = "UVES"

star_spec_name = "HARPS"

hr10_resolution = 18000# hr10 should be around 18,000

for spectra in range(len(star_conv_list)):
    
    spec = np.loadtxt("PLATO/PLATO_spec_cont_norm/" + star_spec_name + "/" + star_conv_list[spectra])
    
    wavelength = spec[:,0]
    flux = spec[:,1]
    error_flux = spec[:,2]
    
    ### need to check if the error was synthetic ###
    
    ### if it was, may need to re-calculate based on rms ###
    
    ### CONVOLVE ###
    
    # Spacing needs to be the same
    
    # Some fluxes had some below zero, those shouldn't be allowed
    
    wavelength_min = min(wavelength)
    wavelength_max = max(wavelength)
    wavelength_length = len(wavelength)
    
    wavelength_regular = np.linspace(wavelength_min,wavelength_max,wavelength_length,endpoint=True)
    
    # interpolate onto regular spacing # 
      
    flux = np.interp(wavelength_regular,wavelength,flux)
    error_flux = np.interp(wavelength_regular,wavelength,error_flux)
    wavelength = np.interp(wavelength_regular,wavelength,wavelength)
        
    flux_con = convolve_python(wavelength,flux,hr10_resolution)
    error_flux_con = convolve_python(wavelength,error_flux,hr10_resolution)
    
#    print(len(flux))
    
#    
#    wavelength, flux_con = instrument_conv_res(wavelength,flux,hr10_resolution)
#    wavelength, error_flux_con = instrument_conv_res(wavelength,error_flux,hr10_resolution)
    
    # interpolate obs onto model
#    
#    flux = np.interp(w0,wavelength,flux)
#    error_flux = np.interp(w0,wavelength,error_flux)
#    wavelength = np.interp(w0,wavelength,wavelength)
#    wavelength = np.interp(w0,wavelength,wavelength)
        
    # should w0 now be the observed wavelength, if cut was done properly

    spectrum_conv_cont_norm_hr10 = np.vstack((wavelength,flux_con,error_flux_con)).T
    
    np.savetxt("PLATO/PLATO_spec_convolve/" + star_spec_name + "/" + star_conv_list[spectra].replace(".txt","") + "_convolved_hr10_.txt",spectrum_conv_cont_norm_hr10,fmt='%10.5f')    
    
    print(f"{star_conv_list[spectra]} spectra interpolated, convolved, and saved")
    
#    break
'''


