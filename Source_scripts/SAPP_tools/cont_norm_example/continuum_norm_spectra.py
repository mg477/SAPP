#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:55:59 2020

@author: gent
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# use LaTex font
from matplotlib.pyplot import rc

# font = {'family': 'serif',
#        'serif': ['Computer Modern'],
#        'size': 22}

# rc('font', **font)
rc('text', usetex=False)

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
    
    geslines_synth_no_secondary_norm = [[4835.0,4895.0],[5118.2,5218.66],[5745.34,5755.0],[5871.95,5906.12],[6325.95,6372.25],[6520.5,6600.0]]

    # The windows above are ones which should not go through secondary normalisation due to broad lines such as Halpha
    ###
    
    ### CHECK IF ERROR EXISTS ###
    
    if len(error_flux) > 0:
    
        ### NAN CHECK FOR ERROR ###
    
        N_nan = len(np.where(np.isnan(error_flux)==True)[0])
        
        if N_nan/len(error_flux) > 0.2:
            
            print("More than 20\% of the errors are NaN values")
            print("Estimating error via SNR")
            
            error_flux = 1/SNR_star * flux
    
    else:
        
        error_flux = 1/SNR_star * flux
        
    # basically an error will exist regardless 
        
    ### Search spectra for zeroes!!! ###
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
                        print("Significant gap in next zone of spectra")
                        print("Gap occurs in windows =",wavelength_z_min_next,",",wavelength_z_max_next)
                        
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
            
            print("Zone contains no data, continuum normalisation not possible")
            print("Wavlength range =",wavelength_z_min,",",wavelength_z_max)
            print("Skipping zone")
            
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

            print("Secondary normalisation performed on zone =",zones)
            
        else: # I.e. forbidden_secondary = True
            
            print("Secondary normalisation not authorised for zone =",zones)
                    
        
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
    
    print("Zone continuum fitting finished")
    
    ### Here I am noising up the gaps between spectra. To account for edge effects
        
    for wvl_pixel in range(len(flux_normalised_stitch)-1):
        
        wvl_diff = wavelength_normalised_stitch[wvl_pixel+1] - wavelength_normalised_stitch[wvl_pixel]
        
        if wvl_diff > 5: # Angstroms, some arbitrary difference, gap could be smaller! 
        
            # multiply error by 1000
                                    
            error_flux_normalised_stitch[wvl_pixel+1] = error_flux_normalised_stitch[wvl_pixel+1] * 1000
            error_flux_normalised_stitch[wvl_pixel] = error_flux_normalised_stitch[wvl_pixel] * 1000
                
    return [wavelength_normalised_stitch,flux_normalised_stitch,error_flux_normalised_stitch,continuum_stitch,geslines_synth]


# filename = "UVES_18Sco.txt"
# filename = "UVES_delEri.txt"
filename = "18_sco_HD146233.HARPS-N.dop.txt"
# filename = "18_scoHD146233.HARPS-S.dop.txt"
# filename = "Vesta.dop.txt"

# filename = "final_coadded_spectrum_k2-138.txt"

star_example = np.loadtxt(filename)
wavelength = star_example[:,0]
flux = star_example[:,1]
error_flux = []
SNR_star = 300
continuum_buffer = 0 # angstroms
secondary_continuum_buffer = 2.5 # angstroms
geslines_synth = np.loadtxt("sapp_seg_v1.txt")
med_bool = True

spec_norm_results = continuum_normalise_spectra(wavelength,\
                            flux,\
                            error_flux,\
                            SNR_star,\
                            continuum_buffer,\
                            secondary_continuum_buffer,\
                            geslines_synth,\
                            med_bool)
    
wavelength_normalised_stitch = spec_norm_results[0]
flux_normalised_stitch = spec_norm_results[1]
error_flux_normalised_stitch = spec_norm_results[2]
continuum_stitch = spec_norm_results[3]
geslines_synth = spec_norm_results[4]


''
### PLOTTING NORMALISATION ZONES ###

flux_dummy = flux
wavelength_dummy = wavelength

# flux_dummy = flux_normalised_stitch
# wavelength_dummy = wavelength_normalised_stitch

fig3 = plt.figure(figsize=(12,12))
ax3 = fig3.add_subplot(111)

flux_cut = flux_dummy[wavelength_dummy >= geslines_synth[0][0]]
wavelength_cut = wavelength_dummy[wavelength_dummy >= geslines_synth[0][0]]

flux_cut = flux_cut[wavelength_cut <= geslines_synth[len(geslines_synth)-1][1]]
wavelength_cut = wavelength_cut[wavelength_cut <= geslines_synth[len(geslines_synth)-1][1]]

ax3.plot(wavelength_cut,flux_cut,linestyle='-',color='dodgerblue')

N_cols = len(geslines_synth)

from matplotlib.pyplot import cm

colours_list = cm.rainbow_r(np.linspace(0,1,N_cols))

for seg_windows in range(len(geslines_synth)):
    
    # for each window, plot an opaque zone 
    
    # param_fill_range = np.linspace(geslines_synth[seg_windows][0],geslines_synth[seg_windows][1],10)
    
    ax3.axvline(x=geslines_synth[seg_windows][0],color='tomato',linestyle='--',alpha=0.5)
    ax3.axvline(x=geslines_synth[seg_windows][1],color='tomato',linestyle='--',alpha=0.5)

    # ax3.fill_between(param_fill_range, min(flux_cut), max(flux_cut), color=colours_list[seg_windows],alpha=0.5)


# ax3.set_xlim([min(wavelength_cut),max(wavelength_cut)])

ax3.set_xlim([6250,max(wavelength_cut)])


ax3.set_ylim([min(flux_cut),max(flux_cut)])

ax3.set_ylabel("Flux",fontsize=45)
ax3.set_xlabel("Wavelength [$\AA$]",fontsize=45)

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['axes.linewidth'] = 2.5

plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

max_wavelength_cut_value = max(wavelength_cut)

title = filename.replace("_","").split(".txt")[0]


# plt.show()
fig3.savefig(f"{title}_segment_example_6250_{max_wavelength_cut_value}.png",dpi=100)
# plt.savefig(f"{title}_sigclipped.pdf",dpi=100)
''



# flux_sigclip = sigclip(flux,2.5)


### PLOTTING ### 
'''

fig = plt.figure(figsize=(10,10))

# fig = plt.figure(figsize=(18,12))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

ax1 = fig.add_subplot(111)



ax1.plot(wavelength,flux,'r-',label='pre-clipped flux',linewidth=2)
ax1.plot(wavelength,flux_sigclip,'k-',label='clipped flux',linewidth=2)

# ax2.plot(wavelength_normalised_stitch,flux_normalised_stitch,'k-',label='normalised flux',linewidth=2)
# ax2.plot(wavelength_normalised_stitch,continuum_stitch,'b-',label='continuum')
# ax2.plot(wavelength_normalised_stitch,error_flux_normalised_stitch,'g-',label='error',linewidth=2)

# ax2.set_ylim([0,2])

ax1.set_ylabel("Flux",fontsize=35)

ax1.set_xlabel("Wavelength [$\AA$]",fontsize=35)


# ax2.set_ylabel("Flux",fontsize=35)

# ax2.set_xlabel("Wavelength [$\AA$]",fontsize=35)

ax1.legend(loc="lower right",fontsize=30)
# ax2.legend(loc="upper right",fontsize=30)

ax1.set_xlim([6060,6072])


# ax1.set_xlim([5328,5345])
# ax2.set_xlim([5328,5345])

title = filename.replace("_","").split(".txt")[0]
ax1.set_title(f"{title}",fontsize=35)

# ax2.set_xticks(fontsize=25)
# ax2.set_yticks(fontsize=25)

# axins = ax2.inset_axes([6250,6500,1,2])
# axins = zoomed_inset_axes(ax2, 1, loc=2) # zoom-factor: 2.5, location: upper-left
# axins.plot(wavelength_normalised_stitch[0], flux_normalised_stitch[0],'k-')
# x1, x2, y1, y2 = 5328, 5340, 0, 1.5 # specify the limits
# axins.set_xlim(x1, x2) # apply the x-limits
# axins.set_ylim(y1, y2) # apply the y-limits
# plt.yticks(visible=False)
# plt.xticks(visible=False)
# mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# flux_gap = flux_normalised_stitch[wavelength_normalised_stitch<=5345]
# wavelength_gap = wavelength_normalised_stitch[wavelength_normalised_stitch<=5345]
# flux_gap = flux_gap[wavelength_gap>=5328]
# wavelength_gap = wavelength_gap[wavelength_gap>=5328]
# plt.plot(wavelength_gap,flux_gap,'k-',markersize=0.5)
# plt.setp(axes, yticks=[])


plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['axes.linewidth'] = 2.5

plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

plt.show()

# plt.savefig(f"{title}_sigclipped.pdf",dpi=100)

# plt.savefig(f"{title}_cont_norm.eps",dpi=100)
# plt.savefig(f"{title}_cont_norm_zoom.eps",dpi=100)

'''
