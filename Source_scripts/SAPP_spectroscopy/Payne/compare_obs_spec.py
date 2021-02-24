#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:55:37 2020

@author: gent
"""

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits as pyfits


# use LaTex font
from matplotlib.pyplot import rc

font = {'family': 'serif',
       'serif': ['Computer Modern'],
       'size': 26}

rc('font', **font)
rc('text', usetex=True)


def RV_correction_vel_to_wvl_non_rel(waveobs,xRV): 
    
    """ 
    
    This applies RV correction assuming it is known e.g. gravitational redshift
    for the Sun is 0.6 km/s
    
    N.B. xRV is RV correction in km/s 
    
    """
    
    CCC  = 299792.458 # SPOL km/s
    
    for pixel in range(len(waveobs)):
    
        waveobs[pixel] = waveobs[pixel]/((xRV)/CCC + 1.)
        
    return waveobs

def load_TS_spec(path):
    """loads spectra from TS"""
    
    # print("THIS IS THE PATHWAY",path)
    
    spec = np.loadtxt(path)
    wavelength = spec[:,0] # angstrom
    flux = spec[:,1] # normalised
    
    return wavelength,flux


star_conv_list_HARPS = np.loadtxt("PLATO/HARPS_stars_conv_list.txt",dtype=str)

### eps vir and k2-138 has not been calculated for most recent data set

# remove index 32 and the last one from this list

star_conv_list_HARPS_edit = []

for star_removal in range(len(star_conv_list_HARPS)):
    
    if star_removal == 32: # eps vir
    
        continue
    
    elif star_removal == len(star_conv_list_HARPS)-1:
        
        continue
    
    else:
    
        star_conv_list_HARPS_edit.append(star_conv_list_HARPS[star_removal])     

star_conv_list_UVES = np.loadtxt("PLATO/UVES_stars_conv_list.txt",dtype=str)

HARPS_rv_corrections = np.loadtxt("PLATO/HARPS_UVES_NLTE_RESULTS_cheb_0/HARPS_NLTE_rv_results_list_all_cheb_0.txt")

UVES_rv_corrections = np.loadtxt("PLATO/HARPS_UVES_NLTE_RESULTS_cheb_0/UVES_NLTE_rv_results_list_all_cheb_0.txt")

### ALF CEN A ###

alfcenA_UVES = np.hstack((star_conv_list_UVES[:5],star_conv_list_UVES[len(star_conv_list_UVES)-4]))
alfcenA_UVES_rv = np.hstack((UVES_rv_corrections[:5],UVES_rv_corrections[len(UVES_rv_corrections)-4]))

# need to add UVES/ as a path

alfcenA_UVES_new = []

for add_path_index in range(len(alfcenA_UVES)):
    
   alfcenA_UVES_new.append( "UVES/" + alfcenA_UVES[add_path_index])

alfcenA_HARPS = [star_conv_list_HARPS_edit[30]]
alfcenA_HARPS_rv = [HARPS_rv_corrections[30]]

# need to add HARPS/ as a path

alfcenA_HARPS_new = []

for add_path_index in range(len(alfcenA_HARPS)):
    
    alfcenA_HARPS_new.append("HARPS/" + alfcenA_HARPS[add_path_index])

alfcenA_list = np.hstack((alfcenA_UVES_new,alfcenA_HARPS_new))
alfcenA_rv_list = np.hstack((alfcenA_UVES_rv,alfcenA_HARPS_rv))

alfcenA_snr = []

for snr_i in range(len(alfcenA_list)):
    
    alfcenA_snr.append(alfcenA_list[snr_i].split("_snr")[1].split("_")[0])

alfcenA_snr = np.array(alfcenA_snr,dtype=float)

### ALF CEN B ###

alfcenB_UVES = star_conv_list_UVES[5:10]
alfcenB_UVES_rv = UVES_rv_corrections[5:10]

# need to add UVES/ as a path

alfcenB_UVES_new = []

for add_path_index in range(len(alfcenB_UVES)):
    
   alfcenB_UVES_new.append( "UVES/" + alfcenB_UVES[add_path_index])

alfcenB_HARPS = [star_conv_list_HARPS_edit[31]]
alfcenB_HARPS_rv = [HARPS_rv_corrections[31]]

# need to add HARPS/ as a path

alfcenB_HARPS_new = []

for add_path_index in range(len(alfcenB_HARPS)):
    
    alfcenB_HARPS_new.append("HARPS/" + alfcenB_HARPS[add_path_index])

alfcenB_list = np.hstack((alfcenB_UVES_new,alfcenB_HARPS_new))
alfcenB_rv_list = np.hstack((alfcenB_UVES_rv,alfcenB_HARPS_rv))

alfcenB_snr = []

for snr_i in range(len(alfcenB_list)):
    
    alfcenB_snr.append(alfcenB_list[snr_i].split("_snr")[1].split("_")[0])

alfcenB_snr = np.array(alfcenB_snr,dtype=float)

### new list ### 

alfcenB_check_list = alfcenB_list[0]
alfcenB_rv_check_list = alfcenB_rv_list[0]
alfcenB_snr_check = alfcenB_snr[0]

alfcenAB_list = np.hstack((alfcenA_list[len(alfcenA_list)-2:],alfcenB_check_list,alfcenB_list[len(alfcenB_list)-1]))
alfcenAB_rv_list = np.hstack((alfcenA_rv_list[len(alfcenA_list)-2:],alfcenB_rv_check_list,alfcenB_rv_list[len(alfcenB_list)-1]))
alfcenAB_snr = np.hstack((alfcenA_snr[len(alfcenA_list)-2:],alfcenB_snr_check,alfcenB_snr[len(alfcenB_list)-1]))

# this combined list should have all Alf Cen A and the plausible alf cen A we have

# note all but 1 alf cen B spec are consistently 600 K off

### SUN ###

SUN_UVES = [star_conv_list_UVES[27]]
SUN_UVES_rv = [UVES_rv_corrections[27]]

SUN_UVES_new = []

for add_path_index in range(len(SUN_UVES)):
    
    SUN_UVES_new.append("UVES/" + SUN_UVES[add_path_index])

SUN_HARPS = np.hstack((star_conv_list_HARPS_edit[27:30],star_conv_list_HARPS_edit[len(star_conv_list_HARPS_edit)-1]))
SUN_HARPS_rv = np.hstack((HARPS_rv_corrections[27:30],HARPS_rv_corrections[len(star_conv_list_HARPS_edit)-1]))

SUN_HARPS_new = []

for add_path_index in range(len(SUN_HARPS)):
    
    SUN_HARPS_new.append("HARPS/" + SUN_HARPS[add_path_index])
    
SUN_list = np.hstack((SUN_UVES_new,SUN_HARPS_new))
SUN_rv_list = np.hstack((SUN_UVES_rv,SUN_HARPS_rv))

SUN_snr = []

for snr_i in range(len(SUN_list)):
    
    SUN_snr.append(SUN_list[snr_i].split("_snr")[1].split("_")[0])

SUN_snr = np.array(SUN_snr,dtype=float)


### 18 Sco ###

Sco18_UVES = [star_conv_list_UVES[25]]
Sco18_UVES_rv = [UVES_rv_corrections[25]]

Sco18_UVES_new = []

for add_path_index in range(len(Sco18_UVES)):
    
    Sco18_UVES_new.append("UVES/" + Sco18_UVES[add_path_index])

Sco18_HARPS = star_conv_list_HARPS_edit[2:9]
Sco18_HARPS_rv = HARPS_rv_corrections[2:9]

Sco18_HARPS_new = []

for add_path_index in range(len(Sco18_HARPS)):
    
    Sco18_HARPS_new.append("HARPS/" + Sco18_HARPS[add_path_index])

Sco18_list = np.hstack((Sco18_UVES_new,Sco18_HARPS_new))
Sco18_rv_list = np.hstack((Sco18_UVES_rv,Sco18_HARPS_rv))

Sco18_snr = []

for snr_i in range(len(Sco18_list)):
    
    Sco18_snr.append(Sco18_list[snr_i].split("_snr")[1].split("_")[0])

Sco18_snr = np.array(Sco18_snr,dtype=float)

# we have the names, rv corrections and snr values all associated with eachother

# now time to plot all of these together 

path = "PLATO/PLATO_spec_convolve/"

# path = "PLATO/PLATO_spec_cont_norm/"


### now we need to load up the spectra, rv correct them, and plot them into a 
# 3-4 panel figure

diagnose_star_list = ['alfcenB','SUN','Sco18','alfcenAB']

diagnose_star_list_spec_names = [alfcenB_list,SUN_list,Sco18_list,alfcenAB_list]

diagnose_star_list_rv = [alfcenB_rv_list,SUN_rv_list,Sco18_rv_list,alfcenAB_rv_list]

diagnose_star_list_snr = [alfcenB_snr,SUN_snr,Sco18_snr,alfcenAB_snr]

# need to load up temperature differences

temp_diff = np.loadtxt("PLATO/Parameter_difference_PLATO_stars_cheb_0.txt",dtype=str) # diff = model - reference

def plot_obs_spec_collection(input_path,spec_names,rv_corrections,snr_values,star_name,temp_diff):
    
    # spec_names_new = []
    # rv_corrections_new = []
    # snr_values_new = []
    
    # for i in range(len(spec_names)):
        
    #     if i == 0:
            
    #         continue
        
    #     elif i == 5:
            
    #         continue
        
    #     else:
            
    #         spec_names_new.append(spec_names[i])
    #         rv_corrections_new.append(rv_corrections[i])
    #         snr_values_new.append(snr_values[i])
    
    # spec_names = np.array(spec_names_new)
    # rv_corrections = np.array(rv_corrections_new)
    # snr_values_new = np.array(snr_values_new)
    
    # create figure space before the loop
    
    fig, ax = plt.subplots(4,1, figsize = (25,35))
    
    spectra_buffer = 10 # angstroms
    
    w0 = np.linspace(5329.,5616,11480)[40:-40]
    w0=w0[::2]
    
    w0_range = max(w0) - min(w0) # observed are interpolated on model scale (and cut)
    
    # create 4 windows for comparison
    
    w0_range_1 = [min(w0) + spectra_buffer,min(w0) + 1*(w0_range/4)]
    w0_range_2 = [min(w0) + 1*(w0_range/4),min(w0) + 2*(w0_range/4)]
    w0_range_3 = [min(w0) + 2*(w0_range/4),min(w0) + 3*(w0_range/4)]
    w0_range_4 = [min(w0) + 3*(w0_range/4),min(w0) + 4*(w0_range/4)]

    ax[0].axhline(y=1,color='gray',linestyle='--')
    ax[1].axhline(y=1,color='gray',linestyle='--')
    ax[2].axhline(y=1,color='gray',linestyle='--')
    ax[3].axhline(y=1,color='gray',linestyle='--')
        
    # import matplotlib as mpl
    
    cm_col = plt.get_cmap('Accent')
    
    NUM_COLORS = len(spec_names)

    for star_loop in range(len(spec_names)):
        
        ### load up the spectra 
        
        spec_dummy_wvl,spec_dummy_flux =\
            load_TS_spec(input_path + f"{spec_names[star_loop]}")
            
        spec_name_temp_diff_check = spec_names[star_loop].replace("HARPS/","").replace("UVES/","").split("_error")[0]
        
        ### go through loop of stars, find spectra, print temp difference ###
        
        for temp_diff_loop in range(len(temp_diff)):
            
            if temp_diff[temp_diff_loop][0] == spec_name_temp_diff_check:
                
                temp_diff_legend = temp_diff[temp_diff_loop][1]
                logg_diff_legend = temp_diff[temp_diff_loop][2]
                feh_diff_legend = temp_diff[temp_diff_loop][3]
            
        legend_name = spec_names[star_loop].replace("UVES/","").replace("HARPS/","").split("_error")[0]
        # rv correct the spectra
        
        spec_dummy_wvl = RV_correction_vel_to_wvl_non_rel(spec_dummy_wvl,rv_corrections[star_loop])
        
        # split spectra into 4 windows
        
        obs_1 = spec_dummy_flux[spec_dummy_wvl <= w0_range_1[1]] 
        wvl_corrected_1 = spec_dummy_wvl[spec_dummy_wvl <= w0_range_1[1]]
        obs_1 = obs_1[wvl_corrected_1 >= w0_range_1[0]] 
        wvl_corrected_1 = wvl_corrected_1[wvl_corrected_1 >= w0_range_1[0]]

        obs_2 = spec_dummy_flux[spec_dummy_wvl <= w0_range_2[1]] 
        wvl_corrected_2 = spec_dummy_wvl[spec_dummy_wvl <= w0_range_2[1]]
        obs_2 = obs_2[wvl_corrected_2 >= w0_range_2[0]] 
        wvl_corrected_2 = wvl_corrected_2[wvl_corrected_2 >= w0_range_2[0]]

        obs_3 = spec_dummy_flux[spec_dummy_wvl <= w0_range_3[1]] 
        wvl_corrected_3 = spec_dummy_wvl[spec_dummy_wvl <= w0_range_3[1]]
        obs_3 = obs_3[wvl_corrected_3 >= w0_range_3[0]] 
        wvl_corrected_3 = wvl_corrected_3[wvl_corrected_3 >= w0_range_3[0]]

        obs_4 = spec_dummy_flux[spec_dummy_wvl <= w0_range_4[1]] 
        wvl_corrected_4 = spec_dummy_wvl[spec_dummy_wvl <= w0_range_4[1]]
        obs_4 = obs_4[wvl_corrected_4 >= w0_range_4[0]] 
        wvl_corrected_4 = wvl_corrected_4[wvl_corrected_4 >= w0_range_4[0]]
        
        # plot 4 windows
        
        ax[0].plot(wvl_corrected_1,obs_1,linestyle = '-',color=cm_col(1.*star_loop/NUM_COLORS),label=legend_name.replace("_","") + f" $\Delta$T = {temp_diff_legend} K")
        ax[1].plot(wvl_corrected_2,obs_2,linestyle = '-',color=cm_col(1.*star_loop/NUM_COLORS),label=legend_name.replace("_","") + f" $\Delta$T = {temp_diff_legend} K")
        ax[2].plot(wvl_corrected_3,obs_3,linestyle = '-',color=cm_col(1.*star_loop/NUM_COLORS),label=legend_name.replace("_","") + f" $\Delta$T = {temp_diff_legend} K")
        ax[3].plot(wvl_corrected_4,obs_4,linestyle = '-',color=cm_col(1.*star_loop/NUM_COLORS),label=legend_name.replace("_","") + f" $\Delta$T = {temp_diff_legend} K")
                
        # break

    ax[0].set_ylabel('Flux')
    ax[1].set_ylabel('Flux')
    ax[2].set_ylabel('Flux')
    ax[3].set_ylabel('Flux')

    ax[3].set_xlabel('$\lambda$ [$\AA$]')
    
    ax[0].legend(loc='lower right',fontsize=14,bbox_to_anchor=(1, 0.00),ncol=1)
    ax[1].legend(loc='lower right',fontsize=14,bbox_to_anchor=(1, 0.00),ncol=1)
    ax[2].legend(loc='lower right',fontsize=14,bbox_to_anchor=(1, 0.00),ncol=1)
    ax[3].legend(loc='lower right',fontsize=14,bbox_to_anchor=(1, 0.00),ncol=1)
        
    plt.savefig(f"PLATO/{star_name}_obs_comparison.pdf",dpi=200)
    plt.close("all")


# plot_obs_spec_collection(path,diagnose_star_list_spec_names[0],diagnose_star_list_rv[0],diagnose_star_list_snr[0],diagnose_star_list[0],temp_diff)         
# plot_obs_spec_collection(path,diagnose_star_list_spec_names[1],diagnose_star_list_rv[1],diagnose_star_list_snr[1],diagnose_star_list[1],temp_diff)         
# plot_obs_spec_collection(path,diagnose_star_list_spec_names[2],diagnose_star_list_rv[2],diagnose_star_list_snr[2],diagnose_star_list[2],temp_diff)         
# plot_obs_spec_collection(path,diagnose_star_list_spec_names[3],diagnose_star_list_rv[3],diagnose_star_list_snr[3],diagnose_star_list[3],temp_diff)         



spec_sun_espresso_name = "PLATO/solar_spec_vardan/sun_espresso_s1d_rv.fits"

spec_sun_harps_name = "PLATO/solar_spec_vardan/sun_harps_rv.fits"

spec_sun_pepsi_name = "PLATO/solar_spec_vardan/sun_pepsi_1d_rv.fits"

def read_fits_spectra(path):
        
    hdulist = pyfits.open(path)
    
    print(hdulist.info())
    
    # scidata = hdulist[0].data
    
    # print(min(scidata),max(scidata))
    
    # wavelength = []
    # flux = []
    # error = []
    
    # for data_loop in range(len(scidata)):
        
    #     wavelength.append(scidata[data_loop][0])
    #     flux.append(scidata[data_loop][1])
    #     error.append(scidata[data_loop][2])
        
#    
    # wavelength = scidata[0][0]
    # flux = scidata[0][1]
#    error = scidata[0][2]
    
    # return [wavelength,flux,error]
    # return [wavelength,flux]



read_fits_spectra(spec_sun_espresso_name)
read_fits_spectra(spec_sun_harps_name)
read_fits_spectra(spec_sun_pepsi_name)


'''
buffer_1 = 30
buffer_2 = 0 # angstroms

fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(111)

for HARPS_loop in range(len(alfcenB_HARPS)):

    spec_name = alfcenB_HARPS[HARPS_loop]#.replace("_convolved_hr10_","")
    
    HARPS_spec_dummy_wvl,HARPS_spec_dummy_flux =\
     load_TS_spec(path + f"HARPS/{spec_name}")
     
    HARPS_spec_dummy_wvl = RV_correction_vel_to_wvl_non_rel(HARPS_spec_dummy_wvl,alfcenB_HARPS_rv[HARPS_loop])
     
    ax1.plot(HARPS_spec_dummy_wvl,HARPS_spec_dummy_flux,label=f'HARPS, snr = {HARPS_snr[HARPS_loop]}')

for UVES_loop in range(len(alfcenB_UVES)):

    spec_name = alfcenB_UVES[UVES_loop]#.replace("_convolved_hr10_","")
    
    UVES_spec_dummy_wvl,UVES_spec_dummy_flux =\
     load_TS_spec(path + f"UVES/{spec_name}")
     
    UVES_spec_dummy_wvl = RV_correction_vel_to_wvl_non_rel(UVES_spec_dummy_wvl,alfcenB_UVES_rv[UVES_loop])
     
    ax1.plot(UVES_spec_dummy_wvl,UVES_spec_dummy_flux,label=f'UVES, snr = {UVES_snr[UVES_loop]}')
    
ax1.legend(loc='lower right',fontsize=12)

#wavelength scale of the models # this is important as obs were cut to these during continuum normalisation
w0=np.linspace(5329.,5616,11480)[40:-40]
w0=w0[::2]


#ax1.set_xlim([min(w0)+buffer_1,min(w0)+buffer_2])

ax1.set_xlim([max(w0)-buffer_1,max(w0)-buffer_2])


ax1.set_xlabel("$\lambda$ [$\AA$]")
ax1.set_ylabel("Flux")
    
#plt.show()
fig.savefig("PLATO/AlfCenB_obs_comparison.pdf",dpi=250)
'''




















