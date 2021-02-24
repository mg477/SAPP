#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:19:38 2020

@author: gent
"""
import numpy as np
from astropy.io import fits as pyfits
import os

# def tbdfits(path):
    
    
    
#     flux = x  # Not really sure what this line is meant to do? It will do something different in Python because of pass-by-reference!
#     cr = "GET FROM ASTROPY, KEY IS CRVAL1"
#     # IDL's WHERE function returns indices, np.where() returns an array where the elements are true
#     # Use a conditional instead as a mask
#     cdelt = "GET FROM ASTROPY, KEY IS CD1_1"
#     wavelength = np.linspace(0, len(flux)) * cdelt + cr

#     return wavelength

def read_fits_spectra(path):
        
    hdulist = pyfits.open(path)
    
    print(hdulist.info())
    
    # print(hdulist[0].header['DATE'])    
    
    
    scidata = hdulist[0].data

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
    
    return wavelength,flux,error


def read_fits_GES_GIR_spectra(path):
        
    hdulist = pyfits.open(path)
        
    flux_norm = hdulist[2].data # this is the normalised spectrum
    
    wav_zero_point = hdulist[0].header['CRVAL1'] # wavelength zero point
    
    wav_increment = hdulist[0].header['CD1_1'] # wavelength increment
    
    wavelength = wavelength = np.arange(0,len(flux_norm))*wav_increment + wav_zero_point
    
    rv_shift = hdulist[5].data['VEL'][0] # 5 is the velocity column, documentation says to use this
    
    rv_shift_err = hdulist[5].data['EVEL'][0] # km/s
    
    SNR_med = hdulist[5].data['S_N'][0] 
    
    error_med = flux_norm/SNR_med # can't seem to find error from fits
    
    return wavelength,flux_norm,error_med,rv_shift,rv_shift_err


path = "../Payne/PLATO/SUN_gaiaeso_gir"

path_list = os.listdir(path)

for i in range(len(path_list)):
        
    spectral_path = f"{path}/{path_list[i]}"
    
    # need to check for hr10
        
    if ("H548.8" in spectral_path):
    
        wavelength,\
        flux_norm,\
        error_med,\
        rv_shift,\
        rv_shift_err = read_fits_GES_GIR_spectra(f"{path}/{path_list[i]}")
        
        print(wavelength,flux_norm)



