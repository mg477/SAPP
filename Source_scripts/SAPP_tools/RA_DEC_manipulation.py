#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:24:41 2020

@author: gent
"""

def current_RA_DEC(RA,DEC,prop_RA,prop_DEC,yrs_init,yrs_current):
    
    """
    RA arr [hh,mm,ss] float
    DEC [deg,arcmin,arcsec]: [hh,mm,ss]
    prop_RA (mas): float, proper motion
    prop_DEC (mas): float, proper motion
    yrs_init: time when RA and DEC were recorded
    yrs_current: time of predicited RA and DEC
    return: new RA and DEC 
    """
        
    RA_hours = RA[0]
    RA_minutes = RA[1]
    RA_seconds = RA[2]
    
    DEC_deg = DEC[0]
    DEC_arcmin = DEC[1]
    DEC_arcsec = DEC[2]
    
    yrs_change = yrs_current - yrs_init
    
    RA_change = prop_RA * yrs_change/1000 # arcseconds
    
    DEC_change = prop_DEC * yrs_change/1000 # arcseconds
            
    if RA_change > 60:
        
        print("Star has moved more than 1 arcminute, are you sure?")
        
        return 0

    elif DEC_change > 60:
        
        print("Star has moved more than 1 arcminute, are you sure?")
        
        return 0 
    
    else:
        
        print("RA change =",RA_change)
        print("DEC change =",DEC_change)
        
        RA_seconds = RA_seconds + RA_change
        
        DEC_arcsec = DEC_arcsec + DEC_change
        
        print([RA_hours,RA_minutes,RA_seconds],[DEC_deg,DEC_arcmin,DEC_arcsec])
        
        return [RA_hours,RA_minutes,RA_seconds],[DEC_deg,DEC_arcmin,DEC_arcsec]
    
def convert_RA_DEC_deg(RA,DEC):
    
    from astropy.coordinates import SkyCoord
#    from astropy import units

    
    RA_hours = RA[0]
    RA_minutes = RA[1]
    RA_seconds = RA[2]
    
    DEC_deg = DEC[0]
    DEC_arcmin = DEC[1]
    DEC_arcsec = DEC[2]
    
    RA_inp = str(RA_hours)+"h"+str(RA_minutes)+"m"+str(RA_seconds)+"s"
    
    DEC_inp = str(DEC_deg)+"d"+str(DEC_arcmin)+"m"+str(DEC_arcsec)+"s"
        
    converted = SkyCoord(RA_inp,DEC_inp, frame='icrs')
    
    print(converted)
    
    return converted

RA_new,DEC_new = current_RA_DEC([7,39,18.11950],[5,13,29.9552],-714.59,-1036.80,2007,2018)

convert_RA_DEC_deg(RA_new,DEC_new)

