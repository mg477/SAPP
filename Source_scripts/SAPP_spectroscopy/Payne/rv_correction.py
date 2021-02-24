#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:38:54 2020

@author: gent
"""

def rv_cross_corelation(spec_obs,spec_template,title):
    
    wvl = spec_obs[0]
    obs = spec_obs[1]
    usert = spec_obs[2]
    
    spec_template_wvl = spec_template[0]
    spec_template_flux = spec_template[1]
        
    ### the spectra need to be on the same wavelength scale ### 
    
    ### interp obs onto model ###
    
    ### model has already been cut to size ### 
    
    obs = np.interp(spec_template_wvl,wvl,obs)
    usert = np.interp(spec_template_wvl,wvl,usert)
    wvl = np.interp(spec_template_wvl,wvl,wvl)
    
    N_zero_pad = 10000 
    
    obs_zero_padded = np.hstack((np.zeros([N_zero_pad]),obs,np.zeros([N_zero_pad])))
    spec_template_flux_zero_padded = np.hstack((np.zeros([N_zero_pad]),spec_template_flux,np.zeros([N_zero_pad])))
            
    corr = fftpack.ifft(fftpack.fft(obs_zero_padded) * fftpack.fft(np.flip(spec_template_flux_zero_padded)))
    
    power = np.abs(corr)

    wvl_step = abs(wvl[0]-wvl[1])
    
    sample_wvl = fftpack.fftfreq(obs_zero_padded.size, d=wvl_step)
    
    loc_peak = sample_wvl[np.where(sample_wvl==sample_wvl[power==max(power)])[0]]
        
    fig = plt.figure(figsize=(8,8))
    
    ax1 = fig.add_subplot(111)
    
    ax1.plot(sample_wvl,power,'ko',markersize=0.5)    
    
    ax1.set_title(title.split('ADP_')[1].split('_snr')[0])
    
    ax1.set_ylabel("C(x)")
    ax1.set_xlabel("x")
    
    
    # An inner plot to show the peak 
    
    axes = plt.axes([0.55, 0.4, 0.3, 0.3])
    plt.title("Peak \n x$_{peak}$ =" + f'{loc_peak[0]:.4f}' +  "\n $\Delta\lambda$ =" + f'{wvl_step:.4f}' + "$\AA$ \n RV =" + f'{wvl_step * loc_peak[0]:.4f}' + "$\AA$",\
              fontsize=17)
    
    # need to find data around peak 
    
    peak_collect_corr = power[sample_wvl>=-1] 
    peak_collect_wvl = sample_wvl[sample_wvl>=-1]
    peak_collect_corr = peak_collect_corr[peak_collect_wvl<=1] 
    peak_collect_wvl = peak_collect_wvl[peak_collect_wvl<=1]
    
    plt.plot(peak_collect_wvl,peak_collect_corr,'ko',markersize=0.5)
    plt.axvline(x=loc_peak,color='gray',linestyle=':')
    plt.axvline(x=0,color='g',linestyle=':')

    plt.setp(axes, yticks=[])
    
#    plt.show()
    
    title_filename = title.split('ADP_')[1].split('_snr')[0]
    
    plt.savefig(f"PLATO/{star_spec_name}_{title_filename}_cross_correlation_zero_padding.pdf",dpi=250) 
           
    return wvl_step * loc_peak[0]
