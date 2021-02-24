import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from astropy.io import fits as pyfits


plt.rc('font', family='serif')
plt.rc('font', serif = 'Computer Modern Roman')
plt.rc('text', usetex='true')
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 18

def clean_spec(wave, flux, number):
	
#	if wave[0] < 3725:
#		i = 0
#		while wave[i] < 3725:
#			i += 1
#		start = i
#	else:
#		i = 0
#		start = 0
#	
#	if wave[len(wave)-1] > 5075:
#		while wave[i] <= 5075:
#			i += 1
#		end = i
#	else:
#		end = len(wave)
	
	start = 0
	end = len(wave)-1
    
	wave_new = []
	flux_new = []
	wave_clean = []
	flux_clean = []
	
	for i in range(start,end,1):
		area = []
		if i < start+10:
			#for j in range(0, i+(20-(i-start))):
			#	area.append(flux[i+j])
			flux_clean.append(flux[i])
			wave_clean.append(wave[i])
		elif i > end-10:
			flux_clean.append(flux[i])
			wave_clean.append(wave[i])
		else:
			for j in range(-10,10):
				area.append(flux[i+j])
			m = ((flux[i-1]-flux[i+1])/(wave[i-1]-wave[i+1]))
			b = flux[i-1] - m*wave[i-1]
			test = m*wave[i] + b
			if test >= np.median(area) + number*np.std(area):
				m = ((flux[i-2]-flux[i+2])/(wave[i-2]-wave[i+2]))
				b = flux[i-2] - m*wave[i-2]
				test = m*wave[i] + b
			if test >= np.median(area) + number*np.std(area):
				m = ((flux[i-3]-flux[i+3])/(wave[i-3]-wave[i+3]))
				b = flux[i-3] - m*wave[i-3]
				test = m*wave[i] + b
			if flux[i] >= test + number*np.std(area):
				flux_clean.append(test)
			else:
				flux_clean.append(flux[i])
			wave_clean.append(wave[i])
		wave_new.append(wave[i])
		flux_new.append(flux[i])

	return wave_clean, flux_clean, wave_new, flux_new

#CODE STARTS HERE

#specin = np.loadtxt('txtlist5', dtype='str')

specin = ["PLATO/PLATO_spec/UVES/ADP_alphacenb_snr939_UVES_52.640gm.fits"]

def read_fits_spectra(path,spec_save_txt_bool):
        
    hdulist = pyfits.open(path)
    
#    print(hdulist.info())
    
    scidata = hdulist[1].data
    
    wavelength = scidata[0][0]
    flux = scidata[0][1]
    error = scidata[0][2]
    
    if spec_save_txt_bool == True:
    
        spec_save = np.vstack((wavelength,flux)).T
        
        np.savetxt(path.replace("fits","txt"),spec_save)
    
    return wavelength,flux,error


#specout = np.loadtxt('cleanedlist5', dtype='str')
    
#specout = ["PLATO/PLATO_spec/UVES/ADP_alphacenb_snr939_UVES_52.640gm_fits_clean.txt"]


k = 0
while k < len(specin):

	number = 2.5

	print("Working on %s \n" % (specin[k]))

	print("The magic number is %f" % (number))

#	wave_orig, flux_orig = np.loadtxt(specin[k], unpack = True)
    
	wave_orig, flux_orig, error_orig = read_fits_spectra(specin[k],True)

	clean_wave, clean_flux, new_wave, new_flux = clean_spec(wave_orig, flux_orig, number)
	
	max_y = 0
	i = 0
	while clean_wave[i] < 5000:
	    if clean_flux[i] >= max_y:
	        max_y = clean_flux[i]
	    i += 1
	
	#fig = plt.figure()
	plt.ylim(-100,max_y)
	#plt.xlim(4400,4600)
	plt.xlim(5586,5616)
	plt.plot(wave_orig, flux_orig, 'k-')
	plt.plot(clean_wave, clean_flux, 'r-')
	plt.show()
	plt.close('all')
	
	wave_ratio = []
	ratio_clean = []
	i=0
	while new_wave[i] <= 4900:
		wave_ratio.append(new_wave[i])
		ratio_clean.append(new_flux[i]/clean_flux[i])
		i += 1
	
	#fig = plt.figure()
	#plt.ylim(0,max_y)
	#plt.xlim(4400,4600)
	plt.xlim(5586,5616)
	plt.plot(wave_ratio, ratio_clean, 'k-')
	plt.show()
	plt.close('all')

	#k+=1

	redo = input('Happy with the result? (y/n)')

	if redo == 'n':
		what_next = input('Clean more or go back? (m/b) ')

	while redo == 'n':
		print("The magic number is %f" % (number))
		if input('Would you like to adjust the magic number? (y/n) ') == 'y':
			number = float(input('New magic number = '))

		#k = k - 1

		if what_next == 'm':
			clean_wave, clean_flux, new_wave, new_flux = clean_spec(clean_wave, clean_flux, number)
		elif what_next == 'b':
			clean_wave = new_wave
			clean_flux = new_flux

		max_y = 0
		i = 0
		while clean_wave[i] < 5000:
		    if clean_flux[i] >= max_y:
		        max_y = clean_flux[i]
		    i += 1
		
		#fig = plt.figure()
		plt.ylim(-100,max_y)
		#plt.xlim(4400,4600)
		plt.xlim(5586,5616)
		plt.plot(wave_orig, flux_orig, 'k-')
		plt.plot(clean_wave, clean_flux, 'r-')
		plt.show()
		plt.close('all')
		
		wave_ratio = []
		ratio_clean = []
		i=0
		while new_wave[i] <= 4900:
			wave_ratio.append(new_wave[i])
			ratio_clean.append(new_flux[i]/clean_flux[i])
			i += 1
		
		#fig = plt.figure()
		#plt.ylim(0,max_y)
		#plt.xlim(4400,4600)
		plt.xlim(5586,5616)
		plt.plot(wave_ratio, ratio_clean, 'k-')
		plt.show()
		plt.close('all')
		
		#k+=1

		redo = input('Happy with the result? (y/n) ')

		if redo == 'n':
			what_next = input('Clean more or go back? (m/b) ')

#	f = open(specout[k], 'w')
#	for i in range(len(clean_wave)):
#		print("%f %f" % (clean_wave[i],clean_flux[i]), file = f)
#	f.close

	k+=1
