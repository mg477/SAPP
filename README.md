# Stellar Abundances and atmospheric Parameters Pipeline (SAPP) Version 1.1

#### Author: Matthew Raymond Gent, Max-Planck Institute for Astronomy, Heidelberg, Germany 
#### Co-Authors: Aldo Serenelli, Maria Bergemann
#### Contributors: Phillip Eitner, Jeffrey Gerber, Mikhail Kovalev, Ekaterina Magg, Ulrike Heiter, Terese Olander, Nicolas Nardetto, Thierry Morel, Nayeem Ebrahimkutty

SAPP is a fully automated python code which culminates different types of 
observation data (photometry, parallax, asteroseismology and spectroscopy) 
in order to determine fundamental and atmospheric parameters.

This pipeline can be used in two different modes: Full or Lite

- Full: Determines parameters from a bayesian framework which calculates and combines 3D 
probability distribtutions.
- Lite: Determines parameters from the Spectroscopy module combined with asteroseismic/surface gravity data

## Requirements
SAPP is developed with Python 3.7

SAPP requires the following python packages:
- numpy 1.18.5
- scipy 1.5.0
- astropy 3.2.3
- matplotlib 3.2.2

## Update notices

The current README needs to be updated inline with the newest commits - M. Gent 

## Installation

There are 3 steps for installation, assuming the above packages have been installed:

- Clone the repository in the directory you want.

- Download stellar evolution models via https://keeper.mpdl.mpg.de/d/5162ad5abd954ae8a457/ into directory:
```bash
SAPP-WEAVE/Input_data/GARSTEC_stellar_evolution_models/evo_tracks
```

There are two memory maps listed in the keeper photometry uses photometry_stellar_track_mmap_v2.npy  and 
asteroseismology uses photometry_stellar_track_mmap.npy in the PLATO setup. 

## Validation and more detail:

Please read the paper at: 

https://www.aanda.org/articles/aa/pdf/2022/02/aa40863-21.pdf

## Input data

You must absolutely have stellar evolution models memory map within the directory

```bash
SAPP-WEAVE/Input_data/GARSTEC_stellar_evolution_models/evo_tracks
```

Currently the code can only run evo tracks tailored to GARSTEC stellar evolution models.

Due to GitHub's file number upload limit, the memory maps are stored on a private keeper (see Installation)
or otherwise contact me. 

## Output data

There are four folders which concern Output data:

- Stars_Lhood_combined_spec_phot

Spectral and photometric PDF grids, statistics as well as best fit parameters
are saved in this directory. Each star/field has a given folder.

- Stars_Lhood_photometry

Photometric and astrometric PDF grids are saved in this directory.
Each star/field has a given folder.

- Stars_Lhood_spectroscopy

Spectroscopy best fit parameters, associated uncertainties and covariance matrix are
saved in this directory. Each star/field has a given folder.

- Stars_Lhood_asteroseismology

Asteroseismic PDF grids/Logg values are saved in this directory. Each star/field has a given folder.

- test_spectra_emasks

Spectra error masks which have been created to be used in this pipeline. 
There are 3 stellar folders representing typical FGK stars in a given temperature and close to solar metallicity. 
These are ```$\delta$``` Eri (low temperature), the Sun (median temperature), and Procyon (high temperature).

## Source scripts

This contains all the scripts necessary to run the pipeline as well as extra. 

- main.py: is the main script used to run the pipeline in different ways, it should be the only
script you will have to edit besides SAPP_best_spec_payne_v1p1.py in the directory SAPP_spectroscopy.

- SAPP_spectroscopy: Spectroscopic module with all the tools to process spectra such 
as RV correction, convolving, cleaning, continuum normalisation as well as calculating best
fit values.

- SAPP_stellar_evolution_scripts: This contains any script used to calculate the photometry and 
asteroseismology grids. These are currently calculated using the same module.

- SAPP_tools: This folder contains functions which are commonly used throughout the pipelines use
and therefore exists for easy access. This also includes querying script for extinction values.

## Photometry module

The script in directory

```bash
Source_scripts/SAPP_stellar_evolution_scripts/photometry_asteroseismology_lhood_TEST.py
```

Calculates the photometric grid. The input data required is located in directory

```bash
Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/PLATO_bmk_phot_data/PLATO_photometry.csv
```

The header is defined in the later section ```Bayesian Framework```. Essentially there are 8 photometric bands in total, 5 non-Gaia i.e. B, V (Johnsons), H, J, Ks (2MASS)
and 3 Gaia i.e. G, Bp, Rp. Included with these are parallaxes from Gaia DR2 (now DR3 is released) and Simbad with associated errors, magnitude extinction from Gaia (if it exists),
 and reddening, E(B-V), queried from the tool Stilism. All of which are used to calculate the photometric chi2
grids for photometry.

There are four grids which can be calculated; Magnitude (Gaia and Non-Gaia); Colour (Gaia and Non-Gaia). 

For the colour grid, only two colours are used, these are V-K and Bp-Rp, this is to ensure the independence
of Gaussian factors i.e. zero covariance is assumed. 

Each grid is calculated by running through stellar evolution models, calculating a chi2 value via the equation

```
$\chi^2$ = $\sum_i \frac{(X_i-\mu)^2}{\sigma^2}$
```

where i represents the model points, $\mu$ is the observed value, $\sigma$ is the uncertainty.

N.B. Photometric uncertainties are calculated by combining the band uncertainty, distance modulus uncertainty
(propagated from parallax), and uncertainty in extinction (propagated from reddening). If the parallax measurement
has no value i.e. is a NaN, the fiducial value is 20$\%$. If the magnitude measurement is a NaN, the fiducial value
is 0.075 magnitude. If the reddening uncertainty measurement is a NaN, the fiducial is 15$\%$ of E(B-V). 
It is unlikely that the values will be NaN, however the option is there incase there are none.

The output of the photometry module is as follows:

[Teff,logg,[Fe/H],non_gaia_mag,gaia_mag,non_gaia_col,gaia_col,Asteroseismolog,Age,Mass,Radius,Age_step,Luminosity]

The units are:

K, dex, dex, --, --, --, --, --, Gyrs, Msol, Rsol, Gyrs, Lsol

The grid is a 13 column array of parameters and chi2 values.

## Asteroseismology module

The Asteroseismic module is similar in structure and the core process which creates
the grid of chi2 values is currently the same script for the photometric values.

Calculates the asteroseismic grid. The input data required is located in directory

```bash
Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/PLATO_bmk_phot_data/Seismology_calculation/PLATO_stars_seism.txt
```

The header is defined in the later section ```Bayesian Framework```. There are two
asteroseismic quantities that the module uses, these are $\nu_{max}$ (maximum frequency) and $\Delta\nu$ (long separation frequency). 

There is only one grid which is calculated which combined these two quantities with associated errors.

Each grid is calculated by running through stellar evolution models, calculating a chi2 value via the same equation as in photometric module.

N.B. Asteroseismic uncertainties are calculated by combining the input variables uncertainty and the corresponding solar value uncertainty.
This is because the solar asteroseismic $\nu_{max}$ and $\Delta\nu$ are used to scale quantities in the model grid. 
If either uncertainty of the variables do not exist i.e. NaN, the fiducial values are: $\Delta\nu_{err}=1.5\ \mu$Hz and $\nu_{max}=150\ \mu$Hz.

The output of the asteroseismic module is as follows:

[Teff,logg,[Fe/H],non_gaia_mag,gaia_mag,non_gaia_col,gaia_col,Asteroseismolog,Age,Mass,Radius,Age_step,Luminosity]

The units are:

K, dex, dex, --, --, --, --, --, Gyrs, Msol, Rsol, Gyrs, Lsol

The grid is a 13 column array of parameters and chi2 values.

## Spectroscopy module

The central script required to perform spectroscopy is called SAPP_best_spec_payne_v1p1.py.

This is located in the directory:

```bash
Source_scripts/SAPP_spectroscopy/
```

The best way to explain how to use spectroscopy is with an example!

First, you need to store your observation spectra in the directory

```bash
Input_data/spectroscopy_observation_data/
```

You can create your own directory to store one or a list of spectra.

Second, you need a Neural Network trained grid of model spectra made by 
The Payne. The training script is currently not in SAPP's architecture, however
we already have a trained file already loaded. This is called ```NN_results_RrelsigN20.npz```
and it is located in the directory 

```bash
Input_data/spectroscopy_model_data/Payne_input_data/
```
This python binary file contains all you need in order to produce a model spectra
in the Gaia ESO Giraffe hr10 format with R=20,000 and wavelength range
5329 ```$\AA$``` to 5616 ```$\AA$```.

Specifically it contains the weights and bias parameters required for the 
Neural Network as well as the limits of the grid it was trained on.

This training grid consists of an 8 Dimensional parameter space with the 
following variables 

```
Teff,logg,[Fe/H],Vmic,Vbrd,[Mg/Fe],[Ti/Fe],[Mn/Fe]
```

Effective temperature, surface gravity, iron metallicity, microturbulence,
velocity broadening (this is effectively vsini), magnesium, titanium and 
manganese metallicity.

There is only two functions you have to have to worry about:

1) find_best_val()
2) "read_spectra_func()"

find_best_val() processes the spectra in several ways, in short it can
continuum normalise, calculate radial velocity correction, convolve the observation
to a lower resolution and match the wavelength scale to the grid of models.


"read_spectra_func()" is in quote marks as this function can vary in structure,
The user should tailor this function specifically the spectra they would like
to analyse.

For example, already in the script there are 3 functions to read the following
spectra:

- Gaia ESO Giraffe fits files, ```read_fits_GES_GIR_spectra``` 
- Generic text files, ```read_txt_spectra```
- HARPS/UVES fits files ```read_fits_spectra```

These all follow similar input which is the variable ```spec_path ```
and output ```wavelength,flux,error_med,rv_shift,rv_shift_err,snr_star ```.
At the very least these outputs should exist, it is possible for rv_shift and 
rv_shift_err to be NaN values. This is because sometimes the RV correction is 
not provided. However the SNR of the star must be known, even a 1st order 
approximation is good enough (For HARPS/UVES you can set the SNR to be 300/200)
respectively.

The input arguments for find_best_val() are described within the doc string
of the function. Below we will go through what all these inputs mean and learn 
what you can do with this module

```python

error_mask_index = 0

emask_kw_instrument = "HARPS" # can be "HARPS" or "UVES"
emask_kw_teff = "solar" # can be "solar","teff_varying"
error_mask_recreate_bool = False # if this is set to True, then emask_kw_teff defaults to "stellar"
error_mask_recreate_arr = [error_mask_recreate_bool,emask_kw_instrument,emask_kw_teff]
error_map_use_bool = True
cont_norm_bool = False
rv_shift_recalc = [False,-100,100,0.05]
conv_instrument_bool = False
input_spec_resolution = 18000
numax_iter_bool = False
niter_numax_MAX = 5
numax_input_arr = [3170,159,niter_numax_MAX]
recalc_metals_bool = False
feh_recalc_fix_bool = False
recalc_metals_inp = [5770,4.44,0,feh_recalc_fix_bool] 

ind_spec_arr = [[spec_path,\
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
                logg_fix_input_arr],stellar_names[inp_index],spec_type]
                
find_best_val(ind_spec_arr)
```

The spectra chosen is a 18 sco HARPS spectra which has been continuum 
normalised and convolved from R = 118,000 to 20,000 already. Hence why
```python
cont_norm_bool = False
conv_instrument_bool = False
``` 
If you would like to test out these processes, there is a unprocessed HARPS
spectra of 18 sco in the ```spectroscopy_observation_data``` directory.

The python code above is the simplest case of processing, finding the best fit
parameters of a spectra using an error mask.

An error mask is a way of introducing model uncertainties to the fitting routine.
Grab the cleanest spectra you have of the star, set the reference parameters and 
you will create a mask of residuals representing how different the best possible
model is from a real observation.

For this example you are giving ```error_map_spec_path``` the same path as ```spec_path```,
this is because with only one spectra this will have to be your cleanest one. 
The variable ```error_mask_index``` refers to a list of reference parameters 
located in directory:

```bash
Input_data/Reference_data/PLATO_stars_lit_params.txt
```

Here we have already made a list of reference parameters for our PLATO benchmark stars.
The first star in the list is 18 sco, hence why ```error_mask_index = 0``` in this
example. If you would like to change this then you just have to edit this parameter list.

However, if you are not sure what the reference parameters are for the star (likely for a big survey),
then you can still use the error mask, just set ```error_mask_recreate_bool = False```. This will then check
to see what temperature the given star is closest to of our three error mask candidates: $\delta$ Eri, the Sun,
and Procyon. All three range in temperature in the FGK space and have almost solar metallicities. These were chosen
for that reason. Therefore, if you have a rough idea of the stellar temperature via means of SBCR, IRFM or other
methods, this can be noted down and be used to help get into the ballpark. Of course having no 
error mask is an option, this just means model uncertainties will be ignored in the fitting routine.

Note, the instrument must be specified, the current options are HARPS or UVES, this is because as well
as model uncertainties, the instrument itself provides uncertainties of measurement, thus an error mask
has been created for each instrument in each temperature range. 

```python
emask_kw_instrument = "HARPS" # can be "HARPS" or "UVES"
emask_kw_teff = "solar" # can be "solar","teff_varying"
error_mask_recreate_bool = False # if this is set to True, then emask_kw_teff defaults to "stellar"
error_mask_recreate_arr = [error_mask_recreate_bool,emask_kw_instrument,emask_kw_teff]
```

Like above, you choose the instrument and whether the emask is just solar or Teff varying. If you 
set ```error_mask_recreate_bool = True``` however, the stellar emask will be assumed i.e. a mask
with all of the reference parameters assumed from the Input_data directory.

We reconmend if you are working with Solar-type stars to use the instrument HARPS.


This fits file did not come with a radial velocity value, therefore we must calculate it.
The "read_spectra function" will give rv_shift as a NaN value, the code will recognise this
and start the procedure for RV correction.

Our RV correction function is called ```rv_cross_corelation_no_fft``` and derives
from PyAstronomy's RV Cross Correlation function with some minor changes. 
To perfom RV correction, you will require at least one "template" to compare your observations to.

If your star is a subgiant/dwarf Main-Sequence star similar to our Sun then having 
a solar template will be good enough. We have already provided a spectral template
in the directory

```bash
Input_data/spectroscopy_model_data/Sun_model_R5e5_TS_hr10/
```

We produced a R ~ 500,000 solar model using TurboSpectrum with the hr10 
wavelength coverage. Due to the high sampling, this had to be done in sections 
and therefore we created a function ```stitch_regions``` to stitch the model together.

Now all the RV correction function requires is for you to specify the intended 
Resolution, RV limits and a step size.

The parameter ```input_spec_resolution``` is where you define the resolution, it 
should match the spectral resolution of the trained grid. 

The array ```rv_shift_recalc``` exists to allow you to decided whether you would
like to re-calculate the rv correction or not, as some files do provide this
information, you may want to re-calculate it. The first parameter in the array 
you will set to True or False. The rest are rv_min, rv_max, and drv i.e. the limits
for the RV correction calculation and step size (in Km/s).

This of course will affect the computation time however I normally pick between
-100 and 100 Km/s for the limits with 0.05 Km/s as the step size. If you would
like to RV correct the spectra as a pre-process, the RV correction function 
can be found in the directory:

```bash
Source_scripts/SAPP_tools/
```

Running this code in the the most basic format will produce something like the following output

```python
SNR 396
BEST fit parameters = [ 5.77216  4.37047  0.02535  1.04010  4.12242  0.02084  0.00929 -0.03453]
parameters ERROR upper = [ 0.00630  0.00842  0.00400  0.01466  0.12393  0.00561  0.00711  0.00615]
parameters ERROR lower = [ 0.00630  0.00842  0.00400  0.01466  0.12393  0.00561  0.00711  0.00615]
```

The SNR of the spectra, then the best fit parameters alongside the respective 
errors. NOTE that the order of the parameters follows the order described above
with effective temperature, surface gravity and [Fe/H] as the first three
parameters. It was found that fitting Teff in units of K/1000 was easier for 
the training, hence why you see 5.77216 and not 5772.16.

Already we can see that the temperature estimation is good but the logg 
estimation could be better. This is where the ```numax_iter_bool``` variable
comes into play.

Our Lite mode is the use of Spectroscopy with Asteroseismic data. This is how 
we do it, you must set ```numax_iter_bool = True``` and fill out the 
parameters ```niter_numax_MAX``` and ```numax_input_arr```. For 18 Sco we
have already filled out the information for you.

The parameter ```niter_numax_MAX``` defines the maximum number of iterations
for the logg refinement. 5 is good enough for most iterations and this goes into 
the parameter ```numax_input_arr``` where the first two elements are
the ```$\Delta\nu_{max}$``` alongside its respective uncertainty.

The process itself is quite simple and can be understood from the code. In short
the code first makes its best guess of all the parameters, which will result in
the exact same output as above. It then uses the effective temperature combined
with ```$\Delta\nu_{max}$``` to calculate a new logg value via the scaling 
relationship below: 

```
$\nu_{max}$ = $\nu_\odot\frac{g}{g_\odot}\left\sqrt{\frac{T_{eff,\odot}}{T_{eff}}]\right}$
```

This logg estimate is then fixed while all the other parameters are 
re-calculated, this gives a new effective temperature and so a new logg value
can be calculated. This iterative process runs for as many loops as the user
gives it.

Once the process is finished, the code finds where the process has a temperature
value oscillating around some central value within 10 K. The process would 
never fully converge, however once we get to the oscillating region we consider
it finished.

The last way to use the code directly relates to how it is 
used in the Full mode which uses the Bayesian framework.

This is concerning the booleans ```recalc_metals_bool``` and ```feh_recalc_fix_bool```.

The Full mode only explores the first 3 dimensions of the spectroscopy PDF while
keeping the other 5 parameters at their best fit solution. Once we combine the
different PDFs alongside priors to create an overall posterior the rest of the
parameters need to be recalculated based on the new best fit solution. 

Therefore ```recalc_metals_bool = True``` will tell the code to look at the 
parameters given in the array ```recalc_metals_inp``` and fix the first two/three
parameters while re-calculating the rest. The effective temperature and logg 
are always fixed, however I give the the user the choice to not fix [Fe/H] as
this is an abundance and you might want to self-consistently re-calculate them all.

The inputs we have chosen are the Solar values, as 18 sco is known as the Solar
twin. NOTE by setting ```recalc_metals_bool = True``` this will negate the nu_max
iterative process since both Teff and logg are being fixed. 

The output of the code in this mode with ```feh_recalc_fix_bool = False``` is shown below:

```python
SNR 396
BEST fit parameters = [ 5.77030  4.43961  0.01079  1.05678  4.27973 -0.00909  0.02424 -0.02365]
parameters ERROR upper = [ 0.00637  0.00828  0.00420  0.01495  0.12486  0.00554  0.00713  0.00617]
parameters ERROR lower = [ 0.00637  0.00828  0.00420  0.01495  0.12486  0.00554  0.00713  0.00617]
```

And what follows is if ```feh_recalc_fix_bool = True```:

```python
SNR 396
BEST fit parameters = [ 5.77030  4.43961  0.00038  1.09698  4.12245  0.00072  0.03523 -0.00777]
parameters ERROR upper = [ 0.00645  0.00843  0.00419  0.01484  0.12359  0.00551  0.00723  0.00618]
parameters ERROR lower = [ 0.00645  0.00843  0.00419  0.01484  0.12359  0.00551  0.00723  0.00618]
```

## Bayesian Framework 

If you would like to run the FULL bayesian scheme then you should look at the 
script located in the following directory:

```bash
Source_scripts/main.py
```

Inside this script you will find several functions, of which only one concerns
you with respect to input, ```main_func_multi_obs()```.

Firstly, you need to understand the inputs for the full mode, these are split up
into 3 sections

- Photometry data
- Asteroseismology data
- Spectroscopy data

These are loaded in from the ```Input_data``` directory and the headers are defined
within the main_FULL.py script.

You should see something like this

```python
Input_phot_data = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/PLATO_bmk_phot_data/PLATO_photometry.csv',delimiter = ',',dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; G
2 ; eG
3 ; Bp
4 ; eBp
5 ; Rp
6 ; eRp
7 ; Gaia Parallax (mas)
8 ; eGaia Parallax (mas)
9 ; AG [mag], Gaia Extinction
10 ; Gaia DR2 ids
11 ; B (Johnson filter) [mag]
12 ; eB (Johnson filter) [mag]
13 ; V (Johnson filter) [mag]
14 ; eV (Johnson filter) [mag]
15 ; H (2MASS filter) [mag]
16 ; eH (2MASS filter) [mag]
17 ; J (2MASS filter) [mag]
18 ; eJ (2MASS filter) [mag]
19 ; K (2MASS filter) [mag]
20 ; eK (2MASS filter) [mag]
21 ; SIMBAD Parallax (mas)
22 ; eSIMBAD Parallax (mas)
23 ; E(B-V), reddening value from stilism 	
24 ; E(B-V)_upp, upper reddening uncertainty from stilism 	
25 ; E(B-V)_low, lower reddening uncertainty from stilism 	
"""

Input_ast_data = np.loadtxt('../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/PLATO_stars_seism_copy.txt',delimiter = ',',dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; nu_max, maximum frequency [microHz]
2 ; err nu_max pos, upper uncertainty
3 ; err nu_max neg, lower uncertainty
4 ; d_nu, large separation frequency [microHz]
5 ; err d_nu pos, upper uncertainty [microHz]
6 ; err d_nu neg, lower uncertainty [microHz]
"""

Input_spec_data = np.loadtxt('../Input_data/spectroscopy_observation_data/spectra_list.txt',delimiter=',',dtype=str)

"""
0 ; source_id (Literature name of the star)
1 ; spectra type (e.g. HARPS or UVES_cont_norm_convolved)
2 ; spectra path
"""

```

The observation data needs to match these headers EXACTLY.

Once the input data is loaded you need to make some decisons based on how the spectra 
are processed (just like in the spectroscopy section).

```python
emask_kw_instrument = "HARPS" # can be "HARPS" or "UVES"
emask_kw_teff = "solar" # can be "solar","teff_varying"
error_mask_recreate_bool = False # if this is set to True, then emask_kw_teff defaults to "stellar"
error_mask_recreate_arr = [error_mask_recreate_bool,emask_kw_instrument,emask_kw_teff]
error_map_use_bool = True
cont_norm_bool = False
rv_shift_recalc = [False,-100,100,0.05]
conv_instrument_bool = False
input_spec_resolution = 20000
numax_iter_bool = False
niter_numax_MAX = 5
recalc_metals_bool = False
feh_recalc_fix_bool = False
logg_fix_bool = False

spec_fix_values = [5770,4.44,0]
```

To understand these settings better, read the spectroscopy documentation above.

The only input for ```main_func_multi_obs()``` is ```inp_index```, which refers to the 
index of a list of stars organised by your input data.

NOTE: all input data including reference data should be in the same order (by rows)
with the exact same identifier, its how cross correlation works. 

After deciding the spectra treatment, you decide on the limits and central values of the core parameter space,
Teff, logg and [Fe/H]. This will dictate how much of the parameter space you want to explore which will be 
later used to create the spectroscopy grid. 

The default limits are

```python
Teff_resolution = 250 # K
logg_resolution = 0.3 # dex
feh_resolution = 0.3 # dex
```

And the central values are decided in two ways

- phot_ast_central_values_bool = True, means that the best fit values calculated from spectroscopy
will be used as a good idea/place to start. 
- phot_ast_central_values_bool = False, means that you decide where the centre of the grid will be

```python
phot_ast_central_values_bool = False # if set to True, values below will be used
phot_ast_central_values_arr = [5900,4,0] # these values are at the centre of the photometry grid

```

The next set of inputs concerns spectroscopy                

```python
spec_type = Input_spec_data[:,1][correlation_arr[inp_index][spec_obs_number]]\
                    + "_" + str(spec_obs_number)
spec_path = Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
error_map_spec_path = Input_spec_data[:,2][correlation_arr[inp_index][spec_obs_number]]
error_mask_index = inp_index
nu_max = Input_ast_data[inp_index][1]
nu_max_err = Input_ast_data[inp_index][2]
numax_input_arr = [float(nu_max),float(nu_max_err),niter_numax_MAX]
recalc_metals_inp = [spec_fix_values[0],spec_fix_values[1],spec_fix_values[2],feh_recalc_fix_bool]
logg_fix_load = np.loadtxt(f"../Input_data/photometry_asteroseismology_observation_data/PLATO_benchmark_stars/Seismology_calculation/seismology_lhood_results/{stellar_filename}_seismic_logg.txt",dtype=str)
logg_fix_input_arr = [float(logg_fix_load[1]),float(logg_fix_load[2]),float(logg_fix_load[3])]
```

You shouldn't need to edit any of these, unless you would like to fix the first 
3 parameters in spectroscopy, if you want to do this, pick what values you want in 
the variable ```recalc_metals_inp``` and set ```feh_recalc_fix_bool=True``` outside the ```main_func_multi_obs()```.

Note that the observation number relates to multiple observations of spectroscopy ONLY.

There are multiple stages in this code that requires all the modules that have been discussed, the
order of stages is reflected in the order of these bool parameters:

```python
best_spec_bool = True 
phot_ast_track_bool = False
spec_track_bool = False
bayes_scheme_bool = False
```

The first parameter will activate the spectroscopy module only, calculating best fit parameters
with the conditions and processes that have been selected to treat the spectra. These parameters, 
degrees of freedom, and covariance matrix for each observation of each star are saved in the directory

```bash
Output_data/Stars_Lhood_spectroscopy/multiobs/stellar_name
```

The second parameter will activate the photometry and asteroseismology module, creating the grids
and saving the tracks in the following directory

```bash
Output_data/Stars_Lhood_photometry/stellar_name
```

N.B. Photometry is only run for the first spectral observation of a given star.
This is because multiple observations (unless the spectra are bad) will give similar best fit parameters
and therefore central values. The limits currently provided should be large enough to take into account
any variation in these values. Thus, computation time is saved.

The third parameter runs once the photometry-asteroseismology grids are completed, these grids are loaded
into the workspace, and are used to calculate the spectroscopy grid. This is done by comparing the best fit
spectroscopy parameters Teff, logg, [Fe/H], to the values tabulated in the grids, the errors used are ones 
calculated from the spectroscopy module. 

The parameter ```chi_2_red_bool``` decides whether a reduced chi2 will be calculated or not. If False
then the chi2 values for all grids are normalised by their minimum chi2, thus rendering their maximum likelihood value
to be 1. If you would like spectroscopic covariance to be included (I would reconmend, it costs nothing in terms
of memory and computation) then set the parameter ```spec_covariance_bool to True```.

The fourth and final parameter runs the bayesian scheme, where all of the chi2 grids are loaded up,
converted to PDFs and combined to make the final PDF with priors included. The priors currently are 
IMF which is calculated from the tabulated Mass and Age_step which is used to reduce the probability of false
PMS stellar ages. Once the grids are created, best fit parameters and associated errors are calculated by
fitting to the combined PDF in each parameter space. See ```Log_prob_distributions_1D function``` for more detail on how 
this is done. 

These values are then combined with the best fit spectroscopic values which were fixed (i.e. Vmic, Vbrd, abundances)
alongside the uncertainties and then saved in the directory

```bash
Output_data/Stars_Lhood_combined_spec_phot/multiobs/stellar_name
```

N.B. if the probability distributions of for example spectroscopy and photometry are separated enough
i.e. they do not overlap in a given parameter, then the fitting function will not be able to fit it 
and will output a NaN value instead. This can happen!

Once you have run it entirey, you can of course set some of these steps to False to 
save re-calculation, for example if you want to recalc spectroscopy but photometry already exists,
set everything to False except spectroscopy.

The errors and best fit parameters are in separate files.

Lastly, there are two naming conventions which are used in the filenames of several
of the results, these are 

- mode_kw
- extra_save_string

mode_kw must be defined, this is to give you can indication of what "mode" of the code you ran,
e.g. "FULL" would be the full process, bayesian and all. "LITE_numax" is the spectroscopic module but with error mask
and nu_max prior applied, "MASK" is the spectroscopic process with just the mask.

extra_save_string can be nothing i.e. "", but I typically use it to indicate what type of error mask I am 
using or if I've medelled with some part of the source code like "_gaia_mags_only". something like that.

### Multiprocessing or Single Run 

You can run the code on a single star such as below:

```python
if __name__ == '__main__':
    
    inp_index = 15
    
    main_func_multi_obs(inp_index)
```

where inp_index refers to the star's source id in the input photometry data, asteroseismology data, and reference data.
E.g. the 16th star in the list (counting from 0) is KIC3427720.

If you would like to run multiple stars at once, the code can be parallelised by the number of stars:

```python
num_processor = 8

inp_index_list = np.arange(len(stellar_names))

if __name__ == '__main__':
#    
    p = mp.Pool(num_processor) # Pool distributes tasks to available processors using a FIFO schedulling (First In, First Out)
    p.map(main_func_multi_obs,inp_index_list) # You pass it a list of the inputs and it'll iterate through them
    p.terminate() # Terminates process 
```

Now, try to run the full code! If there are any issues, please contact me at 

```gent@mpia.de```
