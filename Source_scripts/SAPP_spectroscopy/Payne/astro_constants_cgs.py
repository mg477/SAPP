#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:12:18 2019

@author: gent
"""

import h5py

h5f_PLATO_consts = h5py.File("../Input_data/PLATO_constants/constants.hdf5",'r')


"""
Source: http://physics.rutgers.edu/~abrooks/342/constants.html
"""

# mass_sol = 1.989 * 10 ** 33 # solar mass, [g]
# radius_sol = 6.955 * 10 ** 10 # solar radius, [cm]
# parsec = 3.086 * 10 ** 18 # parsec, [cm]
# AU = 1.496 * 10 ** 13 # astronomical unit, [cm]

mass_sol = h5f_PLATO_consts['CGS/solar/M '][()] # solar mass, [g]
radius_sol = h5f_PLATO_consts['CGS/solar/R'][()] # solar radius, [cm]
parsec = h5f_PLATO_consts['CGS/distances/pc'][()] # parsec, [cm]
AU = h5f_PLATO_consts['CGS/distances/au'][()] # astronomical unit, [cm]

"""
Source: https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/units.html
"""

# lumin_sol = 3.826 * 10 ** 33 # solar luminosity, [ergs/s]
# radian_arcsec = 206265 # 1 radian to arc second ["]
# jansky = 10 ** -23 # Jansky [erg/s/cm^2/Hz]

lumin_sol = h5f_PLATO_consts['CGS/solar/L'][()] # solar luminosity, [ergs/s]
radian_arcsec = 206265 # 1 radian to arc second ["]
jansky = 10 ** -23 # Jansky [erg/s/cm^2/Hz]


"""
Source: http://www.astro.wisc.edu/~dolan/constants.html
"""

# teff_sol = 5780 # solar temperature, [K], this was used before
# grav_constant = 6.67259 * 10 ** -8 # Gravitational constant, [cm^3/g/s^2]

teff_sol = h5f_PLATO_consts['CGS/solar/Teff'][()] # solar temperature, [K], this was used before
grav_constant = h5f_PLATO_consts['CGS/fundamental/G'][()] # Gravitational constant, [cm^3/g/s^2]


"""
Source: https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
"""

# surf_grav_sol = 2.74 * 10 ** 4 # solar surface gravity, [cm^2/s]

surf_grav_sol = h5f_PLATO_consts['CGS/solar/M '][()] * h5f_PLATO_consts['CGS/fundamental/G'][()]/h5f_PLATO_consts['CGS/solar/R'][()] ** 2 # solar surface gravity, [cm^2/s]

"""
Source: https://arxiv.org/abs/1703.10834 'New solar metallicity measurements' Sunny Vagnozzi 2019

Note: The 'solar modelling problem' is the downward revision of the metallicity of the Sun over the years which is due to disagreement between predictions
of standard solar models and inferences from helioseismology. This value was derived via measurements of solar wind emerging from polar coronal holes
"""

z_metal_sol = 0.0196 # Solar metallicity

"""
These are the observed solar asteroseismic quantities that Aldo uses, these will have to change for each
observational result as people presenting delta_nu and nu_max in the literature may use different techniques to determine
the solar value and therefore the model points will need to be re-calibrated for self-consistency 
"""

# delta_nu_sol = 135.1 # [uHz]
# delta_nu_sol_err = 0.1 # [uHz]

# nu_max_sol = 3090 # [uHz]
# nu_max_sol_err = 30 # [uHz]

delta_nu_sol = h5f_PLATO_consts['CGS/solar_seismic/Delta_nu'][()] # [uHz]
delta_nu_sol_err = 0.1 # [uHz]

nu_max_sol = h5f_PLATO_consts['CGS/solar_seismic/nu_max'][()] # [uHz]
nu_max_sol_err = 30 # [uHz]