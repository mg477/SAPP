# Reading instructions for "constants.hdf5"
Olivier Roth - olivier.roth@obspm.fr


## Content of the constants file:
Two groups, for each system of units, S.I. and C.G.S., both containing:
- Fundamental physical constants
- Units of astronomical distances
- Solar constants
- Seismic solar constants
- Terrestrial constants.


## In odrer to collect a constant:
- First get the location of the constant in the file with the help of any visualizer of HDF5 file. <br>
    A python script, h5_view.py, is attached for this purpose if you don't have any visualizer.<br>
    With h5_view, in the terminal:<br>
    `$ python path_to_h5_view/h5_view.py -f path_to_constants/constants.hdf5`


- Then you have to open the file and get the desired constant with the
  following commands in a python script (example with two constants here):

    `import h5py ` # `$ pip install h5py` in the terminal if the package isn't downloaded<br>
    `h5f = h5py.File("constants.hdf5",'r') ` # opening the constants file in 'reading' mode under the name h5f<br>
    `Delta_nu = h5f["SI/solar_seismic/Delta_nu"][()]` # extracting the value of Delta_nu in S.I. units, note that `[()]` is mandatory to get the value<br>
    `c = h5f["CGS/fundamental/c"][()]` # extracting the value of the speed of light in vacuum in C.G.S. units<br>
    `print(h5f['CGS/fundamental/c'].attrs.get("Units"))` # show units of a specific constant, here c in C.G.S<br>
    `print(h5f['CGS/fundamental/c'].attrs.get("fname"))` # show the full name of a specific constant <br>
    `h5f.close() ` # closing the file (hdf5 files need to be closed in order to be opened somewhere else)
