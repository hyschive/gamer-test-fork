"""
This file generates a toy vector potential for import into GAMER using the
OPT__INIT_BFIELD_BYFILE parameter. It does the following:

1. Generates a uniform coordinate grid
2. Defines a vector potential on the coordinate grid
3. Saves the coordinate grid and the vector potential to an HDF5 file

The units of the vector potential and the coordinate arrays should be the same
as those used in GAMER. So:

* coordinates are in UNIT_L
* vector potential components are in UNIT_B*UNIT_L = sqrt(4*pi*UNIT_P)*UNIT_L
  where UNIT_P = UNIT_M/UNIT_L/UNIT_T**2

The file should also be named "B_IC" for GAMER to recognize it.

It requires NumPy, h5py, and HDF5 to be installed.

Use regular expression for parsing the files Input__Parameter and Input__TestProb
to obtain the required parameters
"""

import h5py
import re
import numpy as np


reg_pattern = r"\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)"  # patterns for all numeric numbers

### use regular expression to get the setting in Input__Parameter
par = open("Input__Parameter").read()

# UNIT
UNIT_L = re.findall(r"UNIT_L" + reg_pattern, par)
UNIT_M = re.findall(r"UNIT_M" + reg_pattern, par)
UNIT_T = re.findall(r"UNIT_T" + reg_pattern, par)

UNIT_L = float(UNIT_L[0])
UNIT_M = float(UNIT_M[0])
UNIT_T = float(UNIT_T[0])
UNIT_P = UNIT_M / UNIT_L / UNIT_T**2  # use energy density unit same as that in GAMER code
UNIT_B = np.sqrt(4 * np.pi * UNIT_P)
UNIT_A = UNIT_B  # UNIT_L is not required if A is obtained from dimensionless coordinate (xx, yy, and zz)

# simulation scale (assume the number of base-level cells are the same in each direction)
BOX_SIZE = re.findall(r"BOX_SIZE" + reg_pattern, par)
NX0_TOT  = re.findall(r"NX0_TOT_X" + reg_pattern, par)

BOX_SIZE = float(BOX_SIZE[0])
NX0_TOT  = int(NX0_TOT[0])


### use regular expression to get the setting in Input__TestProb
par_testprob = open("Input__TestProb").read()

# parameters for B field
Bfield_Ab = re.findall(r"Bfield_Ab" + reg_pattern, par_testprob)
Bfield_np = re.findall(r"Bfield_np" + reg_pattern, par_testprob)

Bfield_Ab = float(Bfield_Ab[0]) / UNIT_A
Bfield_np = float(Bfield_np[0])


### print parameters loaded from Input__Parameter and Input__TestProb
print("=" * 12 + " Parameters adopted " + "=" * 12)
print("UNIT_L    in Input__Parameter: {:13.7e}".format(UNIT_L))
print("UNIT_M    in Input__Parameter: {:13.7e}".format(UNIT_M))
print("UNIT_T    in Input__Parameter: {:13.7e}".format(UNIT_T))
print("BOX_SIZE  in Input__Parameter: {:13.7e}".format(BOX_SIZE))
print("NX0_TOT   in Input__Parameter: {:13d}".format(NX0_TOT) + " (assume NX0_TOT_[XYZ] are all the same)")
print("Bfield_Ab in Input__TestProb : {:13.7e}".format(Bfield_Ab * UNIT_A))
print("Bfield_np in Input__TestProb : {:13.7e}".format(Bfield_np))


### Read initial condition of denisty and pressure
radius, dens, pres = np.genfromtxt("tovstar_short", usecols = [0, 2, 3], unpack = 1)
radius /= UNIT_L

# functions for interpolation
interp_pres = lambda r: np.interp(r, radius, pres)
interp_dens = lambda r: np.interp(r, radius, dens)

# central density and pressure
rho_c  = interp_dens(0.0)
pres_c = interp_pres(0.0)


# Number of cells along each dimension of the input grid.
# This is somewhat arbitrary, but should be chosen in
# such a way as to adequately resolve the vector potential.

ddims = np.array([NX0_TOT * 2**3]*3, dtype='int')

# Left edge and right edge coordinates of the desired
# simulation domain which will be used in GAMER.

le = np.zeros(3)
re = np.ones(3) * BOX_SIZE
ce = 0.5 * (le + re)

# Since we need to take derivatives of the vector potential
# to get the magnetic field on the simulation domain, the
# input grid must be extended a bit beyond this boundary.
# We therefore add a buffer of three cells on each side.
# (Three cells are necessary to solve some corner cases
# resulting from round-off errors.)

delta = (re-le)/ddims
ddims += 6
le -= 3.0*delta
re += 3.0*delta

# Construct the grid cell edge coordinates

x = np.linspace(le[0], re[0], ddims[0]+1)
y = np.linspace(le[1], re[1], ddims[1]+1)
z = np.linspace(le[2], re[2], ddims[2]+1)

# Find the grid cell midpoints

x = 0.5*(x[1:]+x[:-1])
y = 0.5*(y[1:]+y[:-1])
z = 0.5*(z[1:]+z[:-1])

# Use the 1-D coordinate arrays to consruct 3D coordinate arrays
# that we will use to compute an analytic vector potential

#xx, yy, zz = np.meshgrid(x, y, z, sparse=False, indexing='ij')
xx, yy, zz = np.meshgrid(x - ce[0], y - ce[1], z - ce[2], sparse=False, indexing='ij')
rr = np.sqrt(xx * xx + yy * yy + zz * zz)


# Toy vector potential which depends on all three coordinates
factor_dens = (1.0 - interp_dens(rr) / rho_c)**Bfield_np
factor_pres = interp_pres(rr) / pres_c

Ax = -yy * Bfield_Ab * factor_dens * factor_pres
Ay =  xx * Bfield_Ab * factor_dens * factor_pres
Az =  0.0 * zz

# Write the ICs to an HDF5 file

f = h5py.File("B_IC", "w")

# Write coordinate arrays

f.create_dataset("x", data=x)
f.create_dataset("y", data=y)
f.create_dataset("z", data=z)

#  Write vector potential arrays

f.create_dataset("magnetic_vector_potential_x", data=Ax)
f.create_dataset("magnetic_vector_potential_y", data=Ay)
f.create_dataset("magnetic_vector_potential_z", data=Az)

# Close the file

f.flush()
f.close()
