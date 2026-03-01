import glob
import matplotlib.pyplot as plt
import numpy as np
import time
from petl_server import *
"""
This demo script shows you how to set the PET geometry and perform some basic operations.
For a full description of each function's arguments, please see the online documentation.
"""

# First we make an instance of the PETLserver class
# Each instance carries its own parameters (PET geometry, PET volume specification, etc).
petl = PETLserver()


# Set PET geometry
# Here the proojection data is composed of 6 planograms which corresponds to a system
# where the panel rotate (like the WVU PEM-PET system).
# Note that each planogram may have difference parameters, e.g., this does support rectangular systems like PET/X
num_planograms = 6
for n in range(num_planograms):
    petl.add_planogram(psi=n*180.0/num_planograms, D=180.0, L=192.0, H=144.0, T=2.0, v_m0=np.tan(90.0/num_planograms*np.pi/180), v_m1=np.tan(15.0*np.pi/180))

# Set PET volume parameters
# Once the PET geometry has been set, one can just use the set_default_volume
# to define the PET volume that fills the field of view and uses a voxel size
# which is the same size as the detector pixel size.
petl.set_default_volume()

# One can also define the PET volume yourself using the set_volume function
# Note the the center of the field of view is the origin of the coordinate system
#petl.set_volume(numX, numY, numZ, voxelWidth, voxelHeight = 0.0, offsetX = 0.0, offsetY = 0.0, offsetZ = 0.0)

# You can check that everything is specified correctly by
# printing the PET geometry and PET volume parameters with the following function
#petl.print_parameters()

# Now that the PET geometry is specified, we can allocate memory to store the projection data
# All memory is managed explicitly in Python.  The C++/CUDA does not manage ANY memory.
# Since each planogram can be a different size, we cannot just use a single numpy arrays
# Thus the projection data is stored as a list of 4D float32 numpy arrays, i.e.,
# order of the dimension is given by: g[planogram index][v1,v0,u1,u0]
g = petl.allocate_projections()

# And now we make a numpy array for the PET volume which is just a 3D float32 numpy array
# The order of the dimensions is given by: f[z,y,x]
# One does not have to use the allocate_volume function; they can certainly make this numpy
# array themselves, the following function is provided for convenience.
f = petl.allocate_volume()

# At this point enough is given so that one can start using
# the various PETL functions to bin, project, reconstruct, or simulate data

# As just an example, we shall use the PETL ray tracing routines to simulate some data and then reconstruct it
# So first we set the phantom as a simple ellipse with a hot spot in the center
petl.add_object('ellipsoid', [0.0, 0.0, 0.0], [90.0, 80.0, 70.0], 100.0)
petl.add_object('ellipsoid', [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 1000.0)

# Now we ray trace through the set of geometric solids
petl.ray_trace(g)

# Now we reconstruct
petl.FBP(g, f)

# Finally, if napari is installed, just use the following command to display
# the result with napari
petl.display(f)
