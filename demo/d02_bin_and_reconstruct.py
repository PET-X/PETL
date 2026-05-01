import glob
import matplotlib.pyplot as plt
import numpy as np
import time
"""
This demo script shows you how to bin a list mode file and reconstruct.
For a full description of each function's arguments, please see the online documentation.
"""

# For conenience, we have provided a wrapper class called PETXserver
# that is wrapped around PETLserver.  This class just automatically
# sets the PETX geometry for you.  You can do this as follows
from petx_server import *
petx = PETXserver()
petx.about()

# The above could also simply be done with the following
# This will give the exact same result as above, it just more explicit
# The following is all commented out, but included here to show you
# what is being done in the PETXserver class
"""
from petl_server import *
petx = PETLserver()
petx.about()

# First we set the PET/X geometry
pixel_pitch = 1.6
petx.add_planogram(psi=0.0, D=101.0, L=251.52, H=156.72, T=pixel_pitch)
petx.add_planogram(psi=90.0, D=266.6, L=125.12, H=156.72, T=pixel_pitch)
petx.set_default_volume()

# Set the PET/X module geometry
# this is used to calculate the system sensitivity
block_size = 30.32
block_gap = 1.28
crystal_thickness = 20.0

c_y = 101.0*0.5
for m in range(5):
    c_z = (m-2)*(block_size+block_gap)
    for n in range(8):
        c_x = (n-3.5)*(block_size+block_gap)
        petx.add_module([c_x, c_y+0.5*crystal_thickness, c_z], [0.5*block_size, 0.5*crystal_thickness, 0.5*block_size])
        petx.add_module([c_x, -c_y-0.5*crystal_thickness, c_z], [0.5*block_size, 0.5*crystal_thickness, 0.5*block_size])

c_x = 266.6*0.5
for m in range(5):
    c_z = (m-2)*(block_size+block_gap)
    for n in range(4):
        c_y = (n-1.5)*(block_size+block_gap)
        petx.add_module([c_x+0.5*crystal_thickness, c_y, c_z], [0.5*crystal_thickness, 0.5*block_size, 0.5*block_size])
        petx.add_module([-c_x-0.5*crystal_thickness, c_y, c_z], [0.5*crystal_thickness, 0.5*block_size, 0.5*block_size])
#"""


# First we calculate the system sensitivity which uses analytic ray-tracing to calculate
# the stopping power through the scintillator of each coincidience line of response.
# This can be a lengthy calculation and it may be useful to calculate this once and then save it to disk.
print('Calculating system sensitivity...')
st = time.time()
petx.calc_response(3)
print('calc_response elapsed time:', time.time()-st)

# Next we bin a collection of list mode files
g = petx.allocate_projections()
lst_files = glob.glob(r"D:\PETX\F18uDerenzo_20251008\*.lst")
for file in lst_files:
    petx.bin(file, g)

# We happen the know that the data above is a small micro Derenzo scan
# so let's make a smaller volume with small voxels for this phantom
petx.set_volume(120, 120, 120, 0.5)

# Allocate the volume data
f = petx.allocate_volume()

# Here are a couple of ways to reconstruct the data
which = 1
if which == 1:

    # Apply the solid and and system sensitivity corrections before reconstructing
    petx.apply_corrections(g)
    
    # Now reconstruct with PFDR + 2D FBP
    petx.FBP(g, f)
    
    # Clip negative numbers
    np.maximum(f, 0.0, out=f)
    
elif which == 2:

    # Now we reconstruct with 50 iterations of MLEM using One Step Late (OSL) regularization
    # The regularizer is the relative differences prior
    # Do not apply corrections here because they are included in the forward and backprojectors
    # of the MLEM algorithm
    petx.MLEM(g,f,50,2.0,4.0e-5)
    
elif which == 3:

    # Now we reconstruct with RWLS (also called PWLS)
    # where we seed the reconstruction with a PFDR+FBP reconstruction
    # For RWLS/PWLS the regularizer uses a Huber loss function
    # Corrections are automatically applied in RWLS algorithm, but we
    # also need to apply corrections for FBP, so in order to not apply
    # the corrections twice, we will need to make a copy of the data
    g_2 = petx.copy_data(g)
    petx.apply_corrections(g_2)
    petx.FBP(g_2, f)
    petx.RWLS(g, f, 50, delta=750.0, beta=9.0e-9)

# Display the result with napari
petx.display(f)
