from petl_server import *
import matplotlib.pyplot as plt

class PETXserver(PETLserver):
    def __init__(self, param_id=None, lib_dir=""):
        super().__init__(param_id, lib_dir)
        
        pixel_pitch = 1.6
        self.add_planogram(psi=0.0, D=101.0, L=251.52, H=156.72, T=pixel_pitch)
        self.add_planogram(psi=90.0, D=266.6, L=125.12, H=156.72, T=pixel_pitch)
        self.set_default_volume()

        # Set the PET/X module geometry
        # this is used to calculate the system sensitivity
        block_size = 30.32
        block_gap = 1.28
        crystal_thickness = 20.0
        
        use_simple_geometry = False
        
        if use_simple_geometry:
            self.add_module([0.0, 0.5*101.0+10.0, 0.0], [0.5*251.52, 10.0, 0.5*156.72])
            self.add_module([0.0, -0.5*101.0-10.0, 0.0], [0.5*251.52, 10.0, 0.5*156.72])
            
            self.add_module([0.5*251.52+10.0, 0.0, 0.0], [10.0, 0.5*101.0+20.0, 0.5*156.72])
            self.add_module([-0.5*251.52-10.0, 0.0, 0.0], [10.0, 0.5*101.0+20.0, 0.5*156.72])
        else:
            c_y = 101.0*0.5
            for m in range(5):
                c_z = (m-2)*(block_size+block_gap)
                for n in range(8):
                    c_x = (n-3.5)*(block_size+block_gap)
                    self.add_module([c_x, c_y+0.5*crystal_thickness, c_z], [0.5*block_size, 0.5*crystal_thickness, 0.5*block_size])
                    self.add_module([c_x, -c_y-0.5*crystal_thickness, c_z], [0.5*block_size, 0.5*crystal_thickness, 0.5*block_size])

            c_x = 266.6*0.5
            for m in range(5):
                c_z = (m-2)*(block_size+block_gap)
                for n in range(4):
                    c_y = (n-1.5)*(block_size+block_gap)
                    self.add_module([c_x+0.5*crystal_thickness, c_y, c_z], [0.5*crystal_thickness, 0.5*block_size, 0.5*block_size])
                    self.add_module([-c_x-0.5*crystal_thickness, c_y, c_z], [0.5*crystal_thickness, 0.5*block_size, 0.5*block_size])
