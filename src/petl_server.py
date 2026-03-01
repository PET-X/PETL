import ctypes
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import site
import glob
import imageio
from sys import platform as _platform
from numpy.ctypeslib import ndpointer
import numpy as np

has_torch = False

class PETLserver:
    def __init__(self, param_id=None, lib_dir=""):
        if len(lib_dir) > 0:
            current_dir = lib_dir
        else:
            current_dir = os.path.abspath(os.path.dirname(__file__))

        if _platform == "linux" or _platform == "linux2":
            import readline
            from ctypes import cdll

            libname = glob.glob(os.path.join(current_dir, "*petl*.so"))
            if len(libname) == 0:
                fullPath = os.path.join(current_dir, 'libpetl.so')
                fullPath_backup = os.path.join(current_dir, '../build/lib/libpetl.so')
            elif len(libname) == 1:
                fullPath = libname[0]
                fullPath_backup = ""
            elif len(libname) >= 2:
                fullPath = libname[0]
                fullPath_backup = libname[1]
            
            if os.path.isfile(fullPath):
                self.libprojectors = cdll.LoadLibrary(fullPath)
            elif os.path.isfile(fullPath_backup):
                self.libprojectors = cdll.LoadLibrary(fullPath_backup)
            else:
                print('Error: could not find PETL dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
            
        elif _platform == "win32":
            from ctypes import windll
        
            libname = glob.glob(os.path.join(current_dir, "*petl*.dll"))
            if len(libname) == 0:
                fullPath = os.path.join(current_dir, 'libpetl.dll')
                fullPath_backup = os.path.join(current_dir, r'..\win_build\bin\Release\libpetl.dll')
            elif len(libname) == 1:
                fullPath = libname[0]
                fullPath_backup = ""
            elif len(libname) >= 2:
                fullPath = libname[0]
                fullPath_backup = libname[1]
        
            if os.path.isfile(fullPath):
                try:
                    self.libprojectors = windll.LoadLibrary(fullPath)
                except:
                    self.libprojectors = ctypes.CDLL(fullPath, winmode=0)
            elif os.path.isfile(fullPath_backup):
                try:
                    self.libprojectors = windll.LoadLibrary(fullPath_backup)
                except:
                    self.libprojectors = ctypes.CDLL(fullPath_backup, winmode=0)
            else:
                print('Error: could not find PETL dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
        
        elif _platform == "darwin":  # Darwin is the name for MacOS in Python's platform module
            from ctypes import cdll
            
            libname = glob.glob(os.path.join(current_dir, "*petl*.dylib"))
            if len(libname) == 0:
                fullPath = os.path.join(current_dir, 'libpetl.dylib')
                fullPath_backup = os.path.join(current_dir, '../build/lib/libpetl.dylib')
            elif len(libname) == 1:
                fullPath = libname[0]
                fullPath_backup = ""
            elif len(libname) >= 2:
                fullPath = libname[0]
                fullPath_backup = libname[1]
            
            if os.path.isfile(fullPath):
                self.libprojectors = cdll.LoadLibrary(fullPath)
            elif os.path.isfile(fullPath_backup):
                self.libprojectors = cdll.LoadLibrary(fullPath_backup)
            else:
                print('Error: could not find PETL dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
        
        if self.libprojectors is None:
            self.param_id = -1
        else:
            if param_id is not None:
                self.param_id = param_id
            else:
                self.param_id = self.create_new_model()
            self.set_model()
        self.print_cost = False
        self.print_warnings = True
        self.response = None
        self.include_response_with_projectors = False
        
    def set_response(self, R):
        """
        Sets the response map (i.e., sensitivity)
        This map can be used to correct data (turn it into line integral data)
        by dividing the planogram data by this response.
        
        Once this response is defined, it will automatically be applied
        after call to project and before every call to backproject
        
        See also: calc_response, apply_corrections, clear_response
        
        Args:
            R (list of numpy array): the response map
        """
        self.response = R
        
    def clear_response(self):
        """
        Clears the response map (i.e., sensitivity)
        """
        self.response = None

    def create_new_model(self):
        """ This should be considered a private class function """
        self.libprojectors.create_new_model.restype = ctypes.c_int
        #self.libprojectors.create_new_model.argtypes = [ctypes.c_int]
        return self.libprojectors.create_new_model()

    def set_model(self, i=None):
        """ This should be considered a private class function """
        self.libprojectors.set_model.restype = ctypes.c_bool
        self.libprojectors.set_model.argtypes = [ctypes.c_int]
        if i is None:
            return self.libprojectors.set_model(self.param_id)
        else:
            return self.libprojectors.set_model(i)

    def volume_defined(self):
        """
        Returns True if the volume parameters are specified, otherwise returns False
        """
        self.set_model()
        self.libprojectors.volume_defined.restype = ctypes.c_bool
        return self.libprojectors.volume_defined()

    def geometry_defined(self):
        """
        Returns True if at least one set of coincience flat panel detectors are specified,
        otherwise returns False
        """
        self.set_model()
        self.libprojectors.geometry_defined.restype = ctypes.c_bool
        return self.libprojectors.geometry_defined()

    def all_defined(self):
        """
        Returns True if the volume parameters are specified and
        at least one set of coincience flat panel detectors are specified,
        otherwise returns False.
        This function must return True before any projector of reconstruction
        algorithm function can be called.
        """
        if self.volume_defined() and self.geometry_defined():
            return True
        else:
            return False

    def print_parameters(self):
        """
        Prints the PET geometry and PET volume parameters to the screen
        """
        self.libprojectors.print_parameters()

    def numX(self, N=None):
        """
        If a positive number is given, then sets the number of voxels
        in the x dimension to this value.
        If no argument is given, simply returns the number of voxels
        in the x dimension.
        
        Args:
            N (int): number of voxels in the x dimension
            
        Returns:
            number of voxels in the x dimension
        """
        if N is None:
            self.set_model()
            self.libprojectors.get_numX.restype = ctypes.c_int
            return self.libprojectors.get_numX()
        else:
            self.set_model()
            self.libprojectors.set_numX.restype = ctypes.c_bool
            self.libprojectors.set_numX.argtypes = [ctypes.c_int]
            if self.libprojectors.set_numX(N):
                return self.numX()
            else:
                return None

    def numY(self, N=None):
        """
        If a positive number is given, then sets the number of voxels
        in the y dimension to this value.
        If no argument is given, simply returns the number of voxels
        in the y dimension.
        
        Args:
            N (int): number of voxels in the y dimension
            
        Returns:
            number of voxels in the y dimension
        """
        if N is None:
            self.set_model()
            self.libprojectors.get_numY.restype = ctypes.c_int
            return self.libprojectors.get_numY()
        else:
            self.set_model()
            self.libprojectors.set_numY.restype = ctypes.c_bool
            self.libprojectors.set_numY.argtypes = [ctypes.c_int]
            if self.libprojectors.set_numY(N):
                return self.numY()
            else:
                return None

    def numZ(self, N=None):
        """
        If a positive number is given, then sets the number of voxels
        in the z dimension to this value.
        If no argument is given, simply returns the number of voxels
        in the z dimension.
        
        Args:
            N (int): number of voxels in the z dimension
            
        Returns:
            number of voxels in the z dimension
        """
        if N is None:
            self.set_model()
            self.libprojectors.get_numZ.restype = ctypes.c_int
            return self.libprojectors.get_numZ()
        else:
            self.set_model()
            self.libprojectors.set_numZ.restype = ctypes.c_bool
            self.libprojectors.set_numZ.argtypes = [ctypes.c_int]
            if self.libprojectors.set_numZ(N):
                return self.numZ()
            else:
                return None
                
    def voxelWidth(self, w=None):
        """
        If a positive number is given, then sets the voxel width to this value.
        If no argument is given, simply returns the voxelWidth value.
        
        Args:
            w (float): voxel width (mm)
            
        Returns:
            the volume voxel width (mm)
        """
        if w is None:
            self.set_model()
            self.libprojectors.get_voxelWidth.restype = ctypes.c_float
            return self.libprojectors.get_voxelWidth()
        else:
            self.set_model()
            self.libprojectors.set_voxelWidth.restype = ctypes.c_bool
            self.libprojectors.set_voxelWidth.argtypes = [ctypes.c_float]
            if self.libprojectors.set_voxelWidth(w):
                return self.voxelWidth()
            else:
                return None
                
    def voxelHeight(self, h=None):
        """
        If a positive number is given, then sets the voxel height to this value.
        If no argument is given, simply returns the voxelHeight value.
        
        Args:
            h (float): voxel height (mm)
            
        Returns:
            the volume voxel height (mm)
        """
        if h is None:
            self.set_model()
            self.libprojectors.get_voxelHeight.restype = ctypes.c_float
            return self.libprojectors.get_voxelHeight()
        else:
            self.set_model()
            self.libprojectors.set_voxelHeight.restype = ctypes.c_bool
            self.libprojectors.set_voxelHeight.argtypes = [ctypes.c_float]
            if self.libprojectors.set_voxelHeight(h):
                return self.voxelHeight()
            else:
                return None
                
    def offsetX(self, x_0=None):
        """
        If a positive number is given, then sets the offsetX to this value.
        If no argument is given, simply returns the offsetX value.
        
        Args:
            x_0 (float): the x-coordinate of the center voxel
            
        Returns:
            the x-coordinate of the center voxel
        """
        if x_0 is None:
            self.set_model()
            self.libprojectors.get_offsetX.restype = ctypes.c_float
            return self.libprojectors.get_offsetX()
        else:
            self.set_model()
            self.libprojectors.set_offsetX.restype = ctypes.c_bool
            self.libprojectors.set_offsetX.argtypes = [ctypes.c_float]
            if self.libprojectors.set_offsetX(x_0):
                return self.offsetX()
            else:
                return None
                
    def offsetY(self, y_0=None):
        """
        If a positive number is given, then sets the offsetY to this value.
        If no argument is given, simply returns the offsetY value.
        
        Args:
            y_0 (float): the y-coordinate of the center voxel
            
        Returns:
            the y-coordinate of the center voxel
        """
        if y_0 is None:
            self.set_model()
            self.libprojectors.get_offsetY.restype = ctypes.c_float
            return self.libprojectors.get_offsetY()
        else:
            self.set_model()
            self.libprojectors.set_offsetY.restype = ctypes.c_bool
            self.libprojectors.set_offsetY.argtypes = [ctypes.c_float]
            if self.libprojectors.set_offsetY(y_0):
                return self.offsetY()
            else:
                return None
                
    def offsetZ(self, z_0=None):
        """
        If a positive number is given, then sets the offsetZ to this value.
        If no argument is given, simply returns the offsetZ value.
        
        Args:
            z_0 (float): the z-coordinate of the center voxel
            
        Returns:
            the z-coordinate of the center voxel
        """
        if z_0 is None:
            self.set_model()
            self.libprojectors.get_offsetZ.restype = ctypes.c_float
            return self.libprojectors.get_offsetZ()
        else:
            self.set_model()
            self.libprojectors.set_offsetZ.restype = ctypes.c_bool
            self.libprojectors.set_offsetZ.argtypes = [ctypes.c_float]
            if self.libprojectors.set_offsetZ(z_0):
                return self.offsetZ()
            else:
                return None

    def get_numPlanograms(self):
        """
        Returns the number of planograms that are currently specified.
        """
        self.set_model()
        self.libprojectors.get_numPlanograms.restype = ctypes.c_int
        return self.libprojectors.get_numPlanograms()

    def get_planogramSize(self, n):
        """
        Returns the shape of the n-th planogram
        """
        if self.geometry_defined():
            shape = np.zeros(4, dtype=np.int32)
            self.set_model()
            self.libprojectors.get_planogramSize.restype = ctypes.c_bool
            self.libprojectors.get_planogramSize.argtypes = [ctypes.c_int, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
            self.libprojectors.get_planogramSize(n, shape)
            return shape
        else:
            return None

    def allocate_projections(self):
        """
        Allocates data for the PET planograms data to be stored
        which is a list of numpy arrays where each item in the list
        corresponds to a planogram.
        A single numpy array cannot be used because each planogram
        may have a different shape
        """
        if self.geometry_defined():
            N = self.get_numPlanograms()
            g = []
            for n in range(N):
                shape = self.get_planogramSize(n)
                g.append(np.zeros((shape[0],shape[1],shape[2],shape[3]), dtype=np.float32))
            return g
        else:
            return None

    def allocate_rebinned_projections(self):
        """
        Allocates data for the rebinned PET planograms data to be stored
        which is a list of numpy arrays where each item in the list
        corresponds to a planogram.
        A single numpy array cannot be used because each planogram
        may have a different shape
        """
        if self.geometry_defined():
            N = self.get_numPlanograms()
            g = []
            for n in range(N):
                shape = self.get_planogramSize(n)
                g.append(np.zeros((shape[1],shape[2],shape[3]), dtype=np.float32))
            return g
        else:
            return None
            
    def copy_data(self, x):
        """
        Makes a deep copy of the provided input
        which should be a numpy array or a list of numpy arrays.
        """
        if isinstance(x, list):
            y = []
            for n in range(len(x)):
                y.append(x[n].copy())
        else:
            y = x.copy()
        return y
        
    def copyData(self, x):
        return self.copy_data(x)

    def allocate_volume(self):
        """
        Allocates a numpy array for the volume.
        """
        if self.volume_defined():
            f = np.zeros((self.numZ(), self.numY(), self.numX()), dtype=np.float32)
            return f
        else:
            print('Error: volume is not defined')
            return None
    
    def clearAll(self):
        """
        Clears all PETL parameters, e.g., all planogram and volume specifications
        """
        self.set_model()
        self.libprojectors.clearAll()

    def add_planogram(self, psi, D, L, H, T, v_m0=None, v_m1=None):
        """
        This function adds a pair of parallel flat panel detectors to the PET model
        which are defined in planogram coordinates.
        
        Args:
            psi (float): azimuthal rotation angle of a detector pair
                a value of zero makes the panels parallel to the x-axis
            D (float): the distance between the front faces of the two panels
            L (float): the full length of the panels
            H (float): the full height (z) of the panels
            v_m0 (float): the maximum slope in the x-y plane for the planogram
            v_m1 (float): the maximum slope in the x-z plane for the planograms
        """
        if v_m0 is None:
            v_m0 = 1.0
        if v_m1 is None:
            v_m1 = min(1.0, (H-1.5*T)/D)
        #bool add_planogram(float psi, float R, float L, float H, float v_m0, float v_m1, float T);
        self.set_model()
        self.libprojectors.add_planogram.restype = ctypes.c_bool
        self.libprojectors.add_planogram.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        if not self.libprojectors.add_planogram(psi, D, L, H, v_m0, v_m1, T):
            raise ValueError('Error: add_planogram failed')

    def remove_planogram(self, n):
        """
        Removes the n-th planogram from the PET geometry specification
        
        Args:
            n (int): the planogram to remove
        """
        #bool remove_planogram(int);
        self.set_model()
        self.libprojectors.remove_planogram.restype = ctypes.c_bool
        self.libprojectors.remove_planogram.argtypes = [ctypes.c_int]
        if not self.libprojectors.remove_planogram(n):
            raise ValueError('Error: remove_planogram failed')
    
    def keep_only_planogram(self, n):
        """
        Removes all planograms except one from the PET geometry specification
        
        Args:
            n (int): the planogram to remove
        """
        #bool keep_only_planogram(int);
        self.set_model()
        self.libprojectors.keep_only_planogram.restype = ctypes.c_bool
        self.libprojectors.keep_only_planogram.argtypes = [ctypes.c_int]
        if not self.libprojectors.keep_only_planogram(n):
            raise ValueError('Error: keep_only_planogram failed')

    def set_default_volume(self, scale=1.0):
        """
        Sets the default values for the volume parameters.
        This volume fills the entire PET field of view and the voxel size
        matches the detector pixel pitch.
        The PET geometry MUST be specified prior to calling this function.
        
        scale (float): optional parameter the scales the voxel size by the given amount
            small numbers make smaller voxels (and thus more voxels), while larger numbers
            make larger voxels (and thus fewer voxels)
        """
	    #bool set_default_volume();
        self.set_model()
        self.libprojectors.set_default_volume.restype = ctypes.c_bool
        self.libprojectors.set_default_volume.argtypes = [ctypes.c_float]
        if not self.libprojectors.set_default_volume(scale):
            raise ValueError('Error: set_default_volume failed')
    
    def set_volume(self, numX, numY, numZ, voxelWidth, voxelHeight = 0.0, offsetX = 0.0, offsetY = 0.0, offsetZ = 0.0):
        """
        Sets the PET volume parameters
        
        Args:
            numX (int): number of voxels in x
            numY (int): number of voxels in y
            numZ (int): number of voxels in z
            voxelWidth (float): the width (x and y dimensions) of a voxel (mm)
            voxelHeight (float): the height (z dimension) of a voxel (mm)
            offsetX (float): the x-coordinate of the center voxel
            offsetY (float): the y-coordinate of the center voxel
            offsetZ (float): the z-coordinate of the center voxel
        """
        if voxelHeight == 0.0:
            voxelHeight = voxelWidth
        #bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX = 0.0, float offsetY = 0.0, float offsetZ = 0.0);
        self.set_model()
        self.libprojectors.set_volume.restype = ctypes.c_bool
        self.libprojectors.set_volume.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        if not self.libprojectors.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ):
            raise ValueError('Error: set_volume failed')

    def calc_response(self, oversampling=1, mu=None):
        """
        This function calculate the response (i.e., sensitivity) and stores
        it is composed of a product of the solid angle, attenuation, and detector uniformity
        
        See also: add_module, calc_response, apply_corrections, clear_response
        
        Args:
            oversampling (int): the oversampling factor for calculating the detector uniformity
                larger numbers take longer to compute, but are more accurate as they model the
                partial response for rays at/ near the detector module boundaries
                if 0, this calculation is not made
            mu (numpy array): optional parameter which specifies the attenuation map (mm^-1 at 511 keV)
                is specified, includes attenuation in the response map
        """
        if oversampling > 0:
            self.response = self.stopping_power(self.response, oversampling=oversampling)
        else:
            self.response = None
        self.response = self.apply_solid_angle_model(self.response)
        
        if mu is not None:
            Pmu = self.allocate_projections()
            self.project(Pmu, mu)
            np.exp(Pmu, out=Pmu)
            self.divide(self.response, Pmu)
        
        return self.response
        
    def apply_corrections(self, g, oversampling=1, mu=None):
        """
        
        If the response is already specified, this function simply applies the correction
        If the response is not specified, this function calculates the response and then
        applies the correction.
        
        Args:
            g (list of numpy array): projection data
            oversampling (int): the oversampling factor for calculating the detector uniformity
                larger numbers take longer to compute, but are more accurate as they model the
                partial response for rays at/ near the detector module boundaries
                if 0, this calculation is not made
            mu (numpy array): optional parameter which specifies the attenuation map (mm^-1 at 511 keV)
                is specified, includes attenuation in the response map
        
        Returns:
            g, the corrected projection data
        """
        if self.response is None:
            self.calc_response(oversampling, mu)
        #self.divide(g, self.response)

        if self.response is not None:
            self.libprojectors.apply_corrections.restype = ctypes.c_bool
            self.libprojectors.apply_corrections.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_float]
            if self.libprojectors.apply_corrections(self.floatstarstar(g), self.floatstarstar(self.response), 0.75):
                return g
            else:
                return None
        else:
            return g


    def floatstarstar(self, g):
        """
        Internal (non-user) function used to pass data to the C++ library
        """
        c_float_p = ctypes.POINTER(ctypes.c_float)
        ptrs = [x.ctypes.data_as(c_float_p) for x in g]

        n_arrays = len(ptrs)
        FloatPtrArray = c_float_p * n_arrays
        c_arrays = FloatPtrArray(*ptrs)
        return c_arrays

    def bin(self, file_name, g=None):
        """
        Bins list mode data
        If data is given this function will add to that data (not reset it).
        This allows one to accumulate counts from several files.
        
        Args:
            file_name (string): full path to the list mode file
            g (list of numpy arrays or None): projection data
        """
        if not self.all_defined():
            raise ValueError('Error: volume and PET geometry must be defined')
        if g is None:
            g = self.allocate_projections()
        if sys.version_info[0] == 3:
            file_name = bytes(str(file_name), 'ascii')

        #bool bin(float** g, char* file_name);
        self.set_model()
        self.libprojectors.bin.restype = ctypes.c_bool
        self.libprojectors.bin.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_char_p]
        if not self.libprojectors.bin(self.floatstarstar(g), file_name):
            #raise TypeError('bin failed')
            pass
        return g

    def simulate_scatter(self, g, mu):
        """
        Simulates first order scatter
        
        Args:
            g (list of numpy arrays): projection data
            mu (numpy array): attenuation volume with units of 1/mm at 511 keV
            
        Returns:
            g, the planogram data of the scatter
        """
        if not self.all_defined():
            raise ValueError('Error: volume and PET geometry must be defined')
        if g is None:
            g = self.allocate_projections()

        #bool simulate_scatter(float** g, float* mu);
        self.set_model()
        self.libprojectors.simulate_scatter.restype = ctypes.c_bool
        self.libprojectors.simulate_scatter.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        if not self.libprojectors.simulate_scatter(self.floatstarstar(g), mu):
            raise TypeError('simulate_scatter failed')
        return g

    def project(self, g, f):
        """
        Performs forward projection
        
        Args:
            g (list of numpy array): projection data
            f (numpy array or None): volume data
            
        Returns:
            g, the projected planogram data
        """
        if not self.all_defined():
            raise ValueError('Error: volume and PET geometry must be defined')

        #bool project(float** g, float* f);
        self.set_model()
        self.libprojectors.project.restype = ctypes.c_bool
        self.libprojectors.project.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        if not self.libprojectors.project(self.floatstarstar(g), f):
            raise TypeError('project failed')
        if self.response is not None and self.include_response_with_projectors:
            self.multiply(g, self.response)
        return g
    
    def backproject(self, g, f=None):
        """
        Performs backprojection
        
        Args:
            g (list of numpy array): projection data
            f (numpy array or None): volume data
            
        Returns:
            f, the backprojected volume
        """
        if not self.all_defined():
            raise ValueError('Error: volume and PET geometry must be defined')
        if f is None:
            f = self.allocate_volume()
            
        if self.response is not None and self.include_response_with_projectors:
            Rg = self.copyData(g)
            self.multiply(Rg,self.response)
        else:
            Rg = g

        #bool backproject(float** g, float* f);
        self.set_model()
        self.libprojectors.backproject.restype = ctypes.c_bool
        self.libprojectors.backproject.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        if not self.libprojectors.backproject(self.floatstarstar(Rg), f):
            raise TypeError('backproject failed')
        return f
    
    def FBP(self, g, f=None):
        """
        Performs analytic reconstruction.
        If the data has already been rebinned, it will just perform FBP reconstruction.
        If the data has not been rebinned, it will rebin it (with PFDR)
        and then perform FBP reconstruction.
        Even if a rebinned step is performed, the original geometry is maintained
        i.e., it does not modify it to the rebinned geometry.
        
        Args:
            g (list of numpy array): projection data
            f (numpy array or None): volume data
            
        Returns:
            f, the reconstructed data
        """
        if not self.all_defined():
            raise ValueError('Error: volume and PET geometry must be defined')
        if f is None:
            f = self.allocate_volume()

        #bool FBP(float** g, float* f);
        self.set_model()
        self.libprojectors.FBP.restype = ctypes.c_bool
        self.libprojectors.FBP.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        if not self.libprojectors.FBP(self.floatstarstar(g), f):
            raise TypeError('FBP failed')
        return f
    
    def PFDR(self, g, g_reb=None):
        """
        Perform the Planogram Frequency-Distance Rebinning Algorithm
        Note that this algorithm changes the planogram specification
        So if you wish to retain the original geometry you should create
        a copy of this class and perform this function there.
        
        Args:
            g (list of numpy arrays): projection data
            g_reb (list of numpy arrays or None): rebinned data
        
        Returns:
            g_reb, the rebinned data
        """
        if not self.all_defined():
            raise ValueError('Error: volume and PET geometry must be defined')
        if g_reb is None:
            g_reb = self.allocate_rebinned_projections()

        #bool PFDR(float** g, float** g_out);
        self.set_model()
        self.libprojectors.PFDR.restype = ctypes.c_bool
        self.libprojectors.PFDR.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float))]
        if not self.libprojectors.PFDR(self.floatstarstar(g), self.floatstarstar(g_reb)):
            raise TypeError('PFDR failed')
        return g_reb

    def add_object(self, typeOfObject, c, r, val, A=None, clip=None):
        """Adds a geometric object to the phantom
        
        This function operates in two modes: (1) specifying a voxelized phantom and (2) specifying a phantom
        to be used in an analytic ray-tracing simulation (see rayTrace).
        The object will be added to a stack of objects that define a phantom to be used for
        an analytic ray-tracing simulation.
        
        The order in which multiple object are defined is important.  Background objects must be specified first
        and foreground objects defined last.  If you reverse the order, then the foreground objects will be effectively overriden
        by the background objects.
        
        Args:
            type (int): ELLIPSOID=0, PARALLELEPIPED=1, CYLINDER_X=2, CYLINDER_Y=3, CYLINDER_Z=4, CONE_X=5, CONE_Y=6, CONE_Z=7
            c (3X1 numpy array): x,y,z coordinates of the center of the object
            r (3X1 numpy array): radii in the x,y,z directions of the object
            val (float): the values to ascribe inside this object
            A (3x3 numpy array): rotation matrix to rotate the object about its center
            clip (3X1 numpy array): specifies the clipping planes, if any
        """
        self.libprojectors.add_object.restype = ctypes.c_bool
        if A is None:
            A = np.zeros((3,3),dtype=np.float32)
            A[0,0] = 1.0
            A[1,1] = 1.0
            A[2,2] = 1.0
        if clip is None:
            clip = np.zeros(3,dtype=np.float32)
        
        if isinstance(c, int) or isinstance(c, float):
            c = [c, c, c]
        if isinstance(r, int) or isinstance(r, float):
            r = [r, r, r]
        
        if isinstance(typeOfObject, str):
            typeOfObject = typeOfObject.lower()
            if typeOfObject == 'ball' or typeOfObject == 'sphere' or typeOfObject == 'ellipsoid':
                typeOfObject = 0
            elif typeOfObject == 'box' or typeOfObject == 'parallelepiped':
                typeOfObject = 1
            elif typeOfObject == 'can_x':
                typeOfObject = 2
            elif typeOfObject == 'can_y':
                typeOfObject = 3
            elif typeOfObject == 'can_z' or typeOfObject == 'can':
                typeOfObject = 4
            elif typeOfObject == 'cone_x':
                typeOfObject = 5
            elif typeOfObject == 'cone_y':
                typeOfObject = 6
            elif typeOfObject == 'cone_z' or typeOfObject == 'cone':
                typeOfObject = 7
            else:
                print('Error: unknown object type')
                return False
        
        c = np.ascontiguousarray(c, dtype=np.float32)
        r = np.ascontiguousarray(r, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        clip = np.ascontiguousarray(clip, dtype=np.float32)
        self.set_model()
        self.libprojectors.add_object.argtypes = [ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        return self.libprojectors.add_object(int(typeOfObject), c, r, float(val), A, clip)
        
    def add_module(self, c, r, val=0.0815, A=None):
        """
        Adds a module to the scintillator model that can be used to calculate the sensitivity
        (stopping power) of each coincidence.
        
        Note that this is NOT the same as add_planogram
        
        Args:
            c (list of 3 floats): the center of the module
            r (list of 3 floats): half width, half lengthm half depth of the module
            val (float): the attenuation coefficient of the scintillator at 511 keV
            A (3x3 numpy array): rotation matrix
        """
        typeOfObject = 1
        self.libprojectors.add_module.restype = ctypes.c_bool
        if A is None:
            A = np.zeros((3,3),dtype=np.float32)
            A[0,0] = 1.0
            A[1,1] = 1.0
            A[2,2] = 1.0
        clip = np.zeros(3,dtype=np.float32)
        
        if isinstance(c, int) or isinstance(c, float):
            c = [c, c, c]
        if isinstance(r, int) or isinstance(r, float):
            r = [r, r, r]
        
        c = np.ascontiguousarray(c, dtype=np.float32)
        r = np.ascontiguousarray(r, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        clip = np.ascontiguousarray(clip, dtype=np.float32)
        self.set_model()
        self.libprojectors.add_module.argtypes = [ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        return self.libprojectors.add_module(int(typeOfObject), c, r, float(val), A, clip)

    def clear_phantom(self):
        """Clears all phantom objects"""
        self.set_model()
        self.libprojectors.clear_phantom()
        
    def clear_modules(self):
        """Clears all module objects"""
        self.set_model()
        self.libprojectors.clear_modules()

    def scale_phantom(self, c):
        r"""Scales the size of the phantom by the provided factor
        
        One must have a phantom already defined before running this function.
        
        Args:
            c (float or list of three float): the scaling values (values greater than one make the phantom larger)
        """
        if isinstance(c, int) or isinstance(c, float):
            c_x = c
            c_y = c
            c_z = c
        elif isinstance(c, np.ndarray) and c.size == 3:
            c_x = c[0]
            c_y = c[1]
            c_z = c[2]
        else:
            return False

        self.set_model()
        self.libprojectors.scale_phantom.restype = ctypes.c_bool
        self.libprojectors.scale_phantom.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        return self.libprojectors.scale_phantom(c_x, c_y, c_z)

    def voxelize(self, f=None, oversampling=1):
        r"""Voxelizes a phantom defined by geometric objects.
        
        One must have a phantom already defined before running this function.
        
        Args:
            f (C contiguous float32 numpy array): volume data
            oversampling (int): the oversampling factor of the voxelization
        """
        if not self.volume_defined():
            raise ValueError("Error: volume parameters not specified")
        if f is None:
            f = self.allocate_volume()
        self.set_model()
        self.libprojectors.voxelize.restype = ctypes.c_bool
        self.libprojectors.voxelize.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
        if self.libprojectors.voxelize(f, oversampling):
            return f
        else:
            return None

    def ray_trace(self, g=None, oversampling=1):
        """Performs analytic ray-tracing simulation through a phantom composed of geometrical objects

        See the addObject function for how to build the phantom description
        The CT geometry parameters must be specified prior to running this functions
        
        Args:
            g (C contiguous float32 numpy array): CT projection data
            oversampling (int): the oversampling factor for each ray
            
        Returns:
            g
        
        """
        if not self.geometry_defined():
            raise ValueError("Error: PET scanner not specified")
        if g is None:
            g = self.allocate_projections()
        self.set_model()
        self.libprojectors.ray_trace.restype = ctypes.c_bool
        #self.libprojectors.ray_trace.argtypes = [ctypes.POINTER(c_float_p), ctypes.c_int]
        #c_float_p = ctypes.POINTER(ctypes.c_float)
        self.libprojectors.ray_trace.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
        if self.libprojectors.ray_trace(self.floatstarstar(g), int(oversampling)):
            return g
        else:
            return None
            
    def stopping_power(self, g=None, oversampling=1):
        """Performs analytic ray-tracing simulation through a phantom composed of geometrical objects

        See the addObject function for how to build the phantom description
        The CT geometry parameters must be specified prior to running this functions
        
        Args:
            g (C contiguous float32 numpy array): CT projection data
            oversampling (int): the oversampling factor for each ray
            
        Returns:
            g
        
        """
        if not self.geometry_defined():
            raise ValueError("Error: PET scanner not specified")
        if g is None:
            g = self.allocate_projections()
        self.set_model()
        self.libprojectors.stopping_power.restype = ctypes.c_bool
        #self.libprojectors.stopping_power.argtypes = [ctypes.POINTER(c_float_p), ctypes.c_int]
        #c_float_p = ctypes.POINTER(ctypes.c_float)
        self.libprojectors.stopping_power.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
        if self.libprojectors.stopping_power(self.floatstarstar(g), int(oversampling)):
            self.gaussian_filter(g, 0.6571)
            return g
        else:
            return None
            
    def apply_solid_angle_correction(self, g=None, do_inverse=False):
        """"
        Applies solid angle correction to given projection data
        
        Args:
            g (list of numpy arrays): projection data
            
        Returns:
            g, the same as the input
        """
        if g is None:
            g = self.allocate_projections()
            self.setToOne(g)
        self.set_model()
        self.libprojectors.set_solid_angle_correction.restype = ctypes.c_bool
        self.libprojectors.set_solid_angle_correction.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_bool]
        if self.libprojectors.set_solid_angle_correction(self.floatstarstar(g), do_inverse):
            return g
        else:
            return None
            
    def apply_solid_angle_model(self, g=None):
        """
        Applies solid angle model to given projection data
        
        Args:
            g (list of numpy arrays): projection data
            
        Returns:
            g, the same as the input
        """
        return self.apply_solid_angle_correction(g, True)
        
    def save_data(self, file_name, x):
        """
        Saves data in an npy or nrrd file
        
        Args:
            file_name (string): full path to npy or nrrd file you want to save
            x (numpy array): data to be saved
        """
        if file_name.endswith('.npy'):
            np.save(file_name, x)
        elif file_name.endswith('.nrrd'):
            try:
                import nrrd
                compression_level = 1
                # https://pynrrd.readthedocs.io/en/latest/examples.html
                #header = {'units': ['mm', 'mm', 'mm'], 'spacings': [T, T, T], 'axismins': [offset_0, offset_1, offset_2], 'thicknesses': [T, T, T],}
                if isinstance(x, list):
                    base_name, file_extension = os.path.splitext(file_name)
                    for n in range(len(x)):
                        nrrd.write(base_name + '_' + str(int(n)) + file_extension, x[n], compression_level=compression_level)#, header)
                else:
                    nrrd.write(file_name, x, compression_level=compression_level)
            except:
                raise TypeError('Error: Failed to load nrrd library!\n To install this package do: pip install pynrrd')
        else:
            raise ValueError('currently only npy and nrrd file types are supported')
            
    def load_data(self, file_name, x=None):
        """
        Loads data from an npy or nrrd file
        
        Args:
            file_name (string): full path to an npy or nrrd file
            x (numpy array or None): where to store the data, if None allocates the data for you
            
        Returns:
            x, the same as the input
        """
        if os.path.isfile(file_name) == False:
            print('file does not exist')
            return None
        if file_name.endswith('.npy'):
            if x is None:
                return np.load(file_name)
            else:
                x[:] = np.load(file_name)
                return x
        elif file_name.endswith('.nrrd'):
            try:
                import nrrd
                if x is not None:
                    x[:], header = nrrd.read(file_name)
                else:
                    x, header = nrrd.read(file_name)
                #T_fromFile = header['spacings'][0]
                return x
            except:
                raise TypeError('Error: Failed to load nrrd library!\n To install this package do: pip install pynrrd')
        else:
            raise ValueError('currently only npy and nrrd file types are supported')

    def display(self, x):
        """
        Uses napari to display the provided 3D data
        
        Args:
            x (numpy array): data
        """
        try:
            import napari
            if len(x.shape) == 3 and (x.shape[0] == 1 or x.shape[1] == 1 or x.shape[2] == 1):
                viewer = napari.view_image(np.squeeze(x), rgb=False)
            else:
                viewer = napari.view_image(x, rgb=False)
            napari.run()
        except:
            print('Cannot load napari, to install run this command:')
            print('pip install napari[all]')

    def as_float_ptr(self, arr):
        if arr is None:
            return None
        else:
            arr = np.asarray(arr, dtype=np.float32, order="C")
            return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def inner_product_helper(self, x, y, w=None):
        if len(x.shape) == 4:
            N_1, N_2, N_3, N_4 = x.shape
            self.set_model()
            self.libprojectors.inner_product4D.restype = ctypes.c_float
            self.libprojectors.inner_product4D.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            return self.libprojectors.inner_product4D(self.as_float_ptr(x), self.as_float_ptr(y), self.as_float_ptr(w), N_1, N_2, N_3, N_4)
        elif len(x.shape) == 3:
            N_1, N_2, N_3 = x.shape
            self.set_model()
            self.libprojectors.inner_product3D.restype = ctypes.c_float
            self.libprojectors.inner_product3D.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            return self.libprojectors.inner_product3D(self.as_float_ptr(x), self.as_float_ptr(y), self.as_float_ptr(w), N_1, N_2, N_3)
        else:
            return np.sum(x*y*w)

    def inner_product(self, x, y, w=None):
        """
        Computes sum(x*y) or sum(x*y*w)
        
        Args:
            x (list of numpy arrays or a numpy array): projection or volume data
            y (list of numpy arrays or a numpy array): projection or volume data
            w (list of numpy arrays or a numpy array or None): projection or volume data
            
        Returns:
            scalar of the calculation
        """
        if isinstance(x, list):
            accum = 0.0
            for n in range(len(x)):
                if w is None:
                    #accum += np.sum(x[n]*y[n])
                    accum += self.inner_product_helper(x[n], y[n])
                elif isinstance(w, (int, float)):
                    #accum += np.sum(x[n]*y[n])*w
                    accum += self.inner_product_helper(x[n], y[n])*w
                else:
                    #accum += np.sum(x[n]*y[n]*w[n])
                    accum += self.inner_product_helper(x[n], y[n], w[n])
            return accum
        else:
            return self.inner_product_helper(x, y, w)

    def innerProd(self, x, y, w=None):
        """
        Computes sum(x*y) or sum(x*y*w)
        
        Args:
            x (list of numpy arrays or a numpy array): projection or volume data
            y (list of numpy arrays or a numpy array): projection or volume data
            w (list of numpy arrays or a numpy array or None): projection or volume data
            
        Returns:
            scalar of the calculation
        """
        return self.inner_product(x, y, w)
    
    def isAllZeros(self, f):
        """
        Returns whether all elements in a numpy array are zero
        
        Args:
            f (numpy array): data
            
        Returns:
            True if all elements of the input are zero, False otherwise
        """
        if not np.any(f):
            return True
        else:
            return False
            
    def setToConstant(self, x, c=0.0):
        """
        Computes x[:] = c
        
        Args:
            x (list of numpy arrays or a numpy array): data
            c (float): scalar
            
        Returns:
            the same as the input
        """
        if isinstance(x, list):
            for n in range(len(x)):
                x[n][:] = c
        else:
            x[:] = c
            
    def setToZero(self, x):
        """
        Computes x[:] = 0.0
        
        Args:
            x (list of numpy arrays or a numpy array): data
            
        Returns:
            the same as the input
        """
        self.setToConstant(x, 0.0)
        
    def setToOne(self, x):
        """
        Computes x[:] = 1.0
        
        Args:
            x (list of numpy arrays or a numpy array): data
            
        Returns:
            the same as the input
        """
        self.setToConstant(x, 1.0)
    
    def sensitivity(self, Pstar1=None):
        """
        Computes P*1, the backprojection of data that is all ones
        
        Args:
            Pstar1 (numpy array or None): volume data
            
        Returns:
            the same as the input
        """
        if Pstar1 is None:
            Pstar1 = self.allocate_volume()
        ones = self.allocate_projections()
        for n in range(len(ones)):
            ones[n][:] = 1.0
        self.backproject(ones, Pstar1)
        return Pstar1
    
    def clip(self, x, min_val=0.0, max_val=None):
        if isinstance(x, list):
            for n in range(len(x)):
                np.clip(x[n], min_val, max_val, out=x[n])
        else:
            np.clip(x, min_val, max_val, out=x)
        return x
    
    def scale(self, x, c=1.0):
        """
        Computes x = c*x
        
        Args:
            x (list of numpy arrays or numpy array): projection or volume data
            c (float): scalar
            
        Returns:
            the same as the first argument
        """
        if isinstance(x, list):
            for n in range(len(x)):
                x[n][:] *= c
        else:
            x[:] *= c
        return x
    
    def square(self, x):
        """
        Computes x = x*x
        
        Args:
            x (list of numpy arrays or numpy array): projection or volume data
            
        Returns:
            the same as the first argument
        """
        return self.multiply(x, x)
    
    def subtract(self, x, y):
        """
        Computes x = x - y
        
        Args:
            x (list of numpy arrays or numpy array): projection or volume data
            y (list of numpy arrays or numpy array): projection or volume data
            
        Returns:
            the same as the first argument
        """
        if isinstance(x, list):
            for n in range(len(x)):
                x[n][:] -= y[n][:]
        else:
            x[:] -= y[:]
        return x
        
    def scalarAdd(self, x, c, y):
        """
        Computes x = x + c*y
        
        Args:
            x (list of numpy arrays or numpy array): projection or volume data
            c (float): scalar
            y (list of numpy arrays or numpy array): projection or volume data
            
        Returns:
            the same as the first argument
        """
        if isinstance(x, list):
            for n in range(len(x)):
                x[n][:] += c*y[n][:]
        else:
            x[:] += c*y[:]
        return x
    
    def multiply(self, x, y):
        """
        Computes x = x * y
        
        Args:
            x (list of numpy arrays or a numpy array): projections or volume data
            x (list of numpy arrays or a numpy array): projections or volume data
            
        Returns:
            the same as the first argument
        """
        self.set_model()
        if isinstance(x, list):
            if len(x[0].shape) > 4 or len(x[0].shape) < 3:
                raise ValueError('inputs must be lists of 3D or 4D numpy arrays')
            count = len(x)
            if len(x[0].shape) == 4:
                self.libprojectors.multiply4D.restype = ctypes.c_bool
                self.libprojectors.multiply4D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for n in range(count):
                    N_1, N_2, N_3, N_4 = x[n].shape
                    if not self.libprojectors.multiply4D(x[n], y[n], N_1, N_2, N_3, N_4):
                        print('multiply failed!')
                        return None
                return x
            else:
                self.libprojectors.multiply3D.restype = ctypes.c_bool
                self.libprojectors.multiply3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for n in range(count):
                    N_1, N_2, N_3 = x[n].shape
                    if not self.libprojectors.multiply3D(x[n], y[n], N_1, N_2, N_3):
                        print('multiply failed!')
                        return None
                return x
        else:
            if len(x.shape) != 3:
                raise ValueError('inputs must be 3D numpy arrays')
            N_1, N_2, N_3 = x.shape
            self.libprojectors.multiply3D.restype = ctypes.c_bool
            self.libprojectors.multiply3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            if self.libprojectors.multiply3D(x, y, N_1, N_2, N_3):
                return x
            else:
                print('multiply failed!')
                return None
                
    def divide(self, num, denom):
        """
        Computes num = num / denom
        
        Args:
            num (list of numpy arrays or a numpy array): the numerator
            denom (list of numpy arrays or a numpy array): the denominator
            
        Returns:
            the same as the first argument
        """
        self.set_model()
        if isinstance(num, list):
            if len(num[0].shape) > 4 or len(num[0].shape) < 3:
                raise ValueError('inputs must be lists of 3D or 4D numpy arrays')
            count = len(num)
            if len(num[0].shape) == 4:
                self.libprojectors.divide4D.restype = ctypes.c_bool
                self.libprojectors.divide4D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for n in range(count):
                    N_1, N_2, N_3, N_4 = num[n].shape
                    if not self.libprojectors.divide4D(num[n], denom[n], N_1, N_2, N_3, N_4):
                        print('divide failed!')
                        return None
                return num
            else:
                self.libprojectors.divide3D.restype = ctypes.c_bool
                self.libprojectors.divide3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for n in range(count):
                    N_1, N_2, N_3 = num[n].shape
                    if not self.libprojectors.divide3D(num[n], denom[n], N_1, N_2, N_3):
                        print('divide failed!')
                        return None
                return num
        else:
            if len(num.shape) != 3:
                raise ValueError('inputs must be 3D numpy arrays')
            N_1, N_2, N_3 = num.shape
            self.libprojectors.divide3D.restype = ctypes.c_bool
            self.libprojectors.divide3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            if self.libprojectors.divide3D(num, denom, N_1, N_2, N_3):
                return num
            else:
                print('divide failed!')
                return None
    
    def rdivide(self, denom, num):
        """
        Computes denom = num / denom
        
        Args:
            denom (list of numpy arrays or a numpy array): the denominator
            num (list of numpy arrays or a numpy array): the numerator
            
        Returns:
            the same as the first argument
        """
        self.set_model()
        if isinstance(num, list):
            if len(num[0].shape) > 4 or len(num[0].shape) < 3:
                raise ValueError('inputs must be lists of 3D or 4D numpy arrays')
            count = len(num)
            if len(num[0].shape) == 4:
                self.libprojectors.rdivide4D.restype = ctypes.c_bool
                self.libprojectors.rdivide4D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for n in range(count):
                    N_1, N_2, N_3, N_4 = num[n].shape
                    if not self.libprojectors.rdivide4D(num[n], denom[n], N_1, N_2, N_3, N_4):
                        print('rdivide failed!')
                        return None
                return denom
            else:
                self.libprojectors.rdivide3D.restype = ctypes.c_bool
                self.libprojectors.rdivide3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
                for n in range(count):
                    N_1, N_2, N_3 = num[n].shape
                    if not self.libprojectors.rdivide3D(num[n], denom[n], N_1, N_2, N_3):
                        print('rdivide failed!')
                        return None
                return denom
        else:
            if len(num.shape) != 3:
                raise ValueError('inputs must be 3D numpy arrays')
            N_1, N_2, N_3 = num.shape
            self.libprojectors.rdivide3D.restype = ctypes.c_bool
            self.libprojectors.rdivide3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            if self.libprojectors.rdivide3D(num, denom, N_1, N_2, N_3):
                return denom
            else:
                print('rdivide failed!')
                return None
                
    def reciprocal(self, x, divide_by_zero_value=0.0):
        """
        Computes the reciprocal of a numpy array or list of numpy arrays in-place
        Args:
            x (list of numpy arrays or numpy array): projections or volume
            divide_by_zero_value (float): the value to assign when the denominator is zero
            
        Returns:
            same as the input
        
        """
        self.set_model()
        if isinstance(x, list):
            if len(x[0].shape) > 4 or len(x[0].shape) < 3:
                raise ValueError('inputs must be lists of 3D or 4D numpy arrays')
            count = len(x)
            if len(x[0].shape) == 4:
                self.libprojectors.reciprocal4D.restype = ctypes.c_bool
                self.libprojectors.reciprocal4D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
                for n in range(count):
                    N_1, N_2, N_3, N_4 = x[n].shape
                    if not self.libprojectors.reciprocal4D(x[n], N_1, N_2, N_3, N_4, divide_by_zero_value):
                        print('reciprocal failed!')
                        return None
                return x
            else:
                self.libprojectors.reciprocal3D.restype = ctypes.c_bool
                self.libprojectors.reciprocal3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
                for n in range(count):
                    N_1, N_2, N_3 = x[n].shape
                    if not self.libprojectors.reciprocal3D(x[n], N_1, N_2, N_3, divide_by_zero_value):
                        print('reciprocal failed!')
                        return None
                return x
        else:
            if len(x.shape) != 3:
                raise ValueError('input must be 3D numpy arrays')
            N_1, N_2, N_3 = x.shape
            self.libprojectors.reciprocal3D.restype = ctypes.c_bool
            self.libprojectors.reciprocal3D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
            if self.libprojectors.reciprocal3D(x, N_1, N_2, N_3, divide_by_zero_value):
                return x
            else:
                print('reciprocal failed!')
                return None
    
    def MLEM(self, g, f, numIter, delta=2.0, beta=0.0, mask=None):
        r"""Maximum Likelihood-Expectation Maximization reconstruction
        
        This algorithm performs reconstruction with the following update equation
        
        .. math::
           \begin{eqnarray}
             f_{n+1} &:=& \frac{f_n}{P^T 1 + R'(f_n)} P^T\left[ \frac{g}{Pf_n} \right]
           \end{eqnarray}
           
        where R'(f) is the gradient of the regularization term(s).
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This reconstruction algorithms assumes the projection data, g, is Poisson distributed which is the
        correct model for SPECT data.
        CT projection data is not Poisson distributed because of the application of the -log
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            filters (filterSequence object): list of differentiable regularization filters
            mask (C contiguous float32 numpy or torch array): projection data to mask out bad data, etc.
            
        
        Returns:
            f, the same as the input with the same name
        """
        if mask is not None and mask.shape != g.shape:
            print('Error: mask must be the same shape as the projection data!')
            return None
            
        include_response_with_projectors_save = self.include_response_with_projectors
        self.include_response_with_projectors = True
            
        #if self.verify_inputs(g,f) == False:
        #    return None
        if f is None:
            f = self.allocate_volume()
            self.setToOne(f)
        elif self.isAllZeros(f) == True:
            self.setToOne(f)
        else:
            f[f<0.0] = 0.0
 
        if mask is not None:
            Pstar1 = np.zeros_like(f)
            self.backproject(mask,Pstar1)
        else:
            Pstar1 = self.sensitivity()
        Pstar1[Pstar1==0.0] = 1.0
        d = np.zeros_like(f)
        Pd = self.allocate_projections()

        for n in range(numIter):
            if self.print_warnings:
                print('MLEM iteration ' + str(n+1) + ' of ' + str(numIter))
            self.project(Pd, f)
            self.rdivide(Pd, g)
            self.backproject(Pd, d)
            if beta <= 0.0:
                #f *= d/Pstar1
                self.multiply(f, self.divide(d, Pstar1))
            else:
                #TVf = self.TVgradient(f, delta, beta)
                TVf = self.relative_differences_grad(f, delta, beta)
                TVf[:] += Pstar1[:]
                np.maximum(0.1*Pstar1, TVf, out=TVf)
                self.multiply(f, self.divide(d, TVf))
            
        self.include_response_with_projectors = include_response_with_projectors_save
            
        return f
        
    def LS(self, g, f, numIter, preconditioner=None, nonnegativityConstraint=True):
        r"""Least Squares reconstruction

        This function minimizes the Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*P1)^-1, where 1 is a volume of all ones, P is forward projection, and P* is backprojection.
        The Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{LS}(f) &:=& \frac{1}{2} \| Pf - g \|^2
           \end{eqnarray}
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, 1.0, 0.0, 1.0, preconditioner, nonnegativityConstraint)
        
    def WLS(self, g, f, numIter, W=None, preconditioner=None, nonnegativityConstraint=True):
        r"""Weighted Least Squares reconstruction
        
        This function minimizes the Weighted Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*WP1)^-1, where 1 is a volume of all ones, W are the weights, P is forward projection, and P* is backprojection.
        The Weighted Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{WLS}(f) &:=& \frac{1}{2} (Pf - g)^T W (Pf - g)
           \end{eqnarray}
           
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            W (C contiguous float32 numpy array): weights, should be the same size as g, if not given, W=exp(-g); can also be used to mask out bad data
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, 1.0, 0.0, W, preconditioner, nonnegativityConstraint)
        
    def RLS(self, g, f, numIter, delta=1.0, beta=0.0, preconditioner=None, nonnegativityConstraint=True):
        r"""Regularized Least Squares reconstruction
        
        This function minimizes the Regularized Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*P1)^-1, where 1 is a volume of all ones, P is forward projection, and P* is backprojection.
        The Regularized Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{RLS}(f) &:=& \frac{1}{2} \| Pf - g \|^2 + R(f)
           \end{eqnarray}

        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            filters (filterSequence object): list of differentiable regularization filters
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, delta, beta, 1.0, preconditioner, nonnegativityConstraint)
    
    def RWLS(self, g, f, numIter, delta=1.0, beta=0.0, W=None, preconditioner=None, nonnegativityConstraint=True, include_response_with_projectors=False):
        r"""Regularized Weighted Least Squares reconstruction
        
        This function minimizes the Regularized Weighted Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*WP1)^-1, where 1 is a volume of all ones, W are the weights, P is forward projection, and P* is backprojection.
        The Regularized Weighted Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{RWLS}(f) &:=& \frac{1}{2} (Pf - g)^T W (Pf - g) + R(f)
           \end{eqnarray}

        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            filters (filterSequence object): list of differentiable regularization filters
            W (C contiguous float32 numpy array): weights, should be the same size as g, if not given, W:=exp(-g); can also be used to mask out bad data
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        
        conjGradRestart = 50
        include_response_with_projectors_save = self.include_response_with_projectors
        self.include_response_with_projectors = include_response_with_projectors # default is False
        if W is None:
            W = self.copy_data(g)
            self.gaussian_filter(W, 3.0)
            self.clip(W, 7.0)
            self.reciprocal(W)
            if self.response is not None and self.include_response_with_projectors == False:
                self.multiply(W, self.response)
                self.multiply(W, self.response)
        
        if self.response is not None and self.include_response_with_projectors == False:
            self.apply_corrections(g)
        
        Pf = self.copyData(g)
        
        #ones = self.allocate_volume()
        #ones[:] = 1.0
        #self.project(Pf,ones)
        #for n in range(len(Pf)):
        #    g[n][Pf[n]==0.0] = 0.0
        
        if self.isAllZeros(f) == False:
            # fix scaling
            #print('fixing scaling...')
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
            self.project(Pf,f)
            #print(np.sum(Pf[0])+np.sum(Pf[1]))
            #print(np.sum(g[0])+np.sum(g[1]))
            #Pf_dot_Pf = self.innerProd(Pf, Pf)
            #g_dot_Pf = self.innerProd(g, Pf)
            #if Pf_dot_Pf > 0.0 and g_dot_Pf > 0.0:
            #    print('fixed scaling by', g_dot_Pf / Pf_dot_Pf)
            #    self.scale(f, g_dot_Pf / Pf_dot_Pf)
            #    self.scale(Pf, g_dot_Pf / Pf_dot_Pf)
        else:
            self.setToZero(Pf)
        Pf_minus_g = Pf
        self.subtract(Pf_minus_g, g)
        
        grad = self.allocate_volume()
        u = self.allocate_volume()
        
        d = self.allocate_volume()
        Pd = self.allocate_projections()
        
        grad_old_dot_grad_old = 0.0
        grad_old = self.allocate_volume()
        
        if preconditioner == True:
            preconditioner = 'SQS'
        if preconditioner == 'SQS':
            # Calculate the SQS preconditioner
            # Reuse some of the memory allocated above
            #Q = 1.0 / P*WP1
            Q = self.allocate_volume()
            
            Q[:] = 1.0
            self.project(Pd, Q)
            if W is np.ndarray or isinstance(W, list):
                self.multiply(Pd, W)
            self.backproject(Pd, Q)
            self.reciprocal(Q, 1.0)
        else:
            Q = None
        
        for n in range(numIter):
            if self.print_warnings:
                print('RWLS iteration ' + str(n+1) + ' of ' + str(numIter))
            WPf_minus_g = Pf_minus_g
            if W is np.ndarray or isinstance(W, list):
                self.multiply(WPf_minus_g, W)
            self.backproject(WPf_minus_g, grad)
            if beta > 0.0:
                Sf1 = self.TVgradient(f, delta, beta)
                grad[:] += Sf1[:]
                
            u[:] = grad[:]
            if Q is not None:
                u[:] = Q[:]*u[:]
            
            if n == 0 or (n % conjGradRestart) == 0:
                d[:] = u[:]
            else:
                gamma = (self.innerProd(u,grad) - self.innerProd(u,grad_old)) / grad_old_dot_grad_old

                d[:] = u[:] + gamma*d[:]

                if self.innerProd(d, grad) <= 0.0:
                    if self.print_warnings:
                        print('\tRLWS-CG: CG descent condition violated, must use GD descent direction')
                    d[:] = u[:]
            
            grad_old_dot_grad_old = self.innerProd(u, grad)
            grad_old[:] = grad[:]
            
            self.project(Pd, d)
            
            num = self.innerProd(d,grad)
            stepSize = self.RWLSstepSize(f, grad, d, Pd, W, delta, beta, num)
            if stepSize <= 0.0:
                if self.print_warnings:
                    print('invalid step size; quitting!')
                break
            
            f[:] = f[:] - stepSize*d[:]
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
                self.project(Pf,f)
            else:
                self.scalarAdd(Pf, -stepSize, Pd)
            self.subtract(Pf_minus_g, g)
            if self.print_cost:
                dataFidelity = 0.5*self.innerProd(Pf_minus_g, Pf_minus_g, W)
                if beta > 0.0:
                    regularizationCost = self.TVcost(f, delta, beta)
                    print('\tcost = ' + str(dataFidelity+regularizationCost) + ' = ' + str(dataFidelity) + ' + ' + str(regularizationCost))
                else:
                    print('\tcost = ' + str(dataFidelity))
                
        self.include_response_with_projectors = include_response_with_projectors_save
        if self.response is not None and self.include_response_with_projectors == False:
            self.multiply(g, self.response)
        
        return f

    def RWLSstepSize(self, f, grad, d, Pd, W, delta, beta, num=None):
        """Calculates the step size for an RWLS iteration

        Args:
            f (C contiguous float32 numpy or torch array): volume data
            grad (C contiguous float32 numpy or torch array): gradient of the RWLS cost function
            d (C contiguous float32 numpy or torch array): descent direction of the RWLS cost function
            Pd (C contiguous float32 numpy or torch array): forward projection of d
            W (C contiguous float32 numpy or torch array): weights, should be the same size as g, if not given, assumes is all ones
            filters (filterSequence object): list of filters to use as a regularizer terms
        
        Returns:
            step size (float)
        """
        if num is None:
            num = self.innerProd(d,grad)
        if W is not None:
            denomA = self.innerProd(Pd,Pd,W)
        else:
            denomA = self.innerProd(Pd,Pd)
        denomB = 0.0
        if beta > 0.0:
            denomB = self.TVquadForm(f, d, delta, beta)
        denom = denomA + denomB

        stepSize = 0.0
        if np.abs(denom) > 1.0e-16:
            stepSize = num / denom
        if self.print_warnings:
            print('\tlambda = ' + str(stepSize))
        return stepSize
        
    def TVcost(self, f, delta, beta=0.0, p=1.0):
        r"""Calculates the anisotropic Total Variation (TV) functional, i.e., cost of the provided numpy array
        
        This function uses a Huber-like loss function applied to the differences of neighboring samples (in 3D).
        One can switch between using 6 or 26 neighbors using the \"set_numTVneighbors\" function.
        The aTV functional with Huber-like loss function is given by
        
        .. math::
           \begin{eqnarray}
             R(x) &:=& \sum_{\boldsymbol{i}} \sum_{\boldsymbol{j} \in N_{\boldsymbol{i}}} \|\boldsymbol{i} - \boldsymbol{j}\|^{-1} h(x_\boldsymbol{i} - x_\boldsymbol{j}) \\
             h(t) &:=& \begin{cases} \frac{1}{2}t^2, & \text{if } |t| \leq delta \\ \frac{delta^{2 - p}}{p}|t|^p + delta^2\left(\frac{1}{2} - \frac{1}{p}\right), & \text{if } |t| > delta \end{cases}
           \end{eqnarray}

        where :math:`N_{\boldsymbol{i}}` is a neighborhood around the 3D pixel index :math:`\boldsymbol{i} = (i_1, i_2, i_3)`.
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size.
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            TV functional value
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TV_cost.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TV_cost.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TV_cost(f.data_ptr(), N_1, N_2, N_3, delta, beta, p, f.is_cuda == False)
        else:
            self.libprojectors.TV_cost.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TV_cost(f, N_1, N_2, N_3, delta, beta, p, True)
        
    def TVgradient(self, f, delta, beta=0.0, p=1.0):
        r"""Calculates the gradient of the anisotropic Total Variation (TV) functional of the provided numpy array
        
        This function uses a Huber-like loss function applied to the differences of neighboring samples (in 3D).
        One can switch between using 6 or 26 neighbors using the \"set_numTVneighbors\" function.
        The aTV functional with Huber-like loss function is given by
        
        .. math::
           \begin{eqnarray}
             R(x) &:=& \sum_{\boldsymbol{i}} \sum_{\boldsymbol{j} \in N_{\boldsymbol{i}}} \|\boldsymbol{i} - \boldsymbol{j}\|^{-1} h(x_\boldsymbol{i} - x_\boldsymbol{j}) \\
             h(t) &:=& \begin{cases} \frac{1}{2}t^2, & \text{if } |t| \leq delta \\ \frac{delta^{2 - p}}{p}|t|^p + delta^2\left(\frac{1}{2} - \frac{1}{p}\right), & \text{if } |t| > delta \end{cases} \\
             h'(t) &=& \begin{cases} t, & \text{if } |t| \leq delta \\ delta^{2 - p}sgn(t)|t|^{p-1}, & \text{if } |t| > delta \end{cases}
           \end{eqnarray}

        where :math:`N_{\boldsymbol{i}}` is a neighborhood around the 3D pixel index :math:`\boldsymbol{i} = (i_1, i_2, i_3)`.
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            Df (C contiguous float32 numpy array): the gradient of the TV functional applied to the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TV_gradient.restype = ctypes.c_bool
        
        if has_torch == True and type(f) is torch.Tensor:
            Df = f.clone()
            self.set_model()
            self.libprojectors.TV_gradient.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.TV_gradient(f.data_ptr(), Df.data_ptr(), N_1, N_2, N_3, delta, beta, p, f.is_cuda == False)
            return Df
        else:
            Df = np.ascontiguousarray(np.zeros(f.shape,dtype=np.float32), dtype=np.float32)
            self.set_model()
            self.libprojectors.TV_gradient.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.TV_gradient(f, Df, N_1, N_2, N_3, delta, beta, p, True)
            return Df
    
    def TVquadForm(self, f, d, delta, beta=0.0, p=1.0):
        r"""Calculates the quadratic form of the anisotropic Total Variation (TV) functional of the provided numpy arrays
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size
        This function calculates the following inner product <d, R''(f)d>, where R'' is the Hessian of the TV functional
        The quadraitc surrogate is used here, so this function can be used to calculate the step size of a cost function
        that includes a TV regularization term.
        See the same  cost in the diffuse function below for an example of its usage.
        
        This function uses a Huber-like loss function applied to the differences of neighboring samples (in 3D).
        One can switch between using 6 or 26 neighbors using the \"set_numTVneighbors\" function.
        The aTV functional with Huber-like loss function is given by
        
        .. math::
           \begin{eqnarray}
             R(x) &:=& \sum_{\boldsymbol{i}} \sum_{\boldsymbol{j} \in N_{\boldsymbol{i}}} \|\boldsymbol{i} - \boldsymbol{j}\|^{-1} h(x_\boldsymbol{i} - x_\boldsymbol{j}) \\
             h(t) &:=& \begin{cases} \frac{1}{2}t^2, & \text{if } |t| \leq delta \\ \frac{delta^{2 - p}}{p}|t|^p + delta^2\left(\frac{1}{2} - \frac{1}{p}\right), & \text{if } |t| > delta \end{cases}
           \end{eqnarray}

        where :math:`N_{\boldsymbol{i}}` is a neighborhood around the 3D pixel index :math:`\boldsymbol{i} = (i_1, i_2, i_3)`.
        To make this calculate a quadraitc surrogate (upper bound), LEAP uses h'(t)/t instead of h''(t).
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            d (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            Df (C contiguous float32 numpy array): the gradient of the TV functional applied to the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TV_quadForm.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TV_quadForm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TV_quadForm(f.data_ptr(), d.data_ptr(), N_1, N_2, N_3, delta, beta, p, f.is_cuda == False)
        else:
            self.libprojectors.TV_quadForm.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TV_quadForm(f, d, N_1, N_2, N_3, delta, beta, p, True)
        
    def diffuse(self, f, delta, numIter, p=1.0):
        r"""Performs anisotropic Total Variation (TV) smoothing to the provided 3D numpy array
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size.
        This function performs a specifies number of iterations of minimizing the aTV functional using gradient descent.
        The step size calculation uses the method of Separable Quadratic Surrogate (see also TVquadForm).
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            numIter (int): number of iterations
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            f, the same array as the input denoised
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        self.libprojectors.diffuse.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.diffuse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            self.libprojectors.diffuse(f.data_ptr(), N_1, N_2, N_3, delta, p, numIter, f.is_cuda == False)
        else:
            self.libprojectors.diffuse.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            self.libprojectors.diffuse(f, N_1, N_2, N_3, delta, p, numIter, True)
        return f
        ''' Here is equivalent code to run this algorithm using the TV functions above
        for n in range(N):
            d = self.TVgradient(f, delta, p)
            num = np.sum(d**2)
            denom = self.TVquadForm(f, d, delta, p)
            if denom <= 1.0e-16:
                break
            stepSize = num / denom
            f -= stepSize * d
        return f
        '''
        
    def TV_denoise(self, f, delta, beta, numIter, p=1.0, meanOverFirstDim=False):
        r"""Performs anisotropic Total Variation (TV) denoising to the provided 3D numpy array
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size.
        This function performs a specifies number of iterations of minimizing the sum of an L2 loss and aTV functional using gradient descent.
        The step size calculation uses the method of Separable Quadratic Surrogate (see also TVquadForm).
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
            numIter (int): number of iterations
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            f, the same array as the input denoised
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        self.libprojectors.TV_denoise.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TV_denoise.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.TV_denoise(f.data_ptr(), N_1, N_2, N_3, delta, beta, p, numIter, meanOverFirstDim, f.is_cuda == False)
        else:
            self.libprojectors.TV_denoise.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.TV_denoise(f, N_1, N_2, N_3, delta, beta, p, numIter, meanOverFirstDim, True)
        return f
    
    def relative_differences_grad(self, f, delta, beta):
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        Df = np.zeros(f.shape,dtype=np.float32)
        self.libprojectors.relativeDifferences_gradient.restype = ctypes.c_bool
        self.set_model()
        self.libprojectors.relativeDifferences_gradient.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
        self.libprojectors.relativeDifferences_gradient(f, Df, N_1, N_2, N_3, delta, beta)
        return Df
        
    def gaussian_filter(self, f, FWHM=2.0, numDims=3):
    
        if isinstance(f, list):
            for n in range(len(f)):
                self.gaussian_filter(f[n], FWHM, numDims)
            return
        if len(f.shape) == 4:
            N_1 = f.shape[0]*f.shape[1]
            N_2 = f.shape[2]
            N_3 = f.shape[3]
            numDims = 2
        elif len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
            
        self.libprojectors.gaussian_filter.restype = ctypes.c_bool
        self.set_model()
        self.libprojectors.gaussian_filter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
        self.libprojectors.gaussian_filter(f, N_1, N_2, N_3, FWHM, numDims)
        