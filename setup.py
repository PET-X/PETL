################################################################################
# Copyright 2026 PET/X and Kyle Champey 
# PETL project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Planogram Emission Tomography Library (PETL)
################################################################################
import os
import pathlib

from setuptools import setup, find_packages
from setuptools.command.install import install

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    lib_fname = 'build/lib/libpetl.so'
    retVal = os.system(r'sh ./etc/build.sh')
    if retVal != 0:
        print('Failed to compile!')
        quit()
    
elif _platform == "win32":
    lib_fname = r'win_build\bin\Release\libpetl.dll'
    retVal = os.system(r'.\etc\win_build_agn.bat')
    if retVal != 0:
        print('Failed to compile!')
        quit()
    
    import site
    copy_text = 'copy ' + str(lib_fname) + ' ' + str(os.path.join(site.getsitepackages()[1], 'libpetl.dll'))
    os.system(copy_text)
    

setup(
    name='petl',
    version='1.0', 
    author='Kyle Champley', 
    author_email='champley@gmail.com', 
    description='Planogram Emission Tomography Library (PETL)', 
    keywords='Positron Emission Tomography, PET, planogram, PFDR', 
    python_requires='>=3.6', 
    packages=find_packages("src"), 
    package_dir={'': 'src'},
    install_requires=['numpy'], 
    py_modules=['petl_server', 'petx_server'], 
    package_data={'': [lib_fname]},
)
