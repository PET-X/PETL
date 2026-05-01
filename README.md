# Planogram Emission Tomography Library (PETL)
This is a C++/CUDA library (Linux or Windows) with Python bindings of 3D Positron Emission Tomography (PET) reconstruction algorithms using the planogram data format.

## Installation

To install PETL just type the following in the root directory
```
pip install .
```

## Dependencies
[cmake version 3.23 or newer](https://cmake.org/download/)

[CUDA toolkit 11.7 or newer](https://developer.nvidia.com/cuda-downloads) (See section below about compiling without a GPU or on Mac)

Linux: gcc compiler

Windows: [Visual Studio 2019](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads)
(be sure to check the box that says "Desktop development with C++")

Python version 3.6 or newer


## Documentation and Usage

Documentation is available [here](https://petl.readthedocs.io/)

Demo scripts for most functionality in the [demo](https://github.com/PET-X/PETL/tree/main/demo) directory

## Authors
Kyle Champley (champley@gmail.com)

## License
PETL is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.
See [LICENSE](LICENSE) for more details.  
SPDX-License-Identifier: MIT  

Please cite our work by referencing this github page and citing our [article](https://iopscience.iop.org/article/10.1088/0266-5611/26/4/045008/meta):

Champley, Kyle M., Raymond R. Raylman, and Paul E. Kinahan. "Advancements to the planogram frequency–distance rebinning algorithm." Inverse problems 26, no. 4 (2010): 045008.

