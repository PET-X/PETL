#ifndef __RELATIVE_DIFFERENCES_H
#define __RELATIVE_DIFFERENCES_H

#ifdef WIN32
#pragma once
#endif

//#include "device_launch_parameters.h"

// calculate aTV cost with relative differences loss function
float aTV_RelativeDifferencesLoss_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, int whichGPU = 0);

// calculate aTV gradient with relative differences loss function
bool aTV_RelativeDifferencesLoss_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta,
                                          int whichGPU = 0);

// calculate aTV quadratic form with relative differences loss function
float aTV_RelativeDifferencesLoss_quadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta,
                                           int whichGPU = 0);

// calculate aTV diagonal Hessian elements with relative differences loss function
bool aTV_RelativeDifferencesLoss_curvature(float* f, float* DDf, int N_1, int N_2, int N_3, float delta, float beta,
                                           int whichGPU = 0);

bool GaussianFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims = 3, int whichGPU = 0);
//dim3 setBlockSize(int4);
//dim3 setBlockSize(int3);

#endif
