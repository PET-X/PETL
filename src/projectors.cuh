/*
How to handle 4D planogram data

FORWARD PROJECTION
cast threads over v1, v0, u1 and for loop over u0

BACKPROJECTION
cast threads over x,y,z
copy all data to GPU
for iv1
	copy v1 planogram into texture
	perform backprojection
//*/

#ifndef __PROJECTORS_SF_H
#define __PROJECTORS_SF_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool setConstantMemory(parameters* params);
bool project_SF(float* g, float* f, parameters* params);
bool backproject_SF(float* g, float* f, parameters* params);


#endif
