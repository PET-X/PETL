#ifndef __RAMP_FILTER_H
#define __RAMP_FILTER_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

#include <cufft.h>
float* rampFilterFrequencyResponseMagnitude(int N, parameters* params);

float* rampImpulseResponse(int N, parameters* params);

bool rampFilter1D(float*& g, parameters* params, bool data_on_cpu, float scalar = 1.0);

#endif
