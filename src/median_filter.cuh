#ifndef __MEDIAN_FILTER_H
#define __MEDIAN_FILTER_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool bad_pixel_correction(float* g, float* r, parameters* params, float threshold);

#endif
