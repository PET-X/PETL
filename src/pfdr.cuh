#ifndef __PFDR_H
#define __PFDR_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool setConstantMemory_pfdr(parameters* params);
bool PFDR(float* g, float* g_reb, parameters* params);


#endif
