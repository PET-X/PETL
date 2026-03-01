#ifndef __ANALYTIC_RAY_TRACING_H
#define __ANALYTIC_RAY_TRACING_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"
#include "phantom.h"

/**
 * This class provides CPU-based implementations (accelerated by OpenMP) to perform analytic ray tracing simulation through geometric solids.
 */

class analyticRayTracing
{
public:

    // Constructor and destructor; these do nothing
    analyticRayTracing();
    ~analyticRayTracing();

    bool rayTrace(float** g, parameters* params_in, phantom* aPhantom, int oversampling = 1);

private:

    bool setSourcePosition(int iplan, int iu1, int iu0, double* sourcePos, double du1 = 0.0, double du0 = 0.0);
    bool setTrajectory(int iplan, int iv1, int iv0, double* r);
    
    parameters* params;
};

#endif
