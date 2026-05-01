#ifndef __PETL_H
#define __PETL_H

#ifdef WIN32
#pragma once
#endif

#define PETL_VERSION "1.0"

#include "petl_defines.h"
#include "parameters.h"
#include "phantom.h"
#include <vector>

class PETL
{
public:
	PETL();
	~PETL();

    void clearAll();
    const char* about();

    bool add_planogram(float psi, float R, float L, float H, float v_m0, float v_m1, float T);
	bool remove_planogram(int);
	bool keep_only_planogram(int);

	bool set_default_volume(float scale = 1.0);
    bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX = 0.0, float offsetY = 0.0, float offsetZ = 0.0);

    bool bin(float** g, char* file_name);
    bool ray_trace(float** g, int oversampling = 1);
    bool stopping_power(float** g, int oversampling = 1);
    bool set_solid_angle_correction(float** g, bool do_inverse = false);
    bool apply_corrections(float** g, float** r, float threshold = 0.5);

    bool simulate_scatter(float** g, float* mu);

    bool project(float** g, float* f);
    bool backproject(float** g, float* f, parameters* params_FBP = nullptr, bool FBPweight = false);
    bool doFBP(float** g, float* f, parameters* params_local = nullptr);
    bool doPFDR(float** g, float** g_out, parameters* params_reb = nullptr);

    float** malloc_rebinned_data(parameters* params_local = nullptr);
    bool free_rebinned_data(float** g_reb, parameters* params_local = nullptr);

    parameters params;
    phantom geometricPhantom;
    phantom modules;
};

#endif
