#include <omp.h>
#include "petl.h"
#include "binning.h"
#include "data_cube.h"
#include "projectors.cuh"
#include "ramp_filter.cuh"
#include "analytic_ray_tracing.h"
#include "analytic_ray_tracing_gpu.cuh"
#include "pfdr.cuh"
#include "median_filter.cuh"

PETL::PETL()
{

}

PETL::~PETL()
{
    clearAll();
}

void PETL::clearAll()
{
    params.clearAll();
}

bool PETL::add_planogram(float psi, float R, float L, float H, float v_m0, float v_m1, float T)
{
    return params.add_planogram(psi, R, L, H, v_m0, v_m1, T);
}

bool PETL::remove_planogram(int which)
{
    return params.remove_planogram(which);
}

bool PETL::keep_only_planogram(int which)
{
    return params.keep_only_planogram(which);
}

bool PETL::set_default_volume(float scale)
{
    return params.set_default_volume(scale);
}

bool PETL::set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    return params.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool PETL::bin(float** g, char* file_name)
{
    if (g == nullptr || file_name == nullptr || file_name[0] == 0 || !params.geometryDefined())
        return false;
    else
    {
        for (int i = 0; i < params.num_planograms(); i++)
        {
            params.planogramSet[i]->set_data(g[i]);
        }
        binning binner;
        binner.init(&params);
        return binner.binAcqdata(file_name);
    }
}

bool PETL::ray_trace(float** g, int oversampling)
{
    if (g == nullptr || !params.geometryDefined())
        return false;
    if (params.whichGPU < 0)
    {
        analyticRayTracing rayTracingRoutines;
        return rayTracingRoutines.rayTrace(g, &params, &geometricPhantom, oversampling);
    }
    else
    {
        omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
        #pragma omp parallel for schedule(dynamic)
        for (int iplan = 0; iplan < params.num_planograms(); iplan++)
        {
            //printf("projecting planogram %d...\n", iplan);
            parameters params_i = params;
            params_i.whichGPU = params.whichGPUs[omp_get_thread_num()];
            params_i.whichGPUs.clear();
            params_i.keep_only_planogram(iplan);
            //params_i.printAll();
            float* g_i = g[iplan];
            rayTrace_gpu(g_i, &params_i, &geometricPhantom, true, oversampling);
        }
        return true;
        //return rayTrace_gpu(float* g, parameters * params, phantom * aPhantom, bool data_on_cpu, int oversampling = 1);
    }
}

bool PETL::stopping_power(float** g, int oversampling)
{
    if (g == nullptr || !params.geometryDefined())
        return false;
    if (params.whichGPU < 0)
    {
        //analyticRayTracing rayTracingRoutines;
        //return rayTracingRoutines.rayTrace(g, &params, &modules, oversampling);
        printf("Error: CPU version not yet implemented\n");
        return false;
    }
    else
    {
        if (modules.numObjects() > 0)
        {
            omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
            #pragma omp parallel for schedule(dynamic)
            for (int iplan = 0; iplan < params.num_planograms(); iplan++)
            {
                //printf("projecting planogram %d...\n", iplan);
                parameters params_i = params;
                params_i.whichGPU = params.whichGPUs[omp_get_thread_num()];
                params_i.whichGPUs.clear();
                params_i.keep_only_planogram(iplan);
                //params_i.printAll();
                float* g_i = g[iplan];
                stoppingPower_gpu(g_i, &params_i, &modules, true, oversampling);
            }
        }
        else
        {
            //printf("Model not set\n");
            for (int iplan = 0; iplan < params.num_planograms(); iplan++)
            {
                planogram* plan = params.planogramSet[iplan];
                plan->set_data(g[iplan]);
                plan->set_constant(1.0);
            }
        }
        return true;
        //return rayTrace_gpu(float* g, parameters * params, phantom * aPhantom, bool data_on_cpu, int oversampling = 1);
    }
}

bool PETL::set_solid_angle_correction(float** g, bool do_inverse)
{
    for (int iplan = 0; iplan < params.num_planograms(); iplan++)
    {
        params.planogramSet[iplan]->set_data(g[iplan]);
        params.planogramSet[iplan]->apply_solid_angle_correction(do_inverse);
    }
    return true;
}

bool PETL::apply_corrections(float** g, float** r, float threshold)
{
    if (g == nullptr || r == nullptr || params.num_planograms() <= 0)
        return false;

    for (int i = 0; i < params.num_planograms(); i++)
    {
        planogram* plan_i = params.planogramSet[i];
        dataCube G(g[i], plan_i->N_v1, plan_i->N_v0, plan_i->N_u1, plan_i->N_u0);
        dataCube R(r[i], plan_i->N_v1, plan_i->N_v0, plan_i->N_u1, plan_i->N_u0);
        G.divide(&R);
    }

    omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
    #pragma omp parallel for schedule(dynamic)
    for (int iplan = 0; iplan < params.num_planograms(); iplan++)
    {
        //printf("projecting planogram %d...\n", iplan);
        parameters params_i = params;
        params_i.whichGPU = params.whichGPUs[omp_get_thread_num()];
        params_i.whichGPUs.clear();
        params_i.keep_only_planogram(iplan);
        //params_i.printAll();
        float* g_i = g[iplan];
        float* r_i = r[iplan];
        bad_pixel_correction(g_i, r_i, &params_i, threshold);
    }

    return true;
}

bool PETL::simulate_scatter(float** g, float* mu)
{
    if (g == nullptr || mu == nullptr || !params.allDefined())
        return false;
    else
    {
        return true;
    }
}

bool PETL::project(float** g, float* f)
{
    if (g == nullptr || f == nullptr || !params.allDefined())
        return false;

    omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
	#pragma omp parallel for schedule(dynamic)
    for (int iplan = 0; iplan < params.num_planograms(); iplan++)
    {
        //printf("projecting planogram %d...\n", iplan);
        parameters params_i = params;
        params_i.whichGPU = params.whichGPUs[omp_get_thread_num()];
        params_i.whichGPUs.clear();
        params_i.keep_only_planogram(iplan);
        //params_i.printAll();
        float* g_i = g[iplan];
        project_SF(g_i, f, &params_i);
    }
    return true;
}

bool PETL::backproject(float** g, float* f, parameters* params_local, bool FBPweight)
{
    if (params_local == nullptr)
        params_local = &params;
    if (g == nullptr || f == nullptr || !params_local->allDefined())
        return false;

    float** f_stack = (float**) malloc(sizeof(float*)*size_t(params_local->planogramSet.size()));
    
    omp_set_num_threads(std::min(int(params_local->whichGPUs.size()), omp_get_num_procs()));
	#pragma omp parallel for schedule(dynamic)
    for (int iplan = 0; iplan < int(params_local->planogramSet.size()); iplan++)
    {
        parameters params_i;
        params_i.assign(*params_local);
        params_i.whichGPU = params_local->whichGPUs[omp_get_thread_num()];
        params_i.whichGPUs.clear();
        params_i.keep_only_planogram(iplan);
        float* g_i = g[iplan];
        float* f_i = nullptr;
        if (iplan == 0)
            f_i = f;
        else
            f_i = (float*) calloc(params_local->volumeData_numberOfElements(), sizeof(float));
        f_stack[iplan] = f_i;

        backproject_SF(g_i, f_i, &params_i);
        //*
        if (FBPweight)
        {
            dataCube X(f_i, params_local->numZ, params_local->numY, params_local->numX);
            //X.scale(params_i.planogramSet[0]->T_v0 / (2.0 * PI * params_local->voxelWidth));
            X.scale(params_i.planogramSet[0]->T_v0 * params_i.planogramSet[0]->T_u0 / params_local->voxelWidth); // correct
        }
        //*/
    }

    dataCube primary(f, params_local->numZ, params_local->numY, params_local->numX);
    for (int iplan = 1; iplan < int(params_local->planogramSet.size()); iplan++)
    {
        if (f_stack[iplan] != nullptr)
        {
            dataCube temp(f_stack[iplan], params_local->numZ, params_local->numY, params_local->numX);
            primary.add(&temp);
            free(f_stack[iplan]);
            f_stack[iplan] = nullptr;
        }
    }
    free(f_stack);

    return true;
}

float** PETL::malloc_rebinned_data(parameters* params_local)
{
    if (params_local == nullptr)
        params_local = &params;
    float** g_reb = (float**)malloc(sizeof(float*) * params_local->num_planograms());
    for (int i = 0; i < params_local->num_planograms(); i++)
    {
        planogram* plan = params_local->planogramSet[i];
        uint64 reb_sz = uint64(plan->N_v0) * uint64(plan->N_u1) * uint64(plan->N_u0);
        g_reb[i] = (float*)malloc(sizeof(float) * reb_sz);
    }
    return g_reb;
}

bool PETL::free_rebinned_data(float** g_reb, parameters* params_local)
{
    if (g_reb == nullptr)
        return false;
    if (params_local == nullptr)
        params_local = &params;

    for (int i = 0; i < params_local->num_planograms(); i++)
    {
        if (g_reb[i] != nullptr)
            free(g_reb[i]);
        g_reb[i] = nullptr;
    }
    free(g_reb);
    return true;
}

bool PETL::doFBP(float** g, float* f, parameters* params_local)
{
    if (params_local == nullptr)
        params_local = &params;
    if (g == nullptr || f == nullptr || !params_local->allDefined())
        return false;
    else
    {
        if (params_local->planogramSet[0]->N_v1 > 1)
        {
            parameters params_reb;
            params_reb.assign(*params_local);

            printf("PFDR...\n");
            float** g_reb = malloc_rebinned_data(params_local);
            doPFDR(g, g_reb, &params_reb);
            bool retVal = doFBP(g_reb, f, &params_reb);
            free_rebinned_data(g_reb, &params_reb);
            return retVal;
        }
        else
        {
            float** g_reb = g;

            //* Ramp Filter
            omp_set_num_threads(std::min(int(params_local->whichGPUs.size()), omp_get_num_procs()));
            #pragma omp parallel for schedule(dynamic)
            for (int iplan = 0; iplan < int(params_local->planogramSet.size()); iplan++)
            {
                parameters params_i;
                params_i.assign(*params_local);
                params_i.whichGPU = params_local->whichGPUs[omp_get_thread_num()];
                params_i.whichGPUs.clear();
                params_i.keep_only_planogram(iplan);
                float* g_reb_i = g_reb[iplan];

                rampFilter1D(g_reb_i, &params_i, true);
                params_i.planogramSet[0]->set_data(g_reb_i);
                params_i.planogramSet[0]->apply_planogram_weight(true);
            }
            //*/

            return backproject(g_reb, f, params_local, true);
        }
    }
}

bool PETL::doPFDR(float** g, float** g_reb, parameters* params_reb)
{
    if (params_reb == nullptr)
        params_reb = &params;
    if (g == nullptr || g_reb == nullptr || !params_reb->geometryDefined())
        return false;
    else
    {
        omp_set_num_threads(std::min(int(params_reb->whichGPUs.size()), omp_get_num_procs()));
        #pragma omp parallel for schedule(dynamic)
        for (int iplan = 0; iplan < int(params_reb->planogramSet.size()); iplan++)
        {
            //parameters params_i = *params_reb;
            parameters params_i;
            params_i.assign(*params_reb);
            params_i.whichGPU = params_reb->whichGPUs[omp_get_thread_num()];
            params_i.whichGPUs.clear();
            params_i.keep_only_planogram(iplan);
            float* g_i = g[iplan];
            float* g_reb_i = g_reb[iplan];

            PFDR(g_i, g_reb_i, &params_i);
            params_reb->planogramSet[iplan]->N_v1 = 1;
            params_reb->planogramSet[iplan]->v1_0 = 0.0;
        }

        return true;
    }
}
