////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#include "analytic_ray_tracing.h"
#include "petl_defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>

using namespace std;

analyticRayTracing::analyticRayTracing()
{
    params = NULL;
}

analyticRayTracing::~analyticRayTracing()
{
}

bool analyticRayTracing::rayTrace(float** g, parameters* params_in, phantom* aPhantom, int oversampling)
{
    params = params_in;
    if (g == NULL || params == NULL || aPhantom == NULL)
        return false;
    
    oversampling = max(1, min(oversampling, 11));
    if (oversampling % 2 == 0)
        oversampling += 1;
    oversampling = max(1, min(oversampling, 11));
    int os_radius = (oversampling - 1) / 2;

    int num_threads = omp_get_num_procs();
    aPhantom->makeTempData(num_threads);

    for (int iplan = 0; iplan < int(params->planogramSet.size()); iplan++)
    {
        planogram* plan = params->planogramSet[iplan];
        float* p = g[iplan];

        double T_u1_os = double(plan->T_u1) / double(oversampling + 1);
        double T_u0_os = double(plan->T_u0) / double(oversampling + 1);

        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic)
        for (int iv1 = 0; iv1 < plan->N_v1; iv1++)
        {
            float v1 = plan->v1(iv1);
            double sourcePos[3];
            double r[3];
            float* p_v1 = &p[uint64(iv1) * uint64(plan->N_v0 * plan->N_u1 * plan->N_u0)];

            for (int iv0 = 0; iv0 < plan->N_v0; iv0++)
            {
                float v0 = plan->v0(iv0);
                setTrajectory(iplan, iv1, iv0, r);
                float* p_v1_v0 = &p_v1[uint64(iv0) * uint64(plan->N_u1 * plan->N_u0)];

                for (int iu1 = 0; iu1 < plan->N_u1; iu1++)
                {
                    float u1 = plan->u1(iu1);

                    float* p_v1_v0_u1 = &p_v1_v0[iu1 * plan->N_u0];
                    for (int iu0 = 0; iu0 < plan->N_u0; iu0++)
                    {
                        float u0 = plan->u0(iu0);
                        if (fabs(u1) - 0.5 * plan->T_u1 > plan->H - plan->R * fabs(v1) || fabs(u0) - 0.5 * plan->T_u0 > plan->L + plan->R * fabs(v0))
                        {
                            p_v1_v0_u1[iu0] = 0.0;
                        }
                        else
                        {
                            if (oversampling == 1)
                            {
                                setSourcePosition(iplan, iu1, iu0, sourcePos);
                                p_v1_v0_u1[iu0] = float(aPhantom->lineIntegral(sourcePos, r));
                            }
                            else
                            {
                                float accum = 0.0;
                                for (int ios1 = -os_radius; ios1 <= os_radius; ios1++)
                                {
                                    double du1 = ios1 * T_u1_os;
                                    for (int ios0 = -os_radius; ios0 <= os_radius; ios0++)
                                    {
                                        double du0 = ios0 * T_u0_os;
                                        setSourcePosition(iplan, iu1, iu0, sourcePos, du1, du0);
                                        accum += float(aPhantom->lineIntegral(sourcePos, r));
                                    }
                                }
                                p_v1_v0_u1[iu0] = accum / float(oversampling * oversampling);
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

bool analyticRayTracing::setSourcePosition(int iplan, int iu1, int iu0, double* sourcePos, double du1, double du0)
{
    if (sourcePos == NULL)
        return false;

    planogram* plan = params->planogramSet[iplan];
    sourcePos[0] = (plan->u0(iu0) + du0) * cos(plan->psi);
    sourcePos[1] = (plan->u0(iu0) + du0) * sin(plan->psi);
    sourcePos[2] = plan->u1(iu1) + du1;
    
    return true;
}

bool analyticRayTracing::setTrajectory(int iplan, int iv1, int iv0, double* r)
{
    if (r == NULL)
        return false;

    planogram* plan = params->planogramSet[iplan];
    float cos_psi = cos(plan->psi);
    float sin_psi = sin(plan->psi);
    r[0] = -(plan->v0(iv0)*cos_psi + sin_psi);
    r[1] = -(plan->v0(iv0)*sin_psi - cos_psi);
    r[2] = -plan->v1(iv1);

    double mag = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    r[0] = r[0] / mag;
    r[1] = r[1] / mag;
    r[2] = r[2] / mag;

    return true;
}
