#include "cuda_runtime.h"
#include "projectors.cuh"
#include "cuda_utils.cuh"

__constant__ float d_cos_psi;
__constant__ float d_sin_psi;
__constant__ float d_R;
__constant__ float d_L;
__constant__ float d_H;

__global__ void project_planogram(float* __restrict__ g, const int4 N_g, const float4 T_g, const float4 startVal_g, cudaTextureObject_t f, const int4 N_f, const float4 T_f, const float4 startVal_f, const int iv1)
{
    const int iv0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int iu1 = threadIdx.y + blockIdx.y * blockDim.y;
    const int iu0 = threadIdx.z + blockIdx.z * blockDim.z;
    if (iv0 >= N_g.x || iu1 >= N_g.y || iu0 >= N_g.z)
        return;

    const float v1 = iv1*T_g.w + startVal_g.w;
    const float v0 = iv0*T_g.x + startVal_g.x;
    const float u1 = iu1*T_g.y + startVal_g.y;
    const float u0 = iu0*T_g.z + startVal_g.z;

    uint64 g_ind = uint64(iv0)*uint64(N_g.y*N_g.z) + uint64(iu1*N_g.z + iu0);

    //if (iv0 == 0 && iu1 == 0 && iu0 == 0)
    //    printf("H = %f, R = %f, L = %f\n", d_H, d_R, d_L);

    //*
    if (fabsf(u0)-0.5f*T_g.z > d_L + d_R * fabsf(v0))
    {
        // does not intersect volume
        g[g_ind] = 0.0f;
		return;
    }
    //*/
    //*
    if (fabsf(u1)-0.5f*T_g.y > d_H - d_R * fabsf(v1))
    {
        // does not intersect volume
        g[g_ind] = 0.0f;
		return;
    }
    //*/

    const float iu0_min = iu0 - 0.5f;
    const float iu0_max = iu0 + 0.5f;
    const float iu1_min = iu1 - 0.5f;
    const float iu1_max = iu1 + 0.5f;

    // p + t*r
    const float3 p = make_float3(u0*d_cos_psi, u0*d_sin_psi, u1);
    const float3 r = make_float3(-(v0*d_cos_psi+d_sin_psi), -(v0*d_sin_psi-d_cos_psi), -v1);

    const float Tx_inv = 1.0f / T_f.x;
    const float Ty_inv = 1.0f / T_f.y;
    const float Tz_inv = 1.0f / T_f.z;

    const float Tu1_inv = 1.0f / T_g.y;
    const float Tu0_inv = 1.0f / T_g.z;

    bool do_print = false;
    if (fabsf(v0) < 1.0e-5 && fabsf(v1) < 1.0e-5)
    {
        do_print = true;
        //printf("v1[%d] = %f, v0[%d] = %f\n", iv1, v1, iv0, v0);
    }

    const float Tx_over_Tu0 = T_f.x * Tu0_inv;

    const float width_z = 0.5f * T_g.y;
    const float Tz_over_Tu1 = T_f.z * Tu1_inv;
    const float z_footprint_width = 0.5f * T_f.z * Tu1_inv;

    float val = 0.0f;
    //*
    if (fabsf(r.x) >= fabsf(r.y))
    {
        const float width_y = 0.5f * T_g.z * fabsf(d_sin_psi);
        const float sin_psi_inv = 1.0f / d_sin_psi;
        const float rx_inv = 1.0f / r.x;
        const float l_phi = T_f.x* fabsf(sin_psi_inv);

        const float y_footprint_width = 0.5f * T_f.y * fabs(sin_psi_inv) * Tu0_inv;
        for (int ix = 0; ix < N_f.x; ix++)
        {
            const float x = ix*T_f.x + startVal_f.x;
            const float t = (x-p.x) * rx_inv;

            const float y_c = p.y + t*r.y;
            const float z_c = p.z + t*r.z;

            const float y_A = y_c - width_y;
            const float y_B = y_c + width_y;
            const int iy_min = max(0, int(ceilf((y_A - 0.5f*T_f.y - startVal_f.y) * Ty_inv)));
            const int iy_max = min(N_f.y-1, int(floorf((y_B + 0.5f*T_f.y - startVal_f.y) * Ty_inv)));

            const float z_A = z_c - width_z;
            const float z_B = z_c + width_z;
            const int iz_min = max(0, int(ceilf((z_A - 0.5f*T_f.z - startVal_f.z) * Tz_inv)));
            const int iz_max = min(N_f.z-1, int(floorf((z_B + 0.5f*T_f.z - startVal_f.z) * Tz_inv)));

            float y = iy_min * T_f.y + startVal_f.y - t * r.y;
            const float z_start = (iz_min * T_f.z + startVal_f.z - t * r.z - startVal_g.y) * Tu1_inv;
            for (int iy = iy_min; iy <= iy_max; iy++)
            {
                //const float u0Footprint_lo = ((y - 0.5f * T_f.y) * sin_psi_inv - startVal_g.z) * Tu0_inv;
                //const float u0Footprint_hi = ((y + 0.5f * T_f.y) * sin_psi_inv - startVal_g.z) * Tu0_inv;

                const float u0Footprint_lo = (y * sin_psi_inv - startVal_g.z) * Tu0_inv - y_footprint_width;
                const float u0Footprint_hi = (y * sin_psi_inv - startVal_g.z) * Tu0_inv + y_footprint_width;

                const float u0Footprint = fminf(iu0_max, u0Footprint_hi) - fmaxf(iu0_min, u0Footprint_lo);
                if (u0Footprint > 0.0f)
                {
                    float z = z_start;
                    for (int iz = iz_min; iz <= iz_max; iz++)
                    {
                        const float u1Footprint = fmaxf(0.0f, fminf(iu1_max, z + z_footprint_width) - fmaxf(iu1_min, z - z_footprint_width));

                        if (u1Footprint > 0.0f)
                            val += tex3D<float>(f, ix + 0.5f, iy + 0.5f, iz + 0.5f) * u0Footprint * u1Footprint;
                        z += Tz_over_Tu1;
                    }
                }
                y += T_f.y;
            }
        }
        val *= l_phi;
    }
    else
    {
        const float width_x = 0.5f * T_g.z * fabsf(d_cos_psi);
        const float cos_psi_inv = 1.0f / d_cos_psi;
        const float ry_inv = 1.0f / r.y;
        const float l_phi = T_f.x * fabsf(cos_psi_inv);

        const float x_footprint_width = 0.5f * T_f.x * fabs(cos_psi_inv) * Tu0_inv;

        for (int iy = 0; iy < N_f.y; iy++)
        {
            const float y = iy*T_f.y + startVal_f.y;
            const float t = (y-p.y) * ry_inv;

            const float x_c = p.x + t * r.x;
            const float z_c = p.z + t * r.z;

            const float x_A = x_c - width_x;
            const float x_B = x_c + width_x;
            const int ix_min = max(0, int(ceilf((x_A - 0.5f * T_f.x - startVal_f.x) * Tx_inv)));
            const int ix_max = min(N_f.x-1, int(floorf((x_B + 0.5f * T_f.x - startVal_f.x) * Tx_inv)));

            const float z_A = z_c - width_z;
            const float z_B = z_c + width_z;
            const int iz_min = max(0, int(ceilf((z_A - 0.5f * T_f.z - startVal_f.z) * Tz_inv)));
            const int iz_max = min(N_f.z-1, int(floorf((z_B + 0.5f * T_f.z - startVal_f.z) * Tz_inv)));

            float x = ix_min * T_f.x + startVal_f.x - t * r.x;
            const float z_start = (iz_min * T_f.z + startVal_f.z - t * r.z - startVal_g.y) * Tu1_inv;
            for (int ix = ix_min; ix <= ix_max; ix++)
            {
                //const float u0Footprint_lo = ((x - 0.5f * T_f.x) * cos_psi_inv - startVal_g.z) * Tu0_inv;
                //const float u0Footprint_hi = ((x + 0.5f * T_f.x) * cos_psi_inv - startVal_g.z) * Tu0_inv;

                const float u0Footprint_lo = (x * cos_psi_inv - startVal_g.z) * Tu0_inv - x_footprint_width;
                const float u0Footprint_hi = (x * cos_psi_inv - startVal_g.z) * Tu0_inv + x_footprint_width;

                const float u0Footprint = fminf(iu0_max, u0Footprint_hi) - fmaxf(iu0_min, u0Footprint_lo);
                if (u0Footprint > 0.0f)
                {
                    float z = z_start;
                    for (int iz = iz_min; iz <= iz_max; iz++)
                    {
                        const float u1Footprint = fmaxf(0.0f, fminf(iu1_max, z + z_footprint_width) - fmaxf(iu1_min, z - z_footprint_width));

                        if (u1Footprint > 0.0f)
                            val += tex3D<float>(f, ix + 0.5f, iy + 0.5f, iz + 0.5f) * u0Footprint * u1Footprint;
                        z += Tz_over_Tu1;
                    }
                }
                x += T_f.x;
            }
        }
        val *= l_phi;
    }
    //*/
    /*

    const float x_dot_theta_dy = 0.5f * T_f.y * d_sin_psi;
    const float t_dy = 0.5f * T_f.y * d_cos_psi;

    const float x_dot_theta_dx = 0.5f * T_f.x * d_cos_psi;
    const float t_dx = -0.5f * T_f.x * d_sin_psi;

    if (fabsf(r.x) >= fabsf(r.y))
    {
        const float rx_inv = 1.0f / r.x;
        for (int ix = 0; ix < N_f.x; ix++)
        {
            const float x = ix * T_f.x + startVal_f.x;
            const float t = (x - p.x) * rx_inv;

            const float y_c = p.y + t * r.y;
            const float z_c = p.z + t * r.z;

            // where does the ray hit the detector?
            float u_A = x*d_cos_psi + y_c*d_sin_psi 

            //const int diy = fmaxf(1, int(ceil(0.5f * T_g.z / (T_f.y * fabsf(cos_phi)))));
            //const int iy_c = int(0.5f + (y_c - startVals_f.y) * T_x_inv);


            const float iy = (y_c - startVal_f.y) * Ty_inv;
            const float iz = (z_c - startVal_f.z) * Tz_inv;

            val += tex3D<float>(f, ix + 0.5f, iy + 0.5f, iz + 0.5f);
        }
    }
    else
    {
        const float ry_inv = 1.0f / r.y;
        for (int iy = 0; iy < N_f.y; iy++)
        {
            const float y = iy * T_f.y + startVal_f.y;
            const float t = (y - p.y) * ry_inv;

            const float x_c = p.x + t * r.x;
            const float z_c = p.z + t * r.z;

            const float ix = (x_c - startVal_f.x) * Tx_inv;
            const float iz = (z_c - startVal_f.z) * Tz_inv;

            val += tex3D<float>(f, ix + 0.5f, iy + 0.5f, iz + 0.5f);
        }
    }
    //*/
    g[g_ind] = val * rsqrtf(1.0f + v0 * v0 + v1 * v1); // same as rsqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
}

__global__ void backproject_planogram(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVal_g, float* __restrict__ f, const int4 N_f, const float4 T_f, const float4 startVal_f, const int iv1)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= N_f.x || iy >= N_f.y || iz >= N_f.z)
        return;

    const float v1 = iv1*T_g.w + startVal_g.w;

    const float x = ix*T_f.x + startVal_f.x;
    const float y = iy*T_f.y + startVal_f.y;
    const float z = iz*T_f.z + startVal_f.z;

    
    /*
    const float t = y * d_cos_psi - x * d_sin_psi;

    const float u1 = z + v1 * t;

    float val = 0.0f;
    if (fabsf(u1)-0.5f*T_g.y <= d_H - d_R * fabsf(v1))
    {
        const float iu1 = (u1 - startVal_g.y) / T_g.y;

        const float x_dot_theta = x*d_cos_psi + y*d_sin_psi;

        const float Tu0_inv = 1.0f / T_g.z;

        for (int iv0 = 0; iv0 < N_g.x; iv0++)
        {
            const float v0 = iv0*T_g.x + startVal_g.x;

            const float u0 = x_dot_theta + v0*t;
            if (fabsf(u0)-0.5f*T_g.z <= d_L + d_R * fabsf(v0))
            {
                const float iu0 = (u0 - startVal_g.z) * Tu0_inv;
                val += tex3D<float>(g, iu0 + 0.5f, iu1 + 0.5f, iv0 + 0.5f);
            }
        }
    }
    //*/
    
    //*
    const float t = y * d_cos_psi - x * d_sin_psi;

    const float u1 = z + v1 * t;

    float val = 0.0f;
    if (fabsf(u1) - 0.5f * T_g.y <= d_H - d_R * fabsf(v1))
    {
        const float x_dot_theta = x * d_cos_psi + y * d_sin_psi;

        const float Tu1_inv = 1.0f / T_g.y;
        const float Tu0_inv = 1.0f / T_g.z;

        const float iu1_mid = (u1 - startVal_g.y) * Tu1_inv;
        const float iu1_A = iu1_mid - 0.5f * T_f.z * Tu1_inv;
        const float iu1_B = iu1_mid + 0.5f * T_f.z * Tu1_inv;
        const int iu1_min = int(ceil(iu1_A - 0.5f));
        const int iu1_max = int(floor(iu1_B + 0.5f));

        const float x_dot_theta_dy = 0.5f * T_f.y * d_sin_psi;
        const float t_dy = 0.5f * T_f.y * d_cos_psi;

        const float x_dot_theta_dx = 0.5f * T_f.x * d_cos_psi;
        const float t_dx = -0.5f * T_f.x * d_sin_psi;

        for (int iv0 = 0; iv0 < N_g.x; iv0++)
        {
            const float v0 = iv0 * T_g.x + startVal_g.x;
            //const float l_phi = T_f.x * rsqrtf(1.0f + v1*v1 + v0*v0);
            //const float l_phi = T_f.x;// *sqrtf(1.0f + v1 * v1)* sqrtf(1.0f + v0 * v0);

            const float width_y = v0 * d_cos_psi + d_sin_psi;
            const float width_x = d_cos_psi - v0 * d_sin_psi;
            //const float l_phi = T_f.x / fmaxf(fabsf(width_y), fabsf(width_x));
            const float l_phi = 1.0f / fmaxf(fabsf(width_y), fabsf(width_x)) * rsqrtf(1.0f + v0 * v0 + v1 * v1);

            float iu0_A, iu0_B;
            if (fabsf(width_y) > fabsf(width_x))
            {
                // primarily traveling in x-direction
                if (width_y >= 0.0f)
                {
                    iu0_A = (x_dot_theta - x_dot_theta_dy + v0 * (t - t_dy) - startVal_g.z) * Tu0_inv;
                    iu0_B = (x_dot_theta + x_dot_theta_dy + v0 * (t + t_dy) - startVal_g.z) * Tu0_inv;
                }
                else
                {
                    iu0_B = (x_dot_theta - x_dot_theta_dy + v0 * (t - t_dy) - startVal_g.z) * Tu0_inv;
                    iu0_A = (x_dot_theta + x_dot_theta_dy + v0 * (t + t_dy) - startVal_g.z) * Tu0_inv;
                }
            }
            else
            {
                // primarily traveling in y-direction
                if (width_x >= 0.0f)
                {
                    iu0_A = (x_dot_theta - x_dot_theta_dx + v0 * (t - t_dx) - startVal_g.z) * Tu0_inv;
                    iu0_B = (x_dot_theta + x_dot_theta_dx + v0 * (t + t_dx) - startVal_g.z) * Tu0_inv;
                }
                else
                {
                    iu0_B = (x_dot_theta - x_dot_theta_dx + v0 * (t - t_dx) - startVal_g.z) * Tu0_inv;
                    iu0_A = (x_dot_theta + x_dot_theta_dx + v0 * (t + t_dx) - startVal_g.z) * Tu0_inv;
                }
            }

            const int iu0_min = int(ceil(iu0_A - 0.5f));
            const int iu0_max = int(floor(iu0_B + 0.5f));
            for (int iu0 = iu0_min; iu0 <= iu0_max; iu0 += 2)
            {
                //const float u0Weight = fmaxf(0.0f, fminf(float(iu0) + 0.5f, iu0_B) - fmaxf(float(iu0) - 0.5f, iu0_A));
                const float u0Weight_2 = fmaxf(0.0f, fminf(float(iu0 + 1) + 0.5f, iu0_B) - fmaxf(float(iu0 + 1) - 0.5f, iu0_A));
                const float u0Weight_sum = u0Weight_2 + fmaxf(0.0f, fminf(float(iu0) + 0.5f, iu0_B) - fmaxf(float(iu0) - 0.5f, iu0_A));

                if (u0Weight_sum > 0.0f)
                {
                    const float u0shift_12 = u0Weight_2 / u0Weight_sum;
                    for (int iu1 = iu1_min; iu1 <= iu1_max; iu1 += 2)
                    {
                        //const float u1Weight = fmaxf(0.0f, fminf(float(iu1) + 0.5f, iu1_B) - fmaxf(float(iu1) - 0.5f, iu1_A));
                        const float u1Weight_2 = fmaxf(0.0f, fminf(float(iu1 + 1) + 0.5f, iu1_B) - fmaxf(float(iu1 + 1) - 0.5f, iu1_A));
                        const float u1Weight_sum = u1Weight_2 + fmaxf(0.0f, fminf(float(iu1) + 0.5f, iu1_B) - fmaxf(float(iu1) - 0.5f, iu1_A));
                        if (u1Weight_sum > 0.0f)
                        {
                            const float u1shift_12 = u1Weight_2 / u1Weight_sum;
                            val += tex3D<float>(g, iu0 + u0shift_12 + 0.5f, iu1 + u1shift_12 + 0.5f, iv0 + 0.5f) * l_phi * u0Weight_sum * u1Weight_sum;
                        }
                    }
                }
            }
        }
    }
    //*/
    f[iz*N_f.y*N_f.x + iy*N_f.x + ix] += val * T_f.x;
}

bool project_SF(float* g, float* f, parameters* params)
{
    if (g == nullptr || f == nullptr || params == nullptr)
        return false;
    if (params->planogramSet.size() != 1)
    {
        printf("usage error: project_SF was written to project one planogram at a time\n");
        return false;
    }

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    //float* dev_f = 0;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    //printf("psi = %f\n", params->planogramSet[0]->psi);
    //printf("N_g = %d, %d, %d, %d, T_g = %f, %f, %f, %f, startVal_g = %f, %f, %f, %f\n", N_g.w, N_g.x, N_g.y, N_g.z, T_g.w, T_g.x, T_g.y, T_g.z, startVal_g.w, startVal_g.x, startVal_g.y, startVal_g.z);

    setConstantMemory(params);

    if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(projections) failed!\n");
        return false;
    }

    //dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    int4 N_f_swap = make_int4(N_f.z, N_f.y, N_f.x, N_f.w);
    cudaTextureObject_t f_data_txt = 0;
    cudaArray* f_data_array = nullptr;
    f_data_array = loadTexture_from_cpu(f_data_txt, f, N_f_swap, false, true);

    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);

    planogram* plan = params->planogramSet[0];
    uint64 plan_sz = uint64(plan->N_v0) * uint64(plan->N_u1) * uint64(plan->N_u0);
    for (int iv1 = 0; iv1 < plan->N_v1; iv1++)
    {
        project_planogram <<< dimGrid, dimBlock >>> (&dev_g[uint64(iv1)*plan_sz], N_g, T_g, startVal_g, f_data_txt, N_f, T_f, startVal_f, iv1);
    }
    cudaStatus = cudaDeviceSynchronize();

    pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    
    // Clean Up
    //cudaFree(dev_f);
    cudaFreeArray(f_data_array);
    cudaDestroyTextureObject(f_data_txt);
    cudaFree(dev_g);

    return true;
}

bool backproject_SF(float* g, float* f, parameters* params)
{
    if (g == nullptr || f == nullptr || params == nullptr)
        return false;
    if (params->planogramSet.size() != 1)
    {
        printf("usage error: backproject_SF was written to backproject one planogram at a time\n");
        return false;
    }
    //printf("backproject_SF: psi = %f, N_v1 = %d\n", params->planogramSet[0]->psi*180.0/PI, params->planogramSet[0]->N_v1);

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    setConstantMemory(params);

    dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);

    if ((cudaStatus = cudaMalloc((void**)&dev_f, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return false;
    }
    setToConstant(dev_f, 0.0, make_int3(N_f.x, N_f.y, N_f.z), params->whichGPU);

    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);
    
    setToConstant(dev_f, 0.0, make_int3(N_f.x, N_f.y, N_f.z), params->whichGPU);
    planogram* plan = params->planogramSet[0];
    uint64 plan_sz = uint64(plan->N_v0) * uint64(plan->N_u1) * uint64(plan->N_u0);
    for (int iv1 = 0; iv1 < plan->N_v1; iv1++)
    {
        cudaTextureObject_t g_data_txt = 0;
        cudaArray* g_data_array = nullptr;
        g_data_array = loadTexture(g_data_txt, &dev_g[uint64(iv1)*plan_sz], N_g, false, true);

        backproject_planogram <<< dimGrid, dimBlock >>> (g_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, iv1);
        cudaFreeArray(g_data_array);
        cudaDestroyTextureObject(g_data_txt);
    }
    cudaStatus = cudaDeviceSynchronize();

    pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);

    cudaFree(dev_g);
    cudaFree(dev_f);
    return true;
}

bool setConstantMemory(parameters* params)
{
    planogram* plan = params->planogramSet[0];
    float cos_psi = cos(plan->psi);
    float sin_psi = sin(plan->psi);
    cudaMemcpyToSymbol(d_cos_psi, &cos_psi, sizeof(float));
    cudaMemcpyToSymbol(d_sin_psi, &sin_psi, sizeof(float));
    cudaMemcpyToSymbol(d_R, &(plan->R), sizeof(float));
    cudaMemcpyToSymbol(d_L, &(plan->L), sizeof(float));
    cudaMemcpyToSymbol(d_H, &(plan->H), sizeof(float));
    return true;
}
