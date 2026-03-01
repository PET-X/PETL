#include "median_filter.cuh"
#include "cuda_utils.cuh"
#include "cuda_runtime.h"
#include "vector_ops.h"

__constant__ float d_R;
__constant__ float d_L;
__constant__ float d_H;

__global__ void bad_pixel_correction_kernel(float* g, const float* r, const int4 N, const float4 T, const float4 startVal, const float* max_r_proj, const float relative_threshold, const int iv1)
{
    const int iv0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int iu1 = threadIdx.y + blockIdx.y * blockDim.y;
    const int iu0 = threadIdx.z + blockIdx.z * blockDim.z;
    if (iv0 >= N.x || iu1 >= N.y || iu0 >= N.z)
        return;

    const float v1 = iv1 * T.w + startVal.w;
    const float v0 = iv0 * T.x + startVal.x;
    const float u1 = iu1 * T.y + startVal.y;
    const float u0 = iu0 * T.z + startVal.z;

    const int ii = uint64(iv0) * uint64(N.y * N.z) + uint64(iu1 * N.z + iu0);

    //*
    if (fabsf(u0) - 0.5f * T.z > d_L + d_R * fabsf(v0))
    {
        // does not intersect volume
        g[ii] = 0.0f;
        return;
    }
    //*/
    //*
    if (fabsf(u1) - 0.5f * T.y > d_H - d_R * fabsf(v1))
    {
        // does not intersect volume
        g[ii] = 0.0f;
        return;
    }
    //*/

    const float threshold = relative_threshold * max_r_proj[iv0];

    /*
    if (r[ii] <= 0.01f * max_r_proj[iv0])
    {
        g[ii] = 0.0f;
        return;
    }
    //*/
    if (r[ii] < threshold)
    {
        float* g_proj = &g[uint64(iv0)*uint64(N.y*N.z)];
        float* r_proj = &g[uint64(iv0)*uint64(N.y*N.z)];

        //g_proj[iu1 * N.z + iu0] = -1.0f;

        //*
        //##########################################################################
        const int windowRadius = 2;
        float v[25];
        int ind = 0;
        for (int dj = -windowRadius; dj <= windowRadius; dj++)
        {
            const int j_shift = max(0, min(iu1 + dj, N.y - 1));
            for (int dk = -windowRadius; dk <= windowRadius; dk++)
            {
                const int k_shift = max(0, min(iu0 + dk, N.z - 1));
                if (r_proj[j_shift * N.z + k_shift] > threshold) // pixel is good, store it
                {
                    v[ind] = g_proj[j_shift * N.z + k_shift];
                    ind += 1;
                }
            }
        }

        if (ind == 0)
        {
            g_proj[iu1 * N.z + iu0] = 0.0f;
        }
        if (ind == 1)
        {
            g_proj[iu1 * N.z + iu0] = v[0];
        }
        else if (ind == 2)
        {
            g_proj[iu1 * N.z + iu0] = 0.5f * (v[0] + v[1]);
        }
        else if (ind > 2)
        {
            // 3 ==> 2
            // 4 ==> 3
            // 5 ==> 3
            // 6 ==> 4 (need 2 and 3)
            // 7 ==> 4
            // 8 ==> 5
            // 9 ==> 5
            const int ind_mid = (ind - (ind % 2)) / 2 + 1;

            // bubble-sort for first half of samples
            for (int i = 0; i < ind_mid; i++)
            {
                for (int j = i + 1; j < ind; j++)
                {
                    if (v[i] > v[j])
                    {  // swap?
                        const float tmp = v[i];
                        v[i] = v[j];
                        v[j] = tmp;
                    }
                }
            }
            if (ind % 2 == 0)
                g_proj[iu1 * N.z + iu0] = 0.5f * (v[ind_mid - 1] + v[ind_mid - 2]);
            else
                g_proj[iu1 * N.z + iu0] = v[ind_mid-1];
        }
        //##########################################################################
        //*/
    }
}

bool bad_pixel_correction(float* g, float* r, parameters* params, float threshold)
{
    if (g == nullptr || r == nullptr || params == nullptr)
        return false;

    if (params->planogramSet.size() != 1)
    {
        printf("usage error: backproject_SF was written to backproject one planogram at a time\n");
        return false;
    }

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    cudaMemcpyToSymbol(d_R, &(params->planogramSet[0]->R), sizeof(float));
    cudaMemcpyToSymbol(d_L, &(params->planogramSet[0]->L), sizeof(float));
    cudaMemcpyToSymbol(d_H, &(params->planogramSet[0]->H), sizeof(float));

    float* dev_g = 0;
    float* dev_r = 0;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    dev_r = copyProjectionDataToGPU(r, params, params->whichGPU);

    /*
    N.w = plan->N_v1;
    N.x = plan->N_v0;
    N.y = plan->N_u1;
    N.z = plan->N_u0;
    //*/
    uint64 proj_sz = uint64(N_g.y*N_g.z);
    float* max_r_proj = new float[N_g.w*N_g.x];

    for (int i = 0; i < N_g.w*N_g.x; i++)
    {
        float* r_proj = &r[uint64(i)*proj_sz];

        //float cur_max = r_proj[0];
        //for (int j = 1; j < proj_sz; j++)
        //    cur_max = max(cur_max, r_proj[j]);
        //max_r_proj[i] = cur_max;
        max_r_proj[i] = median(r_proj, proj_sz);
    }
    float* dev_max_r_proj = copy1DdataToGPU(max_r_proj, N_g.w*N_g.x, params->whichGPU);

    //printf("31,22 = %f\n", max_r_proj[31 * N_g.x + 22]);

    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);

    planogram* plan = params->planogramSet[0];
    uint64 plan_sz = uint64(plan->N_v0) * uint64(plan->N_u1) * uint64(plan->N_u0);
    for (int iv1 = 0; iv1 < plan->N_v1; iv1++)
    {
        bad_pixel_correction_kernel <<< dimGrid, dimBlock >>> (&dev_g[uint64(iv1)*plan_sz], &dev_r[uint64(iv1)*plan_sz], N_g, T_g, startVal_g, &dev_max_r_proj[iv1*N_g.x], threshold, iv1);
    }
    cudaStatus = cudaDeviceSynchronize();

    pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    cudaFree(dev_g);
    cudaFree(dev_r);
    cudaFree(dev_max_r_proj);

    delete [] max_r_proj;

    return false;
}
