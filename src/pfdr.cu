#include "cuda_runtime.h"
#include "pfdr.cuh"
#include "cuda_utils.cuh"
#include <cufft.h>
#include "vector_ops.h"

__constant__ int d_N_u1;
__constant__ int d_N_V;
__constant__ int d_N_U;
__constant__ int d_N_U_over2;
__constant__ float d_T_V;
__constant__ float d_T_U;
__constant__ float d_apertureConstant;

//*
bool setConstantMemory_pfdr(planogram* p, int N_V, int N_U)
{
	if (p == nullptr)
		return false;

    float T_U = 1.0 / (N_U * p->T_u0);
    float T_V = 1.0 / (N_V * p->T_v0);

    int N_U_over2 = N_U / 2 + 1;

    cudaMemcpyToSymbol(d_N_u1, &(p->N_u1), sizeof(int));
    cudaMemcpyToSymbol(d_N_V, &N_V, sizeof(int));
    cudaMemcpyToSymbol(d_N_U, &N_U, sizeof(int));
    cudaMemcpyToSymbol(d_N_U_over2, &N_U_over2, sizeof(int));
    cudaMemcpyToSymbol(d_T_V, &T_V, sizeof(float));
    cudaMemcpyToSymbol(d_T_U, &T_U, sizeof(float));

    float apertureConstant = 0.0;
    //if (settings->algorithm == settings->PFDRX || settings->algorithm == settings->PWLS)
    apertureConstant = 1 / (2.0 * p->T_u1 * 2.0 * p->v_m0);
    cudaMemcpyToSymbol(d_apertureConstant, &apertureConstant, sizeof(float));


	return true;
}
//*/

__device__ float U(const int i)
{
    if (i < d_N_U / 2)
        return float(i) * d_T_U;
    else
        return float(i - d_N_U) * d_T_U;
}

__device__ float V(const int i)
{
    if (i < d_N_V / 2)
        return float(i) * d_T_V;
    else
        return float(i - d_N_V) * d_T_V;
}

__global__ void copy_data_to_2d_fft_buffer(float* d_real, const int B, const int NY, const int NX, const float* p, const int4 N_g, const float4 T_g, const float4 startVal_g, const float v1)
{
    const int iu1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int iv0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int iu0 = threadIdx.z + blockIdx.z * blockDim.z;
    if (iu1 >= B || iv0 >= NY || iu0 >= NX)
        return;

    const uint64 ind_out = uint64(iu1 * NY * NX) + uint64(iv0 * NX + iu0);
    const uint64 ind_in = uint64(iv0 * N_g.y * N_g.z) + uint64(iu1 * N_g.z + iu0);
    if (iv0 >= N_g.x || iu0 >= N_g.z)
        d_real[ind_out] = 0.0f;
    else
    {
        const float v0 = T_g.x * iv0 + startVal_g.x;
        d_real[ind_out] = p[ind_in] * rsqrtf(1.0f + v1 * v1 + v0 * v0);
    }
}

__global__ void rebin(cufftComplex* G_reb, const cufftComplex* G, float* weights, const float4 T_g, const float4 startVal_g, const float v1)
{
    const int iu1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int iV0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int iU0 = threadIdx.z + blockIdx.z * blockDim.z;
    if (iu1 >= d_N_u1 || iV0 >= d_N_V || iU0 >= d_N_U_over2)
        return;

    const float U_0 = U(iU0);
    const float V_0 = V(iV0);

    const float u1 = iu1 * T_g.y + startVal_g.y;

    const float T_v1 = T_g.w;
    const float abs_v1 = fabsf(v1);
    if (abs_v1 <= 0.5f * T_v1 || fabs(U_0) > abs_v1 * d_apertureConstant)
    {
        const uint64 ind_out = uint64(iu1 * d_N_V * d_N_U_over2) + uint64(iV0 * d_N_U_over2 + iU0);
        if (abs_v1 <= 0.5f * T_v1)
        {
            G_reb[ind_out].x += G[ind_out].x;
            G_reb[ind_out].y += G[ind_out].y;
            weights[ind_out] += 1.0f;
        }
        else
        {
            const float arg = v1 * (V_0 / U_0);

            const float u1_oblique = u1 - arg;
            const float iu1_oblique = (u1 - startVal_g.y) / T_g.y;
            if (iu1_oblique >= 0.0f && iu1_oblique <= float(d_N_u1 - 1))
            {

                const int iu1_oblique_lo = int(iu1_oblique);
                const int iu1_oblique_hi = min(d_N_u1-1, iu1_oblique_lo + 1);
                const float d = iu1_oblique - float(iu1_oblique_lo);

                const uint64 ind_in_lo = uint64(iu1_oblique_lo * d_N_V * d_N_U_over2) + uint64(iV0 * d_N_U_over2 + iU0);
                const uint64 ind_in_hi = uint64(iu1_oblique_hi * d_N_V * d_N_U_over2) + uint64(iV0 * d_N_U_over2 + iU0);

                G_reb[ind_out].x += (1.0 - d) * G[ind_in_lo].x + d * G[ind_in_hi].x;
                G_reb[ind_out].y += (1.0 - d) * G[ind_in_lo].y + d * G[ind_in_hi].y;
                weights[ind_out] += 1.0f;
            }
        }
    }
}

__global__ void normalize(cufftComplex* G_reb, const float* weights)
{
    const int iu1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int iV0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int iU0 = threadIdx.z + blockIdx.z * blockDim.z;
    if (iu1 >= d_N_u1 || iV0 >= d_N_V || iU0 >= d_N_U_over2)
        return;

    const uint64 ind = uint64(iu1 * d_N_V * d_N_U_over2) + uint64(iV0 * d_N_U_over2 + iU0);

    double rebinningWeight = 1.0f / float(d_N_V * d_N_U);
    if (weights[ind] > 0.0f && iU0 != 0)
        rebinningWeight *= 1.0f / weights[ind];

    G_reb[ind].x *= rebinningWeight;
    G_reb[ind].y *= rebinningWeight;
}

__global__ void copy_data_from_2d_fft_buffer(const float* real, float* g_reb, const int4 N_g, const float4 T_g, const float4 startVal_g)
{
    const int iv0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int iu1 = threadIdx.y + blockIdx.y * blockDim.y;
    const int iu0 = threadIdx.z + blockIdx.z * blockDim.z;
    if (iv0 >= N_g.x || iu1 >= N_g.y || iu0 >= N_g.z)
        return;

    const float v0 = T_g.x * iv0 + startVal_g.x;

    const uint64 ind_in = uint64(iu1 * d_N_V * d_N_U) + uint64(iv0 * d_N_U + iu0);
    const uint64 ind_out = uint64(iv0 * N_g.y * N_g.z) + uint64(iu1 * N_g.z + iu0);
    g_reb[ind_out] = real[ind_in] * sqrtf(1.0f + v0*v0);
}

bool PFDR(float* g, float* g_reb, parameters* params)
{
	if (g == nullptr || g_reb == nullptr || params == nullptr)
		return false;

    if (params->planogramSet.size() != 1)
    {
        printf("usage error: project_SF was written to project one planogram at a time\n");
        return false;
    }

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g);

    //setConstantMemory(params);

    dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    /*
    if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(projections) failed!\n");
        return false;
    }
    //*/

    planogram* p = params->planogramSet[0];
    uint64 rebin_sz = uint64(p->N_v0) * uint64(p->N_u1) * uint64(p->N_u0);
    float* dev_g_reb;
    cudaMalloc(&dev_g_reb, rebin_sz * sizeof(float));
    
    // SET FFT PLANS
    const int NX = optimalFFTsize(p->N_u0);  // fastest dimension
    const int NXH = NX / 2 + 1; // complex width for Hermitian spectrum
    const int NY = optimalFFTsize(p->N_v0);   // slow dimension
    const int B = p->N_u1;    // batch count

    setConstantMemory_pfdr(p, NY, NX);

    // Element counts per batch
    const int real_elems_per_batch = NY * NX;
    const int cplx_elems_per_batch = NY * NXH;

    // Total element counts across the batch
    const int real_total = B * real_elems_per_batch;
    const int cplx_total = B * cplx_elems_per_batch;

    // Device buffers:
    // - d_freq: complex spectrum
    // - d_real: real spatial-domain output
    cufftComplex* d_freq = nullptr;
    cufftComplex* d_G_reb = nullptr;
    float* d_real = nullptr;
    float* d_weights = nullptr;

    cudaMalloc(&d_freq, cplx_total * sizeof(cufftComplex));
    cudaMalloc(&d_G_reb, cplx_total * sizeof(cufftComplex));
    cudaMalloc(&d_real, real_total * sizeof(float));
    cudaMalloc(&d_weights, cplx_total * sizeof(float));

    cudaMemset(d_G_reb, 0, cplx_total * sizeof(cufftComplex));
    cudaMemset(d_weights, 0, cplx_total * sizeof(float));

    // Plan a batched 2D C2R:
    // n[] is [NY, NX] (slow-to-fast). This matches row-major where X is contiguous.
    int n[2] = { NY, NX };

    // For R2C, the OUTPUT has width NXH in the fastest dimension.
    // Embed arrays describe the physical layout (pitches) in elements, not bytes.
    int inembed[2] = { NY, NX };
    int onembed[2] = { NY, NXH };

    int istride = 1;
    int ostride = 1;

    // Distance between consecutive batches (in elements)
    int idist = real_elems_per_batch;
    int odist = cplx_elems_per_batch;

    cufftHandle forward_plan;
    cufftPlanMany(&forward_plan,
        2,        // rank (2D)
        n,
        inembed,   // input embedding
        istride,
        idist,
        onembed,   // output embedding
        ostride,
        odist,
        CUFFT_R2C, // complex-to-real
        B);       // batch count

    cufftHandle backward_plan;
    if (CUFFT_SUCCESS != cufftPlan2d(&backward_plan, NY, NX, CUFFT_C2R))  // do I use N_H_over2?
    {
        fprintf(stderr, "Failed to plan 2d c2r ifft");
        return false;
    }

    //################################################################################################################
    // PERFORM REBINNING
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);

    dim3 dimBlock_fft = setBlockSize(make_int3(B, NY, NX));
    dim3 dimGrid_fft = setGridSize(make_int3(B, NY, NX), dimBlock_fft);

    dim3 dimBlock_fft2 = setBlockSize(make_int3(B, NY, NXH));
    dim3 dimGrid_fft2 = setGridSize(make_int3(B, NY, NXH), dimBlock_fft2);

    for (int iv1 = 0; iv1 < p->N_v1; iv1++)
    {
        float v1 = p->v1(iv1);
        //printf("v1 = %f\n", p->v1(iv1));
        //project_planogram <<< dimGrid, dimBlock >>> (&dev_g[uint64(iv1) * rebin_sz], N_g, T_g, startVal_g, iv1);
        // copy data into d_real buffer
        copy_data_to_2d_fft_buffer <<< dimGrid_fft, dimBlock_fft >>> (d_real, B, NY, NX, &dev_g[uint64(iv1) * rebin_sz], N_g, T_g, startVal_g, v1);
        
        // perform 2D FFTs
        cufftExecR2C(forward_plan, d_real, d_freq);

        // rebin
        //__global__ void rebin(cufftComplex* d_rebin, const cufftComplex* d_freq, const float4 T_g, const float4 startVal_g, const float v1)
        rebin <<< dimGrid_fft2, dimBlock_fft2 >>> (d_G_reb, d_freq, d_weights, T_g, startVal_g, p->v1(iv1));
    }
    cudaStatus = cudaDeviceSynchronize();

    // NORMALIZE
    normalize <<< dimGrid_fft2, dimBlock_fft2 >>> (d_G_reb, d_weights);

    // IFFT
    for (int iu1 = 0; iu1 < p->N_u1; iu1++)
        cufftExecC2R(backward_plan, &d_G_reb[iu1* NY * NXH], &d_real[iu1* NY * NX]);

    // copy data back
    copy_data_from_2d_fft_buffer <<< dimGrid, dimBlock >>> (d_real, dev_g_reb, N_g, T_g, startVal_g);

    // copy rebinned data back to cpu
    cudaStatus = cudaDeviceSynchronize();
    pull3DdataFromGPU(g_reb, make_int3(p->N_v0, p->N_u1, p->N_u0), dev_g_reb, params->whichGPU);

    // Clean Up
    cudaFree(dev_g);
    cudaFree(d_freq);
    cudaFree(d_real);
    cudaFree(d_G_reb);
    cudaFree(d_weights);
    cudaFree(dev_g_reb);
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);

    return true;
}
