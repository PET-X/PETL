////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for GPU-based ramp and Hilbert filters
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <algorithm>
#include <cufft.h>
#include "ramp_filter.cuh"
#include "cuda_runtime.h"
#include "cuda_utils.cuh"
#include "vector_ops.h"
//#include "cpu_utils.h"

#ifndef PI
#define PI 3.141592653589793
#endif

#define NUM_RAYS_PER_THREAD 8

__global__ void multiplyRampFilterKernel(cufftComplex* G, const float* H, int3 N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z)
        return;
    uint64 ind = uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k);
    G[ind].x *= H[k];
    G[ind].y *= H[k];
}

__global__ void setPaddedDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView, int numExtrapolate)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x + startView;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    //int j = threadIdx.x;
    //int i = blockIdx.x + startView;
    if (i > endView || j > N.y - 1)
        return;
    float* data_padded_block = &data_padded[uint64(i - startView) * uint64(N_pad * N.y) + uint64(j * N_pad)];
    float* data_block = &data[uint64(i) * uint64(N.z * N.y) + uint64(j * N.z)];

    for (int k = 0; k < N.z; k++)
        data_padded_block[k] = data_block[k];
    for (int k = N.z; k < N_pad; k++)
        data_padded_block[k] = 0.0f;

    if (numExtrapolate > 0)
    {
        const float leftVal = data_block[0];
        const float rightVal = data_block[N.z - 1];
        for (int k = N.z; k < N.z + numExtrapolate; k++)
            data_padded_block[k] = rightVal;
        for (int k = N_pad - numExtrapolate; k < N_pad; k++)
            data_padded_block[k] = leftVal;
    }
}

__global__ void setFilteredDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x + startView;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    //int j = threadIdx.x;
    //int i = blockIdx.x + startView;
    if (i > endView || j > N.y - 1)
        return;
    float* data_padded_block = &data_padded[uint64(i - startView) * uint64(N_pad * N.y) + uint64(j * N_pad)];
    float* data_block = &data[uint64(i) * uint64(N.z * N.y) + uint64(j * N.z)];

    for (int k = 0; k < N.z; k++)
        data_block[k] = data_padded_block[k];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LAUNCHING FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float* rampFilterFrequencyResponseMagnitude(int N, parameters* params)
{
    cudaError_t cudaStatus;
    float* h = rampImpulseResponse(N, params);

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N, CUFFT_R2C, 1))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft");
        return NULL;
    }

    float* dev_h = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_h, N * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(ramp filter) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_h, h, N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(ramp filter) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_h);
        return NULL;
    }

    // Make data for the result of the FFT
    int N_over2 = N / 2 + 1;
    cufftComplex* dev_H = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_H, N_over2 * sizeof(cufftComplex))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of ramp filter) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_h);
        return NULL;
    }

    // FFT
    result = cufftExecR2C(forward_plan, (cufftReal*)dev_h, dev_H);
    if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed!\n");
        printf("cudaDeviceSynchronize Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }

    // get result
    cufftComplex* H_ramp = new cufftComplex[N_over2];
    if ((cudaStatus = cudaMemcpy(H_ramp, dev_H, N_over2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }

    float* H_real = new float[N_over2];
    for (int i = 0; i < N_over2; i++)
        H_real[i] = H_ramp[i].x / float(N);

    // Clean up
    cufftDestroy(forward_plan);
    cudaFree(dev_h);
    cudaFree(dev_H);
    delete[] h;
    delete[] H_ramp;

    return H_real;
}

bool rampFilter1D(float*& g, parameters* params, bool data_on_cpu, float scalar)
{
    if (g == nullptr || params == nullptr)
        return false;
    if (params->num_planograms() != 1 || params->planogramSet[0]->N_v1 != 1)
    {
        printf("Error: rampFilter1D only applies to a single rebinned planogram\n");
        return false;
    }

    bool retVal = true;
    cudaError_t cudaStatus;
    cudaSetDevice(params->whichGPU);

    float* dev_g = 0;
    if (data_on_cpu)
    {
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    }
    else
    {
        dev_g = g;
    }

    int N_v0 = params->planogramSet[0]->N_v0;
    int N_u1 = params->planogramSet[0]->N_u1;
    int N_u0 = params->planogramSet[0]->N_u0;

    // PUT CODE HERE
    //int N_H = int(pow(2.0, ceil(log2(2 * params->numCols))));
    int N_H = optimalFFTsize(2 * N_u0);
    //printf("FFT size = %d\n", N_H);
    int N_H_over2 = N_H / 2 + 1;
    float* H_real = NULL;
    H_real = rampFilterFrequencyResponseMagnitude(N_H, params);
    if (scalar != 1.0)
    {
        for (int i = 0; i < N_H_over2; i++)
        {
            if (H_real != NULL)
                H_real[i] *= scalar;
        }
    }

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g);

    int numRows = N_u1;
    int numAngles = N_v0;
    if (numAngles == 1)
    {
        numRows = 1;
        numAngles = N_u1;
    }

    //int N_viewChunk = params->numAngles;
    int N_viewChunk = max(1, numAngles / 40); // number of views in a chunk (needs to be optimized)
    int numChunks = int(ceil(double(numAngles) / double(N_viewChunk)));

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N_H, CUFFT_R2C, uint64(N_viewChunk) * uint64(numRows)))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft (size %d)\n", N_H);
        return false;
    }
    cufftHandle backward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&backward_plan, N_H, CUFFT_C2R, uint64(N_viewChunk) * uint64(numRows))) // do I use N_H_over2?
    {
        fprintf(stderr, "Failed to plan 1d c2r ifft\n");
        return false;
    }
    //return true;

    float* dev_g_pad = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_g_pad, uint64(N_viewChunk) * uint64(numRows) * uint64(N_H) * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }

    // Make data for the result of the FFT
    cufftComplex* dev_G = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_G, uint64(N_viewChunk) * uint64(numRows) * uint64(N_H_over2) * sizeof(cufftComplex))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of padded projection data) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }

    // Copy filter to device
    float* dev_H = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_H, N_H_over2 * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    cudaStatus = cudaMemcpy(dev_H, H_real, N_H_over2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus)
    {
        fprintf(stderr, "cudaMemcpy(H) failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }
    int3 dataSize; dataSize.x = N_viewChunk; dataSize.y = numRows; dataSize.z = N_H_over2;
    int3 origSize; origSize.x = numAngles; origSize.y = numRows; origSize.z = N_u0;

    int numExtrapolate = 0;
    //if (params->truncatedScan)
    //    numExtrapolate = min(N_H - params->numCols - 1, 100);

    if (retVal == true)
    {
        dim3 dimBlock_viewChunk = setBlockSize(dataSize);
        dim3 dimGrid_viewChunk = setGridSize(dataSize, dimBlock_viewChunk);

        for (int iChunk = 0; iChunk < numChunks; iChunk++)
        {
            int startView = iChunk * N_viewChunk;
            int endView = min(numAngles - 1, startView + N_viewChunk - 1);
            //printf("filtering %d to %d\n", startView, endView);

            dim3 dimBlock_setting(min(8, endView - startView + 1), min(8, numRows));
            dim3 dimGrid_setting(int(ceil(double(endView - startView + 1) / double(dimBlock_setting.x))), int(ceil(double(numRows) / double(dimBlock_setting.y))));

            setPaddedDataKernel <<< dimGrid_setting, dimBlock_setting >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, numExtrapolate);
            //cudaDeviceSynchronize();

            // FFT
            result = cufftExecR2C(forward_plan, (cufftReal*)dev_g_pad, dev_G);
            if (result != CUFFT_SUCCESS)
            {
                printf("cufftExecR2C failed!\n");
            }

            // Multiply Filter
            multiplyRampFilterKernel <<< dimGrid_viewChunk, dimBlock_viewChunk >>> (dev_G, dev_H, dataSize);
            //cudaDeviceSynchronize();

            // IFFT
            result = cufftExecC2R(backward_plan, (cufftComplex*)dev_G, (cufftReal*)dev_g_pad);
            if (result != CUFFT_SUCCESS)
            {
                printf("cufftExecC2R failed!\n");
            }

            setFilteredDataKernel <<< dimGrid_setting, dimBlock_setting >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView);
            //cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        if (data_on_cpu)
        {
            // Copy result back to host
            cudaStatus = cudaMemcpy(g, dev_g, uint64(numAngles) * uint64(numRows) * uint64(N_u0) * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaSuccess != cudaStatus)
            {
                fprintf(stderr, "failed to copy result back to host!\n");
                fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
                fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            }
        }
    }

    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    cudaFree(dev_g_pad);
    if (data_on_cpu)
        cudaFree(dev_g);
    if (dev_H != 0)
        cudaFree(dev_H);
    cudaFree(dev_G);
    if (H_real != NULL)
        delete[] H_real;

    return retVal;
}

float* rampImpulseResponse(int N, parameters* params)
{
    float T = params->planogramSet[0]->T_u0;
    float* h = new float[N];
    for (int i = 0; i < N; i++)
    {
        float s;
        if (i < N / 2)
            s = float(i);
        else
            s = float(i - N);
        h[i] = 1.0 / (T*PI * (0.25 - s * s));
    }
    return h;
}
