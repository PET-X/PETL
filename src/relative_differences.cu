
#include "relative_differences.cuh"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__forceinline__ __device__ float relativeDifferences(const float x, const float y, const float delta)
{
    if (x == y)  //(x == 0 && y == 0)
        return 0.0f;
    else
        return (x - y) * (x - y) / ((x + y) + delta * fabs(x - y));
}

__forceinline__ __device__ float DrelativeDifferences(const float x, const float y, const float delta)
{
    if (x == 0.0f && y == 0.0f)
        return 0.0f;
    else
        return (x - y) * (delta * fabs(x - y) + x + 3.0f * y) /
               ((x + y + delta * fabs(x - y)) * (x + y + delta * fabs(x - y)));
}

__forceinline__ __device__ float DDrelativeDifferences(const float x, const float y, const float delta)
{
    if (x == 0.0f && y == 0.0f)
        return 0.0f;
    else
        return 16.0f * y * y /
               ((x + y + delta * fabs(x - y)) * (x + y + delta * fabs(x - y)) * (x + y + delta * fabs(x - y)));
}

__forceinline__ __device__ float DDrelativeDifferences_quad(const float f_cur, const float f_n, const float d_cur,
                                                            const float d_n, const float delta)
{
    if (f_cur == 0.0f && f_n == 0.0f)
        return 0.0f;
    else
        return 8.0f * (f_n * d_cur - f_cur * d_n) * (f_n * d_cur - f_cur * d_n) /
               ((f_cur + f_n + delta * fabs(f_cur - f_n)) * (f_cur + f_n + delta * fabs(f_cur - f_n)) *
                (f_cur + f_n + delta * fabs(f_cur - f_n)));
}

__device__ float aTV_RelativeDifferences_costTerm(float* f, const int i, const int j, const int k, int4 N, float delta,
                                                  float beta)
{
    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    const float curVal = f[i * N.y * N.z + j * N.z + k];

    return (relativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j * N.z + k_minus], delta)) *
               dist_1 +
           (relativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
            relativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
            relativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
            relativeDifferences(curVal, f[i_minus * N.z + j_plus * N.z + k], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
               dist_2 +
           (relativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            relativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            relativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
               dist_3;
}

__global__ void aTV_RelativeDifferences_cost(float* f, float* d, int4 N, float delta, float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    d[i * N.y * N.z + j * N.z + k] = aTV_RelativeDifferences_costTerm(f, i, j, k, N, delta, beta);
}

__global__ void clip(float* f, int4 N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    // if (f[i * N.y * N.z + j * N.z + k] < 0.0f)
    if (f[i * N.y * N.z + j * N.z + k] < 0.00000001f) f[i * N.y * N.z + j * N.z + k] = 0.0f;
}

__global__ void aTV_RelativeDifferences_gradient(float* f, float* Df, int4 N, float delta, float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    //Df[i * N.y * N.z + j * N.z + k] = float(i * N.y * N.z + j * N.z + k);
    //return;

    const float curVal = f[i * N.y * N.z + j * N.z + k];

    // dist 1: 6
    // dist 2: 12
    // dist 3: 8
    Df[i * N.y * N.z + j * N.z + k] =
        (DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j * N.z + k_minus], delta)) *
            dist_1 +
        (DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
            dist_2 +
        (DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
         DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
         DrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
            dist_3;
}

__device__ float aTV_RelativeDifferences_quadFormTerm(float* f, float* d, const int i, const int j, const int k, int4 N,
                                                      float delta, float beta)
{
    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    const float curVal = f[i * N.y * N.z + j * N.z + k];
    const float curVal_d = d[i * N.y * N.z + j * N.z + k];

    return (DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j * N.z + k], curVal_d,
                                       d[i_plus * N.y * N.z + j * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j * N.z + k], curVal_d,
                                       d[i_minus * N.y * N.z + j * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j_plus * N.z + k], curVal_d,
                                       d[i * N.y * N.z + j_plus * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j_minus * N.z + k], curVal_d,
                                       d[i * N.y * N.z + j_minus * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j * N.z + k_plus], curVal_d,
                                       d[i * N.y * N.z + j * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j * N.z + k_minus], curVal_d,
                                       d[i * N.y * N.z + j * N.z + k_minus], delta)) *
               dist_1 +
           (DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k], curVal_d,
                                       d[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k], curVal_d,
                                       d[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j * N.z + k_plus], curVal_d,
                                       d[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j * N.z + k_minus], curVal_d,
                                       d[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k], curVal_d,
                                       d[i_minus * N.y * N.z + j_plus * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k], curVal_d,
                                       d[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j * N.z + k_plus], curVal_d,
                                       d[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j * N.z + k_minus], curVal_d,
                                       d[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j_plus * N.z + k_plus], curVal_d,
                                       d[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j_plus * N.z + k_minus], curVal_d,
                                       d[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j_minus * N.z + k_plus], curVal_d,
                                       d[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i * N.y * N.z + j_minus * N.z + k_minus], curVal_d,
                                       d[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
               dist_2 +
           (DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_plus], curVal_d,
                                       d[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_minus], curVal_d,
                                       d[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_plus], curVal_d,
                                       d[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_minus], curVal_d,
                                       d[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_plus], curVal_d,
                                       d[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_minus], curVal_d,
                                       d[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_plus], curVal_d,
                                       d[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DDrelativeDifferences_quad(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_minus], curVal_d,
                                       d[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
               dist_3;
}

__global__ void aTV_RelativeDifferences_quadForm(float* f, float* d, float* quad, int4 N, float delta, float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    quad[i * N.y * N.z + j * N.z + k] = aTV_RelativeDifferences_quadFormTerm(f, d, i, j, k, N, delta, beta);
}

__global__ void aTV_RelativeDifferences_curvature(float* f, float* DDf, int4 N, float delta, float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    const float curVal = f[i * N.y * N.z + j * N.z + k];

    // dist 1: 6
    // dist 2: 12
    // dist 3: 8
    DDf[i * N.y * N.z + j * N.z + k] =
        (DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j * N.z + k_minus], delta)) *
            dist_1 +
        (DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
            dist_2 +
        (DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
         DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
         DDrelativeDifferences(curVal, f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
            dist_3;
}

__global__ void GaussianFilterKernel(float* f, float* f_filtered, int4 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    const float denom = 1.0f / (2.0f * sigma);

    float val = 0.0f;
    float sum = 0.0f;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        const float i_dist_sq = float((i - i_shift) * (i - i_shift));
        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            const float j_dist_sq = float((j - j_shift) * (j - j_shift));
            for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                const float k_dist_sq = float((k - k_shift) * (k - k_shift));

                const float theWeight = expf(-denom * (i_dist_sq + j_dist_sq + k_dist_sq));

                val += theWeight * f[i_shift * N.y * N.z + j_shift * N.z + k_shift];
                sum += theWeight;
            }
        }
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

__global__ void GaussianFilter2DKernel(float* f, float* f_filtered, int4 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    const float denom = 1.0f / (2.0f * sigma);

    float val = 0.0f;
    float sum = 0.0f;

    for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
    {
        const int j_shift = max(0, min(j + dj, N.y - 1));
        const float j_dist_sq = float((j - j_shift) * (j - j_shift));
        for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));
            const float k_dist_sq = float((k - k_shift) * (k - k_shift));

            const float theWeight = expf(-denom * (j_dist_sq + k_dist_sq));

            val += theWeight * f[i * N.y * N.z + j_shift * N.z + k_shift];
            sum += theWeight;
        }
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

__global__ void GaussianFilter1DKernel(float* f, float* f_filtered, int4 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    const float denom = 1.0f / (2.0f * sigma);

    float val = 0.0f;
    float sum = 0.0f;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));

        const float theWeight = expf(-denom * float((i - i_shift) * (i - i_shift)));

        val += theWeight * f[i_shift * N.y * N.z + j * N.z + k];
        sum += theWeight;
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

__global__ void sum1D(float* f, int4 N)
{
    // return;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    // const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y) return;
    for (int iz = 1; iz < N.z; iz++) f[j * N.x + i] += f[iz * N.x * N.y + j * N.x + i];
    // now all values are at: f[j * N.x + i]
}

__global__ void sum2D(float* f, int4 N, float* sumAll)
{
    if (threadIdx.x + blockIdx.x * blockDim.x > 0) return;
    sumAll[0] = f[0];
    for (int i = 1; i < N.x * N.y; i++) sumAll[0] += f[i];
}

/*
dim3 setBlockSize(int4 N)
{
    dim3 dimBlock(8, 8, 8);  // needs to be optimized
    if (N.z < 8)
    {
        dimBlock.x = 16;
        dimBlock.y = 16;
        dimBlock.z = 1;
    }
    else if (N.y < 8)
    {
        dimBlock.x = 16;
        dimBlock.y = 1;
        dimBlock.z = 16;
    }
    else if (N.x < 8)
    {
        dimBlock.x = 1;
        dimBlock.y = 16;
        dimBlock.z = 16;
    }
    return dimBlock;
}

dim3 setBlockSize(int3 N)
{
    dim3 dimBlock(8, 8, 8);  // needs to be optimized
    if (N.z < 8)
    {
        dimBlock.x = 16;
        dimBlock.y = 16;
        dimBlock.z = 1;
    }
    else if (N.y < 8)
    {
        dimBlock.x = 16;
        dimBlock.y = 1;
        dimBlock.z = 16;
    }
    else if (N.x < 8)
    {
        dimBlock.x = 1;
        dimBlock.y = 16;
        dimBlock.z = 16;
    }
    return dimBlock;
}
//*/

bool aTV_RelativeDifferencesLoss_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta,
                                          int whichGPU)
{
    if (f == NULL) return false;

    cudaSetDevice(whichGPU);

    int4 N;
    N.x = N_1;
    N.y = N_2;
    N.z = N_3;

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, make_int3(N.x, N.y, N.z), whichGPU);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return false;
    }

    // Call kernel
    // dim3 dimBlock(8, 8, 8); // needs to be optimized
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    clip<<<dimGrid, dimBlock>>>(dev_f, N);
    cudaDeviceSynchronize();
    aTV_RelativeDifferences_gradient<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, delta, beta);
    cudaDeviceSynchronize();

    // pull result off GPU
    if (Df == NULL) Df = f;
    //pullVolumeDataFromGPU(Df, N, dev_Df, whichGPU);
    pull3DdataFromGPU(Df, make_int3(N_1, N_2, N_3), dev_Df, whichGPU);
    cudaDeviceSynchronize();

    // Clean up
    if (dev_f != 0)
    {
        cudaFree(dev_f);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}

float aTV_RelativeDifferencesLoss_quadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta,
                                           int whichGPU)
{
    if (f == NULL || d == NULL) return -1.0;

    cudaSetDevice(whichGPU);

    int4 N;
    N.x = N_1;
    N.y = N_2;
    N.z = N_3;

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, make_int3(N.x, N.y, N.z), whichGPU);

    // Copy step direction to GPU
    float* dev_d = 0;
    dev_d = copy3DdataToGPU(d, make_int3(N.x, N.y, N.z), whichGPU);

    // Allocate space on GPU for the un-collapsed quadratic form
    float* dev_quad = 0;
    if (cudaMalloc((void**)&dev_quad, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return -1.0;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    clip<<<dimGrid, dimBlock>>>(dev_f, N);
    cudaDeviceSynchronize();
    aTV_RelativeDifferences_quadForm<<<dimGrid, dimBlock>>>(dev_f, dev_d, dev_quad, N, delta, beta);
    cudaDeviceSynchronize();

    float retVal = 0.0;

    /*
    float* dev_sumAll = 0;
    if (cudaMalloc((void**)&dev_sumAll, 1 * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(sum) failed!\n");
        return 0.0;
    }

    dim3 dimBlock2D(8, 8); // needs to be optimized
    dim3 dimGrid2D(int(ceil(double(N.x) / double(dimBlock2D.x))), int(ceil(double(N.y) / double(dimBlock2D.y))));
    sum1D <<< dimGrid2D, dimBlock2D >>> (dev_quad, N);
    cudaStatus = cudaDeviceSynchronize();

    sum2D <<< N.x*N.y, 1 >>> (dev_quad, N, dev_sumAll);
    cudaStatus = cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(&retVal, dev_sumAll, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaSuccess != cudaStatus)
    {
        fprintf(stderr, "failed to copy data back to host!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    cudaFree(dev_sumAll);
    //*/

    //* pull result off GPU
    float* quadTerms = (float*)malloc(sizeof(float) * N.x * N.y * N.z);
    pull3DdataFromGPU(quadTerms, make_int3(N.x, N.y, N.z), dev_quad, whichGPU);
    for (int i = 0; i < N.x; i++)
    {
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) retVal += quadTerms[i * N.y * N.z + j * N.z + k];
        }
    }
    free(quadTerms);
    //*/

    // Clean up
    if (dev_f != 0)
        cudaFree(dev_f);
    if (dev_d != 0)
        cudaFree(dev_d);
    if (dev_quad != 0)
        cudaFree(dev_quad);

    return retVal;
}

bool aTV_RelativeDifferencesLoss_curvature(float* f, float* DDf, int N_1, int N_2, int N_3, float delta, float beta,
                                           int whichGPU)
{
    if (f == NULL) return false;

    cudaSetDevice(whichGPU);

    int4 N;
    N.x = N_1;
    N.y = N_2;
    N.z = N_3;

    // copy volume to gpu
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, make_int3(N.x, N.y, N.z), whichGPU);

    // allocate space on gpu for the hessian diagonal
    float* dev_DDf = 0;
    if (cudaMalloc((void**)&dev_DDf, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "aTV_HuberLoss_curvature: cudaMalloc failed!\n");
        return false;
    }

    // call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    aTV_RelativeDifferences_curvature<<<dimGrid, dimBlock>>>(dev_f, dev_DDf, N, delta, beta);
    cudaDeviceSynchronize();

    // copy result from gpu
    if (DDf == NULL) DDf = f;
    pull3DdataFromGPU(DDf, make_int3(N.x, N.y, N.z), dev_DDf, whichGPU);

    // deallocate gpu memory
    if (dev_f != 0)
        cudaFree(dev_f);
    if (dev_DDf != 0)
        cudaFree(dev_DDf);
    return true;
}

float aTV_RelativeDifferencesLoss_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, int whichGPU)
{
    if (f == NULL) return -1.0;

    cudaSetDevice(whichGPU);

    int4 N;
    N.x = N_1;
    N.y = N_2;
    N.z = N_3;

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, make_int3(N.x, N.y, N.z), whichGPU);
    
    // Allocate space on GPU for the un-collapsed quadratic form
    float* dev_d = 0;
    if (cudaMalloc((void**)&dev_d, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return -1.0;
    }

    float retVal = 0.0;

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    clip<<<dimGrid, dimBlock>>>(dev_f, N);
    cudaDeviceSynchronize();

    aTV_RelativeDifferences_cost<<<dimGrid, dimBlock>>>(dev_f, dev_d, N, delta, beta);
    cudaDeviceSynchronize();

    /*
    float* dev_sumAll = 0;
    if (cudaMalloc((void**)&dev_sumAll, 1 * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(sum) failed!\n");
        return 0.0;
    }

    dim3 dimBlock2D(8, 8);  // needs to be optimized
    dim3 dimGrid2D(int(ceil(double(N.x) / double(dimBlock2D.x))), int(ceil(double(N.y) / double(dimBlock2D.y))));
    sum1D<<<dimGrid2D, dimBlock2D>>>(dev_d, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    sum2D<<<N.x * N.y, 1>>>(dev_d, N, dev_sumAll);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t cudaStatus = cudaMemcpy(&retVal, dev_sumAll, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaSuccess != cudaStatus)
    {
        fprintf(stderr, "failed to copy data back to host!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    cudaFree(dev_sumAll);
    //*/

    //* pull result off GPU
    float* costTerms = (float*)malloc(sizeof(float) * N.x * N.y * N.z);
    pull3DdataFromGPU(costTerms, make_int3(N.x, N.y, N.z), dev_d, whichGPU);
    for (int i = 0; i < N.x; i++)
    {
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) retVal += costTerms[i * N.y * N.z + j * N.z + k];
        }
    }
    free(costTerms);
    //*/

    // Clean up
    if (dev_f != 0)
        cudaFree(dev_f);
    if (dev_d != 0)
        cudaFree(dev_d);

    return retVal;
}

bool GaussianFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int whichGPU)
{
    if (f == NULL) return false;

    cudaSetDevice(whichGPU);

    int4 N;
    N.x = N_1;
    N.y = N_2;
    N.z = N_3;

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, make_int3(N.x, N.y, N.z), whichGPU);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    if (numDims == 1)
        GaussianFilter1DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);
    else if (numDims == 2)
        GaussianFilter2DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);
    else
        GaussianFilterKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);
    // medianFilterKernel(float* f, float* f_filtered, int4 N, float threshold)

    // wait for GPU to finish
    cudaDeviceSynchronize();

    pull3DdataFromGPU(f, make_int3(N.x, N.y, N.z), dev_Df, whichGPU);

    // Clean up
    if (dev_f != 0)
        cudaFree(dev_f);
    if (dev_Df != 0)
        cudaFree(dev_Df);

    return true;
}
