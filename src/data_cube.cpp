#include <omp.h>
#include "petl_defines.h"
#include "data_cube.h"

dataCube::dataCube()
{
    clearAll();
}

dataCube::dataCube(float* data_in, int N_1_in, int N_2_in, int N_3_in)
{
    init(data_in, N_1_in, N_2_in, N_3_in);
}

dataCube::dataCube(float* data_in, int N_1_in, int N_2_in, int N_3_in, int N_4_in)
{
    init(data_in, N_1_in, N_2_in, N_3_in, N_4_in);
}

dataCube::~dataCube()
{
    clearAll();
}

void dataCube::clearAll()
{
    data = NULL;
    N_1 = 0;
    N_2 = 0;
    N_3 = 0;
    N_4 = 0;
    T_1 = 0.0;
    T_2 = 0.0;
    T_3 = 0.0;
    T_4 = 0.0;
    first_1 = 0.0;
    first_2 = 0.0;
    first_3 = 0.0;
    first_4 = 0.0;
    numDims = 0;
}

bool dataCube::init(float* data_in, int N_1_in, int N_2_in, int N_3_in, float T_1_in, float T_2_in, float T_3_in, float first_1_in, float first_2_in, float first_3_in)
{
    if (data_in == NULL || N_1_in <= 0 || N_2_in <= 0 || N_3_in <= 0)
        return false;

    data = data_in;
    N_1 = N_1_in;
    N_2 = N_2_in;
    N_3 = N_3_in;
    T_1 = T_1_in;
    T_2 = T_2_in;
    T_3 = T_3_in;
    first_1 = first_1_in;
    first_2 = first_2_in;
    first_3 = first_3_in;
    numDims = 3;

    return true;
}

bool dataCube::init(float* data_in, int N_1_in, int N_2_in, int N_3_in, int N_4_in, float T_1_in, float T_2_in, float T_3_in, float T_4_in, float first_1_in, float first_2_in, float first_3_in, float first_4_in)
{
    if (data_in == NULL || N_1_in <= 0 || N_2_in <= 0 || N_3_in <= 0 || N_4_in <= 0)
        return false;

    data = data_in;
    N_1 = N_1_in;
    N_2 = N_2_in;
    N_3 = N_3_in;
    N_4 = N_4_in;
    T_1 = T_1_in;
    T_2 = T_2_in;
    T_3 = T_3_in;
    T_4 = T_4_in;
    first_1 = first_1_in;
    first_2 = first_2_in;
    first_3 = first_3_in;
    first_4 = first_4_in;
    numDims = 4;

    return true;
}

bool dataCube::dimensionsMatch(dataCube* x)
{
    if (data == nullptr || x == nullptr || x->data == nullptr || numDims != x->numDims || N_1 != x->N_1 || N_2 != x->N_2 || N_3 != x->N_3 || (numDims == 4 && N_4 != x->N_4))
    {
        printf("%d ?= %d, %d ?= %d, %d ?= %d, %d ?= %d, %d ?= %d\n", numDims, x->numDims, N_1, x->N_1, N_2, x->N_2, N_3, x->N_3, N_4, x->N_4);
        return false;
    }
    else
        return true;
}

dataCube* dataCube::binary_operation(dataCube* x, int which, float other)
{
    if (!dimensionsMatch(x))
    {
        printf("dataCube: dimension mismatch\n");
        return nullptr;
    }
    
    if (numDims == 4)
    {
        for (uint64 i = 0; i < N_1; i++)
        {
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)*uint64(N_4)];
            float* x_1 = &(x->data[i*uint64(N_2)*uint64(N_3)*uint64(N_4)]);

            dataCube this_view(data_1, N_2, N_3, N_4);
            dataCube x_view(x_1, N_2, N_3, N_4);
            this_view.binary_operation(&x_view, which, other);
        }
        return this;
    }
    else
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int ii = 0; ii < N_1; ii++)
        {
            uint64 i = uint64(ii);
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)];
            float* x_1 = &(x->data[i*uint64(N_2)*uint64(N_3)]);
            for (uint64 j = 0; j < N_2; j++)
            {
                float* data_2 = &data_1[j*uint64(N_3)];
                float* x_2 = &x_1[j*uint64(N_3)];
                for (uint64 k = 0; k < N_3; k++)
                {
                    switch (which)
                    {
                        case ADD:
                            data_2[k] += x_2[k];
                            break;
                        case SUB:
                            data_2[k] -= x_2[k];
                            break;
                        case MULT:
                            data_2[k] *= x_2[k];
                            break;
                        case DIV:
                            if (x_2[k] == 0.0)
                                data_2[k] = 0.0;
                            else
                                data_2[k] /= x_2[k];
                            break;
                        case RDIV:
                            if (data_2[k] == 0.0)
                                data_2[k] = 0.0;
                            else
                                data_2[k] = x_2[k] / data_2[k];
                            break;
                        case SADD:
                            data_2[k] += other*x_2[k];
                            break;
                        default:
                            data_2[k] += 0.0;
                    }
                }
            }
        }
    }
    return this;
}

dataCube* dataCube::add(dataCube* x)
{
    return binary_operation(x, ADD);
}

dataCube* dataCube::sub(dataCube* x)
{
    return binary_operation(x, SUB);
}

dataCube* dataCube::multiply(dataCube* x)
{
    return binary_operation(x, MULT);
}

dataCube* dataCube::divide(dataCube* x)
{
    return binary_operation(x, DIV);
}

dataCube* dataCube::rdivide(dataCube* x)
{
    return binary_operation(x, RDIV);
}

dataCube* dataCube::scalarAdd(float c, dataCube* x)
{
    return binary_operation(x, SADD, c);
}

dataCube* dataCube::unitary_operation(int which, float x)
{
    if (numDims == 4)
    {
        if (data == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || N_4 <= 0)
        {
            printf("Error: dataCube::unitary_operation data not set\n");
            return this;
        }

        for (uint64 i = 0; i < N_1; i++)
        {
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)*uint64(N_4)];

            dataCube this_view(data_1, N_2, N_3, N_4);
            this_view.unitary_operation(which, x);
        }
        return this;
    }
    else
    {
        if (data == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
        {
            printf("Error: dataCube::unitary_operation data not set\n");
            return this;
        }

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int ii = 0; ii < N_1; ii++)
        {
            uint64 i = uint64(ii);
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)];
            for (uint64 j = 0; j < N_2; j++)
            {
                float* data_2 = &data_1[j*uint64(N_3)];
                for (uint64 k = 0; k < N_3; k++)
                {
                    switch (which)
                    {
                        case SCALE:
                            data_2[k] *= x;
                            break;
                        case EXPONENTIAL:
                            data_2[k] = exp(data_2[k]);
                        case LOGARITHM:
                            if (data_2[k] <= 0.0)
                                data_2[k] = x;
                            else
                                data_2[k] = log(data_2[k]);
                        case RECIPROCAL:
                            if (data_2[k] == 0.0)
                                data_2[k] = x;
                            else
                                data_2[k] = 1.0 / data_2[k];
                            break;
                        case CLIP:
                            data_2[k] = max(x, data_2[k]);
                            break;
                        case CONSTANT:
                            data_2[k] = x;
                            break;
                        case EXPNEG:
                            data_2[k] = exp(-data_2[k]);
                            break;
                        case NEGLOG:
                            if (data_2[k] <= 0.0)
                                data_2[k] = x;
                            else
                                data_2[k] = -log(data_2[k]);
                            break;
                        default:
                            data_2[k] += 0.0;
                    }
                }
            }
        }
    }
    return this;
}

dataCube* dataCube::scale(float x)
{
    return unitary_operation(SCALE, x);
}

dataCube* dataCube::exponential()
{
    return unitary_operation(EXPONENTIAL);
}

dataCube* dataCube::logarithm(float negative_value)
{
    return unitary_operation(LOGARITHM, negative_value);
}

dataCube* dataCube::reciprocal(float divide_by_zero_value)
{
    return unitary_operation(RECIPROCAL, divide_by_zero_value);
}

dataCube* dataCube::clip(float low)
{
    return unitary_operation(CLIP, low);
}

dataCube* dataCube::set_constant(float val)
{
    return unitary_operation(CONSTANT, val);
}

dataCube* dataCube::expNeg()
{
    return unitary_operation(EXPNEG);
}

dataCube* dataCube::negLog(float negative_value)
{
    return unitary_operation(NEGLOG, negative_value);
}

float dataCube::innerProduct(dataCube* x, dataCube* w)
{
    if (!dimensionsMatch(x))
    {
        printf("dataCube: dimension mismatch\n");
        return 0.0;
    }
    if (w != nullptr)
    {
        if (!dimensionsMatch(w))
        {
            printf("dataCube: dimension mismatch\n");
            return 0.0;
        }
    }

    if (numDims == 4)
    {
        double accum = 0.0;
        for (uint64 i = 0; i < N_1; i++)
        {
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)*uint64(N_4)];
            float* x_1 = &(x->data[i*uint64(N_2)*uint64(N_3)*uint64(N_4)]);
            float* w_1 = nullptr;
            if (w != nullptr)
                w_1 = &(w->data[i * uint64(N_2) * uint64(N_3) * uint64(N_4)]);

            dataCube this_view(data_1, N_2, N_3, N_4);
            dataCube x_view(x_1, N_2, N_3, N_4);
            if (w_1 != nullptr)
            {
                dataCube w_view(w_1, N_2, N_3, N_4);
                accum += double(this_view.innerProduct(&x_view, &w_view));
            }
            else
                accum += double(this_view.innerProduct(&x_view));
        }
        return float(accum);
    }
    else
    {
        double* sums = new double[N_1];
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int ii = 0; ii < N_1; ii++)
        {
            uint64 i = uint64(ii);
            double sum = 0.0;
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)];
            float* x_1 = &(x->data[i*uint64(N_2)*uint64(N_3)]);
            float* w_1 = nullptr;
            if (w != nullptr)
                w_1 = &(w->data[i * uint64(N_2) * uint64(N_3)]);
            for (uint64 j = 0; j < N_2; j++)
            {
                float* data_2 = &data_1[j*uint64(N_3)];
                float* x_2 = &x_1[j*uint64(N_3)];
                float* w_2 = nullptr;
                if (w_1 != nullptr)
                    w_2 = &w_1[j * uint64(N_3)];
                for (uint64 k = 0; k < N_3; k++)
                {
                    if (w_2 != nullptr)
                        sum += data_2[k] * x_2[k] * w_2[k];
                    else
                        sum += data_2[k] * x_2[k];
                }
            }
            sums[i] = sum;
        }
        double accum = 0.0;
        for (uint64 i = 0; i < N_1; i++)
            accum += sums[i];
        delete [] sums;

        return float(accum);
    }
}

float dataCube::sum()
{
    if (numDims == 4)
    {
        double accum = 0.0;
        for (uint64 i = 0; i < N_1; i++)
        {
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)*uint64(N_4)];

            dataCube this_view(data_1, N_2, N_3, N_4);
            accum += double(this_view.sum());
        }
        return float(accum);
    }
    else
    {
        double* sums = new double[N_1];
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int ii = 0; ii < N_1; ii++)
        {
            uint64 i = uint64(ii);
            double sum = 0.0;
            float* data_1 = &data[i*uint64(N_2)*uint64(N_3)];
            for (uint64 j = 0; j < N_2; j++)
            {
                float* data_2 = &data_1[j*uint64(N_3)];
                for (uint64 k = 0; k < N_3; k++)
                {
                    sum += data_2[k];
                }
            }
            sums[i] = sum;
        }
        double accum = 0.0;
        for (uint64 i = 0; i < N_1; i++)
            accum += sums[i];
        delete [] sums;

        return float(accum);
    }
}
