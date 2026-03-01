#include <omp.h>
#include "planogram.h"

planogram::planogram()
{
    clearAll();
}

planogram::~planogram()
{
    clearAll();
}

void planogram::clearAll()
{
    psi = 0.0;
    R = 0.0;
    L = 0.0;
    H = 0.0;
    v_m0 = 0.0;
    v_m1 = 0.0;

    T_u0 = 0.0;
    T_u1 = 0.0;
    T_v0 = 0.0;
    T_v1 = 0.0;

    N_u0 = 0;
    N_u1 = 0;
    N_v0 = 0;
    N_v1 = 0;

    u0_0 = 0.0;
    u1_0 = 0.0;
    v0_0 = 0.0;
    v1_0 = 0.0;

    cos_psi = 1.0;
    sin_psi = 0.0;

    dataCube::data = nullptr;
}

planogram& planogram::operator = (const planogram& other)
{
    if (this != &other)
        this->assign(other);
    return *this;
}

void planogram::assign(const planogram& other)
{
    this->psi = other.psi;
    this->R = other.R;
    this->L = other.L;
    this->H = other.H;
    this->v_m0 = other.v_m0;
    this->v_m1 = other.v_m1;

    this->T_u0 = other.T_u0;
    this->T_u1 = other.T_u1;
    this->T_v0 = other.T_v0;
    this->T_v1 = other.T_v1;

    this->N_u0 = other.N_u0;
    this->N_u1 = other.N_u1;
    this->N_v0 = other.N_v0;
    this->N_v1 = other.N_v1;

    this->u0_0 = other.u0_0;
    this->u1_0 = other.u1_0;
    this->v0_0 = other.v0_0;
    this->v1_0 = other.v1_0;

    this->cos_psi = cos(this->psi);
    this->sin_psi = sin(this->psi);

    this->data = other.data;
}

void planogram::set_data(float* data_in)
{
    this->data = data_in;
    this->numDims = 4;
    this->N_1 = N_v1;
    this->N_2 = N_v0;
    this->N_3 = N_u1;
    this->N_4 = N_u0;
}

bool planogram::defined(bool doPrint)
{
    if (R > 0.0 && L > 0.0 && H > 0.0 && v_m0 > 0.0 && v_m1 >= 0.0 && T_u0 > 0.0 && T_u1 > 0.0 && T_v0 > 0.0 && T_v1 > 0.0 && N_u0 > 0 && N_u1 > 0 && N_v0 > 0 && N_v1 > 0)
        return true;
    else
        return false;
}

bool planogram::is_within_azimuthal_acceptable_angle(float* r)
{
    if (r == nullptr)
        return false;
    float c = -sin_psi*r[0] + cos_psi*r[1];
    if (c == 0.0)
        return false;
    float v_0 = -(cos_psi*r[0] + sin_psi*r[1]) / c;
    if (fabs(v_0) > v_m0)
        return false;
    else
        return true;
}

bool planogram::is_within_acceptable_angle(float* r, float& v_1, float& v_0)
{
    if (r == nullptr)
        return false;
    float c = -sin_psi*r[0] + cos_psi*r[1];
    if (c == 0.0)
        return false;
    v_0 = -(cos_psi*r[0] + sin_psi*r[1]) / c;
    if (fabs(v_0) > v_m0)
        return false;
    v_1 = -r[2] / c;
    if (fabs(v_1) > v_m1)
        return false;
    else
        return true;
}

bool planogram::get_u_coords(float* p, float v_1, float v_0, float& u_1, float& u_0)
{
    if (p == nullptr)
        return false;
    else
    {
        float t = p[1]*cos_psi - p[0]*sin_psi;
        u_1 = p[2] + v_1*t; // p[2] = u_1 - v_1*t
        u_0 = p[0]*cos_psi + p[1]*sin_psi + v_0*t;
        return true;
    }
}

bool planogram::init(float psi_in, float R_in, float L_in, float H_in, float v_m0_in, float v_m1_in, float T, float* data_ptr)
{
    if (R_in <= 0.0 || L_in <= 0.0 || H_in <= 0.0 || v_m0_in <= 0.0 || v_m1_in < 0.0 || T <= 0.0)
        return false;

    psi = psi_in;
    R = R_in;
    L = L_in;
    H = H_in;
    v_m0 = v_m0_in;
    v_m1 = min(H/R, v_m1_in);

    T_u0 = T;
    T_u1 = T;
    //T_v0 = T_u0 / (2.0*R);
    //T_v1 = T_u1 / (2.0*R);
    T_v0 = T_u0 / R;
    T_v1 = T_u1 / R;

    N_u0 = 2*int((L + R*v_m0) / T_u0);
    //N_u1 = 2*int((H - R*v_m1) / T_u1);
    N_u1 = 2*int(H / T_u1);
    N_v0 = 2*int(v_m0 / T_v0) + 1;
    N_v1 = 2*int(v_m1 / T_v1) + 1;

    //(N_v0-1)*T_v0 - T_v0 * float(N_v0 - 1) / 2.0 + T_v0/2 = v_m0
    //((N_v0-1) - float(N_v0 - 1) / 2.0 + 1/2)*T_v0 = v_m0
    //((N_v0-1)/2 + 1/2)*T_v0 = v_m0
    //(N_v0/2)*T_v0 = v_m0
    //T_v0 = v_m0*2/N_v0
    T_v0 = v_m0 * 2.0 / float(N_v0);

    u0_0 = -T_u0*float(N_u0-1)/2.0;
    u1_0 = -T_u1*float(N_u1-1)/2.0;
    v0_0 = -T_v0*float(N_v0-1)/2.0;
    v1_0 = -T_v1*float(N_v1-1)/2.0;

    cos_psi = cos(psi);
    sin_psi = sin(psi);

    data = data_ptr;

    return true;
}

float planogram::u0(int i)
{
    return i*T_u0 + u0_0;
}

float planogram::u1(int i)
{
    return i*T_u1 + u1_0;
}

float planogram::v0(int i)
{
    return i*T_v0 + v0_0;
}

float planogram::v1(int i)
{
    return i*T_v1 + v1_0;
}

float planogram::u0_inv(float x)
{
    return (x - u0_0) / T_u0;
}

float planogram::u1_inv(float x)
{
    return (x - u1_0) / T_u1;
}

float planogram::v0_inv(float x)
{
    return (x - v0_0) / T_v0;
}

float planogram::v1_inv(float x)
{
    return (x - v1_0) / T_v1;
}

bool planogram::collapse_dimension(int i)
{
    if (0 <= i && i < N_v1)
    {
        v1_0 = v1(i);
        N_v1 = 1;
        return true;
    }
    else
        return false;
}

bool planogram::reduce_dimension(int i_0, int i_end)
{
    if (0 <= i_0 && i_0 < N_v1 && 0 <= i_end && i_end < N_v1 && i_0 <= i_end)
    {
        v1_0 = v1(i_0);
        N_v1 = i_end - i_0 + 1;
        return true;
    }
    else
        return false;
}

float planogram::planogram_weight(float v1_val, float v0_val)
{
    return 1.0 / sqrt(1.0 + v1_val * v1_val + v0_val * v0_val);
}

float planogram::planogram_weight(int iv1, int iv0)
{
    return planogram_weight(v1(iv1), v0(iv0));
}

float planogram::solid_angle_correction_factor(float v1_val, float v0_val)
{
    return 2.0 * PI * pow(1.0 + v0_val * v0_val + v1_val * v1_val, 1.5) / (T_v1 * T_v0);
}

float planogram::solid_angle_correction_factor(int iv1, int iv0)
{
    return solid_angle_correction_factor(v1(iv1), v0(iv0));
}

bool planogram::apply_solid_angle_correction(bool doInverse)
{
    if (data == nullptr)
    {
        printf("planogram::apply_solid_angle_correction: Error data not set!\n");
        return false;
    }

    uint64 proj_sz = uint64(N_u1) * uint64(N_u0);
    uint64 oblique_sz = uint64(N_v0) * proj_sz;

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < N_v1; i++)
    {
        float* data3D = &data[uint64(i) * oblique_sz];
        for (int j = 0; j < N_v0; j++)
        {
            float* aProj = &data3D[uint64(j) * proj_sz];
            float c = solid_angle_correction_factor(i, j);
            if (doInverse)
                c = 1.0 / c;
            for (int k = 0; k < N_u1; k++)
            {
                for (int l = 0; l < N_u0; l++)
                    aProj[k * N_u0 + l] *= c;
            }
        }
    }
    return true;
}

bool planogram::apply_planogram_weight(bool doInverse)
{
    if (data == nullptr)
    {
        printf("planogram::apply_planogram_weight: Error data not set!\n");
        return false;
    }

    uint64 proj_sz = uint64(N_u1) * uint64(N_u0);
    uint64 oblique_sz = uint64(N_v0) * proj_sz;

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < N_v1; i++)
    {
        float* data3D = &data[uint64(i) * oblique_sz];
        for (int j = 0; j < N_v0; j++)
        {
            float* aProj = &data3D[uint64(j) * proj_sz];
            float c = planogram_weight(i, j);
            if (doInverse)
                c = 1.0 / c;
            for (int k = 0; k < N_u1; k++)
            {
                for (int l = 0; l < N_u0; l++)
                    aProj[k*N_u0+l] *= c;
            }
        }
    }
    return true;
}