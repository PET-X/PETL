#ifndef PLANOGRAM_H
#define PLANOGRAM_H

#ifdef WIN32
#pragma once
#endif

#include "petl_defines.h"
#include "data_cube.h"

class planogram: public dataCube
{
public:
	planogram();
    ~planogram();

    planogram& operator = (const planogram& other);
    void assign(const planogram& other);

    void clearAll();
    bool init(float psi_in, float R_in, float L_in, float H_in, float v_m0_in, float v_m1_in, float T, float* data_ptr = nullptr);
    bool defined(bool doPrint=false);

    bool is_within_azimuthal_acceptable_angle(float* r);
    bool is_within_acceptable_angle(float* r, float& v_1, float& v_0);
    bool get_u_coords(float* p, float v_1, float v_0, float& u_1, float& u_0);

    bool reduce_dimension(int, int);
    bool collapse_dimension(int);

    float u0(int);
    float u1(int);
    float v0(int);
    float v1(int);

    float u0_inv(float);
    float u1_inv(float);
    float v0_inv(float);
    float v1_inv(float);
    
    float psi; // panel rotation angle
    float R; // half panel separation
    float L; // half panel length
    float H; // half panel height

    float v_m0; // v_0 max value
    float v_m1; // v_1 max value

    int N_u0, N_u1, N_v0, N_v1;
    float T_u0, T_u1, T_v0, T_v1;
    float u0_0, u1_0, v0_0, v1_0;

    float planogram_weight(float, float);
    float planogram_weight(int, int);

    float solid_angle_correction_factor(float v1_val, float v0_val);
    float solid_angle_correction_factor(int iv1, int iv0);

    bool apply_solid_angle_correction(bool doInverse = false);
    bool apply_planogram_weight(bool doInverse = false);

    void set_data(float*);

//private:
    float cos_psi;
    float sin_psi;
};

#endif
