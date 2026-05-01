#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "petl_c_interface.h"
#include "list_of_models.h"
#include "data_cube.h"
#include "petl.h"
#include "total_variation.cuh"
#include "relative_differences.cuh"

listOfModels list_models;
int whichModel = 0;

bool set_model(int i)
{
	whichModel = i;
	return true;
}

int create_new_model()
{
	whichModel = list_models.append();
	return whichModel;
}

PETL* tomo()
{
	return list_models.get(whichModel);
}

void about()
{
    tomo()->about();
}

bool copy_parameters(int param_id)
{
	if (0 <= param_id && param_id < list_models.size())
	{
		if (whichModel != param_id)
			tomo()->params.assign(list_models.get(param_id)->params);
		return true;
	}
	else
		return false;
}

void clearAll()
{
    tomo()->clearAll();
}

void print_parameters()
{
    tomo()->params.printAll();
}

bool volume_defined()
{
    return tomo()->params.volumeDefined();
}

bool geometry_defined()
{
    return tomo()->params.geometryDefined();
}

int get_numX()
{
    return tomo()->params.numX;
}

bool set_numX(int N)
{
    if (N > 0)
    {
        tomo()->params.numX = N;
        return true;
    }
    else
        return false;
}

int get_numY()
{
    return tomo()->params.numY;
}

bool set_numY(int N)
{
    if (N > 0)
    {
        tomo()->params.numY = N;
        return true;
    }
    else
        return false;
}

int get_numZ()
{
    return tomo()->params.numZ;
}

bool set_numZ(int N)
{
    if (N > 0)
    {
        tomo()->params.numZ = N;
        return true;
    }
    else
        return false;
}

float get_voxelWidth()
{
    return tomo()->params.voxelWidth;
}

bool set_voxelWidth(float w)
{
    if (w > 0.0)
    {
        tomo()->params.voxelWidth = w;
        return true;
    }
    else
        return false;
}

float get_voxelHeight()
{
    return tomo()->params.voxelHeight;
}

bool set_voxelHeight(float h)
{
    if (h > 0.0)
    {
        tomo()->params.voxelHeight = h;
        return true;
    }
    else
        return false;
}

float get_offsetX()
{
    return tomo()->params.offsetX;
}

bool set_offsetX(float x_0)
{
    if (x_0 > 0.0)
    {
        tomo()->params.offsetX = x_0;
        return true;
    }
    else
        return false;
}

float get_offsetY()
{
    return tomo()->params.offsetY;
}

bool set_offsetY(float y_0)
{
    if (y_0 > 0.0)
    {
        tomo()->params.offsetY = y_0;
        return true;
    }
    else
        return false;
}

float get_offsetZ()
{
    return tomo()->params.offsetZ;
}

bool set_offsetZ(float z_0)
{
    if (z_0 > 0.0)
    {
        tomo()->params.offsetZ = z_0;
        return true;
    }
    else
        return false;
}

int get_numPlanograms()
{
    return int(tomo()->params.planogramSet.size());
}

bool get_planogramSize(int which, int* shape)
{
    if (!geometry_defined() || which < 0 || which >= get_numPlanograms() || shape == NULL)
        return false;
    else
    {
        shape[0] = tomo()->params.planogramSet[which]->N_v1;
        shape[1] = tomo()->params.planogramSet[which]->N_v0;
        shape[2] = tomo()->params.planogramSet[which]->N_u1;
        shape[3] = tomo()->params.planogramSet[which]->N_u0;
        return true;
    }
}

bool add_planogram(float psi, float R, float L, float H, float v_m0, float v_m1, float T)
{
    return tomo()->add_planogram(psi*PI/180.0, 0.5*R, 0.5*L, 0.5*H, v_m0, v_m1, T);
}

bool remove_planogram(int which)
{
    return tomo()->remove_planogram(which);
}

bool keep_only_planogram(int which)
{
    return tomo()->keep_only_planogram(which);
}

bool set_default_volume(float scale)
{
    return tomo()->set_default_volume(scale);
}

bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    return tomo()->set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool bin(float** g, char* file_name)
{
    return tomo()->bin(g, file_name);
}

bool simulate_scatter(float** g, float* mu)
{
    return tomo()->simulate_scatter(g, mu);
}

bool project(float** g, float* f)
{
    return tomo()->project(g, f);
}

bool backproject(float** g, float* f)
{
    return tomo()->backproject(g, f);
}

bool FBP(float** g, float* f)
{
    return tomo()->doFBP(g, f);
}

bool PFDR(float** g, float** g_reb)
{
    return tomo()->doPFDR(g, g_reb);
}

bool add_object(int type, float* c, float* r, float val, float* A, float* clip)
{
    return tomo()->geometricPhantom.addObject(type, c, r, val, A, clip);
}

bool add_module(int type, float* c, float* r, float val, float* A, float* clip)
{
    return tomo()->modules.addObject(type, c, r, val, A, clip);
}

bool clear_phantom()
{
    tomo()->geometricPhantom.clearObjects();
	return true;
}

bool clear_modules()
{
    tomo()->modules.clearObjects();
    return true;
}

bool scale_phantom(float scale_x, float scale_y, float scale_z)
{
    return tomo()->geometricPhantom.scale_phantom(scale_x, scale_y, scale_z);
}

bool voxelize(float* f, int oversampling)
{
    return tomo()->geometricPhantom.voxelize(f, &(tomo()->params), oversampling);
}

bool ray_trace(float** g, int oversampling)
{
    return tomo()->ray_trace(g, oversampling);
}

bool stopping_power(float** g, int oversampling)
{
    return tomo()->stopping_power(g, oversampling);
}

bool multiply3D(float* x, float* y, int N_1, int N_2, int N_3)
{
    if (x == nullptr || y == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
    {
        printf("multiply invalid inputs\n");
        return false;
    }
    dataCube X(x, N_1, N_2, N_3);
    dataCube Y(y, N_1, N_2, N_3);
    if (X.multiply(&Y) == nullptr)
        return false;
    else
        return true;
}

bool multiply4D(float* x, float* y, int N_1, int N_2, int N_3, int N_4)
{
    if (x == nullptr || y == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || N_4 <= 0)
    {
        printf("multiply invalid inputs\n");
        return false;
    }
    dataCube X(x, N_1, N_2, N_3, N_4);
    dataCube Y(y, N_1, N_2, N_3, N_4);
    if (X.multiply(&Y) == nullptr)
        return false;
    else
        return true;
}

bool divide3D(float* x, float* y, int N_1, int N_2, int N_3)
{
    if (x == nullptr || y == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
    {
        printf("divide invalid inputs\n");
        return false;
    }
    dataCube X(x, N_1, N_2, N_3);
    dataCube Y(y, N_1, N_2, N_3);
    if (X.divide(&Y) == nullptr)
        return false;
    else
        return true;
}

bool divide4D(float* x, float* y, int N_1, int N_2, int N_3, int N_4)
{
    if (x == nullptr || y == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || N_4 <= 0)
    {
        printf("divide invalid inputs\n");
        return false;
    }
    dataCube X(x, N_1, N_2, N_3, N_4);
    dataCube Y(y, N_1, N_2, N_3, N_4);
    if (X.divide(&Y) == nullptr)
        return false;
    else
        return true;
}

bool rdivide3D(float* num, float* denom, int N_1, int N_2, int N_3)
{
    if (num == nullptr || denom == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
    {
        printf("rdivide invalid inputs\n");
        return false;
    }
    dataCube x(num, N_1, N_2, N_3);
    dataCube y(denom, N_1, N_2, N_3);
    //denom = num / denom
    if (y.rdivide(&x) == nullptr)
        return false;
    else
        return true;
}

bool rdivide4D(float* num, float* denom, int N_1, int N_2, int N_3, int N_4)
{
    if (num == nullptr || denom == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || N_4 <= 0)
    {
        printf("rdivide invalid inputs\n");
        return false;
    }
    dataCube x(num, N_1, N_2, N_3, N_4);
    dataCube y(denom, N_1, N_2, N_3, N_4);
    //denom = num / denom
    if (y.rdivide(&x) == nullptr)
        return false;
    else
        return true;
}

bool reciprocal3D(float* x, int N_1, int N_2, int N_3, float divide_by_zero_value)
{
    if (x == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
    {
        printf("rdivide invalid inputs\n");
        return false;
    }
    dataCube X(x, N_1, N_2, N_3);
    if (X.reciprocal(divide_by_zero_value) == nullptr)
        return false;
    else
        return true;
}

bool reciprocal4D(float* x, int N_1, int N_2, int N_3, int N_4, float divide_by_zero_value)
{
    if (x == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || N_4 <= 0)
    {
        printf("rdivide invalid inputs\n");
        return false;
    }
    dataCube X(x, N_1, N_2, N_3, N_4);
    if (X.reciprocal(divide_by_zero_value) == nullptr)
        return false;
    else
        return true;
}

float inner_product3D(float* x, float* y, float* w, int N_1, int N_2, int N_3)
{
    if (x == nullptr || y == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
    {
        printf("inner_product invalid inputs\n");
        return 0.0;
    }

    dataCube X(x, N_1, N_2, N_3);
    dataCube Y(y, N_1, N_2, N_3);
    if (w == nullptr)
        return X.innerProduct(&Y);
    else
    {
        dataCube W(w, N_1, N_2, N_3);
        return X.innerProduct(&Y, &W);
    }
}

float inner_product4D(float* x, float* y, float* w, int N_1, int N_2, int N_3, int N_4)
{
    if (x == nullptr || y == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || N_4 <= 0)
    {
        printf("inner_product invalid inputs\n");
        return 0.0;
    }

    dataCube X(x, N_1, N_2, N_3, N_4);
    dataCube Y(y, N_1, N_2, N_3, N_4);
    if (w == nullptr)
        return X.innerProduct(&Y);
    else
    {
        dataCube W(w, N_1, N_2, N_3, N_4);
        return X.innerProduct(&Y, &W);
    }
}

bool set_solid_angle_correction(float** g, bool do_inverse)
{
    if (g == nullptr)
        return false;
    else
        return tomo()->set_solid_angle_correction(g, do_inverse);
}

bool apply_corrections(float** g, float** r, float threshold)
{
    if (g == nullptr || r == nullptr)
        return false;
    else
        return tomo()->apply_corrections(g, r, threshold);
}

float TV_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
    //return TVcost(f, N_1, N_2, N_3, delta, beta, p, data_on_cpu);
    return anisotropicTotalVariation_cost(f, N_1, N_2, N_3, delta, beta, p, data_on_cpu, tomo()->params.whichGPU);
}

bool TV_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
    //return TVgradient(f, Df, N_1, N_2, N_3, delta, beta, p, false, data_on_cpu);
    return anisotropicTotalVariation_gradient(f, Df, N_1, N_2, N_3, delta, beta, p, data_on_cpu, tomo()->params.whichGPU);
}

float TV_quadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
    //return TVquadForm(f, d, N_1, N_2, N_3, delta, beta, p, data_on_cpu);
    return anisotropicTotalVariation_quadraticForm(f, d, N_1, N_2, N_3, delta, beta, p, data_on_cpu, tomo()->params.whichGPU);
}

bool diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu)
{
    //return Diffuse(f, N_1, N_2, N_3, delta, p, numIter, data_on_cpu);
    return Diffuse(f, N_1, N_2, N_3, delta, p, numIter, data_on_cpu, tomo()->params.whichGPU);
}

bool TV_denoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool doMean, bool data_on_cpu)
{
    //return TVdenoise(f, N_1, N_2, N_3, delta, beta, p, numIter, doMean, data_on_cpu);
    return TVdenoise(f, N_1, N_2, N_3, delta, beta, p, numIter, data_on_cpu, tomo()->params.whichGPU);
}

bool relativeDifferences_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta)
{
    if (f == nullptr || Df == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
        return false;
    else
        return aTV_RelativeDifferencesLoss_gradient(f, Df, N_1, N_2, N_3, delta, beta, tomo()->params.whichGPU);
}

bool gaussian_filter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims)
{
    if (f == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
        return false;
    else
        return GaussianFilter(f, N_1, N_2, N_3, FWHM, numDims, tomo()->params.whichGPU);
}
