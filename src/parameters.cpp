#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <omp.h>
#include "parameters.h"
#include "cuda_utils.cuh"

using namespace std;

parameters::parameters()
{
	initialize();
}

void parameters::initialize()
{
    numX = 0;
	numY = 0;
	numZ = 0;
	voxelHeight = 0.0;
	voxelWidth = 0.0;
	offsetX = 0.0;
	offsetY = 0.0;
	offsetZ = 0.0;

    whichGPUs.clear();
	int numGPUs = numberOfGPUs();
	if (numGPUs > 0)
	{
		whichGPU = 0;
		for (int i = 0; i < numGPUs; i++)
			whichGPUs.push_back(i);
	}
	else
		whichGPU = -1;

    listModeUnits = MM; //CM;
	listModeOrigin = CHEST_WALL;
    binningType = PRIMARY;
    listModeDataIsWeighted = false;
    binningOrder = 1;
}

void parameters::printAll()
{
    printf("\n");

    printf("======== PET Volume ========\n");
	printf("number of voxels (x, y, z): %d x %d x %d\n", numX, numY, numZ);
	printf("voxel size: %f mm x %f mm x %f mm\n", voxelWidth, voxelWidth, voxelHeight);
	if (offsetX != 0.0 || offsetY != 0.0 || offsetZ != 0.0)
		printf("volume offset: %f mm, %f mm, %f mm\n", offsetX, offsetY, offsetZ);
	printf("FOV: [%f, %f] x [%f, %f] x [%f, %f]\n", x_0()-0.5*voxelWidth, (numX-1)*voxelWidth+x_0()+0.5*voxelWidth, y_0()- 0.5 * voxelWidth, (numY - 1) * voxelWidth + y_0()+ 0.5 * voxelWidth, z_0()- 0.5 * voxelHeight, (numZ - 1) * voxelHeight + z_0()+ 0.5 * voxelHeight);
	//printf("x_0 = %f, y_0 = %f, z_0 = %f\n", x_0(), y_0(), z_0());
	printf("\n");

    printf("======== PET Geometry ========\n");
    for (int i = 0; i < int(planogramSet.size()); i++)
    {
        planogram* plan = planogramSet[i];
        printf("Planogram %d\n", i);
        printf("  angle = %f deg\n", plan->psi*180.0/PI);
        printf("  number of samples: %d x %d x %d x %d\n", plan->N_v1, plan->N_v0, plan->N_u1, plan->N_u0);
        printf("  detector size: %f mm x %f mm\n", plan->T_u1, plan->T_u0);
        printf("  full detector length: %f mm\n", 2.0*plan->L);
        printf("  full detector height: %f mm\n", 2.0*plan->H);
        printf("  distance between panels: %f mm\n", 2.0*plan->R);
    }
    printf("\n");

    printf("======== Settings ========\n");
    if (whichGPUs.size() > 0)
    {
        printf("GPUs: ");
        for (int i = 0; i < int(whichGPUs.size()); i++)
            printf("%d ", whichGPUs[i]);
        printf("\n");
    }
    else if (whichGPU >= 0)
    {
        printf("GPUs: %d\n", whichGPU);
    }
    printf("%d CPU threads\n", omp_get_num_procs());
    printf("\n");
}

void parameters::clearAll()
{
    numX = 0;
    numY = 0;
    numZ = 0;

    voxelWidth = 0.0;
    voxelHeight = 0.0;

    offsetX = 0.0;
    offsetY = 0.0;
    offsetZ = 0.0;
    
    clearPlanograms();
}

void parameters::clearPlanograms()
{
    for (int i = 0; i < int(planogramSet.size()); i++)
    {
        if (planogramSet[i] != nullptr)
            delete planogramSet[i];
        planogramSet[i] = NULL;
    }
    planogramSet.clear();
}

parameters::parameters(const parameters& other)
{
	initialize();
    assign(other);
}

parameters::~parameters()
{
    clearAll();
}

parameters& parameters::operator = (const parameters& other)
{
    if (this != &other)
        this->assign(other);
    return *this;
}

void parameters::assign(const parameters& other, float** g)
{
    this->clearAll();

    this->numX = other.numX;
    this->numY = other.numY;
    this->numZ = other.numZ;
    this->voxelWidth = other.voxelWidth;
    this->voxelHeight = other.voxelHeight;
    this->offsetX = other.offsetX;
    this->offsetY = other.offsetY;
    this->offsetZ = other.offsetZ;

    for (int i = 0; i < int(other.planogramSet.size()); i++)
    {
        planogram* plan = new planogram;
        plan->assign(*(other.planogramSet[i]));
        if (g != nullptr)
            plan->data = g[i];
        else
            plan->data = nullptr;
        this->planogramSet.push_back(plan);
    }

    this->whichGPU = other.whichGPU;
    this->whichGPUs.clear();
	for (int i = 0; i < int(other.whichGPUs.size()); i++)
		this->whichGPUs.push_back(other.whichGPUs[i]);

    this->listModeUnits = other.listModeUnits;
    this->listModeOrigin = other.listModeOrigin;
    this->binningType = other.binningType;
    this->listModeDataIsWeighted = other.listModeDataIsWeighted;
    this->binningOrder = other.binningOrder;
}

bool parameters::allDefined(bool doPrint)
{
    return geometryDefined(doPrint) & volumeDefined(doPrint);
}

bool parameters::geometryDefined(bool doPrint)
{
    if (planogramSet.size() == 0)
        return false;
    else
    {
        for (int i = 0; i < int(planogramSet.size()); i++)
        {
            if (!planogramSet[i]->defined(doPrint))
                return false;
        }
        return true;
    }
}

bool parameters::volumeDefined(bool doPrint)
{
    if (numX <= 0 || numY <= 0 || numZ <= 0 || voxelWidth <= 0.0 || voxelHeight <= 0.0)
	{
		if (doPrint)
		{
			printf("numZ = %d voxelHeight = %f\n", numZ, voxelHeight);
			printf("Error: volume voxel sizes and number of data elements must be positive\n");
		}
		return false;
	}
    else
        return true;
}

bool parameters::set_default_volume(float scale)
{
    if (!geometryDefined())
        return false;
    
    offsetX = 0.0;
    offsetY = 0.0;
    offsetZ = 0.0;
    float X = 0.0;
    float Y = 0.0;
    float Z = 0.0;
    for (int i = 0; i < int(planogramSet.size()); i++)
    {
        float W = planogramSet[i]->L; // x
        float H = planogramSet[i]->R; // y
        float psi = planogramSet[i]->psi;
        float X_new = fabs(W*cos(psi)) + fabs(H*sin(psi));
        float Y_new = fabs(W*sin(psi)) + fabs(H*cos(psi));
        if (i == 0)
        {
            voxelWidth = planogramSet[i]->T_u0;
            voxelHeight = planogramSet[i]->T_u1;
            Z = planogramSet[i]->H;
            X = X_new;
            Y = Y_new;
        }
        else
        {
            voxelWidth = min(voxelWidth, planogramSet[i]->T_u0);
            voxelHeight = min(voxelHeight, planogramSet[i]->T_u1);
            Z = min(Z, planogramSet[i]->H);
            X = min(X, X_new);
            Y = min(Y, Y_new);
        }
    }
    voxelWidth *= scale;
    voxelHeight *= scale;
    
    numZ = 2 * int(Z / voxelHeight);
    numY = 2 * int(Y / voxelWidth);
    numX = 2 * int(X / voxelWidth);

    return true;
}

bool parameters::set_volume(int numX_in, int numY_in, int numZ_in, float voxelWidth_in, float voxelHeight_in, float offsetX_in, float offsetY_in, float offsetZ_in)
{
    if (numX_in > 0 && numY_in > 0 && numZ_in > 0 && voxelWidth_in > 0.0 && voxelHeight_in > 0.0)
    {
        numX = numX_in;
        numY = numY_in;
        numZ = numZ_in;

        voxelWidth = voxelWidth_in;
        voxelHeight = voxelHeight_in;

        offsetX = offsetX_in;
        offsetY = offsetY_in;
        offsetZ = offsetZ_in;

        return true;
    }
    else
        return false;
}

float parameters::x_0()
{
    return (0.0 - 0.5*(numX-1))*voxelWidth + offsetX;
}

float parameters::y_0()
{
    return (0.0 - 0.5*(numY-1))*voxelWidth + offsetY;
}

float parameters::z_0()
{
    return (0.0 - 0.5*(numZ-1))*voxelHeight + offsetZ;
}

int parameters::num_planograms()
{
    return int(planogramSet.size());
}

bool parameters::add_planogram(float psi, float R, float L, float H, float v_m0, float v_m1, float T)
{
    planogram* plan = new planogram;
    if (plan->init(psi, R, L, H, v_m0, v_m1, T))
    {
        planogramSet.push_back(plan);
        return true;
    }
    else
    {
        delete plan;
        return false;
    }
}

bool parameters::remove_planogram(int which)
{
    if (which >= 0 && which < planogramSet.size())
    {
        planogram* plan = planogramSet[which];
        if (plan != nullptr)
            delete plan;
        planogramSet[which] = nullptr;
        planogramSet.erase(planogramSet.begin() + which);
        return true;
    }
    else
        return false;
}

bool parameters::keep_only_planogram(int which)
{
    if (which >= 0 && which < planogramSet.size())
    {
        planogram* plan = planogramSet[which];
        planogramSet[which] = nullptr;
        clearPlanograms();
        planogramSet.push_back(plan);
        return true;
    }
    else
        return false;
}

uint64 parameters::projectionData_numberOfElements(int iplan)
{
    if (planogramSet.size() == 0 || iplan >= num_planograms())
        return 0;
    else
    {
        planogram* plan = planogramSet[iplan];
        return uint64(plan->N_v1) * uint64(plan->N_v0) * uint64(plan->N_u1) * uint64(plan->N_u0);
    }
}

uint64 parameters::volumeData_numberOfElements()
{
    return uint64(numZ) * uint64(numY) * uint64(numX); 
}
