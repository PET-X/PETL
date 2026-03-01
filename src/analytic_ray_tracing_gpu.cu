////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include "analytic_ray_tracing_gpu.cuh"
#include "analytic_ray_tracing.h"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cuda_utils.cuh"

#ifndef PI
#define PI 3.141592653589793f
#endif

#ifndef OUT_OF_BOUNDS
#define OUT_OF_BOUNDS 1.0e12f;
#endif

//enum geometry_list { CONE = 0, PARALLEL = 1, FAN = 2, MODULAR = 3, CONE_PARALLEL = 4 };

__constant__ int d_oversampling;
__constant__ float d_psi;
__constant__ float d_cos_psi;
__constant__ float d_sin_psi;
__constant__ float d_R;
__constant__ float d_L;
__constant__ float d_H;
__constant__ int4 d_N_g;
__constant__ float4 d_T_g;
__constant__ float4 d_startVal_g;

//enum objectType_list { ELLIPSOID = 0, PARALLELEPIPED = 1, CYLINDER_X = 2, CYLINDER_Y = 3, CYLINDER_Z = 4, CONE_X = 5, CONE_Y = 6, CONE_Z = 7 };
#define d_ELLIPSOID 0
#define d_PARALLELEPIPED 1
#define d_CYLINDER_X 2
#define d_CYLINDER_Y 3
#define d_CYLINDER_Z 4
#define d_CONE_X  5
#define d_CONE_Y 6
#define d_CONE_Z 7

__device__ float v1(const int i)
{
	return float(i) * d_T_g.w + d_startVal_g.w;
}

__device__ float v0(const int i)
{
	return float(i) * d_T_g.x + d_startVal_g.x;
}

__device__ float u1(const int i)
{
    return float(i) * d_T_g.y + d_startVal_g.y;
}

__device__ float u0(const int i)
{
	return float(i) * d_T_g.z + d_startVal_g.z;
}

__device__ float3 setSourcePosition(const int iv1, const int iv0, const int iu1, const int iu0, const float du1, const float du0)
{
	return make_float3((u0(iu0) + du0) * d_cos_psi, (u0(iu0) + du0) * d_sin_psi, u1(iu1) + du1);
}

__device__ float3 setTrajectory(const int iv1, const int iv0, const int iu1, const int iu0)
{
	float3 r = make_float3(-(v0(iv0) * d_cos_psi + d_sin_psi), -(v0(iv0) * d_sin_psi - d_cos_psi), -v1(iv1));

	const float mag = rsqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
	r.x = r.x / mag;
	r.y = r.y / mag;
	r.z = r.z / mag;
	return r;
}

__device__ bool parametersOfIntersection_1D(float2& ts, float p, float r)
{
	// finds ts such that p+t*r = +-1
	if (fabs(r) < 1e-12f)
	{
		if (fabs(p) < 1.0f)
		{
			ts.x = -OUT_OF_BOUNDS;
			ts.y = OUT_OF_BOUNDS;

			return true;
		}
		else
			return false;
	}
	else
	{
		if (r > 0.0f)
		{
			ts.x = (-1.0f - p) / r;
			ts.y = (1.0f - p) / r;
		}
		else
		{
			ts.y = (-1.0f - p) / r;
			ts.x = (1.0f - p) / r;
		}
		return true;
	}
}


__device__ bool parametersOfClippingPlaneIntersections(float2& ts, float3 p, float3 r, geometricSolid* solid)
{
	for (int i = 0; i < solid->numClippingPlanes; i++)
	{
		const float p_dot_n = solid->clippingPlanes[i][0] * p.x + solid->clippingPlanes[i][1] * p.y + solid->clippingPlanes[i][2] * p.z;
		const float r_dot_n = solid->clippingPlanes[i][0] * r.x + solid->clippingPlanes[i][1] * r.y + solid->clippingPlanes[i][2] * r.z;
		if (fabs(r_dot_n) < 1.0e-12f)
		{
			if (p_dot_n < solid->clippingPlanes[i][3])
				return false;
		}
		else if (r_dot_n > 0.0f)
		{
			const float temp = (solid->clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			// restriction: t > temp
			if (ts.y < temp)
				return false;
			if (temp > ts.x)
				ts.x = temp;
		}
		else
		{
			// restriction: t < temp
			const float temp = (solid->clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			if (ts.x > temp)
				return false;
			if (ts.y > temp)
				ts.y = temp;
		}
	}

	return true;
}

__device__ bool intersectionEndPoints_centeredAndNormalized(double3& p, double3& r, float2& ts, geometricSolid* solid)
{
	ts.x = -OUT_OF_BOUNDS;
	ts.y = ts.x;

	// r != (0,0,1)
	const double r_dot_r = r.x * r.x + r.y * r.y + r.z * r.z;
	const double p_dot_r = r.x * p.x + r.y * p.y + r.z * p.z;

	if (solid->type == d_ELLIPSOID)
	{
		double disc = p_dot_r * p_dot_r + r_dot_r * (1.0 - (p.x * p.x + p.y * p.y + p.z * p.z));
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			ts.x = (-p_dot_r - disc) / r_dot_r;
			ts.y = (-p_dot_r + disc) / r_dot_r;
		}
		else
			return false;
	}
	else if (solid->type == d_PARALLELEPIPED)
	{
		float2 tx;
		float2 ty;
		float2 tz;
		if (parametersOfIntersection_1D(tx, p.x, r.x) == true)
		{
			if (parametersOfIntersection_1D(ty, p.y, r.y) == true)
			{
				if (parametersOfIntersection_1D(tz, p.z, r.z) == true)
				{
					ts.x = max(max(tx.x, ty.x), tz.x);
					ts.y = min(min(tx.y, ty.y), tz.y);
				}
				else
					return false;
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (solid->type == d_CYLINDER_Z)
	{
		//double r_dot_r_2D = r.x * r.x + r.y * r.y; // 3
		const double r_dot_r_2D = r_dot_r - r.z * r.z; // 2
		const double p_dor_r_2D = p_dot_r - p.z * r.z; // 2
		double disc = p_dor_r_2D * p_dor_r_2D - r_dot_r_2D * (p.x * p.x + p.y * p.y - 1.0); // 7
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			const float tmin = (-p_dor_r_2D - disc) / r_dot_r_2D; // 2
			const float tmax = (-p_dor_r_2D + disc) / r_dot_r_2D; // 2

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.z, r.z) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.x == 0.0f && r.y == 0.0f && p.x * p.x + p.y * p.y <= 1.0f)
		{
			return parametersOfIntersection_1D(ts, p.z, r.z);
		}
		else
			return false;
	}
	else if (solid->type == d_CYLINDER_X) // ellipsoidal cross sections parallel to x-y axis
	{
		const double r_dot_r_2D = r_dot_r - r.x * r.x;
		double disc = (p.z * r.z + p.y * r.y) * (p.z * r.z + p.y * r.y) - r_dot_r_2D * (p.z * p.z + p.y * p.y - 1.0);
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			const float tmin = (-(p.z * r.z + r.y * p.y) - disc) / r_dot_r_2D;
			const float tmax = (-(p.z * r.z + r.y * p.y) + disc) / r_dot_r_2D;

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.x, r.x) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.y == 0.0 && r.z == 0.0 && p.y * p.y + p.z * p.z <= 1.0)
		{
			return parametersOfIntersection_1D(ts, p.x, r.x);
		}
		else
			return false;
	}
	else if (solid->type == d_CYLINDER_Y) // ellipsoidal cross sections parallel to x-y axis
	{
		const double r_dot_r_2D = r_dot_r - r.y * r.y;
		double disc = (p.x * r.x + p.z * r.z) * (p.x * r.x + p.z * r.z) - r_dot_r_2D * (p.x * p.x + p.z * p.z - 1.0);
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			const float tmin = (-(p.x * r.x + r.z * p.z) - disc) / r_dot_r_2D;
			const float tmax = (-(p.x * r.x + r.z * p.z) + disc) / r_dot_r_2D;

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.y, r.y) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.x == 0.0 && r.z == 0.0 && p.x * p.x + p.z * p.z <= 1.0)
		{
			return parametersOfIntersection_1D(ts, p.y, r.y);
		}
		else
			return false;
	}
	else if (solid->type == d_CONE_Z)
	{
		double a = r.x * r.x + r.y * r.y - r.z * r.z;
		const double b_half = p.x * r.x + p.y * r.y - p.z * r.z;
		const double c = p.x * p.x + p.y * p.y - p.z * p.z;
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			float tmin = (-b_half - disc) / a;
			float tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			const double theShift = 0.5 * (solid->clipCone.y + solid->clipCone.x);
			const double theScale = 0.5 * (solid->clipCone.y - solid->clipCone.x);

			float2 tz;
			if (parametersOfIntersection_1D(tz, (p.z - theShift) / theScale, r.z / theScale) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (solid->type == d_CONE_X)
	{
		double a = r.z * r.z + r.y * r.y - r.x * r.x;
		const double b_half = p.z * r.z + p.y * r.y - p.x * r.x;
		const double c = p.z * p.z + p.y * p.y - p.x * p.x;
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			float tmin = (-b_half - disc) / a;
			float tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			const double theShift = 0.5 * (solid->clipCone.y + solid->clipCone.x);
			const double theScale = 0.5 * (solid->clipCone.y - solid->clipCone.x);

			float2 tz;
			if (parametersOfIntersection_1D(tz, (p.x - theShift) / theScale, r.x / theScale) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (solid->type == d_CONE_Y)
	{
		double a = r.x * r.x + r.z * r.z - r.y * r.y;
		const double b_half = p.x * r.x + p.z * r.z - p.y * r.y;
		const double c = p.x * p.x + p.z * p.z - p.y * p.y;
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			double tmin = (-b_half - disc) / a;
			double tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			const double theShift = 0.5 * (solid->clipCone.y + solid->clipCone.x);
			const double theScale = 0.5 * (solid->clipCone.y - solid->clipCone.x);

			float2 tz;
			if (parametersOfIntersection_1D(tz, (p.y - theShift) / theScale, r.y / theScale) == true)
			{
				if (fabs(r.y) > 1.0e-12f)
				{
					bool isInside_0 = false;
					double x_val, y_val, z_val;
					x_val = p.x + tz.x * r.x; x_val *= x_val;
					y_val = p.y + tz.x * r.y; y_val *= y_val;
					z_val = p.z + tz.x * r.z; z_val *= z_val;
					if (x_val + z_val <= y_val)
						isInside_0 = true;

					bool isInside_1 = false;
					x_val = p.x + tz.y * r.x; x_val *= x_val;
					y_val = p.y + tz.y * r.y; y_val *= y_val;
					z_val = p.z + tz.y * r.z; z_val *= z_val;
					if (x_val + z_val <= y_val)
						isInside_1 = true;
					if (isInside_0 == true)
					{
						if (isInside_1 == true)
						{
							//ts.x = max(tmin, tz.x);
							//ts.y = min(tmax, tz.y);
							ts.x = tz.x;
							ts.y = tz.y;
						}
						else
						{
							//insiders: tz.x, tmin, tmax
							if (tz.x <= tmin && tmin <= tz.y)
							{
								ts.x = tz.x;
								ts.y = tmin;
							}
							else
							{
								ts.x = tz.x;
								ts.y = tmax;
							}
						}
					}
					else
					{
						if (isInside_1 == true)
						{
							//insiders: tz.y, tmin, tmax
							if (tz.x <= tmin && tmin <= tz.y)
							{
								ts.x = tmin;
								ts.y = tz.y;
							}
							else
							{
								ts.x = tmax;
								ts.y = tz.y;
							}
						}
						else
						{
							// insiders: tmin, tmax
							if (tz.x <= tmin && tmax <= tz.y)
							{
								ts.x = tmin;
								ts.y = tmax;
							}
							else
							{
								ts.x = -OUT_OF_BOUNDS;
								ts.y = ts.x;
								return false;
							}
						}
					}
				}
				else
				{
					ts.x = tmin;
					ts.y = tmax;
				}
			}
		}
	}
	else
	{
		ts.x = -OUT_OF_BOUNDS;
		ts.y = -OUT_OF_BOUNDS;
		return false;
	}

	if (ts.x >= ts.y)
	{
		ts.x = -OUT_OF_BOUNDS;
		ts.y = ts.x;
		return false;
	}

	return true;
}

__device__ bool intersectionEndPoints(float3& p, float3& r, float2& ts, geometricSolid* solid)
{
    // assume ||r|| == 1 && r != (0,0,1) && axis[i] > 0 for i=0,1,2
    // alpha is rotation around x-y axis, currently there is no rotation for x-z or y-z axes
    double3 q;
	double3 Minv_r;

    if (solid->isRotated == false)
    {
        // Scale; (9,0) ops
        q.x = (p.x - solid->centers.x) / solid->radii.x;
        q.y = (p.y - solid->centers.y) / solid->radii.y;
        q.z = (p.z - solid->centers.z) / solid->radii.z;

        Minv_r.x = r.x / solid->radii.x;
        Minv_r.y = r.y / solid->radii.y;
        Minv_r.z = r.z / solid->radii.z;
    }
    else
    {
		double3 temp;

        // Shift
        q.x = p.x - solid->centers.x;
        q.y = p.y - solid->centers.y;
        q.z = p.z - solid->centers.z;

        // Rotate and Scale; (36, 0) ops
        temp.x = (q.x * solid->A[0 * 3 + 0] + q.y * solid->A[0 * 3 + 1] + q.z * solid->A[0 * 3 + 2]) / solid->radii.x;
        temp.y = (q.x * solid->A[1 * 3 + 0] + q.y * solid->A[1 * 3 + 1] + q.z * solid->A[1 * 3 + 2]) / solid->radii.y;
        temp.z = (q.x * solid->A[2 * 3 + 0] + q.y * solid->A[2 * 3 + 1] + q.z * solid->A[2 * 3 + 2]) / solid->radii.z;

        q.x = temp.x;
        q.y = temp.y;
        q.z = temp.z;

        Minv_r.x = (r.x * solid->A[0 * 3 + 0] + r.y * solid->A[0 * 3 + 1] + r.z * solid->A[0 * 3 + 2]) / solid->radii.x;
        Minv_r.y = (r.x * solid->A[1 * 3 + 0] + r.y * solid->A[1 * 3 + 1] + r.z * solid->A[1 * 3 + 2]) / solid->radii.y;
        Minv_r.z = (r.x * solid->A[2 * 3 + 0] + r.y * solid->A[2 * 3 + 1] + r.z * solid->A[2 * 3 + 2]) / solid->radii.z;
    }

    if (intersectionEndPoints_centeredAndNormalized(q, Minv_r, ts, solid) == false)
        return false;
    if (parametersOfClippingPlaneIntersections(ts, p, r, solid) == false)
    {
        ts.x = -OUT_OF_BOUNDS;
        ts.y = ts.x;
        return false;
    }
    if (ts.y > ts.x)
        return true;
    else
        return false;
}

__device__ void sort(float* v, int count)
{
	for (int i = 0; i < count; i++)
	{
		for (int j = i + 1; j < count; j++)
		{
			if (v[i] > v[j])
			{  // swap?
				const float tmp = v[i];
				v[i] = v[j];
				v[j] = tmp;
			}
		}
	}
}

__device__ float stoppingPower_geometricSolids(float3 p, float3 r, geometricSolid* solids, const int numObjects, float* floatData, int* intData)
{
	//*
	//vector<float> endPoints;
	//vector<int> objectIndices;
	//float* endPoints = (float*)malloc(size_t(2 * numObjects) * sizeof(float));
	//int* objectIndices = (int*)malloc(size_t(numObjects) * sizeof(int));
	//float* intersection_0 = (float*)malloc(size_t(2*numObjects) * sizeof(float));
	//float* intersection_1 = &intersection_0[numObjects];

	float* endPoints = &floatData[0];
	float* intersection_0 = &floatData[2 * numObjects];
	float* intersection_1 = &floatData[3 * numObjects];
	int* objectIndices = intData;

	int count = 0;
	//float* intersection_0 = &intersections[0];
	//float* intersection_1 = &intersections[numObjects];
	for (int i = 0; i < numObjects; i++)
	{
		float2 ts;
		if (intersectionEndPoints(p, r, ts, &solids[i]))
		{
			endPoints[2 * count + 0] = ts.x;
			endPoints[2 * count + 1] = ts.y;
			//endPoints.push_back(ts.x);
			//endPoints.push_back(ts.y);
			intersection_0[i] = ts.x;
			intersection_1[i] = ts.y;
			//objectIndices.push_back(i);
			objectIndices[count] = i;
			count += 1;
			//printf("intersection: %f to %f\n", ts[0], ts[1]);
		}
		else
		{
			//printf("no intersection (%f, %f)\n", ts[0], ts[1]);
			intersection_0[i] = OUT_OF_BOUNDS;
			intersection_1[i] = OUT_OF_BOUNDS;
		}
	}
	float forward_integral = 0.0f;
	float backward_integral = 0.0f;
	if (count > 0)
	{
		sort(endPoints, 2 * count);
		//sort(endPoints.begin(), endPoints.end());
		for (int i = 0; i < 2 * count - 1; i++)
		{
			// Consider the interval (allPoints[i], allPoints[i+1])
			const float midPoint = (endPoints[i + 1] + endPoints[i]) * 0.5f;
			//for (int j = int(objects.size())-1; j >= 0; j--)
			for (int ind = count - 1; ind >= 0; ind--)
			{
				const int j = objectIndices[ind];
				//if (objects[j].val != 0.0)
				{
					// Find which object this interval belongs to
					if (intersection_0[j] <= midPoint && midPoint <= intersection_1[j])
					{
						//if (isnan(arealDensities[j]))
						//	arealDensities[j] = 0.0;
						//arealDensities[j] += objects[j].val * (endPoints[i + 1] - endPoints[i]);
						//retVal += solids[j].val * (endPoints[i + 1] - endPoints[i]);
						if (midPoint >= 0.0f)
							forward_integral += solids[j].val * (endPoints[i + 1] - endPoints[i]);
						else
							backward_integral += solids[j].val * (endPoints[i + 1] - endPoints[i]);
						break;
					}
				}
			}
		}
	}
	//free(endPoints);
	//free(objectIndices);
	//free(intersection_0);
	return (1.0f - expf(-forward_integral)) * (1.0f - expf(-backward_integral));
	//*/
}

__device__ float lineIntegral_geometricSolids(float3 p, float3 r, geometricSolid* solids, const int numObjects, float* floatData, int* intData)
{
	//*
	//vector<float> endPoints;
	//vector<int> objectIndices;
	//float* endPoints = (float*)malloc(size_t(2 * numObjects) * sizeof(float));
	//int* objectIndices = (int*)malloc(size_t(numObjects) * sizeof(int));
	//float* intersection_0 = (float*)malloc(size_t(2*numObjects) * sizeof(float));
	//float* intersection_1 = &intersection_0[numObjects];

	float* endPoints = &floatData[0];
	float* intersection_0 = &floatData[2 * numObjects];
	float* intersection_1 = &floatData[3 * numObjects];
	int* objectIndices = intData;

	int count = 0;
	//float* intersection_0 = &intersections[0];
	//float* intersection_1 = &intersections[numObjects];
	for (int i = 0; i < numObjects; i++)
	{
		float2 ts;
		if (intersectionEndPoints(p, r, ts, &solids[i]))
		{
			endPoints[2 * count + 0] = ts.x;
			endPoints[2 * count + 1] = ts.y;
			//endPoints.push_back(ts.x);
			//endPoints.push_back(ts.y);
			intersection_0[i] = ts.x;
			intersection_1[i] = ts.y;
			//objectIndices.push_back(i);
			objectIndices[count] = i;
			count += 1;
			//printf("intersection: %f to %f\n", ts[0], ts[1]);
		}
		else
		{
			//printf("no intersection (%f, %f)\n", ts[0], ts[1]);
			intersection_0[i] = OUT_OF_BOUNDS;
			intersection_1[i] = OUT_OF_BOUNDS;
		}
	}
	float retVal = 0.0f;
	if (count > 0)
	{
		sort(endPoints, 2 * count);
		//sort(endPoints.begin(), endPoints.end());
		for (int i = 0; i < 2*count - 1; i++)
		{
			// Consider the interval (allPoints[i], allPoints[i+1])
			const float midPoint = (endPoints[i + 1] + endPoints[i]) * 0.5f;
			//for (int j = int(objects.size())-1; j >= 0; j--)
			for (int ind = count - 1; ind >= 0; ind--)
			{
				const int j = objectIndices[ind];
				//if (objects[j].val != 0.0)
				{
					// Find which object this interval belongs to
					if (intersection_0[j] <= midPoint && midPoint <= intersection_1[j])
					{
						//if (isnan(arealDensities[j]))
						//	arealDensities[j] = 0.0;
						//arealDensities[j] += objects[j].val * (endPoints[i + 1] - endPoints[i]);
						retVal += solids[j].val * (endPoints[i + 1] - endPoints[i]);
						break;
					}
				}
			}
		}
	}
	//free(endPoints);
	//free(objectIndices);
	//free(intersection_0);
	return retVal;
	//*/
}

__global__ void rayTracingKernel(float* g, geometricSolid* solids, const int numObjects, float* floatData, int* intData, const uint64 ichunk, const int chunkSize, const int iv1)
{
    //const int i = threadIdx.x + blockIdx.x * blockDim.x;
    //const int j = threadIdx.y + blockIdx.y * blockDim.y;
    //const int k = threadIdx.z + blockIdx.z * blockDim.z;
	const int iprocess = threadIdx.x + blockIdx.x * blockDim.x;

	uint64 ind = ichunk * chunkSize + iprocess;
	int k = ind % d_N_g.z;
	ind = (ind - k) / d_N_g.z;

	int j = ind % d_N_g.y;
	ind = (ind - j) / d_N_g.y;

	int i = ind % d_N_g.x;
	//ind = (ind - i) / d_N_g.x;

	//int iv1 = ind % d_N_g.w;

	//const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_N_g.x || j >= d_N_g.y || k >= d_N_g.z)
        return;

	const uint64 g_ind = uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k);

	const float v_1 = v1(iv1);
	const float v_0 = v0(i);
	const float u_1 = u1(j);
	const float u_0 = u0(k);
	if (fabs(u_1) - 0.5f * d_T_g.y > d_H - d_R * fabs(v_1) || fabs(u_0) - 0.5f * d_T_g.x > d_L + d_R * fabs(v_0))
	{
		g[g_ind] = 0.0f;
		return;
	}

	if (d_oversampling <= 1)
	{
		const float du1 = 0.0f;
		const float du0 = 0.0f;

		const float3 sourcePos = setSourcePosition(iv1, i, j, k, du1, du0);
		const float3 r = setTrajectory(iv1, i, j, k);

		//const float val = lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &allIntersections[k*2*numObjects]);
		const float val = lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects]);
		g[g_ind] = val;
	}
	else
	{
		const float T_v_os = d_T_g.y / float(d_oversampling + 1);
		const float T_u_os = d_T_g.z / float(d_oversampling + 1);

		const int os_radius = (d_oversampling - 1) / 2;

		float accum = 0.0f;
		for (int j_os = -os_radius; j_os <= os_radius; j_os++)
		{
			const float du1 = j_os * T_v_os;

			for (int k_os = -os_radius; k_os <= os_radius; k_os++)
			{
				const float du0 = k_os * T_u_os;

				const float3 sourcePos = setSourcePosition(iv1, i, j, k, du1, du0);
				const float3 r = setTrajectory(iv1, i, j, k);

				accum += lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects]);
			}
		}

		g[g_ind] = accum / float(d_oversampling * d_oversampling);
	}
}

__global__ void stoppingPowerKernel(float* g, geometricSolid* solids, const int numObjects, float* floatData, int* intData, const uint64 ichunk, const int chunkSize, const int iv1)
{
	//const int i = threadIdx.x + blockIdx.x * blockDim.x;
	//const int j = threadIdx.y + blockIdx.y * blockDim.y;
	//const int k = threadIdx.z + blockIdx.z * blockDim.z;
	const int iprocess = threadIdx.x + blockIdx.x * blockDim.x;

	uint64 ind = ichunk * chunkSize + iprocess;
	int k = ind % d_N_g.z;
	ind = (ind - k) / d_N_g.z;

	int j = ind % d_N_g.y;
	ind = (ind - j) / d_N_g.y;

	int i = ind % d_N_g.x;
	//ind = (ind - i) / d_N_g.x;

	//int iv1 = ind % d_N_g.w;

	//const int k = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= d_N_g.x || j >= d_N_g.y || k >= d_N_g.z)
		return;

	const uint64 g_ind = uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k);

	const float v_1 = v1(iv1);
	const float v_0 = v0(i);
	const float u_1 = u1(j);
	const float u_0 = u0(k);
	if (fabs(u_1) - 0.5f * d_T_g.y > d_H - d_R * fabs(v_1) || fabs(u_0) - 0.5f * d_T_g.x > d_L + d_R * fabs(v_0))
	{
		g[g_ind] = 0.0f;
		return;
	}

	if (d_oversampling <= 1)
	{
		const float du1 = 0.0f;
		const float du0 = 0.0f;

		const float3 sourcePos = setSourcePosition(iv1, i, j, k, du1, du0);
		const float3 r = setTrajectory(iv1, i, j, k);

		//const float val = lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &allIntersections[k*2*numObjects]);
		const float val = stoppingPower_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects]);
		g[g_ind] = val;
	}
	else
	{
		const float T_v_os = d_T_g.y / float(d_oversampling + 1);
		const float T_u_os = d_T_g.z / float(d_oversampling + 1);

		const int os_radius = (d_oversampling - 1) / 2;

		float accum = 0.0f;
		for (int j_os = -os_radius; j_os <= os_radius; j_os++)
		{
			const float du1 = j_os * T_v_os;

			for (int k_os = -os_radius; k_os <= os_radius; k_os++)
			{
				const float du0 = k_os * T_u_os;

				const float3 sourcePos = setSourcePosition(iv1, i, j, k, du1, du0);
				const float3 r = setTrajectory(iv1, i, j, k);

				accum += stoppingPower_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects]);
			}
		}

		g[g_ind] = accum / float(d_oversampling * d_oversampling);
	}
}

void setConstantMemoryGeometryParameters(parameters* params, int oversampling)
{
	//cudaError_t cudaStatus;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g);

	cudaMemcpyToSymbol(d_N_g, &N_g, sizeof(int4));
	cudaMemcpyToSymbol(d_T_g, &T_g, sizeof(float4));
	cudaMemcpyToSymbol(d_startVal_g, &startVal_g, sizeof(float4));

	cudaMemcpyToSymbol(d_oversampling, &oversampling, sizeof(int));

	planogram* plan = params->planogramSet[0];
	float cos_psi = cos(plan->psi);
	float sin_psi = sin(plan->psi);
	cudaMemcpyToSymbol(d_cos_psi, &cos_psi, sizeof(float));
	cudaMemcpyToSymbol(d_sin_psi, &sin_psi, sizeof(float));
	cudaMemcpyToSymbol(d_R, &(plan->R), sizeof(float));
	cudaMemcpyToSymbol(d_L, &(plan->L), sizeof(float));
	cudaMemcpyToSymbol(d_H, &(plan->H), sizeof(float));
}

bool stoppingPower_gpu(float* g, parameters* params, phantom* aPhantom, bool data_on_cpu, int oversampling)
{
	return rayTrace_gpu(g, params, aPhantom, data_on_cpu, oversampling, true);
}

bool rayTrace_gpu(float* g, parameters* params, phantom* aPhantom, bool data_on_cpu, int oversampling, bool calc_stopping_power)
{
    if (g == NULL || params == NULL || params->geometryDefined() == false)
        return false;
	oversampling = max(1, min(oversampling, 11));
	if (oversampling % 2 == 0)
		oversampling += 1;
	oversampling = max(1, min(oversampling, 11));

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

	setConstantMemoryGeometryParameters(params, oversampling);

    float* dev_g = 0;
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g);

    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;

	int numObjects = int(aPhantom->objects.size());
	geometricSolid* dev_solids = 0;
	geometricSolid* solids = new geometricSolid[numObjects];
	for (int i = 0; i < numObjects; i++)
	{
		solids[i].type = aPhantom->objects[i].type;
		solids[i].centers = make_float3(aPhantom->objects[i].centers[0], aPhantom->objects[i].centers[1], aPhantom->objects[i].centers[2]);
		solids[i].radii = make_float3(aPhantom->objects[i].radii[0], aPhantom->objects[i].radii[1], aPhantom->objects[i].radii[2]);
		solids[i].val = aPhantom->objects[i].val;
		for (int j = 0; j < 9; j++)
			solids[i].A[j] = aPhantom->objects[i].A[j];
		for (int j = 0; j < 6; j++)
		{
			for (int k = 0; k < 4; k++)
				solids[i].clippingPlanes[j][k] = aPhantom->objects[i].clippingPlanes[j][k];
		}
		solids[i].isRotated = aPhantom->objects[i].isRotated;
		solids[i].numClippingPlanes = aPhantom->objects[i].numClippingPlanes;
		solids[i].clipCone.x = aPhantom->objects[i].clipCone[0];
		solids[i].clipCone.y = aPhantom->objects[i].clipCone[1];
	}
	if ((cudaStatus = cudaMalloc((void**)&dev_solids, numObjects * sizeof(geometricSolid))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}
	//if ((cudaStatus = cudaMemcpy(dev_g, g, N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
	if ((cudaStatus = cudaMemcpy(dev_solids, solids, numObjects * sizeof(geometricSolid), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		fprintf(stderr, "failed to copy phantom data to device!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}
	delete[] solids;

	planogram* plan = params->planogramSet[0];
	uint64 plan_sz = uint64(plan->N_v0) * uint64(plan->N_u1) * uint64(plan->N_u0);

	int num_gpu_cores = max(1024, getSPcores(params->whichGPU));
	if (plan_sz < uint64(num_gpu_cores))
		num_gpu_cores = int(plan_sz);
	uint64 numChunks = uint64(ceil(double(plan_sz) / double(num_gpu_cores)));
	//printf("number of cores = %d, number of chunks = %d\n", num_gpu_cores, int(numChunks));

	int blockSize = 8;
	int numBlocks = int(ceil(double(num_gpu_cores) / double(blockSize)));
	int numDataCopies = numBlocks * blockSize;

	float* dev_floatData = 0;
	if ((cudaStatus = cudaMalloc((void**)&dev_floatData, numDataCopies * 4 * numObjects * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}
	int* dev_intData = 0;
	if ((cudaStatus = cudaMalloc((void**)&dev_intData, numDataCopies * numObjects * sizeof(int))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}

	//*
	for (int iv1 = 0; iv1 < plan->N_v1; iv1++)
	{
		for (uint64 ichunk = 0; ichunk < numChunks; ichunk++)
		{
			if (calc_stopping_power)
			{
				stoppingPowerKernel <<< numBlocks, blockSize >>> (&dev_g[uint64(iv1) * plan_sz], dev_solids, numObjects, dev_floatData, dev_intData, ichunk, num_gpu_cores, iv1);
			}
			else
			{
				rayTracingKernel <<< numBlocks, blockSize >>> (&dev_g[uint64(iv1) * plan_sz], dev_solids, numObjects, dev_floatData, dev_intData, ichunk, num_gpu_cores, iv1);
			}
		}
	}
	//*/

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (data_on_cpu)
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    else
        g = dev_g;

    // Clean up
	cudaFree(dev_solids);
	cudaFree(dev_floatData);
	cudaFree(dev_intData);
    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
    }

    return true;
}
