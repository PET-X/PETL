//#ifndef VECTOROPS_CPP
//#define VECTOROPS_CPP

//#include "pch.h"
#include "vector_ops.h"
#include "allocate.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

#ifndef STRLENGTH
	#define STRLENGTH 1024
#endif

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	else
        return false;
}

int strcmpI(char* stringA, const char* stringB)
{
	char stringB_const[STRLENGTH];
	sprintf(stringB_const, "%s", stringB);
	return strcmpI(stringA, stringB_const);
}

int strcmpI(char* stringA, char* stringB)
{
	int N = min(int(strlen(stringA)), int(strlen(stringB)));
	for (int i = 0; i < N; i++)
	{
		if (toupper(stringA[i]) != toupper(stringB[i]))
			return i+1;
	}
	if (int(strlen(stringA)) != int(strlen(stringB)))
		return N+1;

	return 0;
}

char** fileParts(char* fullPath)
{
	// if fullPath == "../data/sinos.sog", then this function returns:
	// retVal[0] == "../data"
	// retVal[1] == "sinos"
	// retVal[2] = "sog"

	char** retVal = (char**) multialloc(sizeof(char), 2, 3, STRLENGTH);
	retVal[0][0] = 0;
	retVal[1][0] = 0;
	retVal[2][0] = 0;

	int N  = strlen(fullPath);
	int n;
	int startFileType = N;
	int startFileName = N;
    
    // Find and set file type
	for (n = N-1; n >= 0; n--)
    {
        if (fullPath[n] == '.')
		{
			startFileType = n+1;
			break;
		}
	}
    if (startFileType <= N-1)
	{
		for (n = startFileType; n <= N-1; n++)
			retVal[2][n-startFileType] = fullPath[n];
	}
    
    // Check to see if file type is of a known type, if not this is actually not a file type
    if (validFileExtension(retVal[2]) == false)
    {
        startFileType = N;
        retVal[2][0] = 0;
    }
    
	for (n = startFileType-1; n >= 0; n--)
	{
		if (fullPath[n] == '/' || fullPath[n] == '\\')
		{
			startFileName = n+1;
			break;
		}
	}
	if (n <= 0)
		startFileName = 0;

	if (startFileName <= N-1)
	{
		for (n = startFileName; n < startFileType-1; n++)
			retVal[1][n-startFileName] = fullPath[n];
		if (fullPath[startFileType-1] != '.')
			retVal[1][startFileType-1-startFileName] = fullPath[startFileType-1];
	}

	if (startFileName > 0)
	{
		for (n = 0; n < min(startFileName, startFileType)-1; n++)
			retVal[0][n] = fullPath[n];
	}

	return retVal;
}

bool validFileExtension(char* theExtension)
{
    if (strlen(theExtension) > 0 && (strcmpI(theExtension, "raw") == 0 ||
        strcmpI(theExtension, "tiff") == 0 || strcmpI(theExtension, "tif") == 0 || strcmpI(theExtension, "sog") == 0 ||
        strcmpI(theExtension, "od") == 0 || strcmpI(theExtension, "pd") == 0 || strcmpI(theExtension, "cfg") == 0 ||
        strcmpI(theExtension, "txt") == 0 || strcmpI(theExtension, "dat") == 0 || strcmpI(theExtension, "spectrum") == 0 ||
        strcmpI(theExtension, "log") == 0 || strcmpI(theExtension, "bin") == 0))
        return true;
    else
        return false;
}

double* addVectors(double* x, double* x_1, double* x_2, int N)
{
	for (int i = 0; i < N; i++)
		x[i] = x_1[i] + x_2[i];
	
	return x;
}

double* subVectors(double* x, double* x_1, double* x_2, int N)
{
	for (int i = 0; i < N; i++)
		x[i] = x_1[i] - x_2[i];
	
	return x;
}

double dot(double* x_1, double* x_2, int N)
{
	double retVal = 0.0;
	for (int i = 0; i < N; i++)
		retVal = retVal + x_1[i] * x_2[i];

	return retVal;
}

double* cross(double* x, double* x_1, double* x_2)
{ // assumes vectors are of length 3
	
	x[0] = x_1[1]*x_2[2] - x_1[2]*x_2[1];
	x[1] = x_1[2]*x_2[0] - x_1[0]*x_2[2];
	x[2] = x_1[0]*x_2[1] - x_1[1]*x_2[0];

	return x;
}

double* normalize(double* x, int N)
{
	double temp = norm(x, N);
	if (temp != 0.0)
		return scalarMult(x, 1.0/temp, x, N);
	else
		return NULL;
}

double* scalarMult(double* x, double alpha, double* x_1, int N)
{
	for (int i = 0; i < N; i++)
		x[i] = alpha * x_1[i];

	return x;
}

double norm(double* x_1, int N)
{
	return sqrt(dot(x_1, x_1, N));
}

////////////////////////////////////////////////////////////
double* addVectors(double* x, double* x_1, double* x_2)
{
	x[0] = x_1[0] + x_2[0];
	x[1] = x_1[1] + x_2[1];
	x[2] = x_1[2] + x_2[2];
	
	return x;
}

double* subVectors(double* x, double* x_1, double* x_2)
{
	x[0] = x_1[0] - x_2[0];
	x[1] = x_1[1] - x_2[1];
	x[2] = x_1[2] - x_2[2];
	
	return x;
}

double dot(double* x_1, double* x_2)
{
	return (x_1[0] * x_2[0] + x_1[1] * x_2[1] + x_1[2] * x_2[2]);
}

double* normalize(double* x)
{
	double temp = norm(x);
	if (temp != 0.0)
		return scalarMult(x, 1.0/temp, x);
	else
		return NULL;
}

double* scalarMult(double* x, double alpha, double* x_1)
{
	x[0] = alpha * x_1[0];
	x[1] = alpha * x_1[1];
	x[2] = alpha * x_1[2];
	
	return x;
}

double norm(double* x_1)
{
	return sqrt(dot(x_1, x_1));
}
////////////////////////////////////////////////////////////

double** orthonormalBasis(double** vectorTriplet)
{ // Assumes vectorTriplet[0] is a vector of unit length

	if (vectorTriplet[0][0] != 0 || vectorTriplet[0][1] != 0)
	{
		double temp = sqrt(vectorTriplet[0][0]*vectorTriplet[0][0] + vectorTriplet[0][1]*vectorTriplet[0][1]);
		vectorTriplet[1][0] = -1.0*vectorTriplet[0][1] / temp;
		vectorTriplet[1][1] = vectorTriplet[0][0] / temp;
		vectorTriplet[1][2] = 0;
		
		/*
		|	i		j		k	|
		|	x_0		x_1		x_2	|	= (x_1y_2 - x_2y_1)i + (x_2y_0 - x_0y_2)j + (x_0y_1 - x_1y_0)k
		|	y_0		y_1		y_2	|
									= -x_2y_1i + x_2y_0j + (x_0y_1 - x_1y_0)k
									because y_2 = 0
		*/
		vectorTriplet[2][0] = -1.0*vectorTriplet[0][2]*vectorTriplet[1][1];
		vectorTriplet[2][1] = vectorTriplet[0][2]*vectorTriplet[1][0];
		vectorTriplet[2][2] = vectorTriplet[0][0]*vectorTriplet[1][1] - vectorTriplet[0][1]*vectorTriplet[1][0];
	}
	else if (vectorTriplet[0][2] != 0)
	{
		//vectorTriplet[0] = (0,0,1)
		vectorTriplet[1][0] = 1;
		vectorTriplet[1][1] = 0;
		vectorTriplet[1][2] = 0;
		
		vectorTriplet[2][0] = 0;
		vectorTriplet[2][1] = 1;
		vectorTriplet[2][2] = 0;
	}
	else
	{
		printf("Fatal Error vector::orthonormalBasis!");
		exit(1);
	}

	return vectorTriplet;
}

double** copyVectorTriplet(double** to, double** from)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			to[i][j] = from[i][j];
	}
	return to;
}

double* rotateAzimuthal(double* x, double alpha)
{ // assume vector in R^3
	double cos_alpha = cos(alpha);
	double sin_alpha = sin(alpha);
	
	double temp = cos_alpha*x[0] - sin_alpha*x[1];
	x[1] = sin_alpha*x[0] + cos_alpha*x[1];
	x[0] = temp;
	
	return x;
}

double factorial(int k)
{
	double retVal = 1;
	for (int i = 2; i <= k; i++)
		retVal = retVal * double(i);
		
	return retVal;
}

double distance(double* x_1, double* x_2)
{
	return sqrt( pow(x_1[0]-x_2[0],2) + pow(x_1[1]-x_2[1],2) + pow(x_1[2]-x_2[2],2) );
}

double max(double a, double b)
{
	if (a > b)
		return a;
	else
		return b;
}

double min(double a, double b)
{
	if (a < b)
		return a;
	else
		return b;
}

int max(int a, int b)
{
	if (a > b)
		return a;
	else
		return b;
}

int min(int a, int b)
{
	if (a < b)
		return a;
	else
		return b;
}

int max_ind(double* data, int N)
{
	int maxInd = 0;
	double maxValue = data[0];
	for (int i = 1; i < N; i++)
	{
		if (data[i] > maxValue)
		{
			maxValue = data[i];
			maxInd = i;
		}
	}
	return maxInd;
}

float median(const float* data, size_t n)
{
	if (n == 0) return 0.0f; // or throw

	std::vector<float> tmp(data, data + n);
	size_t mid = n / 2;

	std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());

	if (n % 2 == 1)
		return tmp[mid];

	float upper = tmp[mid];
	std::nth_element(tmp.begin(), tmp.begin() + mid - 1, tmp.end());
	return (upper + tmp[mid - 1]) / 2.0f;
}

double linearInterpolationExtrapolation(double* values, int N, double index)
{
	if (index < 0) // extrapolation
		return (values[1] - values[0])*(floor(index)-index) + values[0];
	else if (index < N-1) // interpolation
	{
		int indLow = int(index);
		double T = index - indLow;
		return (1-T)*values[indLow] + T*values[indLow+1];
	}
	else // extrapolation
		return (values[N-1] - values[N-2])*(index - floor(index)) + values[N-1];
}

double inverseLinearInterpolationExtrapolation(double* values, int N, double particularValue)
{
	for (int index = 0; index < N-1; index++)
	{
		if ((values[index] <= particularValue && particularValue <= values[index+1]) ||
			(values[index+1] <= particularValue && particularValue <= values[index]))
		{
			//double T = fabs(particularValue - values[index]) / fabs(values[index+1] - values[index]);
			//return double(index) + T; // == (1-T)*index + T*(index+1)
			
			return double(index) + fabs(particularValue - values[index]) / fabs(values[index+1] - values[index]);
		}
	}
	
	// extrapolation
	if (fabs(particularValue - values[0]) < fabs(particularValue - values[N-1]))
		return -fabs(particularValue - values[0]) / fabs(values[1] - values[0]);
	else
		return double(N-1) + fabs(particularValue - values[N-1]) / fabs(values[N-1] - values[N-2]);
}

bool sort(double* a, int N)
{
	// bubble sort: smallest to largest
	double temp;
	bool swapped;
	do
	{
		swapped = false;
		for (int i = 0; i <= N-2; i++)
		{
			if (a[i] > a[i+1])
			{
				temp = a[i];
				a[i] = a[i+1];
				a[i+1] = temp;
				swapped = true;
			}	
		}
	} while (swapped == true);
	
	return true;
}

int ceil_i(double x)
{
	// incorrect for positive whole numbers (overestimate)
	if (x > 0.0)
		return int(x)+1;
	else
		return int(x);
	/*
     if (x < 0.0 || x == trunc(x))
     return int(x);
     else
     return int(x)+1;
     */
    
	/*
     if (x > 0.0)
     return -Real2Int(-x);
     else
     return int(x);
     */
	//return -floor_i(-x);
	/*
     if (x > 0.0)
     return int(x)+1;
     else
     return int(ceil(x));
     //*/
	//return int(x)+1;
}

int floor_i(double x)
{
	// incorrect for negative whole numbers (underestimate)
	if (x > 0.0)
		return int(x);
	else
		return -int(-x)-1;
	//return Real2Int(x);
	/*
     if (x > 0.0)
     return int(x);
     else
     return Real2Int(x);
     //return int(floor(x));
     //*/
	//return int(x);
}

int optimalFFTsize(int N)
{
	// returns smallest number = 2^(n+1)*3^m such that 2^(n+1)*3^m >= N and n,m >= 0
	if (N <= 2)
		return 2;
	
	double c1 = log2(double(N)/2.0)/log2(3);
	double c2 = 1.0 / log2(3);
	//2^x*3^y = N ==> y = c1-c2*x
	double xbar = log2(double(N)/2.0);
	int x, y;
	int minValue = pow(2,int(ceil(xbar))+1);
	int newValue;
	for (x = 0; x < int(ceil(xbar)); x++)
	{
		y = int(ceil(c1-c2*double(x)));
		newValue = pow(2,x+1)*pow(3,y);
		if (newValue < minValue && y >= 0)
			minValue = newValue;
	}
    
	//printf("%d\n", minValue);
    
	return minValue;
}

//#endif
