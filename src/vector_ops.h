
#ifndef VECTOROPS_H
#define VECTOROPS_H

#include <string>

// All functions (except dotm & norm) are "in-place", i.e. parameter "double* x" is modified

bool hasEnding (std::string const &fullString, std::string const &ending);
int strcmpI(char* stringA, char* stringB);
int strcmpI(char* stringA, const char* stringB);
char** fileParts(char* fullPath);
bool validFileExtension(char* theExtension);

double* addVectors(double* x, double* x_1, double* x_2, int N);
double* subVectors(double* x, double* x_1, double* x_2, int N);
double* scalarMult(double* x, double alpha, double* x_1, int N);
double* cross(double* x, double* x_1, double* x_2);
double* normalize(double* x, int N);
double dot(double* x_1, double* x_2, int N);
double norm(double* x_1, int N);

//////////
double* addVectors(double* x, double* x_1, double* x_2);
double* subVectors(double* x, double* x_1, double* x_2);
double* scalarMult(double* x, double alpha, double* x_1);
double* normalize(double* x);
double dot(double* x_1, double* x_2);
double norm(double* x_1);
//////////

float median(const float* data, size_t n);


double* rotateAzimuthal(double* x, double alpha);

double** orthonormalBasis(double** vectorTriplet);
double** copyVectorTriplet(double** to, double** from);

double distance(double* x_1, double* x_2);

double factorial(int);

double max(double, double);
double min(double, double);

int max(int, int);
int min(int, int);

int max_ind(double*, int N);

double linearInterpolationExtrapolation(double* values, int N, double index);
double inverseLinearInterpolationExtrapolation(double* values, int N, double particularValue);

bool sort(double* a, int N);

int ceil_i(double);
int floor_i(double);

int optimalFFTsize(int N);

#endif
