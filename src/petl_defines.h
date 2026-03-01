#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdlib.h>
#include <math.h>
#include <complex>
#include <stdio.h>


//* Use the following to enable openMP
#ifndef USE_OPENMP
    #define USE_OPENMP
#endif
//*/

#ifndef STRLENGTH
	#define STRLENGTH 1024
#endif


#ifndef PI
    #define PI 3.1415926535897932385
#endif

#ifndef NAN
    #define NAN sqrt(-1)
#endif

#ifndef E
    #define E 2.7182818284590452354
#endif

#ifndef ELECTRON_REST_MASS_ENERGY
#define ELECTRON_REST_MASS_ENERGY 510.975 // keV
#endif

#ifndef CLASSICAL_ELECTRON_RADIUS
#define CLASSICAL_ELECTRON_RADIUS 2.8179403267e-13 // cm
#endif

#ifndef AVOGANDROS_NUMBER
#define AVOGANDROS_NUMBER 6.0221414107e23 // mol^-1
#endif

using namespace std;
typedef complex<double> dcomp;

#ifdef WIN32
	typedef unsigned long long uint64;
#else
	typedef unsigned long int uint64;
#endif

typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

//EXTERN double PI INIT(2*acos(0));
//EXTERN double E INIT(2.718281828459);
//EXTERN double NaN INIT(pow(2, pow(2, 1024)));

//#define complex complex<double>

//#define short float

#endif
