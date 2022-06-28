/**
 * File              : birmingham_method_ext.h
 * License           : License: GNU v3.0
 * Author            : Sam Manger
 * Date              : 21.08.2019
 */


#ifndef BIRMINGHAM_METHOD_EXT
#define BIRMINGHAM_METHOD_EXT


#if defined(_MSC_VER)
	// Support the bloody unconforming mess that MSVC is; allow using fopen and ssize_t
	#define _CRT_SECURE_NO_DEPRECATE
	#include <BaseTsd.h>
	typedef SSIZE_T ssize_t;
#else
	#include <sys/types.h>
#endif


#include <math.h>                   // for sqrt
#include <float.h>                  // for DBL_MAX
#include <stdlib.h>                 // for malloc


void birmingham_method_ext(
    const double *, const ssize_t nrows, const ssize_t ncols,
    double *, int *, const double
);

// void calculate(const double *, const double *);

void calculate(
    double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double *,
    int *, int, int, double *
);

#endif
