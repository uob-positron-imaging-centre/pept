/**
 * File              : birmingham_method_ext.h
 * License           : License: GNU v3.0
 * Author            : Sam Manger
 * Date              : 21.08.2019
 */

#ifndef BIRMINGHAM_METHOD_EXT
#define BIRMINGHAM_METHOD_EXT

#include <sys/types.h>

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
