/**
 * File              : birmingham_method_ext.c
 * License           : License: GNU v3.0
 * Author            : Sam Manger
 * Date              : 21.09.2019
 */


#include "birmingham_method_ext.h"
#include <sys/types.h>              // for ssize_t
#include <math.h>                   // for sqrt
#include <float.h>                  // for DBL_MAX
#include <stdlib.h>                 // for malloc


void free_memory(
        double *ttt, double *xx1, double *xx2, double *yy1,
        double *yy2, double *zz1, double *zz2, double *x12,
        double *y12, double *z12, double *r12, double *q12,
        double *p12, double *a12, double *b12, double *c12,
        double *d12, double *e12, double *f12, double *r2,
        double *dev
        )
{
    /*
       Free all the heap-allocated (malloc'd) memory from
       birmingham_method_ext.
       */
    free(ttt); ttt = NULL;
    free(xx1); xx1 = NULL;
    free(xx2); xx2 = NULL;
    free(yy1); yy1 = NULL;
    free(yy2); yy2 = NULL;
    free(zz1); zz1 = NULL;
    free(zz2); zz2 = NULL;

    free(x12); x12 = NULL;
    free(y12); y12 = NULL;
    free(z12); z12 = NULL;

    free(r12); r12 = NULL;
    free(q12); q12 = NULL;
    free(p12); p12 = NULL;

    free(a12); a12 = NULL;
    free(b12); b12 = NULL;
    free(c12); c12 = NULL;
    free(d12); d12 = NULL;
    free(e12); e12 = NULL;
    free(f12); f12 = NULL;

    free(r2); r2 = NULL;
    free(dev); dev = NULL;
}


void birmingham_method_ext(
        const double *lines,    // flattened 2D numpy array of LoRs
        const ssize_t nrows,    // number of rows in `lines`
        const ssize_t ncols,    // number of columns `lines`
        double *location,       // (5,) numpy array for the found location
        int *used,              // (nrows,) numpy array
        const double fopt       // fraction of LoRs used to find location
        )
{  
    /* 
       lines contains LORs

       location is used to store calculated locations from this function

       used is an array of integers used to store which LORs have been used to
       calculate final position

       nrows is the number of LORs per tracked location
       fopt is the fraction of these LORs to use in the final location
       */

    // Allocate vectors of double with `nrows` elements. They will be freed in
    // free_memory(...);
    double *ttt = (double*)malloc(sizeof(double) * nrows);
    double *xx1 = (double*)malloc(sizeof(double) * nrows);
    double *xx2 = (double*)malloc(sizeof(double) * nrows);
    double *yy1 = (double*)malloc(sizeof(double) * nrows);
    double *yy2 = (double*)malloc(sizeof(double) * nrows);
    double *zz1 = (double*)malloc(sizeof(double) * nrows);
    double *zz2 = (double*)malloc(sizeof(double) * nrows);

    double *x12 = (double*)malloc(sizeof(double) * nrows);
    double *y12 = (double*)malloc(sizeof(double) * nrows);
    double *z12 = (double*)malloc(sizeof(double) * nrows);

    double *r12 = (double*)malloc(sizeof(double) * nrows);
    double *q12 = (double*)malloc(sizeof(double) * nrows);
    double *p12 = (double*)malloc(sizeof(double) * nrows);

    double *a12 = (double*)malloc(sizeof(double) * nrows);
    double *b12 = (double*)malloc(sizeof(double) * nrows);
    double *c12 = (double*)malloc(sizeof(double) * nrows);
    double *d12 = (double*)malloc(sizeof(double) * nrows);
    double *e12 = (double*)malloc(sizeof(double) * nrows);
    double *f12 = (double*)malloc(sizeof(double) * nrows);

    double *r2 = (double*)malloc(sizeof(double) * nrows);
    double *dev = (double*)malloc(sizeof(double) * nrows);

    double error;

    double Con_factor = 150; // Convergence factor when removing LORs from set
    const double *line_i;

    // the following variables are used while iteratively removing lines
    double dismin, dismax;   
    int imin, nused, it;

    int imax, nprev, nfin;

    // an array used by the function "calculate" to store output variables
    double calculate_results[8];

    for (it = 0; it < nrows; ++it)
    {
        line_i = lines + it * ncols;

        ttt[it] = line_i[0];
        xx1[it] = line_i[1];
        yy1[it] = line_i[2];
        zz1[it] = line_i[3];
        xx2[it] = line_i[4];
        yy2[it] = line_i[5];
        zz2[it] = line_i[6];
        used[it] = 1;

        // Calculate vectors for set of LORs, to be used in calculate
        x12[it] = xx1[it] - xx2[it];    // Point 2 -> Point 1 vector in x-axis
        y12[it] = yy1[it] - yy2[it];    // Point 2 -> Point 1 vector in y-axis
        z12[it] = zz1[it] - zz2[it];    // Point 2 -> Point 1 vector in z-axis

        // Magnitude of vector from P2 to P1
        r2[it] = (x12[it] * x12[it]) + (y12[it] * y12[it]) +
            (z12[it] * z12[it]);

        r12[it] = (y12[it] * z12[it]) / r2[it];
        q12[it] = (x12[it] * z12[it]) / r2[it];
        p12[it] = (x12[it] * y12[it]) / r2[it];

        a12[it] = ((y12[it] * y12[it]) + (z12[it] * z12[it])) / r2[it];
        b12[it] = ((x12[it] * x12[it]) + (z12[it] * z12[it])) / r2[it];
        c12[it] = ((y12[it] * y12[it]) + (x12[it] * x12[it])) / r2[it];

        d12[it] = ((yy2[it] * xx1[it] - yy1[it] * xx2[it]) * y12[it] + 
                (zz2[it] * xx1[it] - zz1[it] * xx2[it]) * z12[it]) / r2[it];
        e12[it] = ((zz2[it] * yy1[it] - zz1[it] * yy2[it]) * z12[it] +
                (xx2[it] * yy1[it] - xx1[it] * yy2[it]) * x12[it]) / r2[it];
        f12[it] = -((zz2[it] * yy1[it] - zz1[it] * yy2[it]) * y12[it] +
                (zz2[it] * xx1[it] - zz1[it] * xx2[it]) * x12[it]) / r2[it];
    }

    // The total number of lines used to determine final tracer position.
    // Initially use all of them.
    nused = nrows;

    // The target number of lines to be used to determine the final position.
    nfin = (int)(nrows * fopt); 

    int iteration = 0;

    while (nrows > 0)
    {
        iteration += 1;
        imin = 0;
        imax = 0;

        calculate(a12, b12, c12, d12, e12, f12, p12, q12, r12, xx1, yy1, zz1,
                x12, y12, z12, r2, ttt, dev, used, nused, nrows,
                calculate_results);

        // We used the target number of lines to calculate the tracer position.
        if (nused == nfin)
        {
            free_memory(ttt, xx1, xx2, yy1, yy2, zz1, zz2, x12, y12, z12, r12,
                    q12, p12, a12, b12, c12, d12, e12, f12, r2, dev);

            location[0] = calculate_results[0]; // t
            location[1] = calculate_results[1]; // x
            location[2] = calculate_results[2]; // y
            location[3] = calculate_results[3]; // z
            location[4] = calculate_results[7]; // error
            return;
        }

        error = calculate_results[7];
        nprev = nused;
        nused = 0;

        // Iterate through the lines; if they lie too far away from the
        // calculated centroid then we won't use them anymore.
        for (it = 0; it < nrows; it++)
        {
            if (sqrt(dev[it]) > (Con_factor * error / 100))
                used[it] = 0;
            else
            {
                used[it] = 1;
                nused = nused + 1;
            }
        }

        // If true, we have reduced to too few events, so restore closest
        // unused events
        while (nused < nfin)
        {
            dismin = DBL_MAX;
            for (it = 0; it < nrows; it++)
            {
                if (used[it] == 0 && dev[it] < dismin)
                {
                    imin = it;
                    dismin = dev[it];
                }
            }
            used[imin] = 1;
            nused = nused + 1;
        }

        // If true then we haven't removed any, so remove furthest point
        while (nused >= nprev)
        {
            dismax = 0.0;
            for (it = 0; it < nrows; it++)
            {
                if(used[it] == 1 && dev[it] > dismax)
                {
                    imax = it;
                    dismax = dev[it];
                }
            }
            used[imax] = 0;
            nused = nused - 1;
        }
    }   

    // Free memory in case we didn't enter the while loop above.
    free_memory(ttt, xx1, xx2, yy1, yy2, zz1, zz2, x12, y12, z12, r12, q12,
            p12, a12, b12, c12, d12, e12, f12, r2, dev);
}

void calculate(double *a12, double *b12, double *c12, double *d12,
        double *e12, double *f12, double *p12, double *q12,
        double *r12, double *xx1, double *yy1, double *zz1,
        double *x12, double *y12, double *z12, double *r2, 
        double *ttt, double *dev, int *used, int nused, 
        int nrows, double *calculate_results )
{
    double x, y, z, error, avtime, dx = 0, dy = 0, dz = 0, dd;
    double suma, sumb, sumc, sumd, sume, sumf, sump, sumq, sumr;
    double ab, dq, dp, ar, ac, denom;

    int it;

    suma = sumb = sumc = sumd = sume = sumf = sump = sumq = sumr = 0;

    for (it = 0; it < nrows; it++)
    {
        if (used[it] == 1)
        {
            // Calculate "sum of" for lines in use
            suma = suma + a12[it];
            sumb = sumb + b12[it];
            sumc = sumc + c12[it];
            sumd = sumd + d12[it];
            sume = sume + e12[it];
            sumf = sumf + f12[it];
            sump = sump + p12[it];
            sumq = sumq + q12[it];
            sumr = sumr + r12[it];
        }
    }

    ab = suma * sumb - sump * sump;
    dq = sumd * sumq + suma * sumf;
    dp = sumd * sump + suma * sume;
    ar = suma * sumr + sumq * sump;
    ac = suma * sumc - sumq * sumq;
    denom = (ar * ar - ab * ac);

    if (denom == 0)
        denom = 1.0e-6;

    if (ar == 0)
        ar = 1.0e-6;

    if (suma == 0)
        suma = 1.0e-6;

    z = (ab * dq + dp * ar) / denom;
    y = (z * ac + dq) / ar;
    x = (y * sump + z * sumq - sumd) / suma;

    error = 0;
    avtime = 0;

    //work out errors and time
    for (it = 0; it < nrows; it++)
    {
        dx = x - xx1[it];
        dy = y - yy1[it];
        dz = z - zz1[it];

        dd = (dx * z12[it] - dz * x12[it]) * (dx * z12[it] - dz * x12[it]);
        dd = dd + (dy * x12[it] - dx * y12[it]) * (dy * x12[it] - dx * y12[it]);
        dd = dd + (dz * y12[it] - dy * z12[it]) * (dz * y12[it] - dy * z12[it]);
        dev[it] = dd / r2[it];

        if (used[it] == 1)
        {
            error += dev[it];
            avtime += ttt[it];
        }
    }

    error = sqrt(error / nused);
    avtime = avtime / nused;

    calculate_results[0] = avtime;
    calculate_results[1] = x;
    calculate_results[2] = y;
    calculate_results[3] = z;
    calculate_results[4] = dx;
    calculate_results[5] = dy;
    calculate_results[6] = dz;
    calculate_results[7] = error;
}
