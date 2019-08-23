/**
 * File              : C_findMidpointsAPI.c
 * License           : License: GNU v3.0
 * Author            : Andrei Leonard Nicusan <aln705@student.bham.ac.uk>
 * Date              : 01.07.2019
 */

/*
 *  sampleLines is an (n x 7) array, cast into a memory-contiguous 1D vector
 *  so that it can be passed from and to NumPy
 *  
 *  each 'row' of sampleLines is (time, X1, Y1, Z1, X2, Y2, Z2)
 *
 *
 *  midpoints will be an (m x 4) array, where m is the number of midpoints
 *  found, cast again into a memory-contiguous 1D vector
 *
 *  each 'row' of midpoints will be (TimeM, XM, YM, ZM)
 */

#include "find_cutpoints_ext.h"

void  find_cutpoints_ext(const double *sample_lines, double *cutpoints, const unsigned int max_line, const double max_distance)
{
    unsigned int    i, j;                               // iterators
    int             m;                                  // cutpoint index
    const double    *line1, *line2;                     // (line i), (line j)
    double          P[3], U[3], Q[3], R[3], QP[3];      // position, direction vectors
    double          a, b, c, d, e;                      // collected terms
    double          denom, s0, t0;                      // parameters for lines
    double          A0[3], B0[3], AB0[3];               // perpendicular points 

    m = 0;
    for (i = 0; i < max_line - 1; ++i)
        for(j = i + 1; j < max_line; ++j)
        {
            /*
             * Move the pointer in sample_lines to simulate
             * 2D array behaviour
             */
            
            line1 = sample_lines + i * 7 + 1;    // +1 to skip the time value
            line2 = sample_lines + j * 7 + 1;


            /*
             *  Write each pair of lines in terms of a position vector and a
             *  direction vector.
             *  L1: A(s) = P + s U
             *  L2: B(t) = Q + t R
             */

            P[0] = line1[0];
            P[1] = line1[1];
            P[2] = line1[2];

            U[0] = line1[3] - line1[0];
            U[1] = line1[4] - line1[1];
            U[2] = line1[5] - line1[2];

            
            Q[0] = line2[0];
            Q[1] = line2[1];
            Q[2] = line2[2];

            R[0] = line2[3] - line2[0];
            R[1] = line2[4] - line2[1];
            R[2] = line2[5] - line2[2];


            QP[0] = Q[0] - P[0];
            QP[1] = Q[1] - P[1];
            QP[2] = Q[2] - P[2];

            
            a = U[0] * U[0] + U[1] * U[1] + U[2] * U[2];
            b = U[0] * R[0] + U[1] * R[1] + U[2] * R[2];
            c = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
            d = U[0] * QP[0] + U[1] * QP[1] + U[2] * QP[2];
            e = QP[0] * R[0] + QP[1] * R[1] + QP[2] * R[2];

            denom = b * b - a * c;
            if (denom != 0.0)
            {
                s0 = (b * e - c * d) / denom;
                t0 = (a * e - b * d) / denom;

                A0[0] = P[0] + s0 * U[0];
                A0[1] = P[1] + s0 * U[1];
                A0[2] = P[2] + s0 * U[2];

                B0[0] = Q[0] + t0 * R[0];
                B0[1] = Q[1] + t0 * R[1];
                B0[2] = Q[2] + t0 * R[2];

                AB0[0] = A0[0] - B0[0];
                AB0[1] = A0[1] - B0[1];
                AB0[2] = A0[2] - B0[2];

                // Check the distance is smaller than the tolerance
                if (AB0[0] * AB0[0] + AB0[1] * AB0[1] + AB0[2] * AB0[2] < max_distance * max_distance)
                {
                    // Simulate 2D array behaviour on cutpoints
                    cutpoints[m * 4]        = (line1[-1] + line2[-1]) / 2; // Average of the times of the two lines
                    cutpoints[m * 4 + 1]    = (A0[0] + B0[0]) / 2;
                    cutpoints[m * 4 + 2]    = (A0[1] + B0[1]) / 2;
                    cutpoints[m * 4 + 3]    = (A0[2] + B0[2]) / 2;
                    ++m;
                }
            }


        }
}
