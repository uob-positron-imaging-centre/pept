/**
 * File   : find_minpoints_ext.c
 * License: GNU v3.0
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 20.10.2020
 */


#include "find_minpoints_ext.h"
#include <stdlib.h>


inline double square(double x)
{
    return x * x;
}


void compute_mdp(
    const double *sample_lines,
    const ssize_t nrows,
    const ssize_t ncols,
    const ssize_t *indices,
    const ssize_t nlines,
    const double max_distance,
    const double *cutoffs,
    const int append_indices,
    double *minpoints,
    ssize_t *m,
    ssize_t *mpts_nrows,
    ssize_t *mpts_ncols
)
{
    const double *line;                         // Moving pointer for line
    ssize_t i;                                  // Iterator
    double x1, y1, z1, x2, y2, z2;              // LoR points coordinates
    double x12, y12, z12;                       // Differences (x12 = x1 - x2)
    double r2;                                  // Squared LoR length
    double p12, q12, r12;                       // Math notations
    double a12, b12, c12, d12, e12, f12;        // Math notations
    double suma, sumb, sumc, sumd, sume, sumf;  // Collected (summed) terms
    double sump, sumq, sumr;                    // Collected (summed) terms
    double sumt;                                // Sum of times
    double ab, ac, ar, dp, dq, denom;           // Math notations
    double x, y, z;                             // Minimum distance point
    double d2;                                  // Distance2 from MDP to LoR
    double *minpt;                              // Current minpoint

    suma = sumb = sumc = sumd = sume = sumf = sump = sumq = sumr = 0.0;
    sumt = 0.0;

    for (i = 0; i < nlines; ++i)
    {
        // Line elements: [t, x1, y1, z1, x2, y2, z2]
        line = sample_lines + indices[i] * ncols;

        // Individual points' coordinates (here for readability - they should
        // be optimised away by the compiler)
        x1 = line[1];
        y1 = line[2];
        z1 = line[3];

        x2 = line[4];
        y2 = line[5];
        z2 = line[6];

        // Calculate direction vector from P1 to P2
        x12 = x1 - x2;
        y12 = y1 - y2;
        z12 = z1 - z2;

        // Magnitude of vector from P2 to P1
        r2 = (x12 * x12) + (y12 * y12) + (z12 * z12);

        // Terms from math derivation
        p12 = (x12 * y12) / r2;
        q12 = (x12 * z12) / r2;
        r12 = (y12 * z12) / r2;

        a12 = ((y12 * y12) + (z12 * z12)) / r2;
        b12 = ((x12 * x12) + (z12 * z12)) / r2;
        c12 = ((y12 * y12) + (x12 * x12)) / r2;

        d12 = ((y2 * x1 - y1 * x2) * y12 + (z2 * x1 - z1 * x2) * z12) / r2;
        e12 = ((z2 * y1 - z1 * y2) * z12 + (x2 * y1 - x1 * y2) * x12) / r2;
        f12 = -((z2 * y1 - z1 * y2) * y12 + (z2 * x1 - z1 * x2) * x12) / r2;

        // Summed terms for every selected line
        suma += a12;
        sumb += b12;
        sumc += c12;
        sumd += d12;
        sume += e12;
        sumf += f12;

        sump += p12;
        sumq += q12;
        sumr += r12;

        sumt += line[0];
    }

    // Solve matrix using brute force inversion
    ab = suma * sumb - sump * sump;
    dq = sumd * sumq + suma * sumf;
    dp = sumd * sump + suma * sume;
    ar = suma * sumr + sumq * sump;
    ac = suma * sumc - sumq * sumq;
    denom = (ar * ar - ab * ac);

    // Matrix cannot be inverted if any of these terms is 0
    if (denom == 0)
        return;

    if (ar == 0)
        return;

    if (suma == 0)
        return;

    // Minimum distance point coordinates
    z = (ab * dq + dp * ar) / denom;
    y = (z * ac + dq) / ar;
    x = (y * sump + z * sumq - sumd) / suma;

    // Ensure the point is within the cutoff distances
    if (x < cutoffs[0] || x > cutoffs[1] ||
        y < cutoffs[2] || y > cutoffs[3] ||
        z < cutoffs[4] || z > cutoffs[5])
        return;

    // Ensure the distance to each LoR used is smaller than max_distance
    for (i = 0; i < nlines; ++i)
    {
        // Line elements: [t, x1, y1, z1, x2, y2, z2]
        line = sample_lines + indices[i] * ncols;

        // Individual points' coordinates (here for readability - they should
        // be optimised away by the compiler)
        x1 = line[1];
        y1 = line[2];
        z1 = line[3];

        x2 = line[4];
        y2 = line[5];
        z2 = line[6];

        // Math notations
        x12 = x1 - x2;
        y12 = y1 - y2;
        z12 = z1 - z2;

        // Square LoR length
        r2 = square(x12) + square(y12) + square(z12);

        // Squared distance from MDP (x, y, z) to LoR (`line`)
        d2 = 1 / r2 * (
            square((x - x2) * z12 - (z - z2) * x12) +
            square((y - y2) * x12 - (x - x2) * y12) +
            square((z - z2) * y12 - (y - y2) * z12)
        );

        if (d2 > square(max_distance))
            return;
    }

    // `minpt` is a pointer to the current minpoint
    minpt = minpoints + *m * *mpts_ncols;

    // Save (t, x, y, z) of the MDP in `minpt`. Time is the average of the
    // timestamps of the LoRs used to compute the MDP
    minpt[0] = sumt / nlines;
    minpt[1] = x;
    minpt[2] = y;
    minpt[3] = z;

    if (append_indices)
        for (i = 0; i < nlines; ++i)
            minpt[4 + i] = indices[i];

    // Increment minpoint index
    *m += 1;
}




/*  Function that, given a sample of lines, computes the minimum distance
 *  point (MDP) for every possible combination of `nlines` lines. The MDPs are
 *  stored in a flattened array of doubles if the MDPs:
 *  
 *      1. Are within the `cutoffs`.
 *      2. Are closer to all the constituent LoRs than `max_distance`.
 *
 *  The flattened array is returned and its ownership is given to the caller.
 *  The size of the MDP array is saved in the `mpts_nrows` and `mpts_ncols`
 *  input parameters.
 *
 *  Function parameters
 *  -------------------
 *  sample_lines: const double*
 *      A flattened 2D array of lines, where each line is defined by two points
 *      such that every "row" is `[t, x1, y1, z1, x2, y2, z2, etc.]`. Its
 *      dimensions are given by `nrows` and `ncols`. It *must* have at least
 *      `nlines` lines, each with at least 7 columns.
 *
 *  nrows: const ssize_t
 *      The number of rows in the input array `sample_lines`. It *must* have at
 *      least `nlines` rows, satisfying 2 <= nlines <= nrows.
 *
 *  ncols: const ssize_t
 *      The number of columns in the input array `sample_lines`. It *must* have
 *      at least 7 columns such that every row contains the LoR time and the
 *      coordinates of the defining points: `[t, x1, y1, z1, x2, y2, z2, etc.]`
 *
 *  nlines: const ssize_t
 *      The number of lines in each combination of LoRs used to compute the
 *      MDP. This function considers every combination of `nlines` from the
 *      input `sample_lines`.
 *
 *  max_distance: const double
 *      The maximum allowed distance between an MDP and its constituent lines.
 *      If any distance from the MDP to one of its lines is larger than
 *      `max_distance`, the MDP is thrown away.
 *
 *  cutoffs: const double*
 *      An array of spatial cutoff coordinates with *exactly 6 elements* as
 *      [x_min, x_max, y_min, y_max, z_min, z_max]. If any MDP lies outside
 *      this region, it is thrown away.
 *
 *  append_indices: const int
 *      A boolean specifying whether to include the indices of the lines used
 *      to compute each MDP. If `0` (False), the output array will only contain
 *      the [time, x, y, z] of the MDPs. If `1` (True), the output array will
 *      have extra columns [time, x, y, z, i1, i2, ..., in] where n = nlines.
 *
 *  mpts_nrows: ssize_t*
 *      The number of rows in the flattened output array.
 *
 *  mpts_ncols: ssize_t*
 *      The number of columns in the flattened output array.
 *
 *  Returns
 *  -------
 *  minpoints: double*
 *      A flattened 2D array of `double`s containing the time and coordinates
 *      of the MDPs [time, x, y, z]. The time is computed as the average of the
 *      constituent lines. If `append_indices` is `1` (True), then `nlines`
 *      indices of the constituent LoRs are appended as extra columns:
 *      [time, x, y, z, line_idx1, line_idx2, ..].
 *
 *  Notes
 *  -----
 *  This array is heap-allocated and its ownership is returned to the caller -
 *  do not forget to deallocate it (or encapsulate it in a reference-counted
 *  container like a Python object)!
 *
 *  Undefined Behaviour
 *  -------------------
 *  The input pointers `sample_lines`, `mpts_nrows`, `mpts_ncols` and `cutoffs`
 *  must be valid.
 *
 *  There must be at least two lines in `sample_lines` and `nlines` must be
 *  greater or equal to `nrows` (i.e. the number of lines in `sample_lines`).
 *  Put another way: 2 <= nlines <= nrows.
 */
double* find_minpoints_ext(
    const double *sample_lines,
    const ssize_t nrows,
    const ssize_t ncols,
    const ssize_t nlines,
    const double max_distance,
    const double *cutoffs,
    const int append_indices,
    ssize_t *mpts_nrows,
    ssize_t *mpts_ncols
)
{
    // Variables' declarations
    double *minpoints;      // The computed MDPs for combinations of LoRs
    ssize_t m = 0;          // The current minpoint index
    ssize_t *indices;       // The LoR indices used for the MDP
    ssize_t i;              // Iterator index

    // Allocate enough memory for the `minpoints` which will be returned
    // The number of LoR combinations is
    // `factorial(nrows) / (factorial(nlines) * factorial(nrows - nlines))`
    // However we only select MDPs which are closer to the LoRs than
    // `max_distance`; therefore, dynamically reallocate `minpoints` as needed
    // Start with the same length as `sample_lines` (nrows)
    *mpts_nrows = nrows;
    if (append_indices)
        *mpts_ncols = 4 + nlines;
    else
        *mpts_ncols = 4;

    minpoints = (double*)malloc(sizeof(double) * *mpts_nrows * *mpts_ncols);

    // Generate all combinations of LoR indices
    indices = (ssize_t*)malloc(sizeof(ssize_t) * nlines);

    // Initialize first combination
    for (i = 0; i < nlines; ++i)
        indices[i] = i;

    i = nlines - 1;
    while (indices[0] < nrows - nlines + 1)
    {
        // Compute the MDP for the LoRs at `indices`
        compute_mdp(
            sample_lines,
            nrows,
            ncols,
            indices,
            nlines,
            max_distance,
            cutoffs,
            append_indices,
            minpoints,
            &m,
            mpts_nrows,
            mpts_ncols
        );

        // Check `minpoints` is not full - if it is, reallocate it by doubling
        // its size
        if (m >= *mpts_nrows)
        {
            minpoints = (double*)realloc(
                minpoints,
                sizeof(double) * *mpts_ncols * *mpts_nrows * 2
            );
            *mpts_nrows *= 2;
        }

        // LoR index generation
        while (i > 0 && indices[i] == nrows - nlines + i)
            i--;

        indices[i]++;

        // Reset each outer element to prev element + 1
        while (i < nlines - 1)
        {
            indices[i + 1] = indices[i] + 1;
            i++;
        }
    }

    // Reallocate `minpoints` to truncate non-written elements
    minpoints = (double*)realloc(minpoints, sizeof(double) * m * *mpts_ncols);
    *mpts_nrows = m;

    // Free heap-allocated memory and return ownership of `minpoints`
    free(indices);

    return minpoints;
}
