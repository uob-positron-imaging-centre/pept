/**
 * File              : birmingham_method,cu
 * License           : License: GNU v3.0
 * Author            : Dominik Werner
 * Date              : 29.12.2022
 */

#include <float.h>
#include <stdio.h>
__device__ void calculate(float *a12, float *b12, float *c12, float *d12,
        float *e12, float *f12, float *p12, float *q12,
        float *r12, float *xx1, float *yy1, float *zz1,
        float *x12, float *y12, float *z12, float *r2,
        float *ttt, float *dev, int *used, int nused,
        int nrows, float *calculate_results, int sample_idx)
{
    float x, y, z, error, avtime, dx = 0, dy = 0, dz = 0, dd;
    float suma, sumb, sumc, sumd, sume, sumf, sump, sumq, sumr;
    float ab, dq, dp, ar, ac, denom;

    int it;
    int start_data_index = sample_idx * nrows;
    suma = sumb = sumc = sumd = sume = sumf = sump = sumq = sumr = 0;

    for (int line_number_in_sample = 0; line_number_in_sample < nrows; line_number_in_sample++)
    {
        it = start_data_index + line_number_in_sample;
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
    for (int line_number_in_sample = 0; line_number_in_sample < nrows; line_number_in_sample++)
    {
        it = start_data_index + line_number_in_sample;
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


__global__ void birmingham_method(
    float *lines,       // flattened 2D numpy array of LoRs
    int nrows,          // number of rows in `lines` per sample
    int ncols,          // number of columns `lines`
    int overlap,
    int nsamples,       // number of samples
    float fopt,          // fraction of LoRs used to find location
    float *location,    // (5 * max_samples) numpy array for the found location
    int *used,          // (nrows *max_samples ) numpy array
    float *ttt,         // those array have nlors*max_samples elements
    float *xx1,
    float *xx2,
    float *yy1,
    float *yy2,
    float *zz1,
    float *zz2,
    float *x12,
    float *y12,
    float *z12,
    float *r12,
    float *q12,
    float *p12,
    float *a12,
    float *b12,
    float *c12,
    float *d12,
    float *e12,
    float *f12,
    float *r2,
    float *dev)
{
    int verbose = 0;
    int sync = 0;
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Sample index %d, nsamples %d\n", sample_idx, nsamples);
    if (sample_idx < nsamples)
    {
    if (verbose){
        printf("In Kernel! Calculate sample number %d nrows: %d, ncols: %d, nsamples: %d \n", sample_idx, nrows, ncols, nsamples);
    }
    //printf("N samples %d", nrows);
    // index of the first element of the current sample
    int nlors = nrows;
    int start_lor_index = sample_idx * (nlors-overlap) * ncols;
    int start_data_index = sample_idx * nlors;
    float error;

    float Con_factor = 150; // Convergence factor when removing LORs from set

    // the following variables are used while iteratively removing lines
    float dismin, dismax;
    int imin, nused;

    int imax, nprev, nfin;
    float calculate_results[8];
    if (verbose){
    printf("Start loop over lines, Current sample index: %d", sample_idx);

    // all the prins in one print statement:
    printf("In Kernel! Calculate sample number %d nrows: %d, ncols: %d, nsamples: %d\n \
    Nlors: %d, start_lor_index: %d, start_data_index: %d\n \
    First LOR in sample: %f, %f, %f, %f, %f, %f, %f\n \
    Second LOR in sample: %f, %f, %f, %f, %f, %f, %f\n", sample_idx, nrows, ncols, nsamples, nlors, start_lor_index, start_data_index, lines[start_lor_index], lines[start_lor_index + 1], lines[start_lor_index + 2], lines[start_lor_index + 3], lines[start_lor_index + 4], lines[start_lor_index + 5], lines[start_lor_index + 6], lines[start_lor_index + 7], lines[start_lor_index + 8], lines[start_lor_index + 9], lines[start_lor_index + 10], lines[start_lor_index + 11], lines[start_lor_index + 12]);
    }
    for (int lor_idx = 0; lor_idx < nlors; ++lor_idx) // for loop over each line in this sample
    {

        // the real index is the sample_index + the thread index which points to the first element of the sample
        int current_lor_index = start_lor_index + lor_idx * ncols;
        int current_data_index = start_data_index + lor_idx;
        if (verbose){
        printf("Sample index: %d, LOR index: %d current_lor_index: %d, current_data_index: %d\n", sample_idx, lor_idx, current_lor_index, current_data_index);

        printf("Current LOR index: %d, Current data index: %d\n", current_lor_index, current_data_index);
        }
        // lors are stored in array of shape (nrows*ncols*nsamples)
        ttt[current_data_index] = lines[current_lor_index];
        xx1[current_data_index] = lines[current_lor_index + 1];
        yy1[current_data_index] = lines[current_lor_index + 2];
        zz1[current_data_index] = lines[current_lor_index + 3];
        xx2[current_data_index] = lines[current_lor_index + 4];
        yy2[current_data_index] = lines[current_lor_index + 5];
        zz2[current_data_index] = lines[current_lor_index + 6];
        used[current_data_index] = 1;

        // Calculate vectors for set of LORs, to be used in calculate
        x12[ current_data_index] = xx1[ current_data_index] - xx2[ current_data_index];    // Point 2 -> Point 1 vector in x-axis
        y12[ current_data_index] = yy1[ current_data_index] - yy2[ current_data_index];    // Point 2 -> Point 1 vector in y-axis
        z12[ current_data_index] = zz1[ current_data_index] - zz2[ current_data_index];    // Point 2 -> Point 1 vector in z-axis

        // Magn current_data_indexude of vector from P2 to P1
        r2[ current_data_index] = (x12[ current_data_index] * x12[ current_data_index]) + (y12[ current_data_index] * y12[ current_data_index]) +
            (z12[ current_data_index] * z12[ current_data_index]);

        r12[ current_data_index] = (y12[ current_data_index] * z12[ current_data_index]) / r2[ current_data_index];
        q12[ current_data_index] = (x12[ current_data_index] * z12[ current_data_index]) / r2[ current_data_index];
        p12[ current_data_index] = (x12[ current_data_index] * y12[ current_data_index]) / r2[ current_data_index];

        a12[ current_data_index] = ((y12[ current_data_index] * y12[ current_data_index]) + (z12[ current_data_index] * z12[ current_data_index])) / r2[ current_data_index];
        b12[ current_data_index] = ((x12[ current_data_index] * x12[ current_data_index]) + (z12[ current_data_index] * z12[ current_data_index])) / r2[ current_data_index];
        c12[ current_data_index] = ((y12[ current_data_index] * y12[ current_data_index]) + (x12[ current_data_index] * x12[ current_data_index])) / r2[ current_data_index];

        d12[ current_data_index] = ((yy2[ current_data_index] * xx1[ current_data_index] - yy1[ current_data_index] * xx2[ current_data_index]) * y12[ current_data_index] +
                (zz2[ current_data_index] * xx1[ current_data_index] - zz1[ current_data_index] * xx2[ current_data_index]) * z12[ current_data_index]) / r2[ current_data_index];
        e12[ current_data_index] = ((zz2[ current_data_index] * yy1[ current_data_index] - zz1[ current_data_index] * yy2[ current_data_index]) * z12[ current_data_index] +
                (xx2[ current_data_index] * yy1[ current_data_index] - xx1[ current_data_index] * yy2[ current_data_index]) * x12[ current_data_index]) / r2[ current_data_index];
        f12[ current_data_index] = -((zz2[ current_data_index] * yy1[ current_data_index] - zz1[ current_data_index] * yy2[ current_data_index]) * y12[ current_data_index] +
                (zz2[ current_data_index] * xx1[ current_data_index] - zz1[ current_data_index] * xx2[ current_data_index]) * x12[ current_data_index]) / r2[ current_data_index];
    }
    // The total number of lines used to determine final tracer position.
    // Initially use all of them.
    nused = nrows;

    // The target number of lines to be used to determine the final position.
    nfin = (int)(nrows * fopt);

    int iteration = 0;
    if (sync){
        __syncthreads();
        }

    while (nrows > 0)
    {
        iteration += 1;
        imin = 0;
        imax = 0;

        calculate(a12, b12, c12, d12, e12, f12, p12, q12, r12, xx1, yy1, zz1,
                x12, y12, z12, r2, ttt, dev, used, nused, nrows,
                calculate_results, sample_idx);
        if (verbose){
            printf("Sample: %d,iteration : %d, Nused : %d nfin : %d Calculated results to be : %f, %f, %f, %f, %f, %f, %f, %f\n", sample_idx, iteration, nused, nfin, calculate_results[0], calculate_results[1], calculate_results[2], calculate_results[3], calculate_results[4], calculate_results[5], calculate_results[6], calculate_results[7]);
        }
        // We used the target number of lines to calculate the tracer position.
        if (nused == nfin)
        {

            location[sample_idx*5+0] = calculate_results[0]; // t
            location[sample_idx*5+1] = calculate_results[1]; // x
            location[sample_idx*5+2] = calculate_results[2]; // y
            location[sample_idx*5+3] = calculate_results[3]; // z
            location[sample_idx*5+4] = calculate_results[7]; // error
            return;
        }
        if (sync){
        __syncthreads();
        }
        error = calculate_results[7];
        nprev = nused;
        nused = 0;

        // Iterate through the lines; if they lie too far away from the
        // calculated centroid then we won't use them anymore.
        for (int lor_idx = 0; lor_idx < nrows; lor_idx++)
        {
            int current_data_index = lor_idx + start_data_index;
            if (sqrt(dev[current_data_index]) > (Con_factor * error / 100))
                used[current_data_index] = 0;
            else
            {
                used[current_data_index] = 1;
                nused = nused + 1;
            }
        }

        // If true, we have reduced to too few events, so restore closest
        // unused events
        while (nused < nfin)
        {
            dismin = DBL_MAX;
            for (int lor_idx = 0; lor_idx < nrows; lor_idx++)
            {
                int current_data_index = lor_idx + start_data_index;
                if (used[current_data_index] == 0 && dev[current_data_index] < dismin)
                {
                    imin = current_data_index;
                    dismin = dev[current_data_index];
                }
            }
            used[imin] = 1;
            nused = nused + 1;
        }
        if (sync){
        __syncthreads();
        }
        // If true then we haven't removed any, so remove furthest point
        while (nused >= nprev)
        {
            dismax = 0.0;
            for (int lor_idx = 0; lor_idx < nrows; lor_idx++)
            {
                int current_data_index = lor_idx + start_data_index;
                if(used[current_data_index] == 1 && dev[current_data_index] > dismax)
                {
                    imax = current_data_index;
                    dismax = dev[current_data_index];
                }
            }
            used[imax] = 0;
            nused = nused - 1;
        }
        if (sync){
        __syncthreads();
        }
    }

    }
}

