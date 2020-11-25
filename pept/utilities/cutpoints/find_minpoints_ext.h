/**
 * File   : find_minpoints_ext.h
 * License: GNU v3.0
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 20.10.2020
 */

#ifndef FIND_MINPOINTS_EXT
#define FIND_MINPOINTS_EXT

#include <sys/types.h>

double* find_minpoints_ext(
    const double *sample_lines,
    const ssize_t nrows,
    const ssize_t ncols,
    const ssize_t num_lines,
    const double max_distance,
    const double *cutoffs,
    const int append_indices,
    ssize_t *mpts_nrows,
    ssize_t *mpts_ncols
);

#endif
