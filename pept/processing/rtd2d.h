/**
 * File   : rtd2d.h
 * License: MIT
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 06.04.2021
 */


#ifndef _RTD2D_H_
#define _RTD2D_H_


/**
 * User-changeable macros:
 *
 * RTD2D_NUM_VERTS : number of vertices used to approximate circles / cylinders as polygons.
 * RTD2D_MAX_VERTS : maximum number of vertices used in the R2D internal polygon representation.
 * SINGLE_PRECISION : if defined, single-precision floats will be used for calculations.
 */
#define RTD2D_NUM_VERTS 32
#define RTD2D_MAX_VERTS 64
#define SINGLE_PRECISION


/**
 * Macros needed by R2D
 */
#define R2D_MAX_VERTS RTD2D_MAX_VERTS
#include "r2d.h"


/**
 * Compute the occupancy grid of a single circular particle's trajectory.
 *
 * This corresponds to the pixellisation of moving circular particles, such that for every two
 * consecutive particle locations, a 2D cylinder (i.e. convex hull of two circles at the two
 * particle positions), the fraction of its area that intersets a pixel is multiplied with the
 * time between the two particle locations and saved in the input `pixels`.
 *
 * Computing the occupancy grid of a single particle's trajectory by accurately rasterizing its
 * positions onto a pixel grid means:
 *
 * 1. The particle occupies a finite area that is pixellised.
 * 2. The particle movement between two positions is considered as a cylinder (i.e. the convex
 *    hull of the start and end circles).
 * 3. The particle circle / trajectory cylinder areas are pixellised accurately, conserving the
 *    area of polygon-pixel intersections.
 *
 * The particle circle / trajectory cylinder areas are approximated using `RTD2D_NUM_VERTS` points
 * defining a polygon.
 *
 * Important: it is assumed that the particle locations are all fully inside the pixel grid.
 *
 * Types
 * -----
 * The calculations can be done either in double or single precision - by default, the
 * `SINGLE_PRECISION` macro flag is set, so `r2d_real` is equivalent to a `float`:
 *
 * r2d_real :   `float` if SINGLE_PRECISION else `double`
 * r2d_int :    `int32_t`
 *
 * Parameters
 * ----------
 * pixels : r2d_real*
 *      The input flattened 2D grid of pixels, given in row-major order, onto which the particle
 *      trajectory area will be rasterized.
 *
 * dims : r2d_int[2]
 *      The number of rows and columns of the input `pixels`.
 *
 * xlim : r2d_real[2]
 *      The physical limits of the pixel grid in the x-dimension.
 *
 * ylim : r2d_real[2]
 *      The physical limits of the pixel grid in the y-dimension.
 *
 * positions : r2d_real*
 *      The input flattened (`num_positions`, 2) array of 2D particle positions. Has
 *      `2 * num_positions` elements.
 *
 * times : r2d_real*
 *      The timestamps of each particle location in `positions`. Has length `num_positions`.
 *
 * num_positions : r2d_int
 *      The number of rows in the input `positions` array.
 *
 * radius : r2d_real
 *      The particle radius, assumed to be constant throughout the dataset.
 *
 * omit_last : r2d_int
 *      A boolean signifying whether to omit the last circle in the particle positions. Useful
 *      if rasterizing the same trajectory piece-wise; if you split the trajectory and call this
 *      function multiple times, set `omit_last = 0` to avoid considering the last location twice.
 *
 * Undefined Behaviour
 * -------------------
 * Beyond some cheap sanity checks (at least 2x2 pixel grid and two positions), this function does
 * not check the mathematical validity of the inputs.
 */
void    rtd2d_occupancy(r2d_real        *pixels,
                        const r2d_int   dims[2],
                        const r2d_real  xlim[2],
                        const r2d_real  ylim[2],
                        const r2d_real  *positions,
                        const r2d_real  *times,
                        const r2d_int   num_positions,
                        const r2d_real  radius,
                        const r2d_int   omit_last);


#endif  // _RTD2D_H_
