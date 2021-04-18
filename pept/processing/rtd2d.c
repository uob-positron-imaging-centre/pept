/**
 * File   : rtd2d.c
 * License: MIT
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 06.04.2021
 */


#include "rtd2d.h"


#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define RTD2D_PI 3.14159265358979323846


#ifdef SINGLE_PRECISION
    #define RTD2D_SQRT(x) sqrtf(x)
    #define RTD2D_COS(x) cosf(x)
    #define RTD2D_SIN(x) sinf(x)
    #define RTD2D_ATAN2(x, y) atan2f((x), (y))
#else
    #define RTD2D_SQRT(x) sqrt(x)
    #define RTD2D_COS(x) cos(x)
    #define RTD2D_SIN(x) sin(x)
    #define RTD2D_ATAN2(x, y) atan2((x), (y))
#endif


/**
 * Return the Euclidean distance between two points.
 */
r2d_real        rtd2d_distance(const r2d_rvec2 p1, const r2d_rvec2 p2)
{
    r2d_real    dx = p2.x - p1.x;
    r2d_real    dy = p2.y - p2.y;

    return RTD2D_SQRT(dx * dx + dy * dy);
}


/**
 * Approximate a circle as a polygon with `RTD2D_NUM_VERTS` vertices and save it as a `r2d_poly`.
 */
r2d_real        rtd2d_circle(r2d_poly *poly, const r2d_rvec2 centre, const r2d_real radius)
{
    r2d_rvec2   verts[RTD2D_NUM_VERTS];

    r2d_real    inc;
    r2d_int     i;

    inc = 2 * RTD2D_PI / RTD2D_NUM_VERTS;
    for (i = 0; i < RTD2D_NUM_VERTS; ++i)
    {
        verts[i].x = centre.x + RTD2D_COS(i * inc) * radius;
        verts[i].y = centre.y + RTD2D_SIN(i * inc) * radius;
    }

    r2d_init_poly(poly, verts, RTD2D_NUM_VERTS);

    return RTD2D_PI * radius * radius;
}


/**
 * Approximate a 2D cylinder (i.e. the convex hull of two circles) with `RTD2D_NUM_VERTS` vertices
 * and save it as a `r2d_poly`. Omit the second circle's area. Return the full cylinder's area.
 */
r2d_real        rtd2d_half_cylinder(r2d_poly *poly, const r2d_rvec2 p1, const r2d_rvec2 p2,
                                    const r2d_real radius)
{
    r2d_rvec2   verts[RTD2D_NUM_VERTS];

    r2d_real    ang;
    r2d_real    start_ang, end_ang;
    r2d_real    inc;

    r2d_int     NUM_VERTS_2 = RTD2D_NUM_VERTS / 2;
    r2d_int     i;

    // Get the angle between [0, 2pi] using an atan2 trick (atan2 returns [-pi, pi])
    ang = RTD2D_PI - RTD2D_ATAN2(p2.y - p1.y, -(p2.x - p1.x));

    start_ang = RTD2D_PI / 2 + ang;
    end_ang = 3 * RTD2D_PI / 2 + ang;

    inc = (end_ang - start_ang) / NUM_VERTS_2;
    for (i = 0; i < NUM_VERTS_2; ++i)
    {
        verts[i].x = p1.x + RTD2D_COS(start_ang + i * inc) * radius;
        verts[i].y = p1.y + RTD2D_SIN(start_ang + i * inc) * radius;
    }

    inc = (start_ang - end_ang) / NUM_VERTS_2;
    for (i = 0; i < NUM_VERTS_2; ++i)
    {
        verts[i + NUM_VERTS_2].x = p2.x + RTD2D_COS(end_ang + i * inc) * radius;
        verts[i + NUM_VERTS_2].y = p2.y + RTD2D_SIN(end_ang + i * inc) * radius;
    }

    r2d_init_poly(poly, verts, RTD2D_NUM_VERTS);

    return rtd2d_distance(p1, p2) * 2 * radius + RTD2D_PI * radius * radius;
}


/**
 * Rasterize a polygon `poly` onto a grid `grid` with `ny` columns and xy `grid_size`, using a
 * local `lgrid` for temporary calculations in the pixels spanning the rectangular approximation
 * of the polygon.
 *
 * The area ratio is multiplied by `factor` and *added* onto the global `grid`. If you would like
 * to subtract the area from the `grid`, simply make `factor` negative.
 *
 * The local grid `lgrid` is reinitialised to zero at the end of the function.
 */
void            rtd2d_raster(r2d_poly *poly,
                             r2d_real *grid,
                             r2d_real *lgrid,
                             const r2d_int dims[2],
                             const r2d_rvec2 grid_size,
                             const r2d_real factor)
{
    r2d_dvec2   clampbox[2] = {{{0, 0}}, {{dims[0], dims[1]}}};
    r2d_dvec2   ibox[2];        // Local grid's range of indices in the global grid
    r2d_int     lx, ly;         // Local grid's written number of rows and columns
    r2d_int     i, j;           // Iterators

    // Find the range of indices spanned by `poly`, then clamp them if `poly` extends out of `grid`
    r2d_get_ibox(poly, ibox, grid_size);
    r2d_clamp_ibox(poly, ibox, clampbox, grid_size);

    // Initialise local grid for the pixellisation step
    lx = ibox[1].i - ibox[0].i;
    ly = ibox[1].j - ibox[0].j;

    // Rasterize the polygon onto the local grid and compute the total area occupied by `poly`
    r2d_rasterize(poly, ibox, lgrid, grid_size, 0);

    // Add the ratio of one pixel's occupied area and the total area from the local grid to the
    // global grid, multiplied by `factor`.
    for (i = ibox[0].i; i < ibox[1].i; ++i)
        for (j = ibox[0].j; j < ibox[1].j; ++j)
            grid[i * dims[1] + j] += factor * lgrid[(i - ibox[0].i) * ly + j - ibox[0].j];

    // Reinitialise the written local grid to zero
    for (i = 0; i < lx * ly; ++i)
        lgrid[i] = 0.;
}


/**
 * Compute the occupancy grid of a single circular particle's trajectory.
 *
 * This corresponds to the pixellisation of moving circular particles, such that for every two
 * consecutive particle locations, a 2D cylinder (i.e. convex hull of two circles at the two
 * particle positions), the fraction of its area that intersets a pixel is multiplied with the
 * time between the two particle locations and saved in the input `pixels`.
 */
void            rtd2d_occupancy(r2d_real        *pixels,
                                const r2d_int   dims[2],
                                const r2d_real  xlim[2],
                                const r2d_real  ylim[2],
                                const r2d_real  *positions,
                                const r2d_real  *times,
                                const r2d_int   num_positions,
                                const r2d_real  radius,
                                const r2d_int   omit_last)
{
    // Some cheap input parameters' checks
    if (dims[0] < 2 || dims[1] < 2 || num_positions < 2)
    {
        perror("[ERROR]: The input grid should be at least 2 pixels wide and tall, and there "
               "should be at least two particle positions.");
        return;
    }

    // Auxilliaries
    r2d_int     ip;             // Trajectory particle index
    r2d_real    duration;       // Time between two consecutive locations
    r2d_real    area;           // Total area for one 2D cylinder

    // Initialise global pixel grid
    r2d_real    xsize = xlim[1] - xlim[0];
    r2d_real    ysize = ylim[1] - ylim[0];

    r2d_rvec2   grid_size = {{xsize / dims[0], ysize / dims[1]}};

    // Local grid which will be used for rasterising
    r2d_real    *lgrid = (r2d_real*)calloc((size_t)dims[0] * dims[1], sizeof(r2d_real));

    // Polygonal shapes used for the particle trajectories
    r2d_poly    cylinder;
    r2d_poly    circle;

    // Copy `positions` to new local array and translate them such that the grid origin is (0, 0)
    r2d_rvec2   *trajectory = (r2d_rvec2*)malloc(sizeof(r2d_rvec2) * num_positions);

    for (ip = 0; ip < num_positions; ++ip)
    {
        trajectory[ip].x = positions[2 * ip] - xlim[0];
        trajectory[ip].y = positions[2 * ip + 1] - ylim[0];
    }

    // Rasterize particle trajectories
    for (ip = 0; ip < num_positions - 1; ++ip)
    {
        duration = times[ip + 1] - times[ip];

        // Create a polygonal approximation of the convex hull of the two particle locations, minus
        // the second circle's area (because that was already added at the previous iteration)
        area = rtd2d_half_cylinder(&cylinder, trajectory[ip], trajectory[ip + 1], radius);
        rtd2d_raster(&cylinder, pixels, lgrid, dims, grid_size, duration / area);
    }

    if (!omit_last)
    {
        rtd2d_circle(&circle, trajectory[num_positions - 1], radius);
        rtd2d_raster(&circle, pixels, lgrid, dims, grid_size, duration / area);
    }

    free(lgrid);
    free(trajectory);
}
