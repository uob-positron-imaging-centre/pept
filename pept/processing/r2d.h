/*
 *
 *		r2d.h
 *		
 *		Routines for fast, geometrically robust clipping operations
 *		and analytic area/moment computations over polygons in 2D. 
 *		
 *		Devon Powell
 *		31 August 2015
 *
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 *
 */


/**
 *      Andrei Leonard Nicusan inlined all the function declarations from `v2d.h` and definitions from
 *      `r2d.c` and `v2d.c` in April 2021 for the purposes of creating a header-only library. The
 *      function definitions were given `static inline` to maximise optimisation possibilities in user
 *      code at the expense of code bloat - which is arguably reasonable if R2D is used only from a
 *      single translation unit in a library and a `polyorder = 0` is always used.
 *
 *      All rights go to the original authors of the library, whose copyright notice is included at the
 *      top of this file. A sincere thank you for your work.
 */


#ifndef _R2D_H_
#define _R2D_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * \brief Use the single precision power function if `SINGLE_PRECISION` is defined.
 */
#ifdef SINGLE_PRECISION
    #define R2D_POW(x, y) powf((x), (y))
#else 
    #define R2D_POW(x, y) pow((x), (y))
#endif


/**
 * \file r2d.h
 * \author Devon Powell
 * \date 31 August 2015
 * \brief Interface for r2d
 */

/**
 * \brief Real type specifying the precision to be used in calculations
 *
 * Default is `double`. `float` precision is enabled by 
 * compiling with `-DSINGLE_PRECISION`.
 */
#ifdef SINGLE_PRECISION
typedef float r2d_real;
#else 
typedef double r2d_real;
#endif

/**
 * \brief Integer types used for indexing
 */
typedef int32_t r2d_int;
typedef int64_t r2d_long;

/** \struct r2d_rvec2
 *  \brief A 2-vector.
 */
typedef union {
	struct {
		r2d_real x, /*!< \f$x\f$-component. */
				 y; /*!< \f$y\f$-component. */
	};
	r2d_real xy[2]; /*!< Index-based access to components. */
} r2d_rvec2;

/** \struct r2d_dvec2
 *  \brief An integer 2-vector for grid indexing.
 */
typedef union {
	struct {
		r2d_int i, /*!< \f$x\f$-component. */
				j; /*!< \f$y\f$-component. */
	};
	r2d_int ij[2]; /*!< Index-based access to components. */
} r2d_dvec2;

/** \struct r2d_plane
 *  \brief A plane.
 */
typedef struct {
	r2d_rvec2 n; /*!< Unit-length normal vector. */
	r2d_real d; /*!< Signed perpendicular distance to the origin. */
} r2d_plane;

/** \struct r2d_vertex
 * \brief A doubly-linked vertex.
 */
typedef struct {
	r2d_int pnbrs[2]; /*!< Neighbor indices. */
	r2d_rvec2 pos; /*!< Vertex position. */
} r2d_vertex;

/** \struct r2d_poly
 * \brief A polygon. Can be convex, nonconvex, even multiply-connected.
 */
typedef struct {
	r2d_vertex verts[R2D_MAX_VERTS]; /*!< Vertex buffer. */
	r2d_int nverts; /*!< Number of vertices in the buffer. */
} r2d_poly;

/**
 * \brief Clip a polygon against an arbitrary number of clip planes (find its intersection with a set of half-spaces). 
 *
 * \param [in, out] poly 
 * The polygon to be clipped. 
 *
 * \param [in] planes 
 * An array of planes against which to clip this polygon.
 *
 * \param[in] nplanes 
 * The number of planes in the input array. 
 *
 */
static inline void r2d_clip(r2d_poly* poly, r2d_plane* planes, r2d_int nplanes);

/**
 * \brief Splits a list of polygons across a single plane.  
 *
 * \param [in] inpolys 
 * Array input polyhedra to be split 
 *
 * \param [in] npolys 
 * The number of input polygons
 *
 * \param [in] plane 
 * The plane about which to split the input polys 
 *
 * \param[out] out_pos 
 * The output array of fragments on the positive side of the clip plane. Must be at least npolys
 * long. out_pos[i] and out_neg[i] correspond to inpolys[i], where out_neg[i].nverts or
 * out.pos[i].nverts are set to zero if the poly lies entirely in the positive or negative side of
 * the plane, respectively.
 *
 * \param[out] out_neg 
 * The output array of fragments on the negitive side of the clip plane. Must be at least npolys
 * long. 
 *
 */
static inline void r2d_split(r2d_poly* inpolys, r2d_int npolys, r2d_plane plane, r2d_poly* out_pos, r2d_poly* out_neg);

/**
 * \brief Integrate a polynomial density over a polygon using simplicial decomposition.
 * Uses the fast recursive method of Koehl (2012) to carry out the integration.
 *
 * \param [in] poly
 * The polygon over which to integrate.
 *
 * \param [in] polyorder
 * Order of the polynomial density field. 0 for constant (1 moment), 1 for linear
 * (4 moments), 2 for quadratic (10 moments), etc.
 *
 * \param [in, out] moments
 * Array to be filled with the integration results, up to the specified `polyorder`. Must be at
 * least `(polyorder+1)*(polyorder+2)/2` long. A conventient macro,
 * `R2D_NUM_MOMENTS()` is provided to compute the number of moments for a given order.
 * Order of moments is row-major, i.e. `1`, `x`, `y`, `x^2`, `x*y`, `y^2`, `x^3`, `x^2*y`...
 *
 */
#define R2D_NUM_MOMENTS(order) ((order+1)*(order+2)/2)
static inline void r2d_reduce(r2d_poly* poly, r2d_real* moments, r2d_int polyorder);

/**
 * \brief Checks a polygon to see if all vertices have two valid edges, that all vertices are
 * pointed to by two other vertices, and that there are no vertices that point to themselves. 
 *
 * \param [in] poly
 * The polygon to check.
 *
 * \return
 * 1 if the polygon is good, 0 if not. 
 *
 */
static inline r2d_int r2d_is_good(r2d_poly* poly);

/**
 * \brief Calculates a center of a polygon.
 *
 * \param [in] poly
 * The polygon to check.
 *
 * \return
 * coordinates of a polygon center.
 *
 */
static inline r2d_rvec2 r2d_poly_center(r2d_poly* poly);

/**
 * \brief Adjust moments according to the shift of polygon vertices to the origin.
 *
 * \param [in, out] moments
 * The moments of the shifted polygon.
 *
 * \param [in] polyorder
 * Order of the polygon.
 *
 * \param [in] vc
 * Coordinates of the polygon center, which are used to shift the polygon.
 *
 */
static inline void r2d_shift_moments(r2d_real* moments, r2d_int polyorder, r2d_rvec2 vc);

/**
 * \brief Get the signed volume of the triangle defined by the input vertices. 
 *
 * \param [in] pa, pb, pc
 * Vertices defining a triangle from which to calculate an area. 
 *
 * \return
 * The signed volume of the input triangle.
 *
 */
static inline r2d_real r2d_orient(r2d_rvec2 pa, r2d_rvec2 pb, r2d_rvec2 pc);

/**
 * \brief Prints the vertices and connectivity of a polygon. For debugging. 
 *
 * \param [in] poly
 * The polygon to print.
 *
 */
static inline void r2d_print(r2d_poly* poly);

/**
 * \brief Rotate a polygon about the origin. 
 *
 * \param [in, out] poly 
 * The polygon to rotate. 
 *
 * \param [in] theta 
 * The angle, in radians, by which to rotate the polygon.
 *
 */
static inline void r2d_rotate(r2d_poly* poly, r2d_real theta);

/**
 * \brief Translate a polygon. 
 *
 * \param [in, out] poly 
 * The polygon to translate. 
 *
 * \param [in] shift 
 * The vector by which to translate the polygon. 
 *
 */
static inline void r2d_translate(r2d_poly* poly, r2d_rvec2 shift);

/**
 * \brief Scale a polygon.
 *
 * \param [in, out] poly 
 * The polygon to scale. 
 *
 * \param [in] shift 
 * The factor by which to scale the polygon. 
 *
 */
static inline void r2d_scale(r2d_poly* poly, r2d_real scale);

/**
 * \brief Shear a polygon. Each vertex undergoes the transformation
 * `pos.xy[axb] += shear*pos.xy[axs]`.
 *
 * \param [in, out] poly 
 * The polygon to shear. 
 *
 * \param [in] shear 
 * The factor by which to shear the polygon. 
 *
 * \param [in] axb 
 * The axis (0 or 1 corresponding to x or y) along which to shear the polygon.
 *
 * \param [in] axs 
 * The axis (0 or 1 corresponding to x or y) across which to shear the polygon.
 *
 */
static inline void r2d_shear(r2d_poly* poly, r2d_real shear, r2d_int axb, r2d_int axs);

/**
 * \brief Apply a general affine transformation to a polygon. 
 *
 * \param [in, out] poly 
 * The polygon to transform.
 *
 * \param [in] mat 
 * The 3x3 homogeneous transformation matrix by which to transform
 * the vertices of the polygon.
 *
 */
static inline void r2d_affine(r2d_poly* poly, r2d_real mat[3][3]);

/**
 * \brief Initialize a polygon as an axis-aligned box. 
 *
 * \param [out] poly
 * The polygon to initialize.
 *
 * \param [in] rbounds
 * An array of two vectors, giving the lower and upper corners of the box.
 *
 */
static inline void r2d_init_box(r2d_poly* poly, r2d_rvec2* rbounds);

/**
 * \brief Initialize a (simply-connected) general polygon from a list of vertices. 
 * Can use `r2d_is_good` to check that the output is valid.
 *
 * \param [out] poly
 * The polygon to initialize. 
 *
 * \param [in] vertices
 * Array of length `numverts` giving the vertices of the input polygon, in counterclockwise order.
 *
 * \param [in] numverts
 * Number of vertices in the input polygon. 
 *
 */
static inline void r2d_init_poly(r2d_poly* poly, r2d_rvec2* vertices, r2d_int numverts);

/**
 * \brief Get four faces (unit normals and distances to the origin)
 * from a two-vertex description of an axis-aligned box.
 *
 * \param [out] faces
 * Array of four planes defining the faces of the box.
 *
 * \param [in] rbounds
 * Array of two vectors defining the bounds of the axis-aligned box 
 *
 */
static inline void r2d_box_faces_from_verts(r2d_plane* faces, r2d_rvec2* rbounds);

/**
 * \brief Get all faces (unit normals and distances to the origin)
 * from a general boundary description of a polygon.
 *
 * \param [out] faces
 * Array of planes of length `numverts` defining the faces of the polygon.
 *
 * \param [in] vertices
 * Array of length `numverts` giving the vertices of the input polygon, in counterclockwise order. 
 *
 * \param [in] numverts
 * Number of vertices in the input polygon. 
 *
 */
static inline void r2d_poly_faces_from_verts(r2d_plane* faces, r2d_rvec2* vertices, r2d_int numverts);


/**
 * \file v2d.h
 * \author Devon Powell
 * \date 15 October 2015
 * \brief Interface for r2d rasterization routines
 */

/**
 * \brief Rasterize a polygon to the destination grid.
 *
 * \param [in] poly 
 * The polygon to be rasterized.
 *
 * \param [in] ibox 
 * Minimum and maximum indices of the polygon, found with `r2d_get_ibox()`. These indices are
 * from a virtual grid starting at the origin. 
 *
 * \param [in, out] dest_grid 
 * The rasterization buffer. This grid is a row-major grid patch starting at `d*ibox[0]` and ending
 * at `d*ibox[1]. Must have size of at least 
 * `(ibox[1].i-ibox[0].i)*(ibox[1].j-ibox[0].j)*(ibox[1].k-ibox[0].k)*R2D_NUM_MOMENTS(polyorder)`.
 *
 * \param [in] d 
 * The cell spacing of the grid.
 *
 * \param [in] polyorder
 * Order of the polynomial density field to rasterize. 
 * 0 for constant (1 moment), 1 for linear (3 moments), 2 for quadratic (6 moments), etc.
 *
 */
static inline void r2d_rasterize(r2d_poly* poly, r2d_dvec2 ibox[2], r2d_real* dest_grid, r2d_rvec2 d, r2d_int polyorder);

/**
 * \brief Get the minimal box of grid indices for a polygon, given a grid cell spacing,
 * also clamping it to a user-specified range while clipping the polygon to that range.
 *
 * \param [in] poly 
 * The polygon for which to calculate the index box and clip.
 *
 * \param [in, out] ibox 
 * Minimal range of grid indices covered by the polygon.
 *
 * \param [in, out] clampbox 
 * Range of grid indices to which to clamp and clip `ibox` and `poly`, respectively. 
 *
 * \param [in] d 
 * The cell spacing of the grid. The origin of the grid is assumed to lie at the origin in space.
 *
 */
static inline void r2d_clamp_ibox(r2d_poly* poly, r2d_dvec2 ibox[2], r2d_dvec2 clampbox[2], r2d_rvec2 d);

/**
 * \brief Get the minimal box of grid indices for a polygon, given a grid cell spacing.
 *
 * \param [in] poly 
 * The polygon for which to calculate the index box.
 *
 * \param [out] ibox 
 * Minimal range of grid indices covered by the polygon.
 *
 * \param [in] d 
 * The cell spacing of the grid. The origin of the grid is assumed to lie at the origin in space.
 *
 */
static inline void r2d_get_ibox(r2d_poly* poly, r2d_dvec2 ibox[2], r2d_rvec2 d);


// TODO: make this a generic "split" routine that just takes a plane.
static inline void r2d_split_coord(r2d_poly* inpoly, r2d_poly** outpolys, r2d_real coord, r2d_int ax);


// useful macros
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
#define dot(va, vb) (va.x*vb.x + va.y*vb.y)
#define wav(va, wa, vb, wb, vr) {			\
	vr.x = (wa*va.x + wb*vb.x)/(wa + wb);	\
	vr.y = (wa*va.y + wb*vb.y)/(wa + wb);	\
}
#define norm(v) {					\
	r2d_real tmplen = sqrt(dot(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
}


static inline void r2d_clip(r2d_poly* poly, r2d_plane* planes, r2d_int nplanes) {

	// variable declarations
	r2d_int v, p, np, onv, vstart, vcur, vnext, numunclipped; 

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 
	if(*nverts <= 0) return;

	// signed distances to the clipping plane
	r2d_real sdists[R2D_MAX_VERTS];
	r2d_real smin, smax;

	// for marking clipped vertices
	r2d_int clipped[R2D_MAX_VERTS];

	// loop over each clip plane
	for(p = 0; p < nplanes; ++p) {
	
		// calculate signed distances to the clip plane
		onv = *nverts;
		smin = 1.0e30;
		smax = -1.0e30;
		memset(&clipped, 0, sizeof(clipped));
		for(v = 0; v < onv; ++v) {
			sdists[v] = planes[p].d + dot(vertbuffer[v].pos, planes[p].n);
			if(sdists[v] < smin) smin = sdists[v];
			if(sdists[v] > smax) smax = sdists[v];
			if(sdists[v] < 0.0) clipped[v] = 1;
		}

		// skip this face if the poly lies entirely on one side of it 
		if(smin >= 0.0) continue;
		if(smax <= 0.0) {
			*nverts = 0;
			return;
		}

		// check all edges and insert new vertices on the bisected edges 
		for(vcur = 0; vcur < onv; ++vcur) {
			if(clipped[vcur]) continue;
			for(np = 0; np < 2; ++np) {
				vnext = vertbuffer[vcur].pnbrs[np];
				if(!clipped[vnext]) continue;
				vertbuffer[*nverts].pnbrs[1-np] = vcur;
				vertbuffer[*nverts].pnbrs[np] = -1;
				vertbuffer[vcur].pnbrs[np] = *nverts;
				wav(vertbuffer[vcur].pos, -sdists[vnext],
					vertbuffer[vnext].pos, sdists[vcur],
					vertbuffer[*nverts].pos);
				(*nverts)++;
			}
		}

		// for each new vert, search around the poly for its new neighbors
		// and doubly-link everything
		for(vstart = onv; vstart < *nverts; ++vstart) {
			if(vertbuffer[vstart].pnbrs[1] >= 0) continue;
			vcur = vertbuffer[vstart].pnbrs[0];
			do {
				vcur = vertbuffer[vcur].pnbrs[0]; 
			} while(vcur < onv);
			vertbuffer[vstart].pnbrs[1] = vcur;
			vertbuffer[vcur].pnbrs[0] = vstart;
		}

		// go through and compress the vertex list, removing clipped verts
		// and re-indexing accordingly (reusing `clipped` to re-index everything)
		numunclipped = 0;
		for(v = 0; v < *nverts; ++v) {
			if(!clipped[v]) {
				vertbuffer[numunclipped] = vertbuffer[v];
				clipped[v] = numunclipped++;
			}
		}
		*nverts = numunclipped;
		for(v = 0; v < *nverts; ++v) {
			vertbuffer[v].pnbrs[0] = clipped[vertbuffer[v].pnbrs[0]];
			vertbuffer[v].pnbrs[1] = clipped[vertbuffer[v].pnbrs[1]];
		}	
	}
}

static inline void r2d_split(r2d_poly* inpolys, r2d_int npolys, r2d_plane plane, r2d_poly* out_pos, r2d_poly* out_neg) {

	// direct access to vertex buffer
	r2d_int* nverts;
	r2d_int p;
	r2d_vertex* vertbuffer;
	r2d_int v, np, onv, vcur, vnext, vstart, nright, cside;
	r2d_rvec2 newpos;
	r2d_int side[R2D_MAX_VERTS];
	r2d_real sdists[R2D_MAX_VERTS];
	r2d_poly* outpolys[2];

	for(p = 0; p < npolys; ++p) {

		nverts = &inpolys[p].nverts;
		vertbuffer = inpolys[p].verts; 
		outpolys[0] = &out_pos[p];
		outpolys[1] = &out_neg[p];
		if(*nverts <= 0) {
			memset(&out_pos[p], 0, sizeof(r2d_poly));
			memset(&out_neg[p], 0, sizeof(r2d_poly));
			continue;
		} 


		// calculate signed distances to the clip plane
		nright = 0;
		memset(&side, 0, sizeof(side));
		for(v = 0; v < *nverts; ++v) {
			sdists[v] = plane.d + dot(vertbuffer[v].pos, plane.n);
			sdists[v] *= -1;
			if(sdists[v] < 0.0) {
				side[v] = 1;
				nright++;
			}
		}
	
		// return if the poly lies entirely on one side of it 
		if(nright == 0) {
			*(outpolys[0]) = inpolys[p]; 
			outpolys[1]->nverts = 0;
			continue;	
		}
		if(nright == *nverts) {
			*(outpolys[1]) = inpolys[p];
			outpolys[0]->nverts = 0;
			continue;
		}
	
		// check all edges and insert new vertices on the bisected edges 
		onv = *nverts; 
		for(vcur = 0; vcur < onv; ++vcur) {
			if(side[vcur]) continue;
			for(np = 0; np < 2; ++np) {
				vnext = vertbuffer[vcur].pnbrs[np];
				if(!side[vnext]) continue;
				wav(vertbuffer[vcur].pos, -sdists[vnext],
					vertbuffer[vnext].pos, sdists[vcur],
					newpos);
				vertbuffer[*nverts].pos = newpos;
				vertbuffer[vcur].pnbrs[np] = *nverts;
				vertbuffer[*nverts].pnbrs[np] = -1;
				vertbuffer[*nverts].pnbrs[1-np] = vcur;
				(*nverts)++;
				side[*nverts] = 1;
				vertbuffer[*nverts].pos = newpos;
				vertbuffer[*nverts].pnbrs[1-np] = -1;
				vertbuffer[*nverts].pnbrs[np] = vnext;
				vertbuffer[vnext].pnbrs[1-np] = *nverts;
				(*nverts)++;
			}
		}
	
		// for each new vert, search around the poly for its new neighbors
		// and doubly-link everything
		for(vstart = onv; vstart < *nverts; ++vstart) {
			if(vertbuffer[vstart].pnbrs[1] >= 0) continue;
			vcur = vertbuffer[vstart].pnbrs[0];
			do {
				vcur = vertbuffer[vcur].pnbrs[0]; 
			} while(vcur < onv);
			vertbuffer[vstart].pnbrs[1] = vcur;
			vertbuffer[vcur].pnbrs[0] = vstart;
		}
	
		// copy and compress vertices into their new buffers
		// reusing side[] for reindexing
		onv = *nverts;
		outpolys[0]->nverts = 0;
		outpolys[1]->nverts = 0;
		for(v = 0; v < onv; ++v) {
			cside = side[v];
			outpolys[cside]->verts[outpolys[cside]->nverts] = vertbuffer[v];
			side[v] = (outpolys[cside]->nverts)++;
		}
	
		for(v = 0; v < outpolys[0]->nverts; ++v) 
			for(np = 0; np < 2; ++np)
				outpolys[0]->verts[v].pnbrs[np] = side[outpolys[0]->verts[v].pnbrs[np]];
		for(v = 0; v < outpolys[1]->nverts; ++v) 
			for(np = 0; np < 2; ++np)
				outpolys[1]->verts[v].pnbrs[np] = side[outpolys[1]->verts[v].pnbrs[np]];
	}
}

static inline void r2d_reduce(r2d_poly* poly, r2d_real* moments, r2d_int polyorder) {

	// var declarations
	r2d_int vcur, vnext, m, i, j, corder;
	r2d_real twoa;
	r2d_rvec2 v0, v1, vc;

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// zero the moments
	for(m = 0; m < R2D_NUM_MOMENTS(polyorder); ++m) moments[m] = 0.0;

	if(*nverts <= 0) return;

	// flag to translate a polygon to the origin for increased accuracy
	// (this will increase computational cost, in particular for higher moments)
	r2d_int shift_poly = 1;

	if(shift_poly) vc = r2d_poly_center(poly);

	// Storage for coefficients
	// keep two layers of the triangle of coefficients
	r2d_int prevlayer = 0;
	r2d_int curlayer = 1;
    r2d_rvec2 *D = (r2d_rvec2*)malloc(sizeof(r2d_rvec2) * (polyorder + 1));
    r2d_rvec2 *C = (r2d_rvec2*)malloc(sizeof(r2d_rvec2) * (polyorder + 1));

	// r2d_real D[polyorder+1][2];
	// r2d_real C[polyorder+1][2];

	// iterate over edges and compute a sum over simplices 
	for(vcur = 0; vcur < *nverts; ++vcur) {

		vnext = vertbuffer[vcur].pnbrs[0];
		v0 = vertbuffer[vcur].pos;
		v1 = vertbuffer[vnext].pos;

		if(shift_poly) {
			v0.x = v0.x - vc.x;
			v0.y = v0.y - vc.y;
			v1.x = v1.x - vc.x;
			v1.y = v1.y - vc.y;
		}

		twoa = (v0.x*v1.y - v0.y*v1.x);

		// calculate the moments
		// using the fast recursive method of Koehl (2012)
		// essentially building a set of Pascal's triangles, one layer at a time

		// base case
		D[0].xy[prevlayer] = 1.0;
		C[0].xy[prevlayer] = 1.0;
		moments[0] += 0.5*twoa;

		// build up successive polynomial orders
		for(corder = 1, m = 1; corder <= polyorder; ++corder) {
			for(i = corder; i >= 0; --i, ++m) {
				j = corder - i;
				C[i].xy[curlayer] = 0; 
				D[i].xy[curlayer] = 0;  
				if(i > 0) {
					C[i].xy[curlayer] += v1.x*C[i-1].xy[prevlayer];
					D[i].xy[curlayer] += v0.x*D[i-1].xy[prevlayer]; 
				}
				if(j > 0) {
					C[i].xy[curlayer] += v1.y*C[i].xy[prevlayer];
					D[i].xy[curlayer] += v0.y*D[i].xy[prevlayer]; 
				}
				D[i].xy[curlayer] += C[i].xy[curlayer]; 
				moments[m] += twoa*D[i].xy[curlayer];
			}
			curlayer = 1 - curlayer;
			prevlayer = 1 - prevlayer;
		}
	}

	// reuse C to recursively compute the leading multinomial coefficients
	C[0].xy[prevlayer] = 1.0;
	for(corder = 1, m = 1; corder <= polyorder; ++corder) {
		for(i = corder; i >= 0; --i, ++m) {
			j = corder - i;
			C[i].xy[curlayer] = 0.0; 
			if(i > 0) C[i].xy[curlayer] += C[i-1].xy[prevlayer];
			if(j > 0) C[i].xy[curlayer] += C[i].xy[prevlayer];
			moments[m] /= C[i].xy[curlayer]*(corder+1)*(corder+2);
		}
		curlayer = 1 - curlayer;
		prevlayer = 1 - prevlayer;
	}

	if(shift_poly) r2d_shift_moments(moments, polyorder, vc);

    free(D);
    free(C);
}

static inline void r2d_shift_moments(r2d_real* moments, r2d_int polyorder, r2d_rvec2 vc) {

	// var declarations
	r2d_int m, i, j, corder;
	r2d_int mm, mi, mj, mcorder;

	// store moments of a shifted polygon
	r2d_real *moments2 = (r2d_real*)malloc(sizeof(r2d_real) * (R2D_NUM_MOMENTS(polyorder)));
	for(m = 0; m < R2D_NUM_MOMENTS(polyorder); ++m) moments2[m] = 0.0;

	// calculate and save Pascal's triangle
	r2d_real **B = (r2d_real**)malloc(sizeof(r2d_real*) * (polyorder+1));
    for(i = 0; i < polyorder+1; ++i) {
        B[i] = (r2d_real*)malloc(sizeof(r2d_real) * (polyorder+1));
    }

	B[0][0] = 1.0;
	for(corder = 1, m = 1; corder <= polyorder; ++corder) {
		for(i = corder; i >= 0; --i, ++m) {
			j = corder - i;
			B[i][corder] = 1.0;
			if(i > 0 && j > 0) B[i][corder] = B[i][corder-1] + B[i-1][corder-1];
		}
	}

	// shift moments back to the original position using
	// \int_\Omega x^i y^j d\vec r =
	// \int_\omega (x+\xi)^i (y+\eta)^j d\vec r =
	// \sum_{a,b,c=0}^{i,j,k} \binom{i}{a} \binom{j}{b}
	// \xi^{i-a} \eta^{j-b} \int_\omega x^a y^b d\vec r
	for(corder = 1, m = 1; corder <= polyorder; ++corder) {
		for(i = corder; i >= 0; --i, ++m) {
			j = corder - i;
			for(mcorder = 0, mm = 0; mcorder <= corder; ++mcorder) {
				for(mi = mcorder; mi >= 0; --mi, ++mm) {
					mj = mcorder - mi;
					if (mi <= i && mj <= j ) {
						moments2[m] += B[mi][i] * B[mj][j] * R2D_POW(vc.x,(i-mi)) * R2D_POW(vc.y,(j-mj)) * moments[mm];
					}
				}
			}
		}
	}

	// assign shifted moments
	for(m = 1; m < R2D_NUM_MOMENTS(polyorder); ++m)
		moments[m] = moments2[m];

    free(moments2);
    for(i = 0; i < polyorder+1; ++i)
        free(B[i]);
    free(B);
}

static inline r2d_rvec2 r2d_poly_center(r2d_poly* poly) {

	// var declarations
	r2d_int vcur;
	r2d_rvec2 vc;

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts;
	r2d_int* nverts = &poly->nverts;

	vc.x = 0.0;
	vc.y = 0.0;
	for(vcur = 0; vcur < *nverts; ++vcur) {
		vc.x += vertbuffer[vcur].pos.x;
		vc.y += vertbuffer[vcur].pos.y;
	}
	vc.x /= *nverts;
	vc.y /= *nverts;

	return vc;
}

static inline r2d_int r2d_is_good(r2d_poly* poly) {

	r2d_int v;
	r2d_int vct[R2D_MAX_VERTS];

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// consistency check
	memset(&vct, 0, sizeof(vct));
	for(v = 0; v < *nverts; ++v) {

		// return false if vertices share an edge with themselves 
		// or if any edges are obviously invalid
		if(vertbuffer[v].pnbrs[0] == vertbuffer[v].pnbrs[1]) return 0;
		if(vertbuffer[v].pnbrs[0] >= *nverts) return 0;
		if(vertbuffer[v].pnbrs[1] >= *nverts) return 0;

		vct[vertbuffer[v].pnbrs[0]]++;
		vct[vertbuffer[v].pnbrs[1]]++;
	}
	
	// return false if any vertices are pointed to 
	// by more or fewer than two other vertices
	for(v = 0; v < *nverts; ++v) if(vct[v] != 2) return 0;

	return 1;
}

static inline void r2d_rotate(r2d_poly* poly, r2d_real theta) {
	r2d_int v;
	r2d_rvec2 tmp;
	r2d_real sine = sin(theta);
	r2d_real cosine = cos(theta);
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;
		poly->verts[v].pos.x = cosine*tmp.x - sine*tmp.y; 
		poly->verts[v].pos.x = sine*tmp.x + cosine*tmp.y; 
	}
}

static inline void r2d_translate(r2d_poly* poly, r2d_rvec2 shift) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.x += shift.x;
		poly->verts[v].pos.y += shift.y;
	}
}

static inline void r2d_scale(r2d_poly* poly, r2d_real scale) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.x *= scale;
		poly->verts[v].pos.y *= scale;
	}
}

static inline void r2d_shear(r2d_poly* poly, r2d_real shear, r2d_int axb, r2d_int axs) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.xy[axb] += shear*poly->verts[v].pos.xy[axs];
	}
}

static inline void r2d_affine(r2d_poly* poly, r2d_real mat[3][3]) {
	r2d_int v;
	r2d_rvec2 tmp;
	r2d_real w;
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;

		// affine transformation
		poly->verts[v].pos.x = tmp.x*mat[0][0] + tmp.y*mat[0][1] + mat[0][2];
		poly->verts[v].pos.y = tmp.x*mat[1][0] + tmp.y*mat[1][1] + mat[1][2];
		w = tmp.x*mat[2][0] + tmp.y*mat[2][1] + mat[2][2];
	
		// homogeneous divide if w != 1, i.e. in a perspective projection
		poly->verts[v].pos.x /= w;
		poly->verts[v].pos.y /= w;
	}
}


static inline void r2d_init_box(r2d_poly* poly, r2d_rvec2 rbounds[2]) {

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 
	
	*nverts = 4;
	vertbuffer[0].pnbrs[0] = 1;	
	vertbuffer[0].pnbrs[1] = 3;	
	vertbuffer[1].pnbrs[0] = 2;	
	vertbuffer[1].pnbrs[1] = 0;	
	vertbuffer[2].pnbrs[0] = 3;	
	vertbuffer[2].pnbrs[1] = 1;	
	vertbuffer[3].pnbrs[0] = 0;	
	vertbuffer[3].pnbrs[1] = 2;	
	vertbuffer[0].pos.x = rbounds[0].x; 
	vertbuffer[0].pos.y = rbounds[0].y; 
	vertbuffer[1].pos.x = rbounds[1].x; 
	vertbuffer[1].pos.y = rbounds[0].y; 
	vertbuffer[2].pos.x = rbounds[1].x; 
	vertbuffer[2].pos.y = rbounds[1].y; 
	vertbuffer[3].pos.x = rbounds[0].x; 
	vertbuffer[3].pos.y = rbounds[1].y; 

}


static inline void r2d_init_poly(r2d_poly* poly, r2d_rvec2* vertices, r2d_int numverts) {

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// init the poly
	*nverts = numverts;
	r2d_int v;
	for(v = 0; v < *nverts; ++v) {
		vertbuffer[v].pos = vertices[v];
		vertbuffer[v].pnbrs[0] = (v+1)%(*nverts);
		vertbuffer[v].pnbrs[1] = (*nverts+v-1)%(*nverts);
	}
}


static inline void r2d_box_faces_from_verts(r2d_plane* faces, r2d_rvec2* rbounds) {
	faces[0].n.x = 0.0; faces[0].n.y = 1.0; faces[0].d = rbounds[0].y; 
	faces[1].n.x = 1.0; faces[1].n.y = 0.0; faces[1].d = rbounds[0].x; 
	faces[2].n.x = 0.0; faces[2].n.y = -1.0; faces[2].d = rbounds[1].y; 
	faces[3].n.x = -1.0; faces[3].n.y = 0.0; faces[3].d = rbounds[1].x; 
}

static inline void r2d_poly_faces_from_verts(r2d_plane* faces, r2d_rvec2* vertices, r2d_int numverts) {

	// dummy vars
	r2d_int f;
	r2d_rvec2 p0, p1;

	// calculate a centroid and a unit normal for each face 
	for(f = 0; f < numverts; ++f) {

		p0 = vertices[f];
		p1 = vertices[(f+1)%numverts];

		// normal of the edge
		faces[f].n.x = p0.y - p1.y;
		faces[f].n.y = p1.x - p0.x;

		// normalize the normals and set the signed distance to origin
		norm(faces[f].n);
		faces[f].d = -dot(faces[f].n, p0);

	}
}

static inline r2d_real r2d_orient(r2d_rvec2 pa, r2d_rvec2 pb, r2d_rvec2 pc) {
	return 0.5*((pa.x - pc.x)*(pb.y - pc.y) - (pb.x - pc.x)*(pa.y - pc.y)); 
}

static inline void r2d_print(r2d_poly* poly) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		printf("  vertex %d: pos = ( %.10e , %.10e ), nbrs = %d %d\n", 
				v, poly->verts[v].pos.x, poly->verts[v].pos.y, poly->verts[v].pnbrs[0], poly->verts[v].pnbrs[1]);
	}
}


#define wav(va, wa, vb, wb, vr) {			\
	vr.x = (wa*va.x + wb*vb.x)/(wa + wb);	\
	vr.y = (wa*va.y + wb*vb.y)/(wa + wb);	\
}

static inline void r2d_rasterize(r2d_poly* poly, r2d_dvec2 ibox[2], r2d_real* dest_grid, r2d_rvec2 d, r2d_int polyorder) {

	r2d_int i, m, spax, dmax, nstack, siz;
	r2d_int nmom = R2D_NUM_MOMENTS(polyorder);
	r2d_poly* children[2];
	r2d_dvec2 gridsz;

	// return if any parameters are bad 
	for(i = 0; i < 2; ++i) gridsz.ij[i] = ibox[1].ij[i]-ibox[0].ij[i];	
	if(!poly || poly->nverts <= 0 || !dest_grid || 
			gridsz.i <= 0 || gridsz.j <= 0) return;
	
	r2d_real *moments = (r2d_real*)malloc(sizeof(r2d_real) * nmom);

	// explicit stack-based implementation
	// stack size should never overflow in this implementation, 
	// even for large input grids (up to ~512^2) 
	typedef struct {
		r2d_poly poly;
		r2d_dvec2 ibox[2];
	} stack_elem;

    stack_elem *stack = (stack_elem*)malloc(sizeof(stack_elem) * (
        (r2d_int)(ceil(log2(gridsz.i))+ceil(log2(gridsz.j))+1)
    ));

	// push the original polyhedron onto the stack
	// and recurse until child polyhedra occupy single rasters
	nstack = 0;
	stack[nstack].poly = *poly;
	memcpy(stack[nstack].ibox, ibox, 2*sizeof(r2d_dvec2));
	nstack++;
	while(nstack > 0) {

		// pop the stack
		// if the leaf is empty, skip it
		--nstack;
		if(stack[nstack].poly.nverts <= 0) continue;
		
		// find the longest axis along which to split 
		dmax = 0;
		spax = 0;
		for(i = 0; i < 2; ++i) {
			siz = stack[nstack].ibox[1].ij[i]-stack[nstack].ibox[0].ij[i];
			if(siz > dmax) {
				dmax = siz; 
				spax = i;
			}	
		}

		// if all three axes are only one raster long, reduce the single raster to the dest grid
#define gind(ii, jj, mm) (nmom*((ii-ibox[0].i)*gridsz.j+(jj-ibox[0].j))+mm)
		if(dmax == 1) {
			r2d_reduce(&stack[nstack].poly, moments, polyorder);
			// TODO: cell shifting for accuracy
			for(m = 0; m < nmom; ++m)
				dest_grid[gind(stack[nstack].ibox[0].i, stack[nstack].ibox[0].j, m)] += moments[m];
			continue;
		}

		// split the poly and push children to the stack
		children[0] = &stack[nstack].poly;
		children[1] = &stack[nstack+1].poly;
		r2d_split_coord(&stack[nstack].poly, children, d.xy[spax]*(stack[nstack].ibox[0].ij[spax]+dmax/2), spax);
		memcpy(stack[nstack+1].ibox, stack[nstack].ibox, 2*sizeof(r2d_dvec2));
		//stack[nstack].ibox[0].ij[spax] += dmax/2;
		//stack[nstack+1].ibox[1].ij[spax] -= dmax-dmax/2; 

		stack[nstack].ibox[1].ij[spax] -= dmax-dmax/2; 
		stack[nstack+1].ibox[0].ij[spax] += dmax/2;

		nstack += 2;
	}

    free(moments);
    free(stack);
}

static inline void r2d_split_coord(r2d_poly* inpoly, r2d_poly** outpolys, r2d_real coord, r2d_int ax) {

	// direct access to vertex buffer
	if(inpoly->nverts <= 0) return;
	r2d_int* nverts = &inpoly->nverts;
	r2d_vertex* vertbuffer = inpoly->verts; 
	r2d_int v, np, onv, vcur, vnext, vstart, nright, cside;
	r2d_rvec2 newpos;
	r2d_int side[R2D_MAX_VERTS];
	r2d_real sdists[R2D_MAX_VERTS];

	// calculate signed distances to the clip plane
	nright = 0;
	memset(&side, 0, sizeof(side));
	for(v = 0; v < *nverts; ++v) {
		sdists[v] = vertbuffer[v].pos.xy[ax] - coord;
		if(sdists[v] > 0.0) {
			side[v] = 1;
			nright++;
		}
	}

	// return if the poly lies entirely on one side of it 
	if(nright == 0) {
		*(outpolys[0]) = *inpoly;
		outpolys[1]->nverts = 0;
		return;
	}
	if(nright == *nverts) {
		*(outpolys[1]) = *inpoly;
		outpolys[0]->nverts = 0;
		return;
	}

	// check all edges and insert new vertices on the bisected edges 
	onv = inpoly->nverts;
	for(vcur = 0; vcur < onv; ++vcur) {
		if(side[vcur]) continue;
		for(np = 0; np < 2; ++np) {
			vnext = vertbuffer[vcur].pnbrs[np];
			if(!side[vnext]) continue;
			wav(vertbuffer[vcur].pos, -sdists[vnext],
				vertbuffer[vnext].pos, sdists[vcur],
				newpos);
			vertbuffer[*nverts].pos = newpos;
			vertbuffer[vcur].pnbrs[np] = *nverts;
			vertbuffer[*nverts].pnbrs[np] = -1;
			vertbuffer[*nverts].pnbrs[1-np] = vcur;
			(*nverts)++;
			side[*nverts] = 1;
			vertbuffer[*nverts].pos = newpos;
			vertbuffer[*nverts].pnbrs[1-np] = -1;
			vertbuffer[*nverts].pnbrs[np] = vnext;
			vertbuffer[vnext].pnbrs[1-np] = *nverts;
			(*nverts)++;
		}
	}

	// for each new vert, search around the poly for its new neighbors
	// and doubly-link everything
	for(vstart = onv; vstart < *nverts; ++vstart) {
		if(vertbuffer[vstart].pnbrs[1] >= 0) continue;
		vcur = vertbuffer[vstart].pnbrs[0];
		do {
			vcur = vertbuffer[vcur].pnbrs[0]; 
		} while(vcur < onv);
		vertbuffer[vstart].pnbrs[1] = vcur;
		vertbuffer[vcur].pnbrs[0] = vstart;
	}

	// copy and compress vertices into their new buffers
	// reusing side[] for reindexing
	onv = *nverts;
	outpolys[0]->nverts = 0;
	outpolys[1]->nverts = 0;
	for(v = 0; v < onv; ++v) {
		cside = side[v];
		outpolys[cside]->verts[outpolys[cside]->nverts] = vertbuffer[v];
		side[v] = (outpolys[cside]->nverts)++;
	}

	for(v = 0; v < outpolys[0]->nverts; ++v) 
		for(np = 0; np < 2; ++np)
			outpolys[0]->verts[v].pnbrs[np] = side[outpolys[0]->verts[v].pnbrs[np]];
	for(v = 0; v < outpolys[1]->nverts; ++v) 
		for(np = 0; np < 2; ++np)
			outpolys[1]->verts[v].pnbrs[np] = side[outpolys[1]->verts[v].pnbrs[np]];
}

static inline void r2d_get_ibox(r2d_poly* poly, r2d_dvec2 ibox[2], r2d_rvec2 d) {
	r2d_int i, v;
	r2d_rvec2 rbox[2];
	for(i = 0; i < 2; ++i) {
		rbox[0].xy[i] = 1.0e30;
		rbox[1].xy[i] = -1.0e30;
	}
	for(v = 0; v < poly->nverts; ++v) {
		for(i = 0; i < 2; ++i) {
			if(poly->verts[v].pos.xy[i] < rbox[0].xy[i]) rbox[0].xy[i] = poly->verts[v].pos.xy[i];
			if(poly->verts[v].pos.xy[i] > rbox[1].xy[i]) rbox[1].xy[i] = poly->verts[v].pos.xy[i];
		}
	}
	for(i = 0; i < 2; ++i) {
		ibox[0].ij[i] = floor(rbox[0].xy[i]/d.xy[i]);
		ibox[1].ij[i] = ceil(rbox[1].xy[i]/d.xy[i]);
	}
}

static inline void r2d_clamp_ibox(r2d_poly* poly, r2d_dvec2 ibox[2], r2d_dvec2 clampbox[2], r2d_rvec2 d) {
	r2d_int i, nboxclip;
	r2d_plane boxfaces[4];
	nboxclip = 0;
	memset(boxfaces, 0, sizeof(boxfaces));
	for(i = 0; i < 2; ++i) {
		if(ibox[1].ij[i] <= clampbox[0].ij[i] || ibox[0].ij[i] >= clampbox[1].ij[i]) {
			memset(ibox, 0, 2*sizeof(r2d_dvec2));
			poly->nverts = 0;
			return;
		}
		if(ibox[0].ij[i] < clampbox[0].ij[i]) {
			ibox[0].ij[i] = clampbox[0].ij[i];
			boxfaces[nboxclip].d = -clampbox[0].ij[i]*d.xy[i];
			boxfaces[nboxclip].n.xy[i] = 1.0;
			nboxclip++;
		}
		if(ibox[1].ij[i] > clampbox[1].ij[i]) {
			ibox[1].ij[i] = clampbox[1].ij[i];
			boxfaces[nboxclip].d = clampbox[1].ij[i]*d.xy[i];
			boxfaces[nboxclip].n.xy[i] = -1.0;
			nboxclip++;
		}	
	}
	if(nboxclip) r2d_clip(poly, boxfaces, nboxclip);
}
#ifdef __cplusplus
}
#endif

#endif // _R2D_H_
