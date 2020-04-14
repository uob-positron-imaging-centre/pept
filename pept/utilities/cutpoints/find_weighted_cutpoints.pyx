# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools.
#
#    If you used this codebase or any software making use of it in a scientific
#    publication, you must cite the following paper:
#        Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
#        using machine learning. Review of Scientific Instruments.
#        2020 Jan 1;91(1):013329.
#        https://doi.org/10.1063/1.5129251
#
#    Copyright (C) 2020 Andrei Leonard Nicusan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools


# File              : find_weighted_cutpoints.pyx
# License           : License: GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 07.04.2020


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


import numpy as np
cimport numpy as np

from libc.math cimport pi, e, exp, log, sqrt


cdef inline double norm2(double[3] arr) nogil:
    return arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]


cdef inline double norm(double[3] arr) nogil:
    return sqrt(norm2(arr))


cpdef find_weighted_cutpoints(
    double[:, :] sample_lines,  # LoRs in sample
    double d_tracer,            # Tracer diameter
    double sigma_tof,           # Std deviation of ToF *spatial* distribution
    double d_annihilation,      # (Mean) positron annihilation range
    bint append_indices = 0,    # Append LoR indices used for each cutpoint
    double sigma_tof_factor = 2 * sqrt(2 * log(2)),
    double wccw = 1,            # control weight for cutlines
    double wtcw = 1             # control weight for tofpoints
):
    '''Compute the weighted cutpoints from a given sample of LoRs with Time of
    Flight (ToF) data.

    Function signature:
        find_weighted_cutpoints(
            double[:, :] sample_lines,      # LoRs in sample
            double d_tracer,                # Tracer diameter
            double sigma_tof,               # Std deviation of ToF *spatial* distribution
            double d_annihilation,          # (Mean) positron annihilation range
            bint append_indices = False,    # Append LoR indices used for each cutpoint
            double sigma_tof_factor = 2 * sqrt(2 * log(2)),
            double wccw = 1,                # control weight for cutlines
            double wtcw = 1                 # control weight for tofpoints
        )

    This function uses both cutpoints and tofpoints to compute a weighted
    average that is more accurate and precise than the individual datapoints.

    A cutpoint is the point in 3D space that minimises the distance between any
    two lines. For any two non-parallel 3D lines, this point corresponds to
    half the distance between the two lines and is unique.

    A tofpoint is the positron annihilation point as calculated from the
    individual timestamps of the two points defining the LoR. It can be
    calculated using the `pept.LineDataToF` base class.

    Note that the ToF data (i.e. tofpoints) must be calculated before calling
    this function and appended to the LoRs. Read the parameter descriptions
    below to use this function safely.

    Parameters
    ----------
    sample_lines : (N, M >= 12) numpy.ndarray
        The sample of lines (LoRs), where each row is [time1, x1, y1, z1,
        time2,  x2, y2, z2, timeToF, xToF, yToF, zToF, ...], containing two
        points [time1, x1, y1, z1] and [time2, x2, y2, z2] with individual
        timestamps defining an LoR, along with the Time of Flight (ToF) data
        [timeToF, xToF, yToF, zToF] as calculated from the two points..
    d_tracer : float
        The tracer diameter.
    sigma_tof : float
        The standard deviation of the *spatial* distribution of the tofpoint.
        Note that the tofpoint distribution is Gaussian.
    d_annihilation : float
        The mean positron range - the distance that the positron travels
        before hitting an electron and annihilating. For example, the mean
        positron range of F-18 in water is 0.6 mm. This parameter can also be
        used as a fitting parameter for increasing the error sphere diameter.
    append_indices : bool, optional
        If set to `True`, the indices of the individual LoRs that were used
        to compute each cutpoint is also appended to the returned array.
        Default is `False`.
    sigma_tof_factor : float, optional
        The accepted spatial error on the ToF data. Only pairs of LoRs whose
        tofpoints' error spheres touch are considered. The error sphere
        diameter is defined as (d_tracer + d_annihilation + sigma_tof *
        sigma_tof_factor). By default, sigma_tof * sigma_tof_factor is defined
        as the Full Width at Half Maximum (FWHM) of the spatial distribution
        of the tofpoints, corresponding to 2 * sqrt(2 * log(2)) * sigma_tof.
        The default is 2 * sqrt(2 * log(2)).
    wccw : float, optional
        The cutpoints control weight (ccw) - a parameter between 0 and 1 for
        setting the influence of the cutpoints on the final average. This
        exists mainly for testing purposes. The default is 1.
    wtcw : float, optional
        The tofpoint control weight (tcw) - a parameter between 0 and 1 for
        setting the influence of the tofpoints on the final average. This
        exists mainly for testing purposes. The default is 1.

    Returns
    -------
    cutpoints : (M, 4) or (M, 6) numpy.ndarray
        A numpy array of the calculated weighted cutpoints. If `append_indices`
        is `False`, then the columns are [time, x, y, z]. If `append_indices`
        is `True`, then the columns are [time, x, y, z, i, j], where `i` and
        `j` are the LoR indices from `sample_lines` that were used to compute
        the weighted cutpoints. The time is the average between the two LoRs
        that were used to compute the cutpoint. The first column (for time) is
        sorted.

    '''
    # Lines for a single sample => n x 12 array
    # sample_lines cols: [time1, X1, Y1, Z1, time2, X2, Y2, Z2, timeToF, XToF, YToF, ZToF]
    cdef Py_ssize_t n = sample_lines.shape[0]

    # Pre-allocate enough memory
    # weighted_cutpoints cols: [time, x, y, z, [LoR1_index, LoR2_index]]
    if append_indices:
        weighted_cutpoints = np.zeros((n * (n - 1) // 2, 6), order = 'C')
    else:
        weighted_cutpoints = np.zeros((n * (n - 1) // 2, 4), order = 'C')

    cdef double[:, :] wpts = weighted_cutpoints # memoryview of weighted_points

    cdef Py_ssize_t i, j, k                     # iterators
    cdef Py_ssize_t iw                          # wpts index
    cdef double[3] diff1, diff2, diff3          # difference between two 3D vectors
    cdef double dist1, dist2                    # distance between two 3D vectors
    cdef double[3] P, U, Q, R, QP               # position, direction vectors
    cdef double a, b, c, d, e                   # collected terms
    cdef double denom, s0, t0                   # parameters for lines

    cdef double[3] PC1, PC2                     # points of closest approach - PoCAs
    cdef double[3] PT1, PT2                     # tofpoints
    cdef double wc1, wc2, wt1, wt2              # weights for the four points
    cdef double d_pocas                         # distance between PoCAs

    # The accepted spatial error on the ToF data. The ToF point distribution is
    # Gaussian; therefore define an error based on a multiple of the standard
    # deviation. By default we define it to be the Full Width at Half-Maximum
    # (FWHM) which is (2 * sqrt(2 * log(2))) * sigma_tof.
    cdef double tof_err = sigma_tof_factor * sigma_tof
    cdef double sigma_tof2 = sigma_tof * sigma_tof

    # The diameter of the error sphere.
    cdef double err = d_tracer + tof_err + d_annihilation
    cdef double err2 = err * err


    with nogil:
        iw = 0
        for i in range(n - 1):
            for j in range(i + 1, n):

                # Only consider lines i and j if their error spheres touch
                for k in range(3):
                    PT1[k] = sample_lines[i, 9 + k]
                    PT2[k] = sample_lines[j, 9 + k]
                    diff1[k] = PT1[k] - PT2[k]

                # Equivalent to the distance between their tofpoints being smaller
                # than the error sphere diameter
                if not norm2(diff1) < err2:
                    continue

                # Now find the cutpoint from the two lines. If the cutpoint is not
                # defined (e.g. lines are parallel), then the weighted cutpoint will
                # only be calculated in terms of the tofpoints PT1 and PT2.

                # Write each pair of lines in terms of a position vector and a
                # direction vector:
                # L1 : A(s) = P + s U
                # L2 : C(t) = Q + t R
                for k in range(3):
                    P[k] = sample_lines[i, 1 + k]
                    U[k] = sample_lines[i, 5 + k] - P[k]
                    Q[k] = sample_lines[j, 1 + k]
                    R[k] = sample_lines[j, 5 + k] - Q[k]
                    QP[k] = Q[k] - P[k]

                a = U[0] * U[0] + U[1] * U[1] + U[2] * U[2]
                b = U[0] * R[0] + U[1] * R[1] + U[2] * R[2]
                c = R[0] * R[0] + R[1] * R[1] + R[2] * R[2]
                d = U[0] * QP[0] + U[1] * QP[1] + U[2] * QP[2]
                e = QP[0] * R[0] + QP[1] * R[1] + QP[2] * R[2]

                denom = b * b - a * c
                # If denom is 0, the two lines are parallel, so no cutpoint exists.
                # Shortcut the loop and only calculate the weighted cutpoint based on
                # the tofpoints.
                if not denom:
                    # The time of the weighted point is simply the average of
                    # the LoR times
                    wpts[iw, 0] = (sample_lines[i, 8] + sample_lines[j, 8]) / 2
                    wpts[iw, 1] = (PT1[0] + PT2[0]) / 2
                    wpts[iw, 2] = (PT1[1] + PT2[1]) / 2
                    wpts[iw, 3] = (PT1[2] + PT2[2]) / 2

                    if append_indices:
                        wpts[iw, 4] = <double>i
                        wpts[iw, 5] = <double>j

                    iw = iw + 1
                    continue
                else:
                    s0 = (b * e - c * d) / denom
                    t0 = (a * e - b * d) / denom

                    for k in range(3):
                        PC1[k] = P[k] + s0 * U[k]   # These are the PoCAs on LoR 1
                        PC2[k] = Q[k] + t0 * R[k]   # and LoR 2

                # Check that PC1 and PC2 are within the error spheres of PT1 and PT2,
                # respectively. If they're not, shortcut the loop and only calculate
                # the weighted point based on the ToFpoints.
                for k in range(3):
                    diff1[k] = PC1[k] - PT1[k]

                dist1 = norm2(diff1)

                if not dist1 < err2:
                    # The time of the weighted point is simply the average of
                    # the LoR times
                    wpts[iw, 0] = (sample_lines[i, 8] + sample_lines[j, 8]) / 2
                    wpts[iw, 1] = (PT1[0] + PT2[0]) / 2
                    wpts[iw, 2] = (PT1[1] + PT2[1]) / 2
                    wpts[iw, 3] = (PT1[2] + PT2[2]) / 2

                    if append_indices:
                        wpts[iw, 4] = <double>i
                        wpts[iw, 5] = <double>j

                    iw = iw + 1
                    continue

                # Same for PC2
                for k in range(3):
                    diff2[k] = PC2[k] - PT1[k]

                dist2 = norm2(diff2)

                if not dist2 < err2:
                    # The time of the weighted point is simply the average of
                    # the LoR times
                    wpts[iw, 0] = (sample_lines[i, 8] + sample_lines[j, 8]) / 2
                    wpts[iw, 1] = (PT1[0] + PT2[0]) / 2
                    wpts[iw, 2] = (PT1[1] + PT2[1]) / 2
                    wpts[iw, 3] = (PT1[2] + PT2[2]) / 2

                    if append_indices:
                        wpts[iw, 4] = <double>i
                        wpts[iw, 5] = <double>j

                    iw = iw + 1
                    continue

                # Distance between points of closest approach (PoCAs) PC1 and PC2:
                for k in range(3):
                    diff3[k] = PC1[k] - PC2[k]

                d_pocas = norm(diff3)

                # So PC1 and PC2 are within the error spheres. Now calculate the
                # weighted point based on all four points: PT1, PT2, PC1, PC2
                wc1 = 1 / (sigma_tof * sqrt(2 * pi)) * exp(-0.5 * dist1 / sigma_tof2)
                wt1 = 1 / (sigma_tof * sqrt(2 * pi)) - wc1

                # Multiply by control weights (between 0 and 1) for setting the
                # level of influence of the tofpoints and PoCAs on the final
                # weighted cutpoint.
                wc1 *= wccw
                wt1 *= wtcw

                wc2 = 1 / (sigma_tof * sqrt(2 * pi)) * exp(-0.5 * dist2 / sigma_tof2)
                wt2 = 1 / (sigma_tof * sqrt(2 * pi)) - wc2

                wc2 *= wccw
                wt2 *= wtcw


                # Print statements for debugging. They are deliberately left in
                # this source file so that people can experiment with / test / see
                # the weighting technique.
                #
                # Before uncommenting, remove the "with nogil:" before the for loops.
                #
                #print("\n----PoCAs and tofpoints:")
                #print(sample_lines[i])
                #print(sample_lines[j])
                #print("PoCA 1:     ", PC1)
                #print("tofpoint 1: ", PT1)
                #print("Distance^2 between PoCA1 and tofpoint 1: ", dist1)
                #print()
                #print("PoCA 1:     ", PC2)
                #print("tofpoint 2: ", PT2)
                #print("Distance^2 between PoCA2 and tofpoint 2: ", dist2)
                #print()
                #print("Error sphere diameter ^2: ", err2)
                #print("Error sphere diameter   : ", err)
                #print("Distance between PoCAs  : ", d_pocas)
                #print("----\n")
                #print("----Weights:")
                #print("PoCA 1 weight:     ", wc1)
                #print("tofpoint 1 weight: ", wt1)
                #print("PoCA 2 weight:     ", wc2)
                #print("tofpoint 2 weight: ", wt2)
                #print()
                #print("Spatial error of ToF (also weight):   ", tof_err)
                #print("Distance between PoCAs (also weight): ", d_pocas)
                #print("----\n\n")


                # Finally, here's the calculated weighted point:
                wpts[iw, 0] = (sample_lines[i, 8] + sample_lines[j, 8]) / 2
                for k in range(3):
                    wpts[iw, 1 + k] = (tof_err * (wc1 * PC1[k] + wc2 * PC2[k]) + \
                                       d_pocas * (wt1 * PT1[k] + wt2 * PT2[k])) / \
                                      (tof_err * (wc1 + wc2) + d_pocas * (wt1 + wt2))

                if append_indices:
                    wpts[iw, 4] = <double>i
                    wpts[iw, 5] = <double>j

                iw = iw + 1


    # Truncate the cutpoints which were not written
    wpts_truncated = np.delete(weighted_cutpoints, slice(iw, None, None), 0)
    del(weighted_cutpoints)

    # Sort rows based on time (column 0)
    wpts_truncated = wpts_truncated[wpts_truncated[:, 0].argsort()]

    return wpts_truncated


