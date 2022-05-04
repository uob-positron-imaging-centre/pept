#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.09.2021


import  textwrap

import  numpy           as      np
from    scipy.optimize  import  minimize

from    pept            import  LineData, PointData
from    pept.base       import  LineDataFilter, Filter

from    ..peptml        import  get_cutoffs
from    .cutpoints_tof  import  find_cutpoints_tof




class TimeOfFlight(LineDataFilter):
    '''Compute the positron annihilation locations of each LoR as given by the
    Time Of Flight (ToF) data of the two LoR timestamps.

    Filter signature:
    ::

        LineData -> TimeOfFlight.fit_sample -> LineData  (points = False)
        LineData -> TimeOfFlight.fit_sample -> PointData (points = True)

    Importantly, the input LineData must have at least 8 columns, formatted as
    [t1, x1, y1, z1, x2, y2, z2, t2] - notice the different timestamps of the
    two LoR ends.

    If `points = False` (default), the computed ToF points are saved as an
    extra attribute "tof" in the input LineData; otherwise they are returned
    directly.

    The `temporal_resolution` should be set to the FWHM of the temporal
    resolution in the LoR timestamps, in self-consistent units of measure (e.g.
    m / s or mm / ms, but not mm / s). If it is set, the "temporal_resolution"
    and "spatial_resolution" extra attributes are set on the ToF points.

    *New in pept-0.4.2*

    Examples
    --------
    Generate 10 random LoRs between (-100, 100) mm and ms with 8 columns for
    the ToF format.

    >>> import numpy as np
    >>> import pept

    >>> rng = np.random.default_rng(0)
    >>> lors = pept.LineData(
    >>>     rng.uniform(-100, 100, (10, 8)),
    >>>     columns = ["t1", "x1", "y1", "z1", "x2", "y2", "z2", "t2"],
    >>> )
    >>> lors
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 10
    overlap = 0
    lines =
      (rows: 10, columns: 8)
      [[ 57.4196615  -52.1261114  ...  -9.93212667  59.26485406]
       [-53.8715582  -89.59573979 ... -40.26077344  34.39897559]
       ...
       [ 51.59020047   2.55174465 ... -31.13800424 -13.94025361]
       [ 93.21241616  12.44636845 ... -75.08905883 -42.3338486 ]]
    columns = ['t1', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 't2']
    attrs = {}

    Compute Time of Flight annihilation locations from the two timestamps
    in the data above. Assume a temporal resolution of 100 ps - be careful to
    use self-consistent units; in this case we are using mm and ms:

    >>> from pept.tracking import *

    >>> temporal_resolution = 100e-12 * 1000    # ms
    >>> lors_tof = TimeOfFlight(temporal_resolution).fit_sample(lors)
    >>> lors_tof
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 10
    overlap = 0
    lines =
      (rows: 10, columns: 8)
      [[ 57.4196615  -52.1261114  ...  -9.93212667  59.26485406]
       [-53.8715582  -89.59573979 ... -40.26077344  34.39897559]
       ...
       [ 51.59020047   2.55174465 ... -31.13800424 -13.94025361]
       [ 93.21241616  12.44636845 ... -75.08905883 -42.3338486 ]]
    columns = ['t1', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 't2']
    attrs = {
      'tof': pept.PointData (samples: 1)
    ---------------------------
    sample_...
    }

    >>> lors_tof.attrs["tof"]
    pept.PointData (samples: 1)
    ---------------------------
    sample_size = 10
    overlap = 0
    points =
      (rows: 10, columns: 4)
      [[ 5.64970655e+01 -3.22092074e+07  2.41767704e+08 -1.30428351e+08]
       [-9.80068250e+01 -2.48775932e+09 -1.12904720e+10 -6.43480969e+09]
       ...
       [ 1.88249731e+01  3.34819602e+09 -8.78848458e+09  2.83529405e+09]
       [ 2.54392837e+01  1.90343279e+10 -1.92717662e+09 -6.84078611e+09]]
    columns = ['t', 'x', 'y', 'z']
    attrs = {
      'temporal_resolution': 1.0000000000000001e-07
      'spatial_resolution': 29.9792458
    }

    Alternatively, you can extract only the ToF points directly:

    >>> tof = TimeOfFlight(temporal_resolution, points = True).fit_sample(lors)
    >>> tof
    pept.PointData (samples: 1)
    ---------------------------
    sample_size = 10
    overlap = 0
    points =
      (rows: 10, columns: 4)
      [[ 5.64970655e+01 -3.22092074e+07  2.41767704e+08 -1.30428351e+08]
       [-9.80068250e+01 -2.48775932e+09 -1.12904720e+10 -6.43480969e+09]
       ...
       [ 1.88249731e+01  3.34819602e+09 -8.78848458e+09  2.83529405e+09]
       [ 2.54392837e+01  1.90343279e+10 -1.92717662e+09 -6.84078611e+09]]
    columns = ['t', 'x', 'y', 'z']
    attrs = {
      'temporal_resolution': 1.0000000000000001e-07
      'spatial_resolution': 29.9792458
    }

    '''

    def __init__(self, temporal_resolution = None, points = False):
        if temporal_resolution is None:
            self.temporal_resolution = None
        else:
            self.temporal_resolution = float(temporal_resolution)

        self.points = bool(points)


    def fit_sample(self, sample: LineData):
        if not isinstance(sample, LineData):
            sample = LineData(sample)

        if sample.lines.shape[1] < 8:
            raise ValueError(textwrap.fill((
                "The input `sample` of LineData must have at least 8 columns, "
                "formatted as [t1, x1, y1, z1, x2, y2, z2, t2]. Received "
                f"sample with `sample.lines.shape = {sample.lines.shape}`."
            )))

        # The two points defining the LoR
        t1 = sample.lines[:, 0]
        p1 = sample.lines[:, 1:4]

        p2 = sample.lines[:, 4:7]
        t2 = sample.lines[:, 7]

        # Speed of light (mm / ms)
        c = 299792458.

        # The ratio (P1 - tofpoint) / (P1 - P2) for all rows
        distance_ratio = (
            0.5 - 0.5 / np.linalg.norm(p2 - p1, axis = 1) * c * (t2 - t1)
        )

        # [:, np.newaxis] = transform row vector to column vector (i.e. 2D
        # array with one column)
        tof_locations = p1 + (p2 - p1) * distance_ratio[:, np.newaxis]
        tof_time = t1 - np.linalg.norm(tof_locations - p1, axis = 1) / c

        # Encapsulate ToF points in a PointData. Save resolution as attrs
        tof_pts = PointData(np.c_[tof_time, tof_locations])

        if self.temporal_resolution is not None:
            tof_pts.attrs["temporal_resolution"] = self.temporal_resolution
            tof_pts.attrs["spatial_resolution"] = self.temporal_resolution * c

        if self.points:
            return tof_pts

        sample.attrs["tof"] = tof_pts
        return sample




class CutpointsToF(LineDataFilter):
    '''Compute cutpoints from all pairs of lines whose Time Of Flight-predicted
    locations are closer than `max_distance`.

    Filter signature:
    ::

        LineData -> CutpointsToF.fit_sample -> PointData

    If the ``TimeOfFlight`` filter was used and a temporal resolution was
    specified (as a FWHM), then `max_distance` is automatically inferred as
    the minimum between 2 * "spatial_resolution" and the dimension-wise
    standard deviation of the input points.

    The `cutoffs` parameter may be set as [xmin, xmax, ymin, ymax, zmin, zmax]
    for a minimum bounding box outside of which cutpoints are discarded.
    Otherwise it is automatically set to the minimum bounding box containing
    all input LoRs.

    If `append_indices = True`, two extra columns are appended to the result as
    "line_index1" and "line_index2" containing the indices of the LoRs that
    produced each cutpoint; an extra attribute "_lines" is also set to the
    input `LineData`.

    If `cutpoints_only = False` (default), the Time Of Flight-predicted
    positron annihilation locations are also appended to the returned points.

    *New in pept-0.4.2*

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.tracking.HDBSCAN : Efficient, HDBSCAN-based clustering of (cut)points.
    pept.read_csv : Fast CSV file reading into numpy arrays.

    Examples
    --------
    Make sure to use the ``TimeOfFlight`` filter to compute to ToF annihilation
    locations; if you specify a temporal resolution, the `max_distance`
    parameter is automatically computed:

    >>> import pept
    >>> from pept.tracking import *

    >>> line_data = pept.LineData(example_tof_data)
    >>> line_data_tof = TimeOfFlight(100e-9).fit_sample(line_data)
    >>> cutpoints_tof = CutpointsToF().fit_sample(line_data_tof)

    Alternatively, set `max_distance` yourself:

    >>> line_data = pept.LineData(example_tof_data)
    >>> line_data_tof = TimeOfFlight().fit_sample(line_data)
    >>> cutpoints_tof = CutpointsToF(5.0).fit_sample(line_data_tof)
    '''

    def __init__(
        self,
        max_distance = None,
        cutoffs = None,
        append_indices = False,
        cutpoints_only = False,
    ):
        # Setting class attributes. The ones below call setters which do type
        # checking
        self.cutoffs = cutoffs
        self.append_indices = append_indices
        self.max_distance = max_distance

        self.cutpoints_only = bool(cutpoints_only)


    @property
    def max_distance(self):
        return self._max_distance


    @max_distance.setter
    def max_distance(self, max_distance):
        if max_distance is None:
            self._max_distance = None
        else:
            self._max_distance = float(max_distance)


    @property
    def cutoffs(self):
        return self._cutoffs


    @cutoffs.setter
    def cutoffs(self, cutoffs):
        if cutoffs is not None:
            cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
            if cutoffs.ndim != 1 or len(cutoffs) != 6:
                raise ValueError((
                    "\n[ERROR]: cutoffs should be a one-dimensional array "
                    "with values [min_x, max_x, min_y, max_y, min_z, max_z]. "
                    f"Received {cutoffs}.\n"
                ))

            self._cutoffs = cutoffs
        else:
            self._cutoffs = None


    @property
    def append_indices(self):
        return self._append_indices


    @append_indices.setter
    def append_indices(self, append_indices):
        self._append_indices = bool(append_indices)


    def fit_sample(self, sample_lines: LineData):
        if not isinstance(sample_lines, LineData):
            sample_lines = LineData(sample_lines)

        # Ensure ToF data is present and extract it
        if "tof" not in sample_lines.attrs:
            raise AttributeError(textwrap.fill((
                "The input `sample_lines` must have an attribute 'tof' "
                "containing the Time of Flight predicted annihilation point "
                "for each LoR. This is normally set by "
                "`pept.tracking.TimeOfFlight`."
            )))

        tofpoints = sample_lines.attrs["tof"]
        tofattrs = tofpoints.attrs

        # If max_distance was not defined, use FWHM spatial_resolution
        if self.max_distance is not None:
            max_distance = self.max_distance
        elif "spatial_resolution" in tofpoints.attrs:
            max_distance = min(
                tofpoints.attrs["spatial_resolution"] * 2,
                tofpoints.points[:, 1:4].std(axis = 0).max(),
            )
        else:
            max_distance = tofpoints.points[:, 1:4].std(axis = 0).max()

        # If cutoffs were not defined, automatically compute them
        if self.cutoffs is not None:
            cutoffs = self.cutoffs
        else:
            cutoffs = get_cutoffs(sample_lines.lines)

        # Only compute cutpoints if there are at least 2 LoRs
        if len(sample_lines.lines) >= 2:
            sample_cutpoints = find_cutpoints_tof(
                sample_lines.lines,
                tofpoints.points,
                max_distance,
                cutoffs,
                append_indices = self.append_indices,
            )
        else:
            sample_cutpoints = np.empty((0, 6 if self.append_indices else 4))

        # Column names
        columns = ["t", "x", "y", "z"]
        if self.append_indices:
            columns += ["line_index1", "line_index2"]

        # Also include tofpoints if not self.cutpoints_only
        if self.cutpoints_only:
            points = PointData(sample_cutpoints, columns = columns, **tofattrs)
        else:
            tofs = tofpoints.points
            if self.append_indices:
                tofs = np.c_[tofs, np.arange(len(tofs)), np.arange(len(tofs))]

            points = PointData(
                np.vstack((sample_cutpoints, tofs)),
                columns = columns,
                **tofattrs,
            )

        # Add optional metadata to the points; because they have an underscore,
        # they won't be propagated when new objects are constructed
        points.attrs["_max_distance"] = max_distance
        points.attrs["_cutoffs"] = cutoffs

        if self.append_indices:
            points.attrs["_lines"] = sample_lines

        return points




def _gaussian_probs(x, points, sigma):
    '''Probabilities from x to Gaussian-distributed points.'''
    dists = np.sum((x[:3] - points)**2, axis = 1)
    return np.exp(-0.5 * dists / sigma**2)




class GaussianDensity(Filter):
    '''Append weights according to the Gaussian distribution that best fits
    the samples of points.

    Filter signature:
    ::

              PointData -> GaussianDensity.fit_sample -> PointData
          numpy.ndarray -> GaussianDensity.fit_sample -> PointData
        list[PointData] -> GaussianDensity.fit_sample -> list[PointData]

    This is treated as an optimisation problem: find the 3D location that
    maximises the sum of Probability Distributions (PDF) centered at each
    point.

    ::

        Given N points p_1, p_2, ..., p_N:

                  N
        maximise sum( exp( -0.5 * |x - p_i|^2 / sigma^2 ) )
           x      i

    Each point is then assigned a weight corresponding to its PDF - i.e. the
    exponential part - saved in the `weight` column.

    Sigma controls the standard deviation of the Gaussian distribution centred
    at each point; this corresponds to the relative uncertainty in each point's
    location. For ``TimeOfFlight`` data, leave `sigma = None` and it will be
    computed from the "spatial_resolution" attribute.

    You can use ``Centroids`` afterwards to compute the weighted centroid, i.e.
    where the tracer is. For multiple particle tracking (or just more
    robustness to noise) you can use `HDBSCAN + SplitLabels` beforehand.

    *New in pept-0.4.2*
    '''

    def __init__(self, sigma = None):
        self.sigma = None if sigma is None else float(sigma)


    def _add_weights(self, point_data, sigma):
        # Extract the inner NumPy array of points
        points = point_data.points

        # If no points were given, still return an empty array with the correct
        # number of columns
        if len(points) == 0:
            return np.empty((0, points.shape[1] + 1))

        # Find the 3D point that maximises the sum of PDFs
        xyz = np.array(points[:, 1:4])

        res = minimize(
            lambda x: -np.sum(_gaussian_probs(x, xyz, sigma)),
            np.median(xyz, axis = 0),
            method = "SLSQP",
        )

        # The PDF of each point acts as a weight
        weights = _gaussian_probs(res.x, xyz, sigma)
        return np.c_[points, weights]


    def fit_sample(self, points):
        # Type-checking inputs
        return_list = False
        if isinstance(points, PointData):
            list_points = [points]
        elif isinstance(points, np.ndarray):
            list_points = [PointData(points)]
        else:
            return_list = True
            list_points = [
                p if isinstance(p, PointData) else PointData(p)
                for p in list(points)
            ]

        if not len(list_points):
            raise ValueError("Must receive at least one PointData.")

        # Ensure either self.sigma is set or points.attrs["spatial_resolution"]
        # exists
        if self.sigma is None and any(
            ("spatial_resolution" not in lp.attrs for lp in list_points)
        ):
            raise AttributeError(textwrap.fill((
                "If `sigma` is not set, `points.attrs['spatial_resolution']` "
                "is used. However, it was not found (it is normally set by "
                "`pept.tracking.TimeOfFlight` if `temporal_resolution` is "
                "given)."
            )))
        else:
            fwhms = 2 * np.sqrt(2 * np.log(2))
            self.sigma = list_points[0].attrs["spatial_resolution"] / fwhms

        # Add "weight" column to each PointData
        list_weighted = [self._add_weights(p, self.sigma) for p in list_points]
        columns = list_points[0].columns + ["weight"]

        if return_list:
            return [
                points.copy(data = weighted, columns = columns)
                for points, weighted in zip(list_points, list_weighted)
            ]

        return PointData(
            np.vstack(list_weighted),
            columns = columns,
            **list_points[0].attrs,
        )
