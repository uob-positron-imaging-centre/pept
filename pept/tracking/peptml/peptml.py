#!/usr/bin/env python3
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
#    Copyright (C) 2019 Andrei Leonard Nicusan
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


# File   : peptml.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 28.08.2019


'''The `peptml` package implements a hierarchical density-based clustering
algorithm for general Positron Emission Particle Tracking (PEPT).

The PEPT-ML algorithm [1] works using the following steps:
    1. Split the data into a series of individual "samples", each containing
    a given number of LoRs. Use the base class pept.LineData for this.
    2. For every sample of LoRs, compute the *cutpoints*, or the points in
    space that minimise the distance to every pair of lines.
    3. Cluster every sample using HDBSCAN and extract the centres of the
    clusters ("1-pass clustering").
    4. Splirt the centres into samples of a given size.
    5. Cluster every sample of centres using HDBSCAN and extract the centres
    of the clusters ("2-pass clustering").
    6. Construct the trajectory of every particle using the centres from the
    previous step.

A typical workflow for using the `peptml` package would be:
    1. Read the LoRs into a `pept.LineData` class instance and set the
    `sample_size` and `overlap` appropriately.
    2. Compute the cutpoints using the `pept.tracking.peptml.Cutpoints` class.
    3. Instantiate an `pept.tracking.peptml.HDBSCANClusterer` class and cluster
    the cutpoints found previously.

More tutorials and examples can be found on the University of Birmingham
Positron Imaging Centre's GitHub repository.

PEPT-ML was successfuly used at the University of Birmingham to analyse real
Fluorine-18 tracers in air.

'''


import  time
import  sys
import  os

import  numpy               as      np
from    scipy.spatial       import  cKDTree

from    joblib              import  Parallel, delayed
from    tqdm                import  tqdm

from    concurrent.futures  import  ThreadPoolExecutor, ProcessPoolExecutor

# Fix a deprecation warning inside the sklearn library
try:
    sys.modules['sklearn.externals.six'] = __import__('six')
    sys.modules['sklearn.externals.joblib'] = __import__('joblib')
    import hdbscan
except ImportError:
    import hdbscan

import  pept
from    pept.utilities      import  find_cutpoints




def find_cutpoints_sample(sample, max_distance, cutoffs = None):
    '''Find the cutpoints in a sample of LoRs.

    A cutpoint is the point in 3D space that minimises the distance between any
    two lines. For any two non-parallel 3D lines, this point corresponds to
    half the distance between the two lines and is unique.

    This function considers every pair of lines in `sample_lines` and returns
    all the cutpoints that satisfy the following conditions:
        1. The distance between the two lines is smaller than `max_distance`.
        2. The cutpoints is within the `cutoffs`.

    Parameters
    ----------
    sample : (N, 7) numpy.ndarray
        A sample of LoRs, where each row is `[time, x1, y1, z1, x2, y2, z2]`,
        such that every line is defined by the points `[x1, y1, z1]` and
        `[x2, y2, z2]`.
    max_distance : float
        The maximum distance between any pair of lines so that their cutpoint
        will be considered.
    cutoffs : list, optional
        The cutoffs for each dimension, formatted as `[x_min, x_max,
        y_min, y_max, z_min, z_max]`. If not defined, they are computed
        automatically by calling `get_cutoffs`. The default is `None`.

    Returns
    -------
    sample_cutpoints : (N, 4) numpy.ndarray
        The computed cutpoints for the given LoRs, where each row is
        formatted as `[time, x, y, z]` for every cutpoint.

    Raises
    ------
    TypeError
        If `sample` is not a numpy array with shape (N, 7).
    TypeError
        If `cutoffs` is not a `one-dimensional array with values [min_x,
        max_x, min_y, max_y, min_z, max_z]`

    '''

    sample = np.asarray(sample, order = 'C', dtype = float)
    max_distance = float(max_distance)

    # Check sample has shape (N, 7)
    if sample.ndim != 2 or sample.shape[1] != 7:
        raise TypeError('\n[ERROR]: sample should have dimensions (N, 7). Received {}\n'.format(sample.shape))

    if cutoffs is None:
        cutoffs = Cutpoints.get_cutoffs(sample)
    else:
        cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
        if cutoffs.ndim != 1 or len(cutoffs) != 6:
            raise TypeError('\n[ERROR]: cutoffs should be a one-dimensional array with values [min_x, max_x, min_y, max_y, min_z, max_z]\n')

    sample_cutpoints = find_cutpoints(sample, max_distance, cutoffs)
    return sample_cutpoints




class Cutpoints(pept.PointData):
    '''A class that transforms LoRs into *cutpoints* for clustering.

    The `Cutpoints` class transforms LoRs (individual numpy arrays or full `LineData`)
    into cutpoints (individual numpy arrays or full `PointData`) that can then be passed
    to `HDBSCANClusterer`.

    Under typical usage, the class is instantiated with the LoR data (required to be
    an instance of `LineData`) and transforms it into an instance of `PointData` that
    stores the found cutpoints.

    For more control over the operations, the class also provides a static method (i.e.
    it can be used without instantiating the class) `find_cutpoints_sample` that
    receives a generic numpy array of LoRs (one 'sample') and returns a numpy array
    of cutpoints.

    Parameters
    ----------
    line_data : instance of pept.LineData
        The LoRs for which the cutpoints will be computed. It is required to be an
        instance of `pept.LineData`.
    max_distance : float
        The maximum distance between any two lines so that their cutpoint will be
        considered.
    cutoffs : list-like of length 6, optional
        A list (or equivalent) of the cutoff distances for every axis, formatted as
        [x_min, x_max, y_min, y_max, z_min, z_max]. Only consider the cutpoints which
        fall within these cutoff distances. The default is None, in which case they
        are automatically computed using `get_cutoffs`.
    verbose : bool, optional
        Provide extra information when computing the cutpoints: time the operation
        and show a progress bar. The default is `True`.

    Attributes
    ----------
    line_data : instance of pept.LineData
        The LoRs for which the cutpoints will be computed. It is required to be an
        instance of `pept.LineData`.
    max_distance : float
        The maximum distance between any two lines so that their cutpoint will be
        considered.
    cutoffs : list-like of length 6
        A list (or equivalent) of the cutoff distances for every axis, formatted as
        [x_min, x_max, y_min, y_max, z_min, z_max]. Only consider the cutpoints which
        fall within these cutoff distances.
    sample_size, overlap, number_of_lines, etc. : inherited from pept.PointData
        Extra attributes and methods are inherited from the base class `PointData`.

    Raises
    ------
    Exception
        If `line_data` is not an instance of `pept.LineData`.
    TypeError
        If `cutoffs` is not a one-dimensional array with values formatted as
        `[min_x, max_x, min_y, max_y, min_z, max_z]`.

    Example usage
    -------------
    Compute the cutpoints for a `LineData` instance:
        >>> line_data = pept.LineData(example_data)
        >>> cutpts = peptml.Cutpoints(line_data, 0.1)

    Compute the cutpoints for a single sample:
        >>> sample = line_data[0]
        >>> cutpts_sample = peptml.find_cutpoints_sample(sample)

    '''

    def __init__(
        self,
        line_data,
        max_distance,
        cutoffs = None,
        max_workers = None,
        verbose = True
    ):

        # Find the cutpoints when instantiated. The method
        # also initialises the instance as a `PointData` subclass.
        self.find_cutpoints(
            line_data,
            max_distance,
            cutoffs = cutoffs,
            max_workers = max_workers,
            verbose = verbose
        )


    @property
    def line_data(self):
        '''The LoRs for which the cutpoints are computed.

        line_data : instance of pept.LineData

        '''

        return self._line_data


    @line_data.setter
    def line_data(self, new_line_data):
        ''' The LoRs for which the cutpoints are computed.

        Parameters
        ----------
        line_data : instance of pept.LineData
            The LoRs for which the cutpoints will be computed. It is required to be an
            instance of `pept.LineData`.

        Raises
        ------
        Exception
            If `line_data` is not an instance of `pept.LineData`.

        '''

        # Check line_data is an instance (or a subclass!) of pept.LineData
        if not isinstance(line_data, pept.LineData):
            raise Exception('[ERROR]: line_data should be an instance of pept.LineData')

        self._line_data = line_data


    @property
    def max_distance(self):
        '''The maximum distance between any pair of lines for which the cutpoint is considered.

        max_distance : float

        '''
        return self._max_distance


    @max_distance.setter
    def max_distance(self, new_max_distance):
        '''The maximum distance between any pair of lines for which the cutpoint is considered.

        max_distance : float
            The maximum distance between any two lines so that their cutpoint will be
            considered.

        '''
        self._max_distance = new_max_distance


    @property
    def cutoffs(self):
        '''Only consider the cutpoints which fall within these cutoff distances.

        A list (or equivalent) of the cutoff distances for every axis, formatted as
        [x_min, x_max, y_min, y_max, z_min, z_max].

        cutoffs : (6) list or equivalent

        '''

        return self._cutoffs


    @cutoffs.setter
    def cutoffs(self, new_cutoffs):
        '''Only consider the cutpoints which fall within these cutoff distances.

        A list (or equivalent) of the cutoff distances for every axis, formatted as
        [x_min, x_max, y_min, y_max, z_min, z_max].

        Parameters
        ----------
        new_cutoffs : list-like of length 6, optional
            A list (or equivalent) of the cutoff distances for every axis, formatted as
            [x_min, x_max, y_min, y_max, z_min, z_max]. Only consider the cutpoints which
            fall within these cutoff distances. The default is None, in which case they
            are automatically computed using `get_cutoffs`.

        Raises
        ------
        TypeError
            If `cutoffs` is not a one-dimensional array with values formatted as
            `[min_x, max_x, min_y, max_y, min_z, max_z]`.

        '''

        cutoffs = np.asarray(new_cutoffs, order = 'C', dtype = float)
        if cutoffs.ndim != 1 or len(cutoffs) != 6:
            raise TypeError('\n[ERROR]: new_cutoffs should be a one-dimensional array with values [min_x, max_x, min_y, max_y, min_z, max_z]\n')

        self._cutoffs = cutoffs


    @staticmethod
    def get_cutoffs(sample):
        '''Compute the cutoffs from a sample of LoR data.

        This is a static method, meaning it can be called without
        instantiating the `Cutpoints` class. It computes the cutoffs
        from the minimum and maximum values of the LoRs in `sample`
        in each dimension.

        Parameters
        ----------
        sample : (N, 7) numpy.ndarray
            A sample of LoRs, where each row is `[time, x1, y1, z1, x2, y2, z2]`,
            such that every line is defined by the points `[x1, y1, z1]` and
            `[x2, y2, z2]`.
        Returns
        -------
        cutoffs : (6) numpy.ndarray
            The computed cutoffs for each dimension, formatted as `[x_min, x_max,
            y_min, y_max, z_min, z_max]`.

        Raises
        ------
        TypeError
            If `sample` is not a numpy array with shape (N, 7).

        '''

        # Check sample has shape (N, 7)
        if sample.ndim != 2 or sample.shape[1] != 7:
            raise TypeError('\n[ERROR]: sample should have dimensions (N, 7). Received {}\n'.format(sample.shape))

        # Compute cutoffs for cutpoints as the (min, max) values of the lines
        # Minimum value of the two points that define a line
        min_x = min(sample[:, 1].min(),
                    sample[:, 4].min())
        # Maximum value of the two points that define a line
        max_x = max(sample[:, 1].max(),
                    sample[:, 4].max())

        # Minimum value of the two points that define a line
        min_y = min(sample[:, 2].min(),
                    sample[:, 5].min())
        # Maximum value of the two points that define a line
        max_y = max(sample[:, 2].max(),
                    sample[:, 5].max())

        # Minimum value of the two points that define a line
        min_z = min(sample[:, 3].min(),
                    sample[:, 6].min())
        # Maximum value of the two points that define a line
        max_z = max(sample[:, 3].max(),
                    sample[:, 6].max())

        cutoffs = np.array([min_x, max_x, min_y, max_y, min_z, max_z])
        return cutoffs


    def find_cutpoints(
        self,
        line_data,
        max_distance,
        cutoffs = None,
        max_workers = None,
        verbose = False
    ):
        '''Find the cutpoints of the samples in a `LineData` instance.

        Parameters
        ----------
        line_data : instance of pept.LineData
            The LoRs for which the cutpoints will be computed. It is required to be an
            instance of `pept.LineData`.
        max_distance : float
            The maximum distance between any two lines so that their cutpoint will be
            considered.
        cutoffs : list-like of length 6, optional
            A list (or equivalent) of the cutoff distances for every axis, formatted as
            [x_min, x_max, y_min, y_max, z_min, z_max]. Only consider the cutpoints which
            fall within these cutoff distances. The default is None, in which case they
            are automatically computed using `get_cutoffs`.
        verbose : bool, optional
            Provide extra information when computing the cutpoints: time the operation
            and show a progress bar. The default is `False`.

        Returns
        -------
        self : the PointData instance of cutpoints
            The computed cutpoints are stored in the `Cutpoints` class, as a
            subclass of `pept.PointData`.

        Raises
        ------
        Exception
            If `line_data` is not an instance of `pept.LineData`.
        TypeError
            If `cutoffs` is not a one-dimensional array with values formatted as
            `[min_x, max_x, min_y, max_y, min_z, max_z]`.

        '''

        if verbose:
            start = time.time()

        # Check line_data is an instance (or a subclass!) of pept.LineData
        if not isinstance(line_data, pept.LineData):
            raise Exception('[ERROR]: line_data should be an instance of pept.LineData')

        self._line_data = line_data
        self._max_distance = float(max_distance)

        # If cutoffs were not supplied, compute them
        if cutoffs is None:
            cutoffs = self.get_cutoffs(line_data.line_data)
        # Otherwise make sure they are a C-contiguous numpy array
        else:
            cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
            if cutoffs.ndim != 1 or len(cutoffs) != 6:
                raise TypeError('\n[ERROR]: cutoffs should be a one-dimensional array with values [min_x, max_x, min_y, max_y, min_z, max_z]\n')

        self._cutoffs = cutoffs

        # Using ThreadPoolExecutor, asynchronously collect the cutpoints from
        # every sample in a list of arrays. This is more efficient than using
        # ProcessPoolExecutor because find_cutpoints is a Cython function that
        # releases the GIL for most of its computation.
        # If verbose, show progress bar using tqdm.
        if max_workers is None:
            max_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = []
            for sample in line_data:
                futures.append(executor.submit(find_cutpoints, sample, max_distance, cutoffs))

            if verbose:
                futures = tqdm(futures)

            cutpoints = [f.result() for f in futures]

        # cutpoints shape: (n, m, 4), where n is the number of samples, and
        # m is the number of cutpoints in the sample
        cutpoints = np.array(cutpoints)

        number_of_samples = len(cutpoints)
        cutpoints = np.vstack(np.array(cutpoints))
        number_of_cutpoints = len(cutpoints)

        # Average number of cutpoints per sample
        cutpoints_per_sample = int(number_of_cutpoints / number_of_samples)

        super().__init__(
            cutpoints,
            sample_size = cutpoints_per_sample,
            overlap = 0,
            verbose = False
        )

        if verbose:
            end = time.time()
            print("\nFinding the cutpoints took {} seconds\n".format(end - start))

        return self




def findMeanError(truePositions, foundPositions):

    tree = cKDTree(truePositions)

    meanError = 0
    meanErrorX = 0
    meanErrorY = 0
    meanErrorZ = 0
    n = 0
    for centre in foundPositions:
        d, index = tree.query(centre, k = 1,  n_jobs = -1)
        meanError += np.linalg.norm(centre - truePositions[index])

        meanErrorX += np.abs(centre[0] - truePositions[index][0])
        meanErrorY += np.abs(centre[1] - truePositions[index][1])
        meanErrorZ += np.abs(centre[2] - truePositions[index][2])

        n += 1

    meanError /= n

    meanErrorX /= n
    meanErrorY /= n
    meanErrorZ /= n

    return [meanError, meanErrorX, meanErrorY, meanErrorZ]




class HDBSCANClusterer:
    '''HDBSCAN-based clustering for cutpoints from LoRs.

    This class is a wrapper around the `hdbscan` package, providing tools for
    parallel clustering of samples of cutpoints. It can return `PointData`
    classes which can be easily manipulated or visualised.

    Parameters
    ----------
        min_cluster_size : int, optional
            (Taken from hdbscan's documentation): The minimum size of clusters;
            single linkage splits that contain fewer points than this will be
            considered points “falling out” of a cluster rather than a cluster
            splitting into two new clusters. The default is 20.
        min_samples : int, optional
            (Taken from hdbscan's documentation): The number of samples in a
            neighbourhood for a point to be considered a core point. The default
            is None, being set automatically to the `min_cluster_size`.
        allow_single_cluster : bool, optional
            (Taken from hdbscan's documentation): By default HDBSCAN* will not
            produce a single cluster, setting this to True will override this and
            allow single cluster results in the case that you feel this is a valid
            result for your dataset. For PEPT, set this to True if you only have
            one tracer in the dataset. Otherwise, leave it to False, as it will
            provide higher accuracy.

    Attributes
    ----------
        min_cluster_size : int
            (Taken from hdbscan's documentation): The minimum size of clusters;
            single linkage splits that contain fewer points than this will be
            considered points “falling out” of a cluster rather than a cluster
            splitting into two new clusters. The default is 20.
        min_samples : int
            (Taken from hdbscan's documentation): The number of samples in a
            neighbourhood for a point to be considered a core point. The default
            is None, being set automatically to the `min_cluster_size`.
        allow_single_cluster : bool
            (Taken from hdbscan's documentation): By default HDBSCAN* will not
            produce a single cluster, setting this to True will override this and
            allow single cluster results in the case that you feel this is a valid
            result for your dataset. For PEPT, set this to True if you only have
            one tracer in the dataset. Otherwise, leave it to False, as it will
            provide higher accuracy.

    '''

    def __init__(
        self,
        min_cluster_size = 20,
        min_samples = None,
        allow_single_cluster = False
    ):

        if 0 < min_cluster_size < 2:
            print("\n[WARNING]: min_cluster_size was set to 2, as it was {} < 2\n".format(min_cluster_size))
            min_cluster_size = 2

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size = min_cluster_size,
            min_samples = min_samples,
            core_dist_n_jobs = -1,
            allow_single_cluster = allow_single_cluster
        )


    @property
    def min_cluster_size(self):
        return self.clusterer.min_cluster_size


    @min_cluster_size.setter
    def min_cluster_size(self, new_min_cluster_size):
        self.clusterer.min_cluster_size = new_min_cluster_size


    @property
    def min_samples(self):
        return self.clusterer.min_cluster_size


    @min_samples.setter
    def min_samples(self, new_min_samples):
        self.clusterer.min_samples = new_min_samples


    @property
    def allow_single_cluster(self):
        return self.clusterer.allow_single_cluster


    @allow_single_cluster.setter
    def allow_single_cluster(self, option):
        self.clusterer.allow_single_cluster = option


    def fit_sample(
        self,
        sample,
        store_labels = False,
        noise = False,
        as_array = True,
        verbose = False
    ):
        '''Fit one sample of cutpoints and return the cluster centres and
        (optionally) the labelled cutpoints.

        Parameters
        ----------
        sample : (N, M >= 4) numpy.ndarray
            The sample of points that will be clustered. Every point corresponds to
            a row and is formatted as `[time, x, y, z, etc]`. Only columns `[1, 2, 3]`
            are used for clustering.
        store_labels : bool, optional
            If set to True, the clustered cutpoints are returned along with the centres
            of the clusters. Setting it to False speeds up the clustering. The default
            is False.
        noise : bool, optional
            If set to True, the clustered cutpoints also include the points classified
            as noise. Only has an effect if `store_labels` is set to True. The default
            is False.
        as_array : bool, optional
            If set to True, the centres of the clusters and the clustered cutpoints are
            returned as numpy arrays. If set to False, they are returned inside
            instances of `pept.PointData`.
        verbose : bool, optional
            Provide extra information when computing the cutpoints: time the operation
            and show a progress bar. The default is `False`.

        Returns
        -------
        centres : numpy.ndarray or pept.PointData
            The centroids of every cluster found. They are computed as the average
            of every column of `[time, x, y, z, etc]` of the clustered points. Another
            column is added to the initial data in `sample`, signifying the cluster
            size - the number of points included in the cluster. If `as_array` is
            set to True, it is a numpy array, otherwise the centres are stored
            in a pept.PointData instance.
        clustered_cutpoints : numpy.ndarray or pept.PointData
            The points in `sample` that fall in every cluster. A new column is added
            to the points in `sample` that signifies the label of cluster that the
            point was associated with: all points in cluster number 3 will have the
            number 3 as the last element in their row. The points classified as noise
            have the number -1 associated. If `as_array` is set to True, it is a numpy
            array, otherwise the clustered cutpoints are stored in a pept.PointData
            instance.

        Raises
        ------
        TypeError
            If `sample` is not a numpy array of shape (N, M), where M >= 4.

        '''

        if verbose:
            start = time.time()

        # sample row: [time, x, y, z]
        if sample.ndim != 2 or sample.shape[1] < 4:
            raise TypeError('\n[ERROR]: sample should have two dimensions (M, N), where N >= 4. Received {}\n'.format(sample.shape))

        # Only cluster based on [x, y, z]
        labels = self.clusterer.fit_predict(sample[:, 1:4])
        max_label = labels.max()

        centres = []
        clustered_cutpoints = []

        # the centre of a cluster is the average of the time, x, y, z columns
        # and the number of points of that cluster
        # centres row: [time, x, y, z, ..etc.., cluster_size]
        for i in range(0, max_label + 1):
            # Average time, x, y, z of cluster of label i
            centres_row = np.mean(sample[labels == i], axis = 0)
            # Append the number of points of label i => cluster_size
            centres_row = np.append(centres_row, (labels == i).sum())
            centres.append(centres_row)

        centres = np.array(centres)

        if not as_array:
            centres = pept.PointData(
                centres,
                sample_size = 0,
                overlap = 0,
                verbose = False
            )

        # Return all cutpoints as a list of numpy arrays for every label
        # where the last column of an array is the label
        if store_labels:
            # Create a list of numpy arrays with rows: [t, x, y, z, ..etc.., label]
            if noise:
                cutpoints = sample[labels == -1]
                cutpoints = np.insert(cutpoints, cutpoints.shape[1], -1, axis = 1)
                clustered_cutpoints.append(cutpoints)

            for i in range(0, max_label + 1):
                cutpoints = sample[labels == i]
                cutpoints = np.insert(cutpoints, cutpoints.shape[1], i, axis = 1)
                clustered_cutpoints.append(cutpoints)

            clustered_cutpoints = np.vstack(np.array(clustered_cutpoints))

            if not as_array:
                clustered_cutpoints = pept.PointData(
                    clustered_cutpoints,
                    sample_size = 0,
                    overlap = 0,
                    verbose = False
                )

        if verbose:
            end = time.time()
            print("Fitting one sample took {} seconds".format(end - start))

        return [centres, clustered_cutpoints]


    def fit_cutpoints(
        self,
        cutpoints,
        store_labels = False,
        noise = False,
        max_workers = None,
        verbose = True
    ):
        '''Fit cutpoints (an instance of `PointData`) and return the cluster
        centres and (optionally) the labelled cutpoints.

        Parameters
        ----------
        cutpoints : an instance of `pept.PointData`
            The samples of points that will be clustered. In every sample, every point
            corresponds to a row and is formatted as `[time, x, y, z, etc]`. Only
            columns `[1, 2, 3]` are used for clustering.
        store_labels : bool, optional
            If set to True, the clustered cutpoints are returned along with the centres
            of the clusters. Setting it to False speeds up the clustering. The default
            is False.
        noise : bool, optional
            If set to True, the clustered cutpoints also include the points classified
            as noise. Only has an effect if `store_labels` is set to True. The default
            is False.
        verbose : bool, optional
            Provide extra information when computing the cutpoints: time the operation
            and show a progress bar. The default is `False`.

        Returns
        -------
        centres : pept.PointData
            The centroids of every cluster found. They are computed as the average
            of every column of `[time, x, y, z, etc]` of the clustered points. Another
            column is added to the initial data in `sample`, signifying the cluster
            size - the number of points included in the cluster.
        clustered_cutpoints : numpy.ndarray or pept.PointData
            The points in `sample` that fall in every cluster. A new column is added
            to the points in `sample` that signifies the label of cluster that the
            point was associated with: all points in cluster number 3 will have the
            number 3 as the last element in their row. The points classified as noise
            have the number -1 associated.

        Raises
        ------
        Exception
            If `cutpoints` is not an instance (or a subclass) of `pept.PointData`.

        '''

        if verbose:
            start = time.time()

        if not isinstance(cutpoints, pept.PointData):
            raise Exception('[ERROR]: cutpoints should be an instance of pept.PointData (or any class inheriting from it)')

        # Fit all samples in `cutpoints` in parallel using joblib
        # Collect all outputs as a list. If verbose, show progress bar with
        # tqdm
        if verbose:
            cutpoints = tqdm(cutpoints)

        if max_workers is None:
            max_workers = os.cpu_count()

        data_list = Parallel(n_jobs = max_workers)(delayed(self.fit_sample)(
            sample,
            store_labels = store_labels,
            noise = noise,
            as_array = True) for sample in cutpoints
        )

        # Access joblib.Parallel output as list comprehensions
        centres = np.array([row[0] for row in data_list if len(row[0]) != 0])
        if len(centres) != 0:
            centres = pept.PointData(
                np.vstack(centres),
                sample_size = 0,
                overlap = 0,
                verbose = False
            )

        if store_labels:
            clustered_cutpoints = np.array([row[1] for row in data_list if len(row[1]) != 0])
            clustered_cutpoints = pept.PointData(
                np.vstack(np.array(clustered_cutpoints)),
                sample_size = 0,
                overlap = 0,
                verbose = False
            )

        if verbose:
            end = time.time()
            print("\nFitting cutpoints took {} seconds\n".format(end - start))

        if store_labels:
            return [centres, clustered_cutpoints]
        else:
            return [centres, []]




class TrajectorySeparation:

    def __init__(self, centres, pointsToCheck = 25, maxDistance = 20, maxClusterDiff = 500):
        # centres row: [time, x, y, z, clusterSize]
        # Make sure the trajectory is memory-contiguous for efficient
        # KDTree partitioning
        self.centres = np.ascontiguousarray(centres)
        self.pointsToCheck = pointsToCheck
        self.maxDistance = maxDistance
        self.maxClusterDiff = maxClusterDiff

        # For every point in centres, save a set of the trajectory
        # indices of the trajectories that they are part of
        #   eg. centres[2] is part of trajectories 0 and 1 =>
        #   trajectoryIndices[2] = {0, 1}
        # Initialise a vector of empty sets of size len(centres)
        self.trajectoryIndices = np.array([ set() for i in range(len(self.centres)) ])

        # For every trajectory found, save a list of the indices of
        # the centres that are part of that trajectory
        #   eg. trajectory 1 is comprised of centres 3, 5 and 8 =>
        #   centresIndices[1] = [3, 5, 8]
        self.centresIndices = [[]]

        # Maximum trajectory index
        self.maxIndex = 0


    def findTrajectories(self):

        for i, currentPoint in enumerate(self.centres):

            if i == 0:
                # Add the first point to trajectory 0
                self.trajectoryIndices[0].add(self.maxIndex)
                self.centresIndices[self.maxIndex].append(0)
                self.maxIndex += 1
                continue

            # Search for the closest previous pointsToCheck points
            # within a given maxDistance
            startIndex = i - self.pointsToCheck
            endIndex = i

            if startIndex < 0:
                startIndex = 0

            # Construct a KDTree from the x, y, z (1:4) of the
            # selected points. Get the indices for all the points within
            # maxDistance of the currentPoint
            tree = cKDTree(self.centres[startIndex:endIndex, 1:4])
            closestIndices = tree.query_ball_point(currentPoint[1:4], self.maxDistance, n_jobs=-1)
            closestIndices = np.array(closestIndices) + startIndex

            # If no point was found, it is a new trajectory. Continue
            if len(closestIndices) == 0:
                self.trajectoryIndices[i].add(self.maxIndex)
                self.centresIndices.append([i])
                self.maxIndex += 1
                continue

            # For every close point found, search for all the trajectory indices
            #   - If all trajectory indices sets are equal and of a single value
            #   then currentPoint is part of the same trajectory
            #   - If all trajectory indices sets are equal, but of more values,
            #   then currentPoint diverged from an intersection of trajectories
            #   and is part of a single trajectory => separate it
            #
            #   - If every pair of trajectory indices sets is not disjoint, then
            #   currentPoint is only one of them
            #   - If there exists a pair of trajectory indices sets that is
            #   disjoint, then currentPoint is part of all of them

            # Select the trajectories of all the points that were found
            # to be the closest
            closestTrajectories = self.trajectoryIndices[closestIndices]
            #print("closestTrajectories:")
            #print(closestTrajectories)

            # If all the closest points are part of the same trajectory
            # (just one!), then the currentPoint is part of it too
            if (np.all(closestTrajectories == closestTrajectories[0]) and
                len(closestTrajectories[0]) == 1):

                self.trajectoryIndices[i] = closestTrajectories[0]
                self.centresIndices[ next(iter(closestTrajectories[0])) ].append(i)
                continue

            # Otherwise, check the points based on their cluster size
            else:
                # Create a list of all the trajectories that were found to
                # intersect
                #print('\nIntersection:')
                closestTrajIndices = list( set().union(*closestTrajectories) )

                #print("ClosestTrajIndices:")
                #print(closestTrajIndices)

                # For each close trajectory, calculate the mean cluster size
                # of the last lastPoints points
                lastPoints = 50

                # Keep track of the mean cluster size that is the closest to
                # the currentPoint's clusterSize
                currentClusterSize = currentPoint[4]
                #print("currentClusterSize = {}".format(currentClusterSize))
                closestTrajIndex = -1
                clusterSizeDiff = self.maxClusterDiff

                for trajIndex in closestTrajIndices:
                    #print("trajIndex = {}".format(trajIndex))

                    trajCentres = self.centres[ self.centresIndices[trajIndex] ]
                    #print("trajCentres:")
                    #print(trajCentres)
                    meanClusterSize = trajCentres[-lastPoints:][:, 4].mean()
                    #print("meanClusterSize = {}".format(meanClusterSize))
                    #print("clusterSizeDiff = {}".format(clusterSizeDiff))
                    #print("abs diff = {}".format(np.abs( currentClusterSize - meanClusterSize )))
                    if np.abs( currentClusterSize - meanClusterSize ) < clusterSizeDiff:
                        closestTrajIndex = trajIndex
                        clusterSizeDiff = np.abs( currentClusterSize - meanClusterSize )

                if closestTrajIndex == -1:
                    #self.trajectoryIndices[i] = set(closestTrajIndices)
                    #for trajIndex in closestTrajIndices:
                    #    self.centresIndices[trajIndex].append(i)

                    print("\n**** -1 ****\n")
                    break
                else:
                    #print("ClosestTrajIndex found = {}".format(closestTrajIndex))
                    self.trajectoryIndices[i] = set([closestTrajIndex])
                    self.centresIndices[closestTrajIndex].append(i)




            '''
            # If the current point is not part of any trajectory, assign it
            # the maxIndex and increment it
            if len(self.trajectoryIndices[i]) == 0:
                self.trajectoryIndices[i].append(self.maxIndex)
                self.maxIndex += 1

            print(self.trajectoryIndices[i])
            print(self.maxIndex)

            # Construct a KDTree from the numberOfPoints in front of
            # the current point
            tree = cKDTree(self.trajectory[(i + 1):(i + self.numberOfPoints + 2)][1:4])

            # For every trajectory that the current point is part of,
            # find the closest points in front of it
            numberOfIntersections = len(self.trajectoryIndices[i])
            dist, nextPointsIndices = tree.query(currentPoint, k=numberOfIntersections, distance_upper_bound=self.maxDistance, n_jobs=-1)

            print(nextPointsIndices)

            # If the current point is part of more trajectories,
            # an intersection happened. Call subroutine to part
            # the trajectories
            if numberOfIntersections > 1:
                for j in range(0, len(self.trajectoryIndices[i])):
                    trajIndex = self.trajectoryIndices[i][j]
                    self.trajectoryIndices[i + 1 + nextPointsIndices[j]].append(trajIndex)

            else:
                self.trajectoryIndices[i + 1 + nextPointsIndices].append(self.trajectoryIndices[i][0])

            print(self.trajectoryIndices)
            '''


    def getTrajectories(self):

        self.individualTrajectories = []
        for trajCentres in self.centresIndices:
            self.individualTrajectories.append(self.centres[trajCentres])

        self.individualTrajectories = np.array(self.individualTrajectories)
        return self.individualTrajectories

        '''
        self.individualTrajectories = [ [] for i in range(0, self.maxIndex + 1) ]
        for i in range(0, len(self.trajectoryIndices)):
            for trajIndex in self.trajectoryIndices[i]:
                self.individualTrajectories[trajIndex].append(self.centres[i])

        self.individualTrajectories = np.array(self.individualTrajectories)
        for i in range(len(self.individualTrajectories)):
            if len(self.individualTrajectories[i]) > 0:
                self.individualTrajectories[i] = np.vstack(self.individualTrajectories[i])
        return self.individualTrajectories
        '''


    def plotTrajectoriesAltAxes(self, ax):
        trajectories = self.getTrajectories()
        for traj in trajectories:
            if len(traj) > 0:
                ax.scatter(traj[:, 3], traj[:, 1], traj[:, 2], marker='D', s=10)




