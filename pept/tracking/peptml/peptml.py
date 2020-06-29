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


# File   : peptml.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 28.08.2019


import  time
import  sys
import  os
import  warnings
import  textwrap

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




class HDBSCANClusterer:
    '''Efficient, optionally-parallel HDBSCAN-based clustering for cutpoints
    computed from LoRs (or generic 3D points).

    This class is a wrapper around the `hdbscan` package, providing tools for
    parallel clustering of samples of cutpoints. It can return `PointData`
    classes for ease of manipulation and visualisation.

    Two main methods are provided: `fit_sample` for clustering a single numpy
    array of cutpoints (i.e. a single sample) and `fit` which clusters all the
    samples encapsulated in a `pept.PointData` class (such as the one returned
    by the `Cutpoints` class) *in parallel*.

    Attributes
    ----------
    min_cluster_size : int
        (Taken from hdbscan's documentation): The minimum size of clusters;
        single linkage splits that contain fewer points than this will be
        considered points “falling out” of a cluster rather than a cluster
        splitting into two new clusters.
    min_samples : int
        (Taken from hdbscan's documentation): The number of samples in a
        neighbourhood for a point to be considered a core point. The default is
        `None`, being set automatically to the `min_cluster_size`.
    allow_single_cluster : bool or str
        By default HDBSCAN will not produce a single cluster - this creates
        "tighter" clusters (i.e. more accurate positions). Setting this to
        `False` will discard datasets with a single tracer, but will produce
        more accurate positions. Setting this to `True` will also work for
        single tracers, at the expense of lower accuracy for cases with more
        tracers. This class provides a third option, "auto", in which case two
        clusterers will be used: one with `allow_single_cluster` set to `False`
        and another with it set to `True`; the latter will only be used if the
        first did not find any clusters.
    max_workers : int
        The maximum number of threads that will be used for asynchronously
        clustering the samples in `cutpoints`.
    labels : (N,) numpy.ndarray, dtype = int
        A 1D array of the cluster labels for cutpoints fitted using
        `fit_sample` or `fit`. If `fit_sample` is used, `labels` correspond to
        each row in the sample array fitted. If `fit` is used with the setting
        `get_labels = True`, `labels` correpond to the stacked labels for every
        sample in the given `pept.PointData` class.

    Methods
    -------
    fit_sample(\
        sample,\
        get_labels = False,\
        as_array = True,\
        verbose = False,\
        _set_labels = True\
    )
        Fit one sample of cutpoints and return the cluster centres and
        (optionally) the labelled cutpoints.
    fit(\
        cutpoints,\
        get_labels = False,\
        max_workers = None,\
        verbose = True\
    )
        Fit cutpoints (an instance of `PointData`) and return the cluster
        centres and (optionally) the labelled cutpoints.

    Examples
    --------
    A typical workflow would involve reading LoRs from a file, computing their
    cutpoints, clustering them and plotting them.

    >>> import pept
    >>> from pept.tracking import peptml
    >>>
    >>> lors = pept.LineData(...)
    >>> cutpoints = peptml.Cutpoints(lors, 0.1)
    >>> clusterer = peptml.HDBSCANClusterer()
    >>> centres = clusterer.fit(cutpoints)
    >>>
    >>> grapher = PlotlyGrapher()
    >>> grapher.add_points(centres)
    >>> grapher.show()

    For more advanced uses of HDBSCANClusterer such as 2-pass clustering, do
    check out the tutorials available on the Birmingham's Positron Imaging
    Centre's GitHub repository at github.com/uob-positron-imaging-centre.

    See Also
    --------
    pept.tracking.peptml.Cutpoints : Compute cutpoints from `pept.LineData`.
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    def __init__(
        self,
        min_cluster_size = 20,
        min_samples = None,
        allow_single_cluster = "auto",
        max_workers = None
    ):
        '''HDBSCANClusterer class constructor.

        Parameters
        ----------
        min_cluster_size : int, default 20
            (Taken from hdbscan's documentation): The minimum size of clusters;
            single linkage splits that contain fewer points than this will be
            considered points “falling out” of a cluster rather than a cluster
            splitting into two new clusters.
        min_samples : int, optional
            (Taken from hdbscan's documentation): The number of samples in a
            neighbourhood for a point to be considered a core point. The
            default is None, being set automatically to the `min_cluster_size`.
        allow_single_cluster : bool or str, default "auto"
            By default HDBSCAN will not produce a single cluster - this creates
            "tighter" clusters (i.e. more accurate positions). Setting this to
            `False` will discard datasets with a single tracer, but will
            produce more accurate positions. Setting this to `True` will also
            work for single tracers, at the expense of lower accuracy for cases
            with more tracers. This class provides a third option, "auto", in
            which case two clusterers will be used: one with
            `allow_single_cluster` set to `False` and another with it set to
            `True`; the latter will only be used if the first did not find any
            clusters.
        max_workers : int, optional
            The maximum number of threads that will be used for asynchronously
            clustering the samples in `cutpoints`. If unset (`None`), the
            number of threads available on the machine (as returned by
            `os.cpu_count()`) will be used.

        Raises
        ------
        ValueError
            If `allow_single_cluster` is not `True`, `False` or "auto".
        '''

        if min_cluster_size < 2:
            warnings.warn((
                "\n[WARNING]: min_cluster_size was set to 2, as it was "
                f"{min_cluster_size} < 2.\n"
            ), RuntimeWarning)
            min_cluster_size = 2
        else:
            min_cluster_size = int(min_cluster_size)

        if max_workers is None:
            max_workers = os.cpu_count()

        self._labels = None

        if allow_single_cluster == True:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size = min_cluster_size,
                min_samples = min_samples,
                core_dist_n_jobs = max_workers,
                allow_single_cluster = allow_single_cluster
            )

            self.clusterer_single = None

        elif allow_single_cluster == False:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size = min_cluster_size,
                min_samples = min_samples,
                core_dist_n_jobs = max_workers,
                allow_single_cluster = allow_single_cluster
            )

            self.clusterer_single = None

        elif str(allow_single_cluster).lower() == "auto":
            self._allow_single_cluster = "auto"
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size = min_cluster_size,
                min_samples = min_samples,
                core_dist_n_jobs = max_workers,
                allow_single_cluster = False
            )

            self.clusterer_single = hdbscan.HDBSCAN(
                min_cluster_size = min_cluster_size,
                min_samples = min_samples,
                core_dist_n_jobs = max_workers,
                allow_single_cluster = True
            )

        else:
            raise ValueError((
                "\n[ERROR]: `allow_single_cluster` should be either `True`, "
                f"`False` or 'auto'. Received {allow_single_cluster}.\n"
            ))


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
        return self._allow_single_cluster


    @allow_single_cluster.setter
    def allow_single_cluster(self, allow_single_cluster):
        if allow_single_cluster == True:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer.allow_single_cluster = True
            self.clusterer_single = None

        elif allow_single_cluster == False:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer.allow_single_cluster = False
            self.clusterer_single = None

        elif str(allow_single_cluster).lower() == "auto":
            self._allow_single_cluster = "auto"
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size = self.min_cluster_size,
                min_samples = self.min_samples,
                core_dist_n_jobs = self.max_workers,
                allow_single_cluster = False
            )

            self.clusterer_single = hdbscan.HDBSCAN(
                min_cluster_size = self.min_cluster_size,
                min_samples = self.min_samples,
                core_dist_n_jobs = self.max_workers,
                allow_single_cluster = True
            )

        else:
            raise ValueError((
                "\n[ERROR]: `allow_single_cluster` should be either `True`, "
                f"`False` or 'auto'. Received {allow_single_cluster}.\n"
            ))


    @property
    def max_workers(self):
        return self.clusterer.core_dist_n_jobs


    @max_workers.setter
    def max_workers(self, new_max_workers):
        if new_max_workers is None:
            self.clusterer.core_dist_n_jobs = os.cpu_count()
        else:
            self.clusterer.core_dist_n_jobs = new_max_workers


    @property
    def labels(self):
        return self._labels


    def fit_sample(
        self,
        sample,
        get_labels = False,
        as_array = True,
        verbose = False,
        _set_labels = True
    ):
        '''Fit one sample of cutpoints and return the cluster centres and
        (optionally) the labelled cutpoints.

        Parameters
        ----------
        sample : (N, M >= 4) numpy.ndarray
            The sample of points that will be clustered. The expected columns
            are `[time, x, y, z, etc]`. Only columns `[1, 2, 3]` are used for
            clustering.
        get_labels : bool, default False
            If set to True, the input `sample` is also returned with an extra
            column representing the label of the cluster that each point is
            associated with. This label is an `int`, numbering clusters
            starting from 0; noise is represented with the value -1.
        as_array : bool, default True
            If set to True, the centres of the clusters and the labelled
            cutpoints are returned as numpy arrays. If set to False, they are
            returned inside instances of `pept.PointData`.
        verbose : bool, default False
            Provide extra information when computing the cutpoints: time the
            operation and show a progress bar.
        _set_labels : bool, default True
            This is an internal setting that an end-user should not normally
            care about. If `True`, the class property `labels` will be set
            after fitting. Setting this to `False` is helpful for multithreaded
            contexts - when calling `fit_sample` in parallel, it makes sure
            no internal attributes are mutated at the same time.

        Returns
        -------
        centres : numpy.ndarray or pept.PointData
            The centroids of every cluster found with columns
            `[time, x, y, z, ..., cluster_size]`. They are computed as the
            column-wise average of the points included in each cluster (i.e.
            for each label). Another column is added to the initial data in
            `sample`, signifying the cluster size - that is, the number of
            points included in the cluster. If `as_array` is set to True, it is
            a numpy array, otherwise the centres are stored in a
            `pept.PointData` instance.
        sample_labelled : optional, numpy.ndarray or pept.PointData
            Returned if `get_labels` is `True`. It is the input `sample` with
            an appended column representing the label of the cluster that the
            point was associated with. The labels are integers starting from 0.
            The points classified as noise have the number -1 associated. If
            `as_array` is set to True, it is a numpy array, otherwise the
            labelled points are stored in a `pept.PointData` instance.

        Raises
        ------
        ValueError
            If `sample` is not a numpy array of shape (N, M), where M >= 4.

        Notes
        -----
        If no clusters were found (i.e. all labels are -1), the returned values
        are empty numpy arrays.
        '''

        if verbose:
            start = time.time()

        # sample columns: [time, x, y, z, ...]
        sample = np.asarray(sample, dtype = float, order = "C")
        if sample.ndim != 2 or sample.shape[1] < 4:
            raise ValueError((
                "\n[ERROR]: `sample` should have two dimensions (M, N), where "
                f"N >= 4. Received {sample.shape}.\n"
            ))

        # Only cluster based on [x, y, z]. Make a C-contiguous copy to improve
        # cache-locality, then delete it.
        sample_xyz = np.asarray(sample[:, 1:4], dtype = float, order = "C")

        labels = self.clusterer.fit_predict(sample_xyz)
        max_label = labels.max()

        # If `allow_single_cluster` is "auto", check if no clusters were found
        # and try again using the hdbscan option allow_single_cluster = True.
        if max_label == -1 and self._allow_single_cluster == "auto":
            labels = self.clusterer_single.fit_predict(sample_xyz)
            max_label = labels.max()

        del sample_xyz

        if _set_labels:
            self._labels = labels

        # the centre of a cluster is the average of the time, x, y, z columns
        # + the number of points in that cluster (i.e. cluster size)
        # centres columns: [time, x, y, z, ..etc.., cluster_size]
        centres = []
        for i in range(0, max_label + 1):
            # Average time, x, y, z of cluster of label i
            centres_row = np.mean(sample[labels == i], axis = 0)

            # Append the number of points of label i => cluster_size
            centres_row = np.append(centres_row, (labels == i).sum())
            centres.append(centres_row)

        centres = np.array(centres)

        if not as_array and len(centres) != 0:
            centres = pept.PointData(
                centres,
                sample_size = 0,
                overlap = 0,
                verbose = False
            )

        if verbose:
            end = time.time()
            print("Fitting one sample took {} seconds".format(end - start))

        # If labels are requested, also return the initial sample with appended
        # labels. Labels go from 0 to max_label; -1 represents noise.
        if get_labels:
            sample_labelled = np.append(
                sample, labels[:, np.newaxis], axis = 1
            )
            if not as_array and len(samples_labelled) != 0:
                sample_labelled = pept.PointData(
                    samples_labelled,
                    sample_size = 0,
                    overlap = 0,
                    verbose = False
                )
            return centres, sample_labelled

        # Otherwise just return the found centres
        return centres


    def fit(
        self,
        cutpoints,
        get_labels = False,
        max_workers = None,
        verbose = True
    ):
        '''Fit cutpoints (an instance of `PointData`) and return the cluster
        centres and (optionally) the labelled cutpoints.

        This is a convenience function that clusters each sample in an instance
        of `pept.PointData` *in parallel*, using joblib. For more fine-grained
        control over the clustering, the `fit_sample` method can be used for
        each individual sample.

        Parameters
        ----------
        cutpoints : an instance of `pept.PointData`
            The samples of points that will be clustered. Be careful to set the
            appropriate `sample_size` and `overlap` for good results. If the
            `sample_size` is too low, the less radioactive tracers might not be
            found; if it is too high, temporal resolution is decreased. If the
            `overlap` is too small, the tracked points might be very "sparse".
            Note: when transforming LoRs into cutpoints using the `Cutpoints`
            class, the `sample_size` is automatically set based on the average
            number of cutpoints found per sample of LoRs.
        get_labels : bool, default False
            If set to True, the labelled cutpoints are returned along with the
            centres of the clusters. The labelled cutpoints are a list of
            `pept.PointData` for each sample of cutpoints, with an appended
            column representing the cluster labels (starting from 0; noise is
            encoded as -1).
        max_workers : int, optional
            The maximum number of threads that will be used for asynchronously
            clustering the samples in `cutpoints`. If unset (`None`), the
            number of threads available on the machine (as returned by
            `os.cpu_count()`) will be used.
        verbose : bool, default True
            Provide extra information when computing the cutpoints: time the
            operation and show a progress bar.

        Returns
        -------
        centres : pept.PointData
            The centroids of every cluster found with columns
            `[time, x, y, z, ..., cluster_size]`. They are computed as the
            column-wise average of the points included in each cluster (i.e.
            for each label). Another column is added to the initial data in
            `sample`, signifying the cluster size - that is, the number of
            points included in the cluster.
        labelled_cutpoints : optional, pept.PointData
            Returned if `get_labels` is `True`. It is a `pept.PointData`
            instance in which every sample is the corresponding sample in
            `cutpoints`, but with an appended column representing the label of
            the cluster that the point was associated with. The labels are
            integers starting from 0. The points classified as noise have the
            number -1 associated. Note that the labels are only consistent
            within the same sample; that is, for tracers A and B, if in one
            sample A gets the label 0 and B the label 1, in another sample
            their order might be inversed. The trajectory separation module
            might be used to separate them out.

        Raises
        ------
        TypeError
            If `cutpoints` is not an instance (or a subclass) of
            `pept.PointData`.

        Notes
        -----
        If no clusters were found (i.e. all labels are -1), the returned values
        are empty numpy arrays.
        '''

        if verbose:
            start = time.time()

        if not isinstance(cutpoints, pept.PointData):
            raise TypeError((
                "\n[ERROR]: cutpoints should be an instance of "
                "`pept.PointData` (or any class inheriting from it). Received "
                f"{type(cutpoints)}.\n"
            ))

        # Users might forget to set the sample_size, leaving it to the default
        # value of 0; in that case, all points are returned as a single sample;
        # that might not be the intended behaviour.
        if cutpoints.sample_size == 0:
            warnings.warn(
                textwrap.fill((
                    "\n[WARNING]: The `cutpoints.sample_size` was left to the "
                    "default value of 0, in which case all points are returned"
                    " as a single sample. For a very large number of points, "
                    "this might result in a long function execution time.\n"
                ), replace_whitespace = False),
                RuntimeWarning
            )

        get_labels = bool(get_labels)

        # Fit all samples in `cutpoints` in parallel using joblib
        # Collect all outputs as a list. If verbose, show progress bar with
        # tqdm
        if verbose:
            cutpoints = tqdm(cutpoints)

        if max_workers is None:
            max_workers = os.cpu_count()

        data_list = Parallel(n_jobs = max_workers)(
            delayed(self.fit_sample)(
                sample,
                get_labels = get_labels,
                as_array = True,
                verbose = False,
                _set_labels = False
            ) for sample in cutpoints
        )

        if not get_labels:
            # data_list is a list of arrays. Only choose the arrays with at
            # least one row.
            centres = [r for r in data_list if len(r) != 0]
        else:
            # data_list is a list of tuples, in which the first element is an
            # array of the centres, and the second element is an array of the
            # labelled cutpoints.
            centres = [r[0] for r in data_list if len(r[0]) != 0]

        if len(centres) != 0:
            centres = pept.PointData(
                np.vstack(centres),
                sample_size = 0,
                overlap = 0,
                verbose = False
            )

        if verbose:
            end = time.time()
            print("\nFitting cutpoints took {} seconds.\n".format(end - start))

        if get_labels:
            # data_list is a list of tuples, in which the first element is an
            # array of the centres, and the second element is an array of the
            # labelled cutpoints.
            labelled_cutpoints = [r[1] for r in data_list if len(r[1]) != 0]
            if len(labelled_cutpoints) != 0:
                # Encapsulate `labelled_cutpoints` in a `pept.PointData`
                # instance in which every sample is the corresponding sample in
                # `cutpoints`, but with an appended column representing the
                # labels. Therefore, the `sample_size` is the same as for
                # `cutpoints`, which is equal to the length of every array in
                # `labelled_cutpoints`
                labelled_cutpoints = pept.PointData(
                    np.vstack(labelled_cutpoints),
                    sample_size = len(labelled_cutpoints[0]),
                    overlap = 0,
                    verbose = False
                )

            # Set the attribute `labels` to the stacked labels of all the
            # labelled cutpoints; that is, the last column in the
            # labelled_cutpoints internal data:
            self._labels = labelled_cutpoints.points[:, -1]

            return centres, labelled_cutpoints

        return centres


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"clusterer:\n{self.clusterer}\n\n"
            f"clusterer_single:\n{self.clusterer_single}\n\n"
            f"min_cluster_size =     {self.min_cluster_size}\n"
            f"min_samples =          {self.min_samples}\n\n"
            f"allow_single_cluster = {self.allow_single_cluster}\n"
            f"max_workers =          {self.max_workers}\n"
            f"labels =               {self.labels}"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPL
        docstr = (
            "Class instance that inherits from `peptml.HDBSCANClusterer`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n"
            "----------\n"
            f"{self.__str__()}\n"
        )

        return docstr


