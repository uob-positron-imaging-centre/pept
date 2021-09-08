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
#    Copyright (C) 2019-2021 the pept developers
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


import copy
import os
import textwrap
import time
import warnings

import  numpy               as      np

from    joblib              import  Parallel, delayed
from    tqdm                import  tqdm

import  cma
import  hdbscan

import  pept




class HDBSCANClusterer:
    '''Efficient, optionally-parallel HDBSCAN-based clustering for cutpoints
    computed from LoRs (or generic 3D points).

    Given a sample of (cut)points, they are clustered using the HDBSCAN [1]_
    algorithm; the tracers' locations are computed as the "centre" of each
    cluster, corresponding to the average of all the points in the cluster
    (this is the original implementation [2]_). Optionally, the cluster
    exemplars - that is, the most stable points in the cluster - can be used
    for the average (if `select_exemplars` == True).

    This algorithm requires no prior knowledge of the number of tracers within
    the system; it successfully distinguished multiple particles separated by
    distances as small as 2 mm. The technique’s spatial resolution was observed
    to be invariant with the number of tracers used, allowing large numbers of
    particles to be tracked simultaneously, with no loss of data quality [2]_.

    This class is a wrapper around the `hdbscan` package, providing tools for
    parallel clustering of samples of cutpoints. It can return `PointData`
    classes for ease of manipulation and visualisation.

    Two main methods are provided: `fit_sample` for clustering a single numpy
    array of cutpoints (i.e. a single sample) and `fit` which clusters all the
    samples encapsulated in a `pept.PointData` class (such as the one returned
    by the `Cutpoints` class) *in parallel*.

    Optionally, this class can find the optimum clustering parameters for a
    given dataset using the `optimise` method.

    Attributes
    ----------
    clusterer : hdbscan.HDBSCAN object
        The HDBSCAN object that will be used to cluster points. The settings
        below apply to it. If `allow_single_cluster` is set to the default
        "auto", a second clusterer object (`clusterer_single`) will be used for
        samples in which no cluster was found. This is useful for experiments
        in which tracers leave and enter the field of view, or coalesce. The
        `optimise` method can find the optimal settings for a given dataset.

    clusterer_single : hdbscan.HDBSCAN object or None
        The second HDBSCAN object, defined only if `allow_single_cluster` is
        "auto". It can have different settings than `clusterer`, which is
        useful for long experiments in which the data looks very different
        when a single tracer remains in the field of view (e.g. if two tracers
        coalesce, the resulting "single tracer" will be much more active). The
        `optimise` method can find the optimal settings for a given dataset,
        independently from `clusterer`.

    min_cluster_size : int
        (Taken from hdbscan's documentation): The minimum size of clusters;
        single linkage splits that contain fewer points than this will be
        considered points “falling out” of a cluster rather than a cluster
        splitting into two new clusters.

    min_samples : int
        (Taken from hdbscan's documentation): The number of samples in a
        neighbourhood for a point to be considered a core point. The default is
        `None`, being set automatically to the `min_cluster_size`.

    min_cluster_size_single : int or None
        The `min_cluster_size` value for `clusterer_single`, defined only if
        `allow_single_cluster` is "auto".

    min_samples_single : int
        The `min_samples` value for `clusterer_single`, defined only if
        `allow_single_cluster` is "auto".

    allow_single_cluster : bool or "auto"
        By default HDBSCAN will not produce a single cluster - this creates
        "tighter" clusters (i.e. more accurate positions). Setting this to
        `False` will discard datasets with a single tracer, but will produce
        more accurate positions. Setting this to `True` will also work for
        single tracers, at the expense of lower accuracy for cases with more
        tracers. This class provides a third option, "auto", in which case two
        clusterers will be used: one with `allow_single_cluster` set to `False`
        and another with it set to `True`; the latter will only be used if the
        first did not find any clusters.

    select_exemplars : bool, default False
        If `False`, the tracer position is computed as the average of *all* the
        points in a cluster. If `True`, the tracer position is computed as the
        average of the *cluster exemplars*. This can yield tighter clusters in
        the majority of cases, but miss some tracers if the points are very
        sparse.

    labels : (N,) numpy.ndarray, dtype = int
        A 1D array of the cluster labels for cutpoints fitted using
        `fit_sample` or `fit`. If `fit_sample` is used, `labels` correspond to
        each row in the sample array fitted. If `fit` is used with the setting
        `get_labels = True`, `labels` correpond to the stacked labels for every
        sample in the given `pept.PointData` class.

    Methods
    -------
    fit_sample(sample, get_labels = False, as_array = True, verbose = False,\
               _set_labels = True)
        Fit one sample of cutpoints and return the cluster centres and
        (optionally) the labelled cutpoints.

    fit(cutpoints, get_labels = False, max_workers = None, verbose = True)
        Fit cutpoints (an instance of `PointData`) and return the cluster
        centres and (optionally) the labelled cutpoints.

    optimise(points, nsamples = 16, max_workers = None, verbose = True,\
             _stability_params = None)
        Optimise HDBSCANClusterer settings against a given dataset of `points`.

    Examples
    --------
    A typical workflow would involve reading LoRs from a file, computing their
    cutpoints, clustering them and plotting them.

    >>> import pept
    >>> from pept.tracking import peptml
    >>>
    >>> lors = pept.LineData(...)
    >>> cutpoints = peptml.Cutpoints(lors, 0.15)
    >>>
    >>> clusterer = peptml.HDBSCANClusterer()
    >>> clusterer.optimise(cutpoints)           # optional
    >>> centres = clusterer.fit(cutpoints)
    >>>
    >>> grapher = PlotlyGrapher()
    >>> grapher.add_points(centres)
    >>> grapher.show()

    For more advanced uses of HDBSCANClusterer such as 2-pass clustering, do
    check out the tutorials available on the Birmingham's Positron Imaging
    Centre's GitHub repository at github.com/uob-positron-imaging-centre.

    References
    ----------
    .. [1] McInnes L, Healy J. Accelerated Hierarchical Density Based
       Clustering In: 2017 IEEE International Conference on Data Mining
       Workshops (ICDMW), IEEE, pp 33-42. 2017
    .. [2] Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
       using machine learning. Review of Scientific Instruments. 2020 Jan 1;
       91(1):013329.

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
        min_cluster_size_single = None,
        min_samples_single = None,
        allow_single_cluster = "auto",
        select_exemplars = False,
        core_dist_n_jobs = 1,
        **kwargs
    ):
        '''`HDBSCANClusterer` class constructor.

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
            default is `None`, being set automatically to the
            `min_cluster_size`.

        min_cluster_size_single : int, optional
            The `min_cluster_size` value for `clusterer_single`, defined only
            if `allow_single_cluster` is "auto". The default is `None`, being
            set automatically to the `min_cluster_size`.

        min_samples_single : int, optional
            The `min_samples` value for `clusterer_single`, defined only if
            `allow_single_cluster` is "auto". The default is `None`, being set
            automatically to the `min_cluster_size`.

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

        select_exemplars : bool, default False
            If `False`, the tracer position is computed as the average of *all*
            the points in a cluster. If `True`, the tracer position is computed
            as the average of the *cluster exemplars*. This can yield tighter
            clusters in the majority of cases, but miss some tracers if the
            points are very sparse.

        core_dist_n_jobs : int or None, default 1
            (Taken from hdbscan's documentation): Number of parallel jobs to
            run in core distance computations (if supported by the specific
            algorithm). If unset (`None`), the value returned by
            `os.cpu_count()` is used. This should be left to the default 1, as
            parallelism is more advantageous between samples (i.e. when calling
            `fit`) rather than for each clustering algorithm run.

        kwargs : keyword arguments
            Other keyword arguments that will be passed to the HDBSCAN
            instantiation.

        Raises
        ------
        ValueError
            If `allow_single_cluster` is not `True`, `False` or "auto".
        '''

        # Type-check min_cluster_size and min_samples
        min_cluster_size = int(round(min_cluster_size))

        if min_samples is None:
            min_samples = min_cluster_size
        else:
            min_samples = int(round(min_samples))

        if min_cluster_size_single is None:
            min_cluster_size_single = min_cluster_size
        else:
            min_cluster_size_single = int(round(min_cluster_size_single))

        if min_samples_single is None:
            min_samples_single = min_samples
        else:
            min_samples_single = int(round(min_samples_single))

        if core_dist_n_jobs is None:
            core_dist_n_jobs = os.cpu_count()

        self._select_exemplars = bool(select_exemplars)
        self._labels = None

        # Will manually supply allow_single_cluster. Only need prediction_data
        # if select_exemplars is `True`.
        clusterer_options = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "core_dist_n_jobs": core_dist_n_jobs,
            "gen_min_span_tree": True,
            "prediction_data": self._select_exemplars,
            "cluster_selection_method": "leaf",         # Quite harsh
            **kwargs
        }

        clusterer_single_options = {
            "min_cluster_size": min_cluster_size_single,
            "min_samples": min_samples_single,
            "core_dist_n_jobs": core_dist_n_jobs,
            "gen_min_span_tree": True,
            "prediction_data": self._select_exemplars,
            "cluster_selection_method": "eom",          # More lenient
            **kwargs
        }

        # Create inner clusterers (including clusterer_single if
        # allow_single_cluster == "auto")
        if allow_single_cluster is True:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer = hdbscan.HDBSCAN(
                allow_single_cluster = True,
                **clusterer_options
            )

            self.clusterer_single = None

        elif allow_single_cluster is False:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer = hdbscan.HDBSCAN(
                allow_single_cluster = False,
                **clusterer_options
            )

            self.clusterer_single = None

        elif str(allow_single_cluster).lower() == "auto":
            self._allow_single_cluster = "auto"
            self.clusterer = hdbscan.HDBSCAN(
                allow_single_cluster = False,
                **clusterer_options
            )

            self.clusterer_single = hdbscan.HDBSCAN(
                allow_single_cluster = True,
                **clusterer_single_options
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
        if new_min_cluster_size < 2:
            warnings.warn((
                "\n[WARNING]: new_min_cluster_size was set to 2, as it was "
                f"{new_min_cluster_size} < 2.\n"
            ), RuntimeWarning)
            new_min_cluster_size = 2
        else:
            new_min_cluster_size = int(round(new_min_cluster_size))

        self.clusterer.min_cluster_size = new_min_cluster_size


    @property
    def min_samples(self):
        return self.clusterer.min_samples


    @min_samples.setter
    def min_samples(self, new_min_samples):
        if new_min_samples < 2:
            warnings.warn((
                "\n[WARNING]: new_min_samples was set to 2, as it was "
                f"{new_min_samples} < 2.\n"
            ), RuntimeWarning)
            new_min_samples = 2
        else:
            new_min_samples = int(round(new_min_samples))

        self.clusterer.min_samples = new_min_samples


    @property
    def min_cluster_size_single(self):
        if self.clusterer_single is None:
            return None

        return self.clusterer_single.min_cluster_size


    @min_cluster_size_single.setter
    def min_cluster_size_single(self, new_min_cluster_size):
        if new_min_cluster_size < 2:
            warnings.warn((
                "\n[WARNING]: new_min_cluster_size was set to 2, as it was "
                f"{new_min_cluster_size} < 2.\n"
            ), RuntimeWarning)
            new_min_cluster_size = 2
        else:
            new_min_cluster_size = int(round(new_min_cluster_size))

        self.clusterer_single.min_cluster_size = new_min_cluster_size


    @property
    def min_samples_single(self):
        if self.clusterer_single is None:
            return None

        return self.clusterer_single.min_samples


    @min_samples_single.setter
    def min_samples_single(self, new_min_samples):
        if new_min_samples < 2:
            warnings.warn((
                "\n[WARNING]: new_min_samples was set to 2, as it was "
                f"{new_min_samples} < 2.\n"
            ), RuntimeWarning)
            new_min_samples = 2
        else:
            new_min_samples = int(round(new_min_samples))

        self.clusterer_single.min_samples = new_min_samples


    @property
    def allow_single_cluster(self):
        return self._allow_single_cluster


    @allow_single_cluster.setter
    def allow_single_cluster(self, allow_single_cluster):
        if allow_single_cluster is True:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer.allow_single_cluster = True
            self.clusterer_single = None

        elif allow_single_cluster is False:
            self._allow_single_cluster = allow_single_cluster
            self.clusterer.allow_single_cluster = False
            self.clusterer_single = None

        elif str(allow_single_cluster).lower() == "auto":
            self._allow_single_cluster = "auto"
            self.clusterer.allow_single_cluster = False

            # Make a deep copy of the "normal" clusterer, and only change the
            # `allow_single_cluster` option to `True`.
            self.clusterer_single = copy.deepcopy(self.clusterer)
            self.clusterer_single.allow_single_cluster = True

        else:
            raise ValueError((
                "\n[ERROR]: `allow_single_cluster` should be either `True`, "
                f"`False` or 'auto'. Received {allow_single_cluster}.\n"
            ))


    @property
    def select_exemplars(self):
        return self._select_exemplars


    @select_exemplars.setter
    def select_exemplars(self, select_exemplars):
        self._select_exemplars = bool(select_exemplars)

        self.clusterer.prediction_data = self._select_exemplars

        if self.clusterer_single is not None:
            self.clusterer_single.prediction_data = self._select_exemplars


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
            for each label) or the cluster exemplars (if the `select_exemplars`
            attribute is `True`). Another column is added to the initial data
            in `sample`, signifying the cluster size - that is, the number of
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

        labels = self.clusterer.fit_predict(sample[:, 1:4])
        max_label = labels.max()

        # If `allow_single_cluster` is "auto", check if no clusters were found
        # and try again using the hdbscan option allow_single_cluster = True.
        if max_label == -1 and self._allow_single_cluster == "auto":
            labels = self.clusterer_single.fit_predict(sample[:, 1:4])
            max_label = labels.max()

            if self._select_exemplars:
                prediction_data = \
                    self.clusterer_single._prediction_data
                selected_clusters = \
                    self.clusterer_single.condensed_tree_._select_clusters()
                raw_condensed_tree = \
                    self.clusterer_single.condensed_tree_._raw_tree

        elif self._select_exemplars:
            prediction_data = \
                self.clusterer._prediction_data
            selected_clusters = \
                self.clusterer.condensed_tree_._select_clusters()
            raw_condensed_tree = \
                self.clusterer.condensed_tree_._raw_tree

        # Select the cluster exemplars' indices
        if self._select_exemplars:
            exemplars = []
            for cluster in selected_clusters:
                cluster_exemplars = []
                for leaf in prediction_data._recurse_leaf_dfs(cluster):
                    leaf_max_lambda = raw_condensed_tree['lambda_val'][
                        raw_condensed_tree['parent'] == leaf
                    ].max()
                    points = raw_condensed_tree['child'][
                        (raw_condensed_tree['parent'] == leaf) &
                        (raw_condensed_tree['lambda_val'] == leaf_max_lambda)
                    ]
                    cluster_exemplars.append(points)
                exemplars.append(np.hstack(cluster_exemplars))

            exemplars = np.hstack(exemplars)

            # Create a boolean mask for rows in `sample` which are exemplars
            indices_exemplars = np.zeros(len(sample), dtype = bool)
            indices_exemplars[exemplars] = True

        if _set_labels:
            self._labels = labels

        # The tracer location (centre of a cluster's exemplars) is the average
        # of the time, x, y, z columns + the number of points in that cluster
        # (i.e. cluster size)
        # centres columns: [time, x, y, z, ..etc.., cluster_size]
        centres = []
        for i in range(0, max_label + 1):
            # Average time, x, y, z of the exemplars in cluster with label i
            indices_cluster = (labels == i)

            if self._select_exemplars:
                centre = np.mean(sample[indices_cluster & indices_exemplars],
                                 axis = 0)
            else:
                centre = np.mean(sample[indices_cluster], axis = 0)

            # Append the number of points of label i => cluster_size
            centre = np.append(centre, indices_cluster.sum())
            centres.append(centre)

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

            if not as_array and len(sample_labelled) != 0:
                sample_labelled = pept.PointData(
                    sample_labelled,
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
            for each label) or the cluster exemplars (if the `select_exemplars`
            attribute is `True`). Another column is added to the initial data
            in `sample`, signifying the cluster size - that is, the number of
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
            their order might be inversed. The `trajectory_separation` module
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


    def optimise(
        self,
        points,
        nsamples = 16,
        selected_indices = None,
        max_workers = None,
        verbose = True,
        _stability_params = None
    ):
        '''Optimise HDBSCANClusterer settings against a given dataset of
        `points`.

        The `points` are either a single sample (i.e. an (M, N>=4) numpy array)
        or an instance of `PointData` (i.e. multiple samples). In the latter
        case, the settings are optimised against `nsamples` random ramples
        (if `selected_indices` is None), or the samples at indices
        `selected_incides`.

        The CMA-ES (Covariance Matrix Adaptation Evolution Strategy) [1]_
        algorithm is used to find the `min_cluster_size` and
        `min_samples` combination that maximises the overall cluster
        stabilities, quantified with a fitness function that uses:

        1. The probability of each clustered point of being in the assigned
           cluster.
        2. The persistence of each cluster, gauging the relative coherence of
           the clusters.

        As the samples used for optimisation are chosen randomly and CMA-ES is
        a stochastic algorithm, some variance in the results of different
        optimisation passes is expected.

        Parameters
        ----------
        points : (M, N>=4) numpy.ndarray or pept.PointData
            The dataset for which the HDBSCAN parameters will be optimised. It
            is either a single sample (a numpy array) or an instance of
            `PointData` (multiple samples).

        nsamples : int, default 16
            If `points` is an instance of `pept.PointData` and
            `selected_indices` is `None`, `nsamples` random samples are
            selected to optimise against. A larger number may yield better
            parameters, but take more time to find. The default of 16 provides
            a good trade-off between the two.

        selected_indices : list-like of int, optional
            If `None`, `nsamples` random samples are selected from `points`.
            Otherwise, select the samples in `points` at the indices contained
            in `selected_indices`. Only has an effect if `points` is a
            `pept.PointData`.

        max_workers : int, optional
            The maximum number of threads that will be used for asynchronously
            clustering the selected samples in `points`. If unset (`None`), the
            number of threads available on the machine (as returned by
            `os.cpu_count()`) will be used.

        verbose : bool, default True
            Provide extra information during the optimisation pass.

        _stability_params : list-like of length 2 or 4, optional
            If set, the function will no longer run optimisations; rather, it
            will only return the negative fitness score, computed for the
            parameters in `_stability_params`. If `allow_single_cluster` is
            "auto", this parameter should contain the `min_cluster_size` and
            `min_samples` values for `self.clusterer`, followed by the values
            for `self.clusterer_single`. Otherwise, it should contain only the
            values for `self.clusterer`. This is useful for debugging and
            using a different algorithm than CMA-ES.

        Returns
        -------
        stabilities : float or list[float] of length 2
            If `_stability_params` is not `None`, this method will return the
            cluster stability scores. If `allow_single_cluster` is "auto", it
            will return the scores for both the `clusterer` and
            `clusterer_single`; otherwise it returns only the `clusterer`
            score. If `_stability_params` is `None`, nothing is returned.

        Raises
        ------
        ValueError
            If `selected_indices` is not `None` and is not a list-like with a
            single dimension.

        ValueError
            If `points` is a single sample and does not have shape (M, N>=4).

        Notes
        -----
        If the optimisation is successful, this class' `min_cluster_size` and
        `min_samples` attributes will be automatically set to the optimum
        values found.

        For datasets with 2000 points per sample, the optimisation should have
        around 200 function evaluations (i.e. clustering runs) and take
        somewhere under 2 minutes.

        References
        ----------
        .. [1] Nikolaus Hansen, Youhei Akimoto, and Petr Baudis. CMA-ES/pycma
           on Github. Zenodo, DOI:10.5281/zenodo.2559634, February 2019.

        '''

        # Select samples for clustering
        # points is either a single sample or PointData (i.e. multiple samples)
        if isinstance(points, pept.PointData):
            if selected_indices is None:
                # Select `nsamples` random samples
                rng = np.random.default_rng()
                selected_indices = rng.integers(0, len(points) - 1, nsamples)
            else:
                # Ensure `selected_indices` is a list-like of integers
                selected_indices = np.asarray(selected_indices, dtype = int)
                if selected_indices.ndim != 1:
                    raise ValueError(textwrap.fill(
                        "[ERROR]: If `selected_indices` is defined, it must "
                        "be list-like with a single dimension. Received "
                        f"{selected_indices.shape}"
                    ))

            selected = [points[i] for i in selected_indices]

        else:
            # We only have a single sample
            points = np.asarray(points, dtype = float)
            if points.ndim != 2 or points.shape[1] < 4:
                raise ValueError(textwrap.fill(
                    "[ERROR]: If `points` is a single sample, it should have "
                    f"shape (M, N>=4). Received {points.shape}."
                ))

            selected = [points]

        # If `_stability_params` were given, don't optimise and only return the
        # stability scores.
        if _stability_params is not None:
            score = self._optimise(
                self.clusterer, selected, max_workers, verbose,
                _stability_params[:2]
            )

            if self.clusterer_single is not None:
                self.clusterer.min_cluster_size = \
                    int(round(_stability_params[0]))
                self.clusterer.min_samples = int(round(_stability_params[1]))

                score_single = self._optimise_single(
                    self.clusterer_single, self.clusterer, selected,
                    max_workers, verbose, _stability_params[2:]
                )

                return score, score_single

            return score

        # Print optimisation info to the terminal.
        if verbose:
            print((
                "\n" + "-" * 79 + "\n" +
                "Optimising HDBSCANClusterer settings against given "
                "dataset.\n" + "-" * 79 + "\n"
            ))

            if self.clusterer_single is not None:
                print((
                    '`allow_single_cluster` is "auto" - will optimise '
                    "clusterers individually.\n"
                ))

        # Optimise `self.clusterer` settings. `self.clusterer_single` is not
        # used here!
        min_cluster_size, min_samples, score = self._optimise(
            self.clusterer, selected, max_workers, verbose, _stability_params
        )

        # Warning if no feasible settings / clusters were found. Otherwise set
        # min_cluster_size and min_samples to the found optimal values.
        if score == 0.0:
            warnings.warn(
                textwrap.fill((
                    "\n[WARNING]: Could not find settings combination that "
                    "improves the clusters' stability. Are there any clusters "
                    "in the input dataset?\nDid not set `min_cluster_size` "
                    "and `min_samples`.\n"
                ), replace_whitespace = False),
                RuntimeWarning
            )
        else:
            self.min_cluster_size = min_cluster_size
            self.min_samples = min_samples

        # If `allow_single_cluster` == "auto" (i.e. clusterer_single != None),
        # also optimise `self.clusterer_sigle` settings, only for the samples
        # in which `self.clusterer` does not find anything.
        if self.clusterer_single is not None:
            min_cluster_size_single, min_samples_single, score_single = \
                self._optimise_single(
                    self.clusterer_single, self.clusterer, selected,
                    max_workers, verbose, _stability_params
                )

            # Warning if no samples with single clusters were found. Otherwise
            # set `min_cluster_size_single` and `min_samples` to the found
            # optimal values.
            if score_single == 0.0:
                warnings.warn(
                    textwrap.fill((
                        "\n[WARNING]: Could not find settings combination that"
                        " improves the clusters' stability for the single "
                        "cluster case. Are there any samples with a single "
                        "tracer in the input dataset?\n\n"
                        "Setting `min_cluster_size` and `min_samples` to the "
                        "multi-tracer settings.\n"
                    ), replace_whitespace = False),
                    RuntimeWarning
                )
                self.min_cluster_size_single = min_cluster_size
                self.min_samples_single = min_samples
            else:
                self.min_cluster_size_single = min_cluster_size_single
                self.min_samples_single = min_samples_single

        # Print optimisation results to terminal.
        if verbose:
            print((
                "-" * 79 +
                "\nOptimal parameters found:\n"
                f"min_cluster_size = {self.min_cluster_size}\n"
                f"min_samples =      {self.min_samples}"
            ))

            if self.clusterer_single is not None:
                print((
                    "\n"
                    f"min_cluster_size_single = {self.min_cluster_size_single}"
                    f"\nmin_samples_single =      {self.min_samples_single}"
                ))

            print("-" * 79 + "\n\n")


    def _optimise(
        self,
        clusterer,
        selected,
        max_workers = None,
        verbose = True,
        _stability_params = None
    ):
        if verbose:
            start = time.time()

        if max_workers is None:
            max_workers = os.cpu_count()

        def stability_sample(clusterer, sample):
            # Only cluster based on [x, y, z].
            labels = clusterer.fit_predict(sample[:, 1:4])
            max_label = labels.max()

            prob = clusterer.probabilities_
            pers = clusterer.cluster_persistence_

            score = 0.0
            for i, label in enumerate(range(0, max_label + 1)):
                score += prob[labels == label].mean() * pers[i]

            return -score / (max_label + 2)

        def stability(params):
            params = np.asarray(params)
            clusterer.min_cluster_size = int(round(params[0]))
            clusterer.min_samples = int(round(params[1]))

            # Compute stabilities for each selected sample in parallel
            stabilities = Parallel(n_jobs = max_workers)(
                delayed(stability_sample)(clusterer, sample)
                for sample in selected
            )

            stabilities = np.array(stabilities)
            score = stabilities.mean()

            # Decrease score if there is more than 50% difference between
            # `min_cluster_size` and `min_samples`
            correction = params.min() / params.max()

            return score * correction if correction < 0.5 else score

        if _stability_params is not None:
            return stability(_stability_params)

        sample_size = len(selected[0])
        x0 = [2, 2]                 # starting guess = smallest parameters
        sigma0 = 0.1 * sample_size  # initial standard deviation
        bounds = [2, sample_size]   # bounds for min_{cluster_size | samples}

        es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
            bounds = bounds,
            integer_variables = [0, 1],
            tolflatfitness = 10,                # Try 10 times if f is flat
            tolfun = -np.inf,                   # Only stop when tolx < 1
            tolx = 1,                           # Stop if changes in x < 1
            verbose = 3 if verbose else -9
        ))
        es.optimize(stability)

        min_cluster_size = int(round(es.result.xfavorite[0]))
        min_samples = int(round(es.result.xfavorite[1]))
        score = es.result.fbest

        if verbose:
            es.result_pretty()

            end = time.time()
            print(f"\nOptimisation took {end - start} s.\n")

        return min_cluster_size, min_samples, score


    def _optimise_single(
        self,
        clusterer_single,
        clusterer_multi,
        selected,
        max_workers = None,
        verbose = True,
        _stability_params = None
    ):

        if verbose:
            start = time.time()

        if max_workers is None:
            max_workers = os.cpu_count()

        def stability_sample(clusterer_single, clusterer_multi, sample):
            # Only cluster based on [x, y, z].
            labels = clusterer_multi.fit_predict(sample[:, 1:4])
            max_label = labels.max()

            # Only score if single cluster...
            if max_label != -1:
                return 0

            labels = clusterer_single.fit_predict(sample[:, 1:4])
            max_label = labels.max()

            prob = clusterer_single.probabilities_
            pers = clusterer_single.cluster_persistence_

            score = 0.0
            for i, label in enumerate(range(0, max_label + 1)):
                score += prob[labels == label].mean() * pers[i]

            return -score / (max_label + 2)

        def stability(params):
            params = np.asarray(params)
            clusterer_single.min_cluster_size = int(round(params[0]))
            clusterer_single.min_samples = int(round(params[1]))

            # Compute stabilities for each selected sample in parallel
            stabilities = Parallel(n_jobs = max_workers)(
                delayed(stability_sample)(
                    clusterer_single, clusterer_multi, sample
                ) for sample in selected
            )

            stabilities = np.array(stabilities)
            score = stabilities.mean()

            # Decrease score if there is more than 50% difference between
            # `min_cluster_size` and `min_samples`
            correction = params.min() / params.max()

            return score * correction if correction < 0.5 else score

        if _stability_params is not None:
            return stability(_stability_params)

        sample_size = len(selected[0])
        x0 = [2, 2]                 # starting guess = smallest parameters
        sigma0 = 0.1 * sample_size  # initial standard deviation
        bounds = [2, sample_size]   # bounds for min_{cluster_size | samples}

        es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
            bounds = bounds,
            integer_variables = [0, 1],
            tolflatfitness = 10,                # Try 10 times if f is flat
            tolfun = -np.inf,                   # Only stop when tolx < 1
            tolx = 1,                           # Stop if changes in x < 1
            verbose = 3 if verbose else -9
        ))
        es.optimize(stability)

        min_cluster_size = int(round(es.result.xfavorite[0]))
        min_samples = int(round(es.result.xfavorite[1]))
        score = es.result.fbest

        if verbose:
            es.result_pretty()

            end = time.time()
            print(f"\nOptimisation took {end - start} s.\n")

        return min_cluster_size, min_samples, score


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"clusterer:\n{self.clusterer}\n\n"
            f"min_cluster_size =            {self.min_cluster_size}\n"
            f"min_samples =                 {self.min_samples}\n\n"
            f"clusterer_single:\n{self.clusterer_single}\n\n"
            f"min_cluster_size_single =     {self.min_cluster_size_single}\n"
            f"min_samples_single =          {self.min_samples_single}\n\n"
            f"allow_single_cluster =        {self.allow_single_cluster}\n"
            f"labels =                      {self.labels}"
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




# The code below is adapted from the PEPT-EM algorithm developed by Antoine
# Renaud and Sam Manger
def _centroid(lors, weights):
    nx = np.newaxis

    m = np.identity(3)[nx, :, :] - lors[:, nx, 4:7] * lors[:, 4:7, nx]
    m *= weights[:, nx, nx]

    n = np.sum(m, axis = 0)
    v = np.sum(np.sum(m * lors[:, nx, 1:4], axis = -1), axis = 0)

    return np.matmul(np.linalg.inv(n), v)


def _dist_matrix(x, lors):
    y = x[np.newaxis, :3] - lors[:, 1:4]
    return np.sum(y**2, axis = -1) - np.sum(y * lors[:, 4:7], axis = -1)**2


def _latent_weights(d2, s):
    w = np.exp(-d2 / 2 / s**2) / (s * 2.50662827463)
    w /= np.sum(w)
    return w


def _linear_weights(d2):
    return (d2.max() - d2) / (d2.max() - d2.min())


def _predict_gaussian(lines, iters = 2):
    # Rewrite LoRs in the vectorial form y(x) = position + x * direction
    lors = lines[:, :7].copy(order = "C")

    lors[:, 4:7] = lors[:, 4:7] - lors[:, 1:4]
    lors[:, 4:7] /= np.linalg.norm(lors[:, 3:], axis = -1)[:, np.newaxis]

    # Begin with equal weights for all LoRs
    weights = np.ones(len(lors))

    # Run two EM iterations so that a reasonable amount of LoRs remain with
    # significant weights
    for _ in range(iters):
        x = _centroid(lors, weights)
        d2 = _dist_matrix(x, lors)
        # variance = np.sqrt(np.mean(d2 * weights))
        # print(variance)
        weights = _latent_weights(d2, 5.0)

    return x, weights


def _predict_simple(lines):
    # Rewrite LoRs in the vectorial form y(x) = position + x * direction
    lors = lines[:, :7].copy(order = "C")

    lors[:, 4:7] = lors[:, 4:7] - lors[:, 1:4]
    lors[:, 4:7] /= np.linalg.norm(lors[:, 3:], axis = -1)[:, np.newaxis]

    # Begin with equal weights for all LoRs
    weights = np.ones(len(lors))
    x = _centroid(lors, weights)
    d2 = _dist_matrix(x, lors)

    for i in range(6):
        k = int(len(d2) * (1 - 0.1 * (i + 1)))
        part = np.argpartition(d2, k)
        weights[part[k:]] = 0

        x = _centroid(lors, weights)
        d2 = _dist_matrix(x, lors)

    weights[weights != 0] = _linear_weights(d2[weights != 0])
    return x, weights




class HDBSCAN(pept.base.PointDataFilter):
    '''Use HDBSCAN to cluster some ``pept.PointData`` and append a cluster
    label to each point.

    Filter signature:

    ::

        PointData -> HDBSCAN.fit_sample -> PointData


    The only free parameter to select is the ``true_fraction``, a relative
    measure of the ratio of inliers to outliers. A noisy sample - e.g. first
    pass of clustering of cutpoints - may need a value of `0.15`. A cleaned up
    dataset - e.g. a second pass of clustering - can work with `0.6`.

    You can also set the maximum number of tracers visible at any one time in
    the system in ``max_tracers`` (default 1). This is simply an inverse
    scaling factor, but the ``true_fraction`` is quite robust with varying
    numbers of tracers.

    '''

    def __init__(self, true_fraction, max_tracers = 1):
        self.true_fraction = float(true_fraction)
        self.max_tracers = int(max_tracers)

        self.clusterer = hdbscan.HDBSCAN(
            allow_single_cluster = False,
            core_dist_n_jobs = 1,
        )


    def _inject_cluster(self, points):
        # Inject artificial cluster in a clearly outlying region
        quantiles = np.quantile(points[:, 1:4], [0.25, 0.75], axis = 0)
        qrange = quantiles[1, :] - quantiles[0, :]

        # Artificial cluster centre = Tukey's upper fence
        artificial = quantiles[1, :] + 1.5 * qrange

        # Pre-allocate array for the artificial cluster's points
        num_points = int(len(points) * self.true_fraction / self.max_tracers)
        cluster = np.zeros((num_points, points.shape[1]))

        # Fill artificial cluster's points' coordinates with normally
        # distributed points of a variance comparable with that of the points
        cluster[:, 1:4] = np.random.normal(
            artificial,
            0.3 * qrange,
            size = (num_points, 3),
        )

        return np.vstack((points, cluster))


    def fit_sample(self, sample_points):
        # Type-checking inputs
        if not isinstance(sample_points, pept.PointData):
            sample_points = pept.PointData(sample_points)

        points = sample_points.points

        # If there are no points, return empty PointData with the same attrs
        if len(points) <= 1:
            return sample_points.copy(
                data = np.empty((0, points.shape[1] + 1)),
                columns = sample_points.columns + ["label"],
            )

        # Inject artificial cluster outside FoV to enforce characteristic
        # length. Will remove labels given to those points after fitting
        points_art = self._inject_cluster(points)

        # Set HDBSCAN's parameters
        phi = int(self.true_fraction * len(points_art) /
                  (self.max_tracers + 1))

        if phi >= len(points):
            phi = len(points)
        if phi < 2:
            phi = 2

        self.clusterer.min_cluster_size = phi
        self.clusterer.min_samples = phi

        # Cluster the points given and get each point's label
        with warnings.catch_warnings():
            # Ignore deprecation warning from HDBSCAN's use of `np.bool`
            warnings.simplefilter("ignore", category = DeprecationWarning)
            labels = self.clusterer.fit_predict(points_art[:, 1:4])

        # Assign noise to all points included in the artificial cluster
        labels_art = labels[len(points):]
        for label in np.unique(labels_art):
            labels[labels == label] = -1

        # Remove labels given to the artificial cluster's points
        labels = labels[:len(points)]

        # Map labels from [0, 2, 2, 3, 0] to [0, 1, 1, 2, 0]
        good = (labels != -1)
        _, labels[good] = np.unique(labels[good], return_inverse = True)

        # Construct new PointData instance with the same attributes
        return sample_points.copy(
            data = np.c_[points, labels],
            columns = sample_points.columns + ["label"],
        )
