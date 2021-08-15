#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : transformers.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 08.08.2021


from    typing          import  Union
from    collections.abc import  Iterable
import  textwrap

import  numpy           as      np

from    beartype        import  beartype

from    pept.base       import  PointData, LineData
from    pept.base       import  Filter, Reducer, IterableSamples


PointsOrLines = Union[PointData, LineData]




class Stack(Reducer):
    '''Stack iterables - e.g. a ``list[pept.LineData]`` into a single
    ``pept.LineData``, a ``list[list[X]]`` into a flattened ``list[X]``.

    Can optionally set a given `sample_size` and `overlap`. This is useful
    when collecting a list of processed samples back into a single object.
    '''

    def __init__(self, sample_size = None, overlap = None):
        self.sample_size = sample_size
        self.overlap = overlap


    @beartype
    def fit(self, samples: Iterable):
        # If it's a LineData / PointData, the `samples` are already stacked.
        # Simply set the sample_size and overlap if required and return them
        if isinstance(samples, IterableSamples):
            if self.sample_size is not None:
                samples.sample_size = self.sample_size
            if self.sample_size is not None:
                samples.overlap = self.overlap
            return samples

        # If it's an empty iterator, we don't have anything to stack
        if len(samples) == 0:
            return samples

        # Stack Lines into LineData
        elif isinstance(samples[0], LineData):
            samples = LineData(samples)

        # Stack Points into PointData
        elif isinstance(samples[0], PointData):
            samples = PointData(samples)

        # Flatten list of lists
        elif isinstance(samples[0], list):
            samples = [item for sublist in samples for item in sublist]

        # Set new sample_size and overlap if required
        if self.sample_size is not None:
            samples.sample_size = self.sample_size
        if self.sample_size is not None:
            samples.overlap = self.overlap

        return samples




class SplitLabels(Filter):
    '''Split a sample of data into unique `labels` while removing noise.

    Filter signature:

    ::

        # `extract_lines` = False (default)
        LineData -> SplitLabels.fit_sample -> list[LineData]
        PointData -> SplitLabels.fit_sample -> list[PointData]

        # `extract_lines` = True and PointData.lines exists
        PointData -> SplitLabels.fit_sample -> list[LineData]


    The sample of data must have a column named exactly "labels". The filter
    normally removes the "labels" column in the output (if
    ``remove_labels = True``).
    '''

    def __init__(self, remove_labels = True, extract_lines = False):
        self.remove_labels = bool(remove_labels)
        self.extract_lines = bool(extract_lines)


    @beartype
    def fit_sample(self, sample: IterableSamples) -> list[IterableSamples]:
        # Get data attribute name depending on whether it is Points or Lines
        data = sample.data

        # Extract the labels column
        col_idx = sample.columns.index("labels")
        labels = data[:, col_idx]

        # For each unique non-noise label, create a new Points / Lines that
        # maintains / propagates all attributes (which needs a copy)
        labels_unique = np.unique(labels[labels != -1])
        clusters = []

        for label in labels_unique:
            cluster_data = data[labels == label]

            if self.extract_lines:
                indices_cols = [
                    i for i, c in enumerate(sample.columns)
                    if c.startswith("line_index")
                ]

                if len(indices_cols) == 0:
                    raise ValueError(textwrap.fill((
                        "If `extract_lines`, the input samples must have "
                        "columns whose names start with `line_index`."
                    )))

                line_indices = np.unique(cluster_data[:, indices_cols])
                cluster = sample._lines.copy(
                    data = sample._lines.lines[line_indices.astype(int)]
                )

            else:
                cluster = sample.copy(data = cluster_data)

            clusters.append(cluster)

        # If no valid cluster was found, return at least a single empty cluster
        if not len(clusters):
            cluster = sample.copy(data = np.empty((0, sample.data.shape[1])))
            clusters.append(cluster)

        # Remove the "labels" column if needed
        if self.remove_labels and not self.extract_lines:
            for cluster in clusters:
                cluster.columns = cluster.columns.copy()
                cluster.columns.pop(col_idx)
                cluster.data = np.delete(cluster.data, col_idx, axis = 1)

        return clusters




class Centroids(Filter):
    '''Compute the geometric centroids of a list of samples of points.

    Filter signature:

    ::

        PointData -> Centroids.fit_sample -> PointData
        list[PointData] -> Centroids.fit_sample -> PointData
        numpy.ndarray -> Centroids.fit_sample -> PointData

    This filter can be used right after ``pept.tracking.SplitLabels``, e.g.:

    >>> (SplitLabels() + Centroids()).fit(points)

    '''


    def __init__(self, max_error = None):
        self.max_error = None if max_error is None else float(max_error)


    def _empty_centroid(self, points):
        # Return an empty centroid with the correct number of columns
        ncols = points.points.shape[1]
        if self.max_error is not None:
            ncols += 1
        return np.empty((0, ncols))


    def centroid(self, points):
        if len(points.points) == 0:
            return self._empty_centroid(points)

        c = points.points.mean(axis = 0)

        # If max_error is defined, compute std-dev of distances from centroid
        # to all points; if it is larger than max_error, return empty centroid
        if self.max_error is not None:
            err = np.linalg.norm(points.points - c, axis = 1).std()

            if err > self.max_error:
                return self._empty_centroid(points)

            return np.r_[c, err]

        return c


    def fit_sample(self, points):
        # Type-checking inputs
        if isinstance(points, PointData):
            list_points = [points]
        elif isinstance(points, np.ndarray):
            list_points = [PointData(points)]
        else:
            list_points = list(points)

        # Compute centroid for each PointData and stack centroid arrays
        centroids = np.vstack([self.centroid(p) for p in list_points])
        attributes = list_points[0].extra_attributes()

        # If max_error is defined, add "error" column
        if self.max_error is not None:
            attributes["columns"] = attributes["columns"] + ["error"]

        points = PointData(centroids, **attributes)
        return points




class LinesCentroids(Filter):
    '''Compute the minimum distance point of some ``pept.LineData`` while
    iteratively removing a fraction of the furthest lines.

    The code below is adapted from the PEPT-EM algorithm developed by Antoine
    Renaud and Sam Manger
    '''

    def __init__(self, remove = 0.1, iterations = 6):
        self.remove = float(remove)
        self.iterations = int(iterations)


    @staticmethod
    def centroid(lors):
        nx = np.newaxis

        m = np.identity(3)[nx, :, :] - lors[:, nx, 4:7] * lors[:, 4:7, nx]
        n = np.sum(m, axis = 0)
        v = np.sum(np.sum(m * lors[:, nx, 1:4], axis=-1), axis=0)

        return np.matmul(np.linalg.inv(n), v)


    @staticmethod
    def distance_matrix(x, lors):
        y = x[np.newaxis, :3] - lors[:, 1:4]
        return np.sum(y**2, axis=-1) - np.sum(y * lors[:, 4:7], axis=-1)**2


    def predict(self, lines):
        # Rewrite LoRs in the vectorial form y(x) = position + x * direction
        lors = lines.lines[:, :7].copy(order = "C")

        lors[:, 4:7] = lors[:, 4:7] - lors[:, 1:4]
        lors[:, 4:7] /= np.linalg.norm(lors[:, 3:], axis = -1)[:, np.newaxis]

        # Begin with equal weights for all LoRs
        weights = np.ones(len(lors))
        x = LinesCentroids.centroid(lors)

        # Iteratively remove the furthest LoRs and recompute centroid
        for i in range(self.iterations):
            d2 = LinesCentroids.distance_matrix(x, lors)

            k = int(len(d2) * (1 - self.remove * (i + 1)))
            part = np.argpartition(d2, k)
            weights[part[k:]] = 0

            x = LinesCentroids.centroid(lors)

        # Add timestamp as the mean LoRs' time
        return np.hstack((lors[:, 0].mean(), x))


    def fit_sample(self, lines):
        # Type-checking inputs
        if isinstance(lines, LineData):
            list_lines = [lines]
        elif isinstance(lines, np.ndarray):
            list_lines = [LineData(lines)]
        else:
            list_lines = list(lines)

        centroids = [self.predict(lines) for lines in list_lines]
        attributes = list_lines[0].extra_attributes()
        del attributes["columns"]

        return PointData(np.vstack(centroids), **attributes)
