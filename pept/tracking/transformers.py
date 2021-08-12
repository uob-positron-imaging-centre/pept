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

        LineData -> SplitLabels.fit_sample -> list[LineData]
        PointData -> SplitLabels.fit_sample -> list[PointData]


    The sample of data must have a column named exactly "labels". The filter
    normally removes the "labels" column in the output (if
    ``remove_labels = True``).
    '''

    def __init__(self, remove_labels = True):
        self.remove_labels = bool(remove_labels)


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
            cluster = sample.copy()
            cluster.data = data[labels == label]

            # Remove the "labels" column if needed
            if self.remove_labels:
                del cluster.columns[col_idx]
                cluster.data = np.delete(cluster.data, col_idx, axis = 1)

            clusters.append(cluster)

        # If no valid cluster was found, return at least a single empty cluster
        if not len(clusters):
            cluster = sample.copy()
            ncols = cluster.data.shape[1]
            cluster.data = np.empty((0, ncols))

            # Remove the "labels" column if needed
            if self.remove_labels:
                del cluster.columns[col_idx]
                cluster.data = np.delete(cluster.data, col_idx, axis = 1)

            clusters.append(cluster)

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


    def __init__(self):
        self.columns_needed = None


    def centroid(self, points):
        if len(points.points) == 0:
            return np.empty((0, len(points.columns)))

        return points.points.mean(axis = 0)


    def fit_sample(self, points):
        # Type-checking inputs
        if isinstance(points, PointData):
            list_points = [points]
        elif isinstance(points, np.ndarray):
            list_points = [PointData(points)]
        else:
            list_points = list(points)

        centroids = np.vstack([self.centroid(p) for p in list_points])

        return PointData(centroids, **list_points[0].extra_attributes())




class ExtractLines(Filter):
    '''Extract the `._lines` attribute from a sample of ``PointData``, if
    defined.
    '''

    def _extract_cluster(self, points):
        if hasattr(points, "lines"):
            lines = points.lines
        elif hasattr(points, "_lines"):
            lines = points._lines
        else:
            raise ValueError(textwrap.fill((
                "The input `points` must have an attribute `.lines` or "
                "`._lines` to extract lines from. For example, `Cutpoints` "
                "sets this attribute if `append_indices = True`."
            )))

        indices_cols = [
            i for i, c in enumerate(points.columns)
            if c.startswith("line_index")
        ]

        if len(indices_cols) == 0:
            raise AttributeError(textwrap.fill((
                "The input `points` must have columns whose names start with "
                "`line_index` to know which lines to extract. This can be set "
                "in previous filters (e.g. `Cutpoints` if "
                "`append_indices = True`)."
            )))

        line_indices = np.unique(points.points[:, indices_cols]).astype(int)
        attributes = lines.extra_attributes()
        attributes.update(lines.hidden_attributes())
        return LineData(lines.lines[line_indices], **attributes)


    def fit_sample(self, points):
        # Type-checking inputs
        if isinstance(points, PointData):
            list_points = [points]
        elif isinstance(points, np.ndarray):
            list_points = [PointData(points)]
        else:
            list_points = list(points)

        list_lines = [self._extract_cluster(p) for p in list_points]
        return list_lines




class LinesCentroids(Filter):
    '''Compute the minimum distance point of some ``pept.LineData`` while
    iteratively removing a fraction of the furthest lines.
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
