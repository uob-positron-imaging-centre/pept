#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : transformers.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 08.08.2021


import  re
import  sys
import  warnings
from    typing          import  Union

if sys.version_info.minor >= 9:
    # Python 3.9
    from collections.abc import  Iterable
else:
    from typing         import  Iterable

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

    def __init__(
        self,
        remove_labels = True,
        extract_lines = False,
        noise = False,
    ):
        self.remove_labels = bool(remove_labels)
        self.extract_lines = bool(extract_lines)
        self.noise = bool(noise)


    def _get_cluster(self, sample, labels_mask, lines_cols = None):
        # Extract the labels column
        cluster_data = sample.data[labels_mask]

        if lines_cols is not None:
            line_indices = np.unique(cluster_data[:, lines_cols])
            cluster_lines = sample._lines.lines[line_indices.astype(int)]

        if self.extract_lines:
            return sample._lines.copy(data = cluster_lines)

        cluster = sample.copy(data = cluster_data)
        if lines_cols is not None:
            cluster._lines = sample._lines.copy(data = cluster_lines)

        return cluster


    def _empty_cluster(self, sample, lines_cols = None):
        if self.extract_lines:
            # Return empty LineData
            return sample._lines.copy(
                data = np.empty((0, sample._lines.data.shape[1]))
            )

        cluster = sample.copy(data = np.empty((0, sample.data.shape[1])))
        if lines_cols is not None:
            cluster._lines = sample._lines.copy(
                data = np.empty((0, sample._lines.data.shape[1]))
            )

        return cluster


    @beartype
    def fit_sample(self, sample: IterableSamples):
        # Extract the labels column
        col_idx = sample.columns.index("labels")
        labels = sample.data[:, col_idx]

        # Check if there is a `._lines` attribute with `line_index` columns
        lines_cols = None
        if hasattr(sample, "_lines"):
            lines_cols = [
                i for i, c in enumerate(sample.columns)
                if c.startswith("line_index")
            ]

            if len(lines_cols) == 0:
                warnings.warn((
                    "A `sample._lines` attribute was found, but no lines can "
                    "be extracted without columns `line_index<N>`."
                ), RuntimeWarning)
                lines_cols = None

        elif self.extract_lines:
            raise ValueError(textwrap.fill((
                "If `extract_lines` is True, then the input `sample` must "
                "contain a `._lines` attribute."
            )))

        # If noise is requested, also include the noise cluster
        if self.noise:
            labels_unique = np.unique(labels)
        else:
            labels_unique = np.unique(labels[labels != -1])

        # For each unique label, create a new PointData / LineData cluster that
        # maintains / propagates all attributes (which needs a copy)
        labels_unique = np.unique(labels[labels != -1])
        clusters = [
            self._get_cluster(sample, labels == label, lines_cols)
            for label in labels_unique
        ]

        # If no valid cluster was found, return at least a single empty cluster
        if not len(clusters):
            clusters.append(self._empty_cluster(sample, lines_cols))

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


    def __init__(self, error = False, cluster_size = False):
        self.error = bool(error)
        self.cluster_size = bool(cluster_size)


    def _empty_centroid(self, points):
        # Return an empty centroid with the correct number of columns
        ncols = points.points.shape[1]
        if self.error:
            ncols += 1
        if self.cluster_size:
            ncols += 1
        return np.empty((0, ncols))


    def centroid(self, points):
        if len(points.points) == 0:
            return self._empty_centroid(points)

        c = points.points.mean(axis = 0)

        # If error is requested, compute std-dev of distances from centroid
        if self.error:
            err = np.linalg.norm(points.points - c, axis = 1).std()
            c = np.r_[c, err]

        # If cluster_size is requested, also append the number of points
        if self.cluster_size:
            c = np.r_[c, len(points.points)]

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

        # If error or cluster_size are requested, append those columns
        if self.error:
            attributes["columns"] = attributes["columns"] + ["error"]

        if self.cluster_size:
            attributes["columns"] = attributes["columns"] + ["cluster_size"]

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




class Condition(Filter):
    '''Select only data satisfying multiple conditions, given as a string; e.g.
    ``Condition("error < 15")`` selects all points whose "error" column value
    is smaller than 15.

    Filter signature:

    ::

        PointData -> Condition.fit_sample -> PointData
        LineData -> Condition.fit_sample -> LineData


    Multiple conditions may be concatenated using a comma, e.g.
    ``Condition("error < 15, y > 100")`` also selects only points whose "y"
    coordinate is larger than 100.

    Alternatively, the column index may be specified as a number, e.g.
    ``Condition("0 < 150")`` selects all points whose first column is smaller
    than 150.

    '''

    def __init__(self, cond: str):
        # Calls the conditions setter which does parsing
        self.conditions = cond


    @property
    def conditions(self):
        return self._conditions


    @conditions.setter
    def conditions(self, cond):
        conditions = cond.replace(" ", "").split(",")

        for i in range(len(conditions)):
            op = None
            if "<" in conditions[i]:
                op = "<"
            elif ">" in conditions[i]:
                op = ">"

            if op is not None:
                cs = conditions[i].split(op)
                cs[0] = Condition._replace_term(cs[0])
                conditions[i] = op.join(cs)
            else:
                raise ValueError(textwrap.fill((
                    f"The input `conditions[i] = {conditions[i]}` did not "
                    "contain an operator."
                )))

        self._conditions = conditions


    @staticmethod
    def _replace_term(term: str):
        try:
            index = int(term)
            return f"data[:, {index}]"
        except ValueError:
            return f"data[:, sample.columns.index('{term}')]"


    @beartype
    def fit_sample(self, sample: IterableSamples):
        data = sample.data

        for cond in self.conditions:
            data = data[eval(cond, locals())]

        return sample.copy(data = data)




class Expression:

    def __init__(self, cond: str):
        # Calls the conditions setter which does parsing
        self.conditions = cond


    @property
    def conditions(self):
        return self._conditions


    @conditions.setter
    def conditions(self, cond):
        # Remove whitespace and split into individual expressions
        conditions = cond.replace(" ", "").split(",")

        # Compile regex object to find quoted strings
        finder = re.compile(r"'\w+'")

        for i in range(len(conditions)):
            conditions[i] = finder.sub(Condition._replace_term, conditions[i])

        if not len(conditions):
            raise ValueError(textwrap.fill((
                f"The input `conditions[i] = {conditions[i]}` did not contain "
                "quoted terms."
            )))

        self._conditions = conditions


    @staticmethod
    def _replace_term(term):
        # Remove single quotes
        if isinstance(term, re.Match):
            term = term.group()
        term = term.split("'")[1]

        try:
            index = int(term)
            return f"data[:, {index}]"
        except ValueError:
            return f"data[:, sample.columns.index('{term}')]"


    @beartype
    def fit_sample(self, sample: IterableSamples):
        data = sample.data

        for cond in self.conditions:
            if "=" in cond and "<=" not in cond and ">=" not in cond:
                # Assignment
                exec(cond, locals())
            else:
                # Filter
                data = data[eval(cond, locals())]

        return sample.copy(data = data)




class Swap(Filter):

    def __init__(self, expressions: str):
        self._expressions = expressions


    @property
    def expressions(self):
        return self._expressions


    @expressions.setter
    def expressions(self, expressions: str):
        # Remove whitespace and split into individual expressions
        expressions = expressions.replace(" ", "").split(",")

        for i in range(len(expressions)):
            terms = expressions[i].split("=")
            term1 = terms[0]
            term2 = terms[1]

            if "'" in term1:
                pass

            aux1 = f"aux1={Swap._replace_term(term1)}.copy();"
            aux2 = f"aux2={Swap._replace_term(term2)}.copy();"

            expressions[i] = aux1 + aux2


    @staticmethod
    def _replace_term(term):
        # Remove single quotes
        if isinstance(term, re.Match):
            term = term.group()
        term = term.split("'")[1]

        try:
            index = int(term)
            return f"data[:, {index}]"
        except ValueError:
            return f"data[:, sample.columns.index('{term}')]"
