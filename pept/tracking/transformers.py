#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : transformers.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 08.08.2021


import  re
import  warnings
from    numbers         import  Number
import  textwrap

import  numpy           as      np

from    pept.base       import  PointData, LineData
from    pept.base       import  Filter, Reducer, IterableSamples




class Stack(Reducer):
    '''Stack iterables - for example a ``list[pept.LineData]`` into a single
    ``pept.LineData``, a ``list[list]`` into a flattened ``list``.

    Reducer signature:

    ::

             list[LineData] -> Stack.fit -> LineData
            list[PointData] -> Stack.fit -> PointData

            list[list[Any]] -> Stack.fit -> list[Any]
        list[numpy.ndarray] -> Stack.fit -> numpy.ndarray

                      other -> Stack.fit -> other

    Can optionally set a given `sample_size` and `overlap`. This is useful
    when collecting a list of processed samples back into a single object.
    '''

    def __init__(self, sample_size = None, overlap = None):
        self.sample_size = sample_size
        self.overlap = overlap


    def fit(self, samples):
        # If it's a LineData / PointData, the `samples` are already stacked.
        # Simply set the sample_size and overlap if required and return them
        if not hasattr(samples, "__iter__"):
            raise ValueError(textwrap.fill((
                "The input `samples` must be an iterable (e.g. list, tuple, "
                f"PointData, LineData). Received type=`{type(samples)}`."
            )))

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

        # Vertically stack list of NumPy arrays
        elif isinstance(samples[0], np.ndarray):
            samples = np.vstack(samples)

        # Set new sample_size and overlap if required
        if self.sample_size is not None:
            samples.sample_size = self.sample_size
        if self.sample_size is not None:
            samples.overlap = self.overlap

        return samples




class SplitLabels(Filter):
    '''Split a sample of data into unique ``label`` values, optionally removing
    noise and extracting `_lines` attributes.

    Filter signature:

    ::

        # `extract_lines` = False (default)
         LineData -> SplitLabels.fit_sample -> list[LineData]
        PointData -> SplitLabels.fit_sample -> list[PointData]

        # `extract_lines` = True and PointData.attrs["_lines"] exists
        PointData -> SplitLabels.fit_sample -> list[LineData]


    The sample of data must have a column named exactly "label". If
    ``remove_label = True`` (default), the "label" column is removed.
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
            lines = sample.attrs["_lines"].lines
            cluster_lines = lines[line_indices.astype(int)]

        if self.extract_lines:
            return sample.attrs["_lines"].copy(data = cluster_lines)

        cluster = sample.copy(data = cluster_data)
        if lines_cols is not None:
            cluster.attrs["_lines"] = sample.attrs["_lines"].copy(
                data = cluster_lines,
            )

        return cluster


    def _empty_cluster(self, sample, lines_cols = None):
        if self.extract_lines:
            # Return empty LineData
            return sample.attrs["_lines"].copy(
                data = sample.attrs["_lines"][0:0],
            )

        cluster = sample.copy(data = sample[0:0])
        if lines_cols is not None:
            cluster.attrs["_lines"] = sample.attrs["_lines"].copy(
                data = sample.attrs["_lines"][0:0],
            )

        return cluster


    def fit_sample(self, sample: IterableSamples):
        if not isinstance(sample, IterableSamples):
            raise TypeError(textwrap.fill((
                "The input `sample` must be a subclass of `IterableSamples` "
                f"(e.g. PointData, LineData). Received type=`{type(sample)}`."
            )))

        # Extract the labels column
        col_idx = sample.columns.index("label")
        labels = sample.data[:, col_idx]

        # Check if there is a `._lines` attribute with `line_index` columns
        lines_cols = None
        if "_lines" in sample.attrs:
            lines_cols = [
                i for i, c in enumerate(sample.columns)
                if c.startswith("line_index")
            ]

            if len(lines_cols) == 0:
                warnings.warn((
                    "A `_lines` attribute was found, but no lines can "
                    "be extracted without columns `line_index<N>`."
                ), RuntimeWarning)

                lines_cols = None
                self.extract_lines = False

        elif self.extract_lines:
            raise ValueError(textwrap.fill((
                "If `extract_lines` is True, then the input `sample` must "
                "contain a `_lines` attribute."
            )))

        # If noise is requested, also include the noise cluster
        if self.noise:
            labels_unique = np.unique(labels)
        else:
            labels_unique = np.unique(labels[labels != -1])

        # For each unique label, create a new PointData / LineData cluster that
        # maintains / propagates all attributes (which needs a copy)
        clusters = [
            self._get_cluster(sample, labels == label, lines_cols)
            for label in labels_unique
        ]

        # If no valid cluster was found, return at least a single empty cluster
        if not len(clusters):
            clusters.append(self._empty_cluster(sample, lines_cols))

        # Remove the "label" column if needed
        if self.remove_labels and not self.extract_lines:
            for i in range(len(clusters)):
                clusters[i] = clusters[i].copy(
                    data = np.delete(clusters[i].data, col_idx, axis = 1),
                    columns = (clusters[i].columns[:col_idx] +
                               clusters[i].columns[col_idx + 1:]),
                )

        return clusters




def _wstd(x, w):
    '''Weighted standard deviation'''
    avg = np.average(x, weights = w)
    return np.sqrt(np.average((x - avg)**2, weights = w))




class Centroids(Filter):
    '''Compute the geometric centroids of a list of samples of points.

    Filter signature:

    ::

              PointData -> Centroids.fit_sample -> PointData
        list[PointData] -> Centroids.fit_sample -> PointData
          numpy.ndarray -> Centroids.fit_sample -> PointData

    This filter can be used right after ``pept.tracking.SplitLabels``, e.g.:

    >>> (SplitLabels() + Centroids()).fit(points)

    If `error = True`, append a measure of error on the computed centroid as
    the standard deviation in distances from centroid to all points. It is
    saved in an extra column "error".

    If `cluster_size = True`, append the number of points used for each
    centroid in an extra column "cluster_size" - unless `weight = True`, in
    which case it is the sum of weights.

    If `weight = True` and there is a column "weight" in the PointData, compute
    weighted centroids and standard deviations (if `error = True`) and the sum
    of weights (if `cluster_size = True`). The "weight" column is removed in
    the output centroid.

    '''


    def __init__(self, error = False, cluster_size = False, weight = True):
        self.error = bool(error)
        self.cluster_size = bool(cluster_size)
        self.weight = bool(weight)


    def _empty_centroid(self, points, weighted: bool):
        # Return an empty centroid with the correct number of columns
        ncols = points.shape[1]
        if self.error:
            ncols += 1
        if self.cluster_size:
            ncols += 1
        if weighted:
            ncols -= 1
        return np.empty((0, ncols))


    def _centroid(self, point_data, weighted: bool):
        # Extract the NumPy array of points
        points = point_data.points

        if len(points) == 0:
            return self._empty_centroid(points, weighted)

        if weighted:
            weightcol = point_data.columns.index("weight")
            weights = point_data.points[:, weightcol]

            # If all weights are zero, no cluster had been found
            if weights.sum() == 0.:
                return self._empty_centroid(points, weighted)

            c = np.average(points, weights = weights, axis = 0)
            c = np.delete(c, weightcol)
        else:
            c = points.mean(axis = 0)

        # If error is requested, compute std-dev of distances from centroid
        if self.error:
            d = np.linalg.norm(points[:, 1:4] - c[1:4], axis = 1)
            c = np.r_[c, _wstd(d, weights) if weighted else d.std()]

        # If cluster_size is requested, also append the number of points
        if self.cluster_size:
            c = np.r_[c, weights.sum() if weighted else len(points)]

        return c


    def fit_sample(self, points):
        # Type-checking inputs
        if isinstance(points, PointData):
            list_points = [points]
        elif isinstance(points, np.ndarray):
            list_points = [PointData(points)]
        else:
            list_points = [
                p if isinstance(p, PointData) else PointData(p)
                for p in list(points)
            ]

        if not len(list_points):
            raise ValueError("Must receive at least one PointData.")

        # If self.weight and there is a `weight` column, compute weighted
        # centroids
        weigh = (self.weight and "weight" in list_points[0].columns)

        # Compute centroid for each PointData and stack centroid arrays
        centroids = np.vstack([self._centroid(p, weigh) for p in list_points])
        attributes = list_points[0].extra_attrs()

        # If error or cluster_size are requested, append those columns
        columns = list_points[0].columns

        # Omit the `weight` column if weigh
        if weigh:
            weightcol = list_points[0].columns.index("weight")
            columns = columns[:weightcol] + columns[weightcol + 1:]

        if self.error:
            columns.append("error")

        if self.cluster_size:
            columns.append("cluster_size")

        return PointData(centroids, columns = columns, **attributes)




class LinesCentroids(Filter):
    '''Compute the minimum distance point of some ``pept.LineData`` while
    iteratively removing a fraction of the furthest lines.

    Filter signature:

    ::

        list[LineData] -> LinesCentroids.fit_sample -> PointData
              LineData -> LinesCentroids.fit_sample -> PointData
         numpy.ndarray -> LinesCentroids.fit_sample -> PointData

    The code below is adapted from the PEPT-EM algorithm developed by Antoine
    Renaud and Sam Manger.
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
        lors = lines[:, :7].copy(order = "C")

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

        centroids = [self.predict(lines.lines) for lines in list_lines]
        return PointData(np.vstack(centroids), **list_lines[0].extra_attrs())




class Condition(Filter):
    '''Select only data satisfying multiple conditions, given as a string, a
    function or list thereof; e.g. ``Condition("error < 15")`` selects all
    points whose "error" column value is smaller than 15.

    Filter signature:

    ::

        PointData -> Condition.fit_sample -> PointData
         LineData -> Condition.fit_sample -> LineData

    In the simplest case, a column name is specified, plus a comparison, e.g.
    ``Condition("error < 15, y > 100")``; multiple conditions may be
    concatenated using a comma.

    More complex conditions - where the column name is not the first operand -
    can be constructed using single quotes, e.g. using NumPy functions in
    ``Condition("np.isfinite('x')")`` to filter out NaNs and Infs. Quotes can
    be used to index columns too: ``Condition("'0' < 150")`` selects all rows
    whose first column is smaller than 150.

    Generally, you can use any function returning a boolean mask, either as a
    string of code ``Condition("np.isclose('x', 3)")`` or a user-defined
    function receiving a NumPy array ``Condition(lambda x: x[:, 0] < 10)``.

    Finally, multiple such conditions may be supplied separately:
    ``Condition(lambda x: x[:, -1] > 10, "'t' < 50")``.
    '''

    def __init__(self, *conditions):
        # Calls the conditions setter which does parsing
        self.conditions = conditions


    @property
    def conditions(self):
        return self._conditions


    @conditions.setter
    def conditions(self, conditions):
        if isinstance(conditions, str):
            self._conditions = Condition._parse_condition(conditions)
        elif callable(conditions):
            self._conditions = [conditions]
        else:
            cs = []
            for cond in conditions:
                cs.extend(Condition._parse_condition(cond))
            self._conditions = cs


    @staticmethod
    def _parse_condition(cond):
        if callable(cond):
            return [cond]

        conditions = str(cond).replace(" ", "").split(",")

        # Compile regex object to find quoted strings
        finder = re.compile(r"'\w+'")

        for i in range(len(conditions)):
            # Replace single-quoted column numbers / names
            if "'" in conditions[i]:
                conditions[i] = finder.sub(
                    Condition._replace_quoted,
                    conditions[i],
                )
                continue

            # If condition is a simple comparison, allow using non-quoted
            # column names
            op = None
            if "<" in conditions[i]:
                op = "<"
            elif ">" in conditions[i]:
                op = ">"
            elif "!" in conditions[i]:
                op = "!"
            elif "==" in conditions[i]:
                op = "=="

            if op is not None:
                cs = conditions[i].split(op)
                cs[0] = Condition._replace_term(cs[0])
                conditions[i] = op.join(cs)

            else:
                raise ValueError(textwrap.fill((
                    f"The input `conditions[i] = {conditions[i]}` did not "
                    "contain an operator or single-quoted terms."
                )))

        return conditions


    @staticmethod
    def _replace_term(term: str):
        return f"data[:, sample.columns.index('{term}')]"


    @staticmethod
    def _replace_quoted(term):
        # Remove single quotes
        if isinstance(term, re.Match):
            term = term.group()
        term = term.split("'")[1]

        try:
            index = int(term)
            return f"data[:, {index}]"
        except ValueError:
            return f"data[:, sample.columns.index('{term}')]"


    def fit_sample(self, sample: IterableSamples):
        if not isinstance(sample, IterableSamples):
            raise TypeError(textwrap.fill((
                "The input `sample` must be a subclass of `IterableSamples` "
                f"(e.g. PointData, LineData). Received type=`{type(sample)}`."
            )))

        data = sample.data

        for cond in self.conditions:
            if callable(cond):
                data = data[cond(data)]
            else:
                data = data[eval(cond, globals(), locals())]

        return sample.copy(data = data)




class Remove(Filter):
    '''Remove columns (either column names or indices) from `pept.LineData` or
    `pept.PointData`.

    Filter signature:

    ::

         pept.LineData -> Remove.fit_sample -> pept.LineData
        pept.PointData -> Remove.fit_sample -> pept.PointData

    Examples
    --------
    To remove a single column named "line_index":

    >>> import pept
    >>> from pept.tracking import *
    >>> points = pept.PointData(...)    # Some dummy data

    >>> rem = Remove("line_index")
    >>> points_without = rem.fit_sample(points)

    Remove all columns starting with "line_index" using a glob operator (*):

    >>> points_without = Remove("line_index*").fit_sample(points)

    Remove the first column based on its index:

    >>> points_without = Remove(0).fit_sample(points)

    Finally, multiple removals may be chained into a list:

    >>> points_without = Remove(["line_index*", -1]).fit_sample(points)

    '''

    def __init__(self, *columns):
        self._indices = []
        self._filters = []

        # Calls the `columns` setter which does parsing
        self.columns = columns


    @property
    def columns(self):
        return self._columns


    @columns.setter
    def columns(self, columns):
        self._columns = [Remove._parse(col) for col in columns]

        # Split the removers into regex strings and column indices
        for c in self._columns:
            if isinstance(c, str):
                self._filters.append(c)
            else:
                self._indices.append(c)


    @staticmethod
    def _parse(col):
        if isinstance(col, str):
            return col.replace("*", r"\w*")
        elif isinstance(col, Number):
            return int(col)
        else:
            raise ValueError(textwrap.fill((
                "Each input argument in `columns` must be a string or an "
                f"integer. One of them was `type(col) = {type(col)}`."
            )))


    def fit_sample(self, sample: IterableSamples):
        if not isinstance(sample, IterableSamples):
            raise TypeError(textwrap.fill((
                "The input `sample` must be a subclass of `IterableSamples` "
                f"(e.g. PointData, LineData). Received type=`{type(sample)}`."
            )))

        # Extract the relevant `sample` attributes
        columns = sample.columns
        ncols = len(columns)

        # The regex filters to use and column numbers to remove
        filters = self._filters
        indices = self._indices

        # Column indices to remove and remaining column names
        removed = set()
        columns_filtered = []

        for i, c in enumerate(columns):
            # Also handle negative indices
            if any((re.fullmatch(r, c) for r in filters)) or \
                    any((i == ind or i == ind + ncols for ind in indices)):
                removed.add(i)
            else:
                columns_filtered.append(c)

        indices_filtered = [i for i in range(len(columns)) if i not in removed]
        data = sample.data[:, indices_filtered]

        return sample.copy(data = data, columns = columns_filtered)




class SplitAll(Reducer):
    '''Stack all samples and split them into a list according to a named /
    numeric column index.

    Reducer signature:

    ::

                   LineData -> SplitAll.fit -> list[LineData]
             list[LineData] -> SplitAll.fit -> list[LineData]

                  PointData -> SplitAll.fit -> list[PointData]
            list[PointData] -> SplitAll.fit -> list[PointData]

              numpy.ndarray -> SplitAll.fit -> list[numpy.ndarray]
        list[numpy.ndarray] -> SplitAll.fit -> list[numpy.ndarray]

    If using a LineData / PointData, you can use a columns name as a string,
    e.g. ``SplitAll("label")`` or a number ``SplitAll(4)``. If using a NumPy
    array, only numeric indices are accepted.
    '''

    def __init__(self, column):
        try:
            self.column_index = int(column)
            self.column_name = None
        except ValueError:
            self.column_name = str(column)
            self.column_index = None


    def fit(self, samples):
        if not hasattr(samples, "__iter__"):
            raise TypeError(textwrap.fill((
                "The input `samples` must be an iterable (e.g. list, tuple, "
                f"PointData, LineData). Received type=`{type(samples)}`."
            )))

        # Reduce / stack list of samples onto a single IterableSamples / array
        samples = Stack().fit(samples)

        if isinstance(samples, np.ndarray):
            return self._split_numpy(samples)
        elif isinstance(samples, IterableSamples):
            return self._split_iterable_samples(samples)
        else:
            raise TypeError(textwrap.fill((
                "The input samples must be NumPy arrays, PointData / LineData "
                f"or lists thereof. Received `type(samples) = {type(samples)}`"
            )))


    def _split_numpy(self, samples):
        if self.column_index is None:
            raise TypeError(textwrap.fill((
                "If the samples are NumPy arrays, you must use a numeric "
                f"column index; used a named column: `{self.column_name}`."
            )))

        col_data = samples[:, self.column_index]
        labels = np.unique(col_data)

        # If no labels exist, return a list with an empty sample
        if not len(labels):
            return [samples[0:0]]

        return [samples[col_data == label] for label in labels]


    def _split_iterable_samples(self, samples):
        if self.column_index is not None:
            col_data = samples.data[:, self.column_index]
        else:
            col_data = samples.data[:, samples.columns.index(self.column_name)]

        labels = np.unique(col_data)

        # If no labels exist, return a list with an empty sample
        if not len(labels):
            return [samples[0:0]]

        return [
            samples.copy(data = samples.data[col_data == label])
            for label in labels
        ]




# Use standard names, Leonard...
GroupBy = SplitAll




class Swap(Filter):
    '''Swap two columns in a LineData or PointData.

    Filter signature:

    ::

         LineData -> Swap.fit_sample -> LineData
        PointData -> Swap.fit_sample -> PointData

    For example, swap the Y and Z axes: ``Swap("y, z").fit_sample(points)``.
    Add multiple swaps as separate arguments: ``Swap("y, z", "label, x")``.

    You can also swap columns at numerical indices by single-quoting them:
    ``Swap("'0', '1'")``.

    *New in pept-0.4.3*
    '''

    def __init__(self, *swaps, inplace = True):
        # Calls swaps.setter which does type-checking
        self.swaps = swaps
        self.inplace = bool(inplace)


    @property
    def swaps(self):
        return self._swaps


    @swaps.setter
    def swaps(self, swaps):
        commands = []

        # Compile regex object to find quoted strings
        finder = re.compile(r"'\w+'")

        for i in range(len(swaps)):
            s = swaps[i].replace(" ", "").split(",")
            if len(s) != 2:
                raise ValueError(textwrap.fill((
                    f"The input `swaps[{i}] = {swaps[i]}` does not have a "
                    'single comma. It must be formatted as "col1, col2".'
                )))

            # Replace single-quoted column numbers / names
            for j in range(2):
                if "'" in s[j]:
                    s[j] = finder.sub(Swap._replace_quoted, s[j])
                else:
                    s[j] = Swap._replace_term(s[j])

            commands.append(
                f"aux = {s[0]}.copy() ; {s[0]} = {s[1]} ; {s[1]} = aux;\n"
            )

        self._swaps = tuple(commands)


    @staticmethod
    def _replace_term(term: str):
        return f"data[:, sample.columns.index('{term}')]"


    @staticmethod
    def _replace_quoted(term):
        # Remove single quotes
        if isinstance(term, re.Match):
            term = term.group()
        term = term.split("'")[1]

        try:
            index = int(term)
            return f"data[:, {index}]"
        except ValueError:
            return f"data[:, sample.columns.index('{term}')]"


    def fit_sample(self, sample: IterableSamples):
        if not isinstance(sample, IterableSamples):
            raise TypeError(textwrap.fill((
                "The input `sample` must be a subclass of `IterableSamples` "
                f"(e.g. PointData, LineData). Received type=`{type(sample)}`."
            )))

        if not self.inplace:
            sample = sample.copy()

        data = sample.data
        for s in self.swaps:
            exec(s, locals(), globals())

        return sample
