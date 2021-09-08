#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pipelines.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 07.08.2021


import  os
import  sys
import  time
import  textwrap
from    abc                 import  ABC, abstractmethod

if sys.version_info.minor >= 9:
    # Python 3.9
    from collections.abc    import  Iterable
else:
    from typing             import  Iterable

import  numpy               as      np

from    beartype            import  beartype
from    tqdm                import  tqdm
from    joblib              import  Parallel, delayed

from    .iterable_samples   import  PEPTObject
from    .point_data         import  PointData
from    .line_data          import  LineData
from    .voxel_data         import  Voxels




class Transformer(ABC, PEPTObject):
    '''Base class for PEPT filters (transforming a sample into another) and
    reducers (transforming a list of samples).

    You should only need to subclass `Filter` and `Reducer` (or even, better,
    their more specialised subclasses, e.g. `LineDataFilter`).
    '''

    def __add__(self, other):
        # Allow adding (+) multiple filters / reducers into a pipeline
        return Pipeline([self, other])


    def __repr__(self):
        # Select all attributes from dir(obj) that are not callable (i.e. not
        # methods) and don't start with "_" (i.e. they're private)
        docs = []
        for attr in dir(self):
            memb = getattr(self, attr)
            if not callable(memb) and not attr.startswith("_"):
                # If it's a nested collection, print it on a new, indented line
                if (isinstance(memb, np.ndarray) and memb.ndim > 1) or \
                        isinstance(memb, PEPTObject):
                    memb_str = textwrap.indent(str(memb), '  ')
                    docs.append(f"{attr} = \n{memb_str}")
                else:
                    docs.append(f"{attr} = {memb}")

        return f"{type(self).__name__}(" + ", ".join(docs) + ")"




class Filter(Transformer):
    '''Abstract class from which PEPT filters inherit. You only need to define
    a method `def fit_sample(self, sample)`, which processes a *single* sample.

    If you define a filter on `LineData`, you should subclass `LineDataFilter`.
    Same goes for `PointData` with `PointDataFilter`.
    '''

    @abstractmethod
    def fit_sample(self, sample):
        pass


    @beartype
    def fit(
        self,
        samples: Iterable,
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        '''Apply self.fit_sample (implemented by subclasses) according to the
        execution policy. Simply return a list of processed samples. If you
        need a reduction step (e.g. stack all processed samples), apply it
        in the subclass.
        '''

        if executor == "sequential":
            if verbose:
                samples = tqdm(samples, desc = f"{type(self).__name__} :")

            return [self.fit_sample(s) for s in samples]

        elif executor == "joblib":
            if verbose:
                samples = tqdm(samples, desc = f"{type(self).__name__} :")

            if max_workers is None:
                max_workers = os.cpu_count()

            return Parallel(n_jobs = max_workers)(
                delayed(self.fit_sample)(s) for s in samples
            )

        else:
            with executor(max_workers = max_workers) as exe:
                futures = [exe.submit(self.fit_sample, s) for s in samples]

                if verbose:
                    futures = tqdm(futures, desc = f"{type(self).__name__} :")

                return [f.result() for f in futures]




class Reducer(Transformer):
    '''Abstract class from which PEPT reducers inherit. You only need to define
    a method `def fit(self, samples)`, which processes an *iterable* of samples
    (most commonly a `LineData` or `PointData`).
    '''

    @abstractmethod
    def fit(self, samples):
        pass




class LineDataFilter(Filter):
    '''An abstract class that defines a filter for samples of `pept.LineData`.

    An implementor must define the method `def fit_sample(self, sample)`.

    A default `fit` method is provided for convenience, calling `fit_sample`
    on each sample from an iterable according to a given execution policy
    (e.g. "sequential", "joblib", or `concurrent.futures.Executor` subclasses,
    such as `ProcessPoolExecutor` or `MPIPoolExecutor`).
    '''

    @beartype
    def fit(
        self,
        line_data: Iterable[LineData],
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        return Filter.fit(
            self, line_data, executor, max_workers, verbose = verbose
        )




class PointDataFilter(Filter):
    '''An abstract class that defines a filter for samples of `pept.PointData`.

    An implementor must define the method `def fit_sample(self, sample)`.

    A default `fit` method is provided for convenience, calling `fit_sample`
    on each sample from an iterable according to a given execution policy
    (e.g. "sequential", "joblib", or `concurrent.futures.Executor` subclasses,
    such as `ProcessPoolExecutor` or `MPIPoolExecutor`).
    '''

    @beartype
    def fit(
        self,
        point_data: Iterable[PointData],
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        return Filter.fit(
            self, point_data, executor, max_workers, verbose = verbose
        )




class VoxelsFilter(Filter):
    '''An abstract class that defines a filter for samples of `pept.Voxels`.

    An implementor must define the method `def fit_sample(self, sample)`.

    A default `fit` method is provided for convenience, calling `fit_sample`
    on each sample from an iterable according to a given execution policy
    (e.g. "sequential", "joblib", or `concurrent.futures.Executor` subclasses,
    such as `ProcessPoolExecutor` or `MPIPoolExecutor`).
    '''

    @beartype
    def fit(
        self,
        line_data: Iterable[Voxels],
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        return Filter.fit(
            self, line_data, executor, max_workers, verbose = verbose
        )




class Pipeline(PEPTObject):
    '''A PEPT processing pipeline, chaining multiple `Filter` and `Reducer`
    for efficient, parallel execution.

    After a pipeline is constructed, the `fit(samples)` method can be called,
    which will apply the chain of filters and reducers on the samples of data.

    A filter is simply a transformation applied to a sample (e.g. `Voxelliser`
    on a single sample of `LineData`). A reducer is a transformation applied to
    a list of *all* samples (e.g. `Stack` on all samples of `PointData`).

    Note that only filters can be applied in parallel, but the great advantage
    of a `Pipeline` is that it significantly reduces the amount of data copying
    and intermediate results' storage. Reducers will require collecting all
    results.

    There are three execution policies at the moment: "sequential" is
    single-threaded (slower, but easy to debug), "joblib" (very fast on medium
    datasets due to joblib's caching) and any `concurrent.futures.Executor`
    subclass (e.g. MPIPoolExecutor for parallel processing on distributed
    clusters).

    Attributes
    ----------
    transformers : list[pept.base.Filter or pept.base.Reducer]
        A list of transformers that will be applied consecutively to samples of
        data. A `pept.base.Filter` subclass transforms individual samples of
        data (and defines a `fit_sample` method), while a `pept.base.Reducer`
        subclass transforms entire lists of samples (and defines a `fit`
        method).

    Examples
    --------
    A pipeline can be created in two ways: either by adding (+) multiple
    transformers together, or explicitly constructing the `Pipeline` class.

    The first method is the most straightforward:

    >>> import pept

    >>> filter1 = pept.tracking.Cutpoints(max_distance = 0.5)
    >>> filter2 = pept.tracking.HDBSCAN(true_fraction = 0.1)
    >>> reducer = pept.tracking.Stack()
    >>> pipeline = filter1 + filter2 + reducer

    >>> print(pipeline)
    Pipeline
    --------
    transformers = [
        Cutpoints(append_indices = False, cutoffs = None, max_distance = 0.5)
        HDBSCAN(clusterer = HDBSCAN(), max_tracers = 1, true_fraction = 0.1)
        Stack(overlap = None, sample_size = None)
    ]

    >>> lors = pept.LineData(...)        # Some samples of lines
    >>> points = pipeline.fit(lors)

    The chain of filters can also be applied to a single sample:

    >>> point = pipeline.fit_sample(lors[0])

    The pipeline's `fit` method allows specifying an execution policy:

    >>> points = pipeline.fit(lors, executor = "sequential")
    >>> points = pipeline.fit(lors, executor = "joblib")

    >>> from mpi4py.futures import MPIPoolExecutor
    >>> points = pipeline.fit(lors, executor = MPIPoolExecutor)

    The `pept.Pipeline` constructor can also be called directly, which allows
    the enumeration of filters:

    >>> pipeline = pept.Pipeline([filter1, filter2, reducer])

    Adding new filters is very easy:

    >>> pipeline_extra = pipeline + filter2

    '''

    def __init__(self, transformers):
        '''Construct the class from an iterable of ``Filter``, ``Reducer``
        and/or other ``Pipeline`` instances (which will be flattened).
        '''

        # Type-checking inputs
        transformers = list(transformers)

        # The list of transformers might contain entire pipelines; extract
        # their filters into a flat list of transformers
        flattened = []
        for i, t in enumerate(transformers):
            if isinstance(t, Transformer):
                flattened.append(t)

            elif isinstance(t, Pipeline):
                flattened.extend(t.transformers)

            else:
                raise TypeError(textwrap.fill((
                    "All input `transformers` must be subclasses of "
                    "`pept.base.Filter`, `pept.base.Reducer` or "
                    f"`pept.base.Pipeline`. Received `{type(t)}` at index {i}."
                )))

        # Setting class attributes
        self._transformers = flattened


    @property
    def filters(self):
        '''Only the `Filter` instances from the `transformers`. They can be
        applied in parallel.
        '''
        return [t for t in self.transformers if isinstance(t, Filter)]


    @property
    def reducers(self):
        '''Only the `Reducer` instances from the `transformers`. They require
        collecting all parallel results.
        '''
        return [t for t in self.transformers if isinstance(t, Reducer)]


    @property
    def transformers(self):
        '''The list of `Transformer` to be applied; this includes both `Filter`
        and `Reducer` instances.
        '''
        return self._transformers


    def _fit_sample(self, transformers, sample):
        # It is more efficient to fit samples procedurally; recursion in Python
        # is slow and doesn't allow garbage-collecting intermediate states
        for t in transformers:
            # If we need to apply a reducer to a single sample, use [sample]
            if isinstance(t, Reducer):
                sample = t.fit([sample])
            else:
                sample = t.fit_sample(sample)

        return sample


    def fit_sample(self, sample):
        '''Apply all transformers - consecutively - to a single sample of data.
        The output type is simply what the transformers return.
        '''
        return self._fit_sample(self.transformers, sample)


    def _fit(self, transformers, samples, executor, max_workers, desc = None):
        # Select type of execution of the filtering steps and extract a list
        # of processed samples
        if executor == "sequential":
            if desc is not None:
                samples = tqdm(samples, desc = desc)

            return [self._fit_sample(transformers, s) for s in samples]

        elif executor == "joblib":
            if desc is not None:
                samples = tqdm(samples, desc = desc)

            # Joblib's `max_workers` behaviour is different than for Executors.
            # The latter simply sets the maximum number of threads available,
            # while the former uses a max_worker=1. Make behaviour consistent.
            if max_workers is None:
                max_workers = os.cpu_count()

            return Parallel(n_jobs = max_workers)(
                delayed(self._fit_sample)(transformers, s) for s in samples
            )

        else:
            # Otherwise assume `executor` is a `concurrent.futures.Executor`
            # subclass (e.g. ProcessPoolExecutor, MPIPoolExecutor).
            with executor(max_workers = max_workers) as exe:
                futures = [
                    exe.submit(self._fit_sample, transformers, s)
                    for s in samples
                ]

                if desc is not None:
                    futures = tqdm(futures, desc = desc)

                return [f.result() for f in futures]


    @beartype
    def fit(
        self,
        samples: Iterable,
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        '''Apply all transformers defined to all `samples`. Filters are applied
        according to the `executor` policy (e.g. parallel via "joblib"), while
        reducers are applied on a single thread.

        Parameters
        ----------
        samples : IterableSamples
            Any subclass of `IterableSamples` (e.g. `pept.LineData`) that
            allows iterating through samples of data.

        executor : "sequential", "joblib", or `concurrent.futures.Executor` \
                subclass, default "joblib"
            The execution policy controlling how the chain of filters are
            applied to each sample in `samples`; "sequential" is single
            threaded (slow, but easy to debug), "joblib" is multi-threaded
            (very fast due to joblib's caching). Alternatively, a
            `concurrent.futures.Executor` subclass can be used (e.g.
            `MPIPoolExecutor` for distributed computing on clusters).

        max_workers : int, optional
            The maximum number of workers to use for parallel executors. If
            `None` (default), the maximum number of CPUs are used.

        verbose : bool, default True
            If True, show extra information during processing, e.g. loading
            bars.
        '''

        # If verbose, time operation
        if verbose:
            start = time.time()

        # Aggregate processing steps into a list where consecutive filters are
        # collapsed into tuples
        steps = self.steps()
        if verbose:
            nbatches = sum((isinstance(step, tuple) for step in steps))

        nbatch = 0
        for step in steps:
            # Apply continuous sequence of filters (i.e. tuple)
            if isinstance(step, tuple):
                nbatch += 1
                desc = f"Batch {nbatch} / {nbatches} :" if verbose else None
                samples = self._fit(step, samples, executor, max_workers, desc)

            # Apply reducer on all samples in `state`
            else:
                samples = step.fit(samples)

        if verbose:
            end = time.time()
            print(f"\nProcessed samples in {end - start} s\n")

        return samples


    def steps(self):
        '''Return the order of processing steps to apply as a list where all
        consecutive sequences of filters are collapsed into tuples.

        E.g. [F, F, R, F, R, R, F, F, F] -> [(F, F), R, (F), R, R, (F, F, F)].
        '''
        start_filter = 0
        collecting_filters = False

        order = []

        for i, t in enumerate(self.transformers):
            if isinstance(t, Filter) and not collecting_filters:
                # Found the first filter in a new sequence of filters
                collecting_filters = True
                start_filter = i

            elif isinstance(t, Reducer):
                # Reducers are appended unchanged, unless we were collecting a
                # sequence of filters - in which case append filter sequence
                if collecting_filters:
                    filters_seq = tuple(self.transformers[start_filter:i])
                    order.append(filters_seq)

                    collecting_filters = False

                # Append reducer
                order.append(t)

        if collecting_filters:
            # Last running sequence of filters
            filters_seq = tuple(self.transformers[start_filter:])
            order.append(filters_seq)

        return order


    def __add__(self, other):
        if isinstance(other, Transformer):
            return Pipeline(self.transformers + [other])
        elif isinstance(other, Pipeline):
            return Pipeline(self.transformers + other.transformers)
        else:
            raise TypeError(textwrap.fill((
                "All new transformers added to the pipeline must be "
                "subclasses of `pept.base.Filter`, `pept.base.Reducer` or "
                f"`pept.base.Pipeline`. Received `{type(other)}`."
            )))


    def __repr__(self):
        name = type(self).__name__
        underline = "-" * len(name)
        return f"{name}\n{underline}\ntransformers = [\n" + textwrap.indent(
            "\n".join((t.__str__() for t in self.transformers)), '    '
        ) + "\n]"
