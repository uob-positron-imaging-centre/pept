#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pipelines.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 07.08.2021


import  os
import  re
import  time
import  numbers
import  textwrap
import  warnings
from    abc                 import  ABC, abstractmethod
from    concurrent.futures  import  ProcessPoolExecutor

import  numpy               as      np
import  pandas              as      pd
import  cma

from    tqdm                import  tqdm
from    joblib              import  Parallel, delayed

from    .iterable_samples   import  PEPTObject, IterableSamples
from    .point_data         import  PointData
from    .line_data          import  LineData
from    .voxels             import  Voxels
from    .utilities          import  check_iterable




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
        for att in dir(self):
            memb = getattr(self, att)
            if not callable(memb) and not att.startswith("_"):
                # If it's a nested collection, print it on a new, indented line
                if (isinstance(memb, np.ndarray) and memb.ndim > 1) or \
                        isinstance(memb, PEPTObject):
                    memb_str = textwrap.indent(str(memb), '  ')
                    docs.append(f"{att} = \n{memb_str}")
                else:
                    docs.append(f"{att} = {memb}")

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


    def fit(
        self,
        samples,
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        '''Apply self.fit_sample (implemented by subclasses) according to the
        execution policy. Simply return a list of processed samples. If you
        need a reduction step (e.g. stack all processed samples), apply it
        in the subclass.
        '''

        if not hasattr(samples, "__iter__"):
            raise ValueError(textwrap.fill((
                "The input `samples` must be an iterable (e.g. a list, tuple "
                f"or LineData / PointData). Received `{type(samples)}`."
            )))

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

    def fit(
        self,
        line_data,
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        check_iterable(LineData, line_data = line_data)
        return Filter.fit(
            self,
            line_data,
            executor,
            max_workers,
            verbose = verbose,
        )




class PointDataFilter(Filter):
    '''An abstract class that defines a filter for samples of `pept.PointData`.

    An implementor must define the method `def fit_sample(self, sample)`.

    A default `fit` method is provided for convenience, calling `fit_sample`
    on each sample from an iterable according to a given execution policy
    (e.g. "sequential", "joblib", or `concurrent.futures.Executor` subclasses,
    such as `ProcessPoolExecutor` or `MPIPoolExecutor`).
    '''

    def fit(
        self,
        point_data,
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):

        check_iterable(PointData, point_data = point_data)
        return Filter.fit(
            self,
            point_data,
            executor,
            max_workers,
            verbose = verbose,
        )




class VoxelsFilter(Filter):
    '''An abstract class that defines a filter for samples of `pept.Voxels`.

    An implementor must define the method `def fit_sample(self, sample)`.

    A default `fit` method is provided for convenience, calling `fit_sample`
    on each sample from an iterable according to a given execution policy
    (e.g. "sequential", "joblib", or `concurrent.futures.Executor` subclasses,
    such as `ProcessPoolExecutor` or `MPIPoolExecutor`).
    '''

    def fit(
        self,
        voxels,
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        check_iterable(Voxels, voxels = voxels)

        return Filter.fit(
            self,
            voxels,
            executor,
            max_workers,
            verbose = verbose,
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


    def fit(
        self,
        samples,
        executor = "joblib",
        max_workers = None,
        verbose = True,
    ):
        '''Apply all transformers defined to all `samples`. Filters are applied
        according to the `executor` policy (e.g. parallel via "joblib"), while
        reducers are applied on a single thread.

        Parameters
        ----------
        samples : Iterable
            An iterable (e.g. list, tuple, LineData, list[PointData]), whose
            elements will be passed through the pipeline.

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

        if not hasattr(samples, "__iter__"):
            raise ValueError(textwrap.fill((
                "The input `samples` must be an iterable (e.g. list, tuple, "
                f"LineData, list[PointData]). Received `{type(samples)}`."
            )))

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


    def optimise(
        self,
        lines,
        # group_by = "label",
        # error = ["frequency", "error"],
        max_evals = 200,
        executor = "joblib",
        max_workers = None,
        verbose = True,
        **free_parameters,
    ):

        # Type-checking
        if not isinstance(lines, LineData):
            lines = LineData(lines)

        if not len(free_parameters):
            raise ValueError(textwrap.fill((
                "Must define at least one free parameter to optimise, given "
                "as `param_name = [range_min, range_max]`."
            )))

        # The free parameters for each transformer - prepend the input `lors` s
        # sample_size and overlap
        # This is a list[tuple[transformer, set[parameter_name]]]
        pipe_parameters = parameters_list(self)

        # Find transformer free parameters
        # This is a list[OptParam] saving the transformer and param name
        opt_parameters = [
            OptParam(pipe_parameters, param, bounds)
            for param, bounds in free_parameters.items()
        ]

        if verbose:
            _print_pipe_params(opt_parameters)

        bounds = np.array([op.bounds for op in opt_parameters])
        scaling = 0.4 * (bounds[:, 1] - bounds[:, 0])
        bounds_scaled = bounds / scaling[:, None]

        x0 = 0.5 * (bounds[:, 0] + bounds[:, 1]) / scaling
        sigma0 = 1.

        es = cma.CMAEvolutionStrategy(x0, sigma0, dict(
            bounds = [bounds_scaled[:, 0], bounds_scaled[:, 1]],
            tolflatfitness = 10,
            maxfevals = int(max_evals),
            verbose = 3 if verbose else -9,
        ))

        # Store optimisation pipeline parameters
        if max_workers is None:
            max_workers = os.cpu_count()

        with ProcessPoolExecutor(max_workers = max_workers) as executor:
            popt = OptPipeline(self, opt_parameters, lines, executor)

            # Save optimisation evolution
            history = []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                while not es.stop():
                    solutions = es.ask()
                    results = _try_solutions(solutions * scaling, popt)
                    es.tell(solutions, results[:, -1])

                    _print_after_eval(
                        es,
                        opt_parameters,
                        solutions * scaling,
                        results,
                    )
                    history.append(np.c_[solutions * scaling, results])

                    if es.sigma < 0.1:
                        break

        print(es.result)

        best_solution = es.result.xfavorite * scaling
        for op, sol in zip(popt.parameters, best_solution):
            if isinstance(op.default, numbers.Integral):
                sol = int(sol)

            setattr(op.trans, op.param, sol)


        return np.vstack(history)



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




class OptParam:
    '''Class storing a single pipeline optimisation free parameter, including
    its parent transformer (`trans`), name (`param`), range (`bounds`) and
    initial value (`default`).
    '''
    # index: int
    # trans: PEPTObject
    # param: str
    # bounds: np.ndarray
    # default: float

    def __init__(self, pipe_parameters, param, bounds):

        # Type-checking
        param = str(param)
        bounds = np.asarray(bounds, dtype = float)
        if bounds.ndim != 1 or len(bounds) != 2:
            raise ValueError(textwrap.fill((
                f"The parameter range given for `{param}` is incorrect. "
                "It must be a list formatted as [min, max]. Received "
                f"`{bounds}`."
            )))

        # Parameter name may have a number appended, e.g. "sample_size2" means
        # the second pipeline free parameter named `sample_size`
        match = re.search(r"(\d+)$", param)
        if match is None:
            pnum = 1                            # Parameter number
        else:
            pnum = int(match.group())
            param = param[:match.span()[0]]     # Remove parameter number

        self.param = param
        self.bounds = bounds

        # Find index of transformer containing the `param` free parameter
        pcur = 1
        for i, (trans, param_names) in enumerate(pipe_parameters):
            if param in param_names:
                if pcur == pnum:
                    self.index = i
                    self.trans = trans
                    self.default = getattr(trans, param)
                    break
                else:
                    pcur += 1
        else:
            raise ValueError(textwrap.fill((
                f"No free parameter named `{param}` was found in the pipeline "
                f"while looking for the N={pnum} parameter with this name."
            )))




class OptPipeline:
    '''Class storing a single pipeline optimisation free parameter, including
    its parent transformer (`trans`), name (`param`), range (`bounds`) and
    initial value (`default`).
    '''

    def __init__(
        self,
        pipeline: Pipeline,
        parameters: OptParam,
        lines: LineData,
        executor: ProcessPoolExecutor,
    ):
        self.pipeline = pipeline
        self.parameters = parameters
        self.lines = lines
        self.executor = executor




def parameters_list(pipeline: Pipeline):
    '''Return a list[tuple[TransformerName, set[str]]], where for each
    transformer in a pipeline the corresponding set contains its free parameter
    names.
    '''
    parameters = []
    for trans in pipeline.transformers:
        # Save the names of the transformer's attributes that are not hidden
        # (i.e. start with "_") and are not callable (i.e. not methods)
        trans_params = (trans, set())
        for att in dir(trans):
            if not att.startswith("_") and not callable(getattr(trans, att)):
                trans_params[1].add(att)

        parameters.append(trans_params)

    return parameters


def _print_pipe_params(opt_parameters):
    trans_strs = [
        f"{op.trans.__class__.__name__}.{op.param}"
        for op in opt_parameters
    ]
    max_len = max([len(ts) for ts in trans_strs])

    print("\nOptimising Pipeline Parameters:")
    for trans, op in zip(trans_strs, opt_parameters):
        trans = trans + (max_len - len(trans)) * ' '
        print(f"  {op.index} : {trans} : {op.bounds}")
    print()


def _try_solution(pipeline, lines):
    start = time.time()
    try:
        out = pipeline.fit(
            lines,
            executor = "sequential",
            verbose = False,
        )
    except ValueError:
        out = PointData(
            np.empty((0, 5)),
            columns = ["t", "x", "y", "z", "error"],
        )
    end = time.time()
    print(f"Pipe eval: {end - start}")

    print(out)

    # Try to keep the frequency of points ~ freq LoRs / 100
    # Calculate the difference of frequency of points to lines / 100
    times = lines["t"]
    dt = times[-1] - times[0]
    pfreq = len(out.points) / dt
    lfreq = len(lines.lines) / dt

    freq_err = (pfreq - lfreq / 100) ** 2 if pfreq else np.inf

    # Error in particle positions
    spatial_err = np.nanmedian(out["error"]) if pfreq else np.inf

    return [freq_err, spatial_err, freq_err * spatial_err]



def _try_solutions(solutions, popt: OptPipeline):

    # Create pipeline copies holding the different parameter combinations
    pipes = [popt.pipeline.copy() for _ in range(len(solutions))]

    for sol_set in solutions:
        # Set transformers' parameters to the values in `sol_set`
        for i, op in enumerate(popt.parameters):
            sol = sol_set[i]
            if isinstance(op.default, numbers.Integral):
                sol = int(sol)

            setattr(pipes[i].transformers[op.index], op.param, sol)

    # Execute the different pipelines in parallel
    futures = [
        popt.executor.submit(_try_solution, pipe, popt.lines)
        for pipe in pipes
    ]
    results = [f.result() for f in futures]
    del futures

    # Reset pipeline parameters to their default values
    # for op in popt.parameters:
    #     setattr(op.trans, op.param, op.default)

    return np.array(results)


def _print_after_eval(es, parameters, solutions, results):
    # Display evaluation results: solutions, error values, etc.
    cols = [p.param for p in parameters] + ["freq_err", "spatial_err", "error"]
    sols_results = np.c_[solutions, results]

    # Store solutions and results in a DataFrame for easy pretty printing
    sols_results = pd.DataFrame(
        data = sols_results,
        columns = cols,
        index = None,
    )

    # Display all the DataFrame columns and rows
    old_max_columns = pd.get_option("display.max_columns")
    old_max_rows = pd.get_option("display.max_rows")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    print((
        f"{sols_results}\n"
        f"    Overall Convergence: {es.sigma}\n"
        f"  Parameter Convergence: {es.result.stds}\n"
        f"   Function evaluations: {es.result.evaluations}\n---"
    ), flush = True)

    pd.set_option("display.max_columns", old_max_columns)
    pd.set_option("display.max_rows", old_max_rows)
