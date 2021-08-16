#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools.
#
#    If you used this codebase or any software making use of it in a scientific
#    publication, you should cite the following paper:
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


# File   : iterable_samples.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 09.04.2020


import  pickle
import  operator
import  textwrap
from    textwrap            import  indent
from    numbers             import  Number
from    collections.abc     import  Collection
from    concurrent.futures  import  ThreadPoolExecutor

import  numpy               as      np

from    tqdm                import  tqdm




class PEPTObject:
    '''Base class for all PEPT-oriented objects.'''

    def extra_attributes(self, exclude = {}):
        '''Return a dictionary of *extra* attributes defined on this class
        instance; this corresponds to all non-callable attributes whose name
        does not start with an underscore (which are considered hidden).

        You can also define a set `exclude` of attribute names to ignore.
        '''
        exclude = set(exclude)
        attrs = dict()

        for attr in dir(self):
            if not attr.startswith("_") and attr not in exclude:
                memb = getattr(self, attr)
                if not callable(memb):
                    attrs[attr] = memb

        return attrs


    def hidden_attributes(self, exclude = {}):
        '''Return a dictionary of *hidden* attributes defined on this class
        instance; this corresponds to all non-callable attributes whose name
        starts with an underscore, but not *two* underscores (which are
        considered private).

        You can also define a set `exclude` of attribute names to ignore.
        '''
        exclude = set(exclude) | {"_abc_impl"}
        attrs = dict()

        for attr in dir(self):
            if attr.startswith("_") and not attr.startswith("__") and \
                    attr not in exclude:
                memb = getattr(self, attr)
                if not callable(memb):
                    attrs[attr] = memb

        return attrs


    def copy(self):
        '''Create a deep copy of an instance of this class, including all
        inner attributes.
        '''
        return pickle.loads(pickle.dumps(self))


    def __repr__(self, **kwargs):
        # Return pretty string representation of an arbitrary object
        attrs = self.extra_attributes()
        attrs.update(kwargs)

        docs = []
        for attr, memb in attrs.items():
            if (isinstance(memb, np.ndarray) and memb.ndim > 1) or \
                    isinstance(memb, PEPTObject):
                docs.append(f"{attr} = \n{indent(str(memb), '  ')}")
            else:
                docs.append(f"{attr} = {memb}")

        name = type(self).__name__
        underline = "-" * len(name)
        return f"{name}\n{underline}\n" + "\n".join(docs)




def samples_indices_number(data, sample_size, overlap):
    '''Compute the sample indices given some (N, M) `data` and fixed (integer)
    `sample_size` and `overlap`.

    The samples indices are returned in a (N, 2) NumPy array, where the first
    column is each sample's starting index (in `data`), and the second column
    contains the corresponding sample's ending index, so that e.g. sample `n`
    can be extracted as `data[samples_indices[n, 0]:samples_indices[n, 1]]`.
    '''

    if sample_size == 0:
        return np.zeros((0, 2))

    elif sample_size < 0:
        raise ValueError((
            f"\n[ERROR]: `sample_size = {sample_size}` must be positive "
            "(>= 0).\n"
        ))

    elif overlap >= sample_size:
        raise ValueError((
            f"\n[ERROR]: `overlap = {overlap}` must be smaller than "
            f"`sample_size = {sample_size}`.\n"
        ))

    # The first column is each sample's starting index; the second column is
    # the corresponding sample's ending index
    start = np.arange(0, len(data) - sample_size + 1, sample_size - overlap)
    end = start + sample_size

    return np.c_[start, end]




def samples_indices_iterable(data, sample_size):
    '''Compute the sample indices given each sample's length in an iterable
    `sample_size`.
    '''
    end = np.cumsum(sample_size, dtype = int)
    start = end - sample_size

    return np.c_[start, end]




class IterableSamples(PEPTObject, Collection):
    '''An class for iterating through an array (or array-like) in samples with
    potential overlap.

    This class can be used to access samples of data of an adaptive
    ``sample_size`` and ``overlap`` without requiring additional storage.

    The samples from the underlying data can be accessed using both indexing
    (``samples[0]``) and iteration (``for sample in samples: ...``).

    Particular cases:
        1. If sample_size == 0, all data_samples is returned as one single
           sample.
        2. If overlap >= sample_size, an error is raised.
        3. If overlap < 0, lines are skipped between samples.

    Attributes
    ----------
    data : iterable that supports slicing
        An iterable (e.g. numpy array) that supports slicing syntax (data[5:7])
        storing the data that will be iterated over in samples.

    sample_size : int
        The number of rows in `data` to be returned in a single sample. A
        `sample_size` of 0 yields all the data as a single sample.

    overlap : int
        The number of overlapping rows from `data` between two consecutive
        samples. An overlap of 0 implies consecutive samples, while an
        overlap of (`sample_size` - 1) means incrementing the samples by
        one. A negative overlap implies skipping values between samples.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
        must be smaller than `sample_size`. Note that it can also be negative.

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.

    '''

    def __init__(self, data, sample_size = None, overlap = None, **kwargs):
        '''`IterableSamples` class constructor.

        Parameters
        ----------
        data : iterable
            The data that will be iterated over in samples; most commonly a
            NumPy array.

        sample_size : int or Iterable[Int]
            The number of rows in `data` to be returned in a single sample. A
            `sample_size` of 0 yields all the data as a single sample.

        overlap : int, optional
            The number of overlapping rows from `data` between two consecutive
            samples. An overlap of 0 implies consecutive samples, while an
            overlap of (`sample_size` - 1) means incrementing the samples by
            one. A negative overlap implies skipping values between samples.

        '''

        # Allow creating an IterableSamples from a list of samples by stacking
        self._data = np.asarray(data, dtype = float, order = "C")

        # If the overlap is defined, ensure it has the same type as sample_size
        if overlap is not None and not isinstance(overlap, type(sample_size)):
            raise TypeError(textwrap.fill((
                "The input `overlap` (if defined) must have the same type "
                f"as `sample_size`. Received `{type(overlap)}`."
            )))

        # Set sample_size. This calls the setter which does type-checking
        self._overlap = overlap
        self.sample_size = sample_size

        self._index = 0

        # Set extra attributes passed as keyword arguments
        for k, v in kwargs.items():
            setattr(self, k, v)


    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, data):
        self._data = np.asarray(data, dtype = float, order = "C")
        self.sample_size = self.sample_size    # Re-run type-checking


    @property
    def samples_indices(self):
        return self._samples_indices


    @samples_indices.setter
    def samples_indices(self, samples_indices):
        samples_indices = np.asarray(samples_indices, order = "C", dtype = int)

        if samples_indices.ndim != 2 or samples_indices.shape[1] != 2:
            raise ValueError(textwrap.fill((
                "The `samples_indices`, if given as a NumPy array, must be "
                "a (N, 2) matrix where the first column contains each "
                "sample's starting index (in `data`), and the second "
                "column is the corresponding sample's end index. "
                f"Received array with shape `{samples_indices.shape}`."
            )))

        self._samples_indices = samples_indices


    @property
    def sample_size(self):
        return self._sample_size


    @sample_size.setter
    def sample_size(self, sample_size):
        if sample_size is None:
            self._sample_size = None
            self._overlap = None
            self._samples_indices = np.array([[0, len(self.data)]])
        elif isinstance(sample_size, Number):
            # If the overlap is of a different type, reset it
            if not isinstance(self.overlap, Number):
                self._overlap = 0

            self._sample_size = int(sample_size)
            self._samples_indices = samples_indices_number(
                self.data, self._sample_size, self._overlap
            )
        elif hasattr(sample_size, "__iter__"):
            sample_size = np.asarray(sample_size, dtype = int)

            # Special case: if all sample_sizes are equal, set them to that
            if len(sample_size) and (sample_size == sample_size[0]).all():
                self._overlap = 0
                self.sample_size = sample_size[0]
                return

            self._overlap = None
            self._sample_size = sample_size

            self._samples_indices = samples_indices_iterable(
                self.data, self._sample_size
            )
        elif sample_size is None:
            self._overlap = None
            self._sample_size = None
        else:
            raise TypeError("The input `sample_size` has an unknown type.")


    @property
    def overlap(self):
        return self._overlap


    @overlap.setter
    def overlap(self, overlap):
        if overlap is not None and not \
                isinstance(overlap, type(self.sample_size)):
            raise TypeError(textwrap.fill((
                "The input `overlap` must have the same type "
                f"as `sample_size`. Received `{type(overlap)}`."
            )))

        # Call the `sample_size` setter which does type checking
        self._overlap = overlap
        self.sample_size = self._sample_size


    def extra_attributes(self, exclude={}):
        exclude = set(exclude) | {"sample_size", "overlap",
                                  "samples_indices", "data"}
        return PEPTObject.extra_attributes(self, exclude)


    def hidden_attributes(self, exclude={}):
        exclude = set(exclude) | {"_sample_size", "_overlap", "_index",
                                  "_samples_indices", "_data"}
        return PEPTObject.hidden_attributes(self, exclude)


    def copy(self, data = None, sample_size = None, overlap = None, **kwargs):
        '''Construct a similar object, optionally with different `data`,
        `sample_size` and `overlap`.
        '''

        if data is None:
            return PEPTObject.copy(self)

        new_instance = self.__class__(data, sample_size, overlap)

        # Propagate all hidden / extra attributes
        for k, v in self.extra_attributes().items():
            setattr(new_instance, k, v)
        for k, v in self.hidden_attributes().items():
            setattr(new_instance, k, v)

        for k, v in kwargs.items():
            setattr(new_instance, k, v)

        return new_instance


    def __len__(self):
        # Defined so that len(class_instance) returns the number of samples.
        return len(self.samples_indices)


    def __contains__(self, key):
        return self.data.__contains__(key)


    def __getitem__(self, n):
        # Defined so that samples can be accessed as class_instance[0]
        indices = self.samples_indices
        samples_indices = None

        # If n is a slice or iterable, return another IterableSamples
        if isinstance(n, slice):
            # Construct explicit list of indices from slice
            n = np.arange(len(self.samples_indices))[n]

        if hasattr(n, "__iter__"):
            mask = np.full(len(self.data), False)
            samples_indices = np.full((len(n), 2), 0)

            # Create a boolean mask array selecting only array elements we need
            for i, nsample in enumerate(n):
                curi = indices[nsample]
                mask[curi[0]:curi[1]] = True

                # The samples indices must be offset by the number of omitted
                # array elements before their array index
                previous = mask[:curi[0]]
                offset = np.size(previous) - np.count_nonzero(previous)
                samples_indices[i, :] = indices[nsample] - offset

            data = self.data[mask]
            sample_sizes = None

        else:
            # Otherwise return a single sample
            while n < 0:
                n += len(self)

            data = self.data[indices[n, 0]:indices[n, 1]]
            sample_sizes = len(data)

        new_instance = self.copy(data, sample_sizes)

        # Set `samples_indices` directly if needed (e.g. for slices)
        if samples_indices is not None:
            new_instance._samples_indices = samples_indices

        return new_instance


    def __iter__(self):
        # Defined so the class can be iterated as
        # `for sample in class_instance: ...`
        return self


    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration

        self._index += 1
        return self[self._index - 1]




class AsyncIterableSamples(PEPTObject):
    '''Asynchronously apply a function to some samples of data and return those
    processed samples on demand.

    For example, samples of `Cutpoints` are computed from samples of
    `LineData`; `cutpoints_instance[0]` processes the first sample of lines and
    returns it.

    Attributes
    ----------
    samples : instance or subclass of IterableSamples
        The samples of data to be processed; must be a subclass of
        `IterableSamples` to allow iterating over samples (e.g. `LineData`).

    function : callable, signature `func(sample)`
        A function transforming a raw sample from `samples` into a processed
        sample.

    columns : List[str] or None, optional
        The column names of the processed samples.

    executor : concurrent.futures.Executor subclass, default ThreadPoolExecutor
        The executor used

    '''


    def __init__(
        self,
        samples,
        function,
        args = (),
        kwargs = dict(),
        columns = None,
        save_cache = False,
        verbose = True,
    ):
        # Type-checking inputs
        if not isinstance(samples, IterableSamples):
            raise TypeError((
                "The input `samples` must be a collection that allows "
                "iteration in samples - and therefore to be a subclass of "
                "`IterableSamples`"
            ))

        if not callable(function):
            raise TypeError("The input `function` must be callable!")

        # Setting class attributes
        self._samples = samples
        self._function = function
        self._args = tuple(args)
        self._kwargs = dict(kwargs)

        self._columns = None if columns is None else [str(c) for c in columns]

        self._save_cache = bool(save_cache)

        self._index = 0
        self._processed = [None for _ in range(len(self._samples))]


    @property
    def samples(self):
        # The samples of data that will be processed
        return self._samples


    @property
    def columns(self):
        return self._columns


    @property
    def processed(self):
        # Return the list of processed samples
        return self._processed


    @property
    def function(self):
        # The function that transforms `samples` asynchronously
        return self._function


    @property
    def save_cache(self):
        return self._save_cache


    @save_cache.setter
    def save_cache(self, new_save_cache):
        self._save_cache = bool(new_save_cache)


    @property
    def data(self):
        # Accessing all processed data will trigger a full processing run.
        # Defined here to allow e.g. plotting
        return np.vstack(self.traverse(verbose = False))


    def traverse(
        self,
        sample_indices = ...,
        executor = ThreadPoolExecutor,
        max_workers = None,
        verbose = True,
    ):
        '''Apply `self.function` to all samples in `samples` at indices
        `samples_indices`.

        If `save_cache` is `True`, the processed samples are also cached in the
        `data` attribute. Otherwise, they are only returned as a list.

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples to process. An `int` signifies
            the sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all samples).

        verbose : bool, default True
            Show extra information as the processing is done.

        Returns
        -------
        list
            A list of the processed samples, selected by `sample_indices`. The
            type depends on the output of `function`.

        Notes
        -----
        This method is automatically called if the instantiation of the class
        sets `traverse = True`.

        '''

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis.
        if sample_indices is Ellipsis:
            sample_indices = np.arange(len(self.samples))
        elif not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        with executor(max_workers = max_workers) as exe:
            # Use pre-computed voxels and voxellise the other samples
            selected_samples = [None for _ in range(len(sample_indices))]
            selected_futures = [None for _ in range(len(sample_indices))]

            # Apply `function` to each selected sample
            for i, n in enumerate(sample_indices):
                # Optimisation: if this sample was already processed and
                # cached, reuse it
                if self.processed[n] is not None:
                    selected_samples[i] = self.processed[n]
                    continue

                # Otherwise, process the sample asynchronously
                selected_futures[i] = exe.submit(
                    self.function,
                    self.samples[n],
                    *self._args,
                    **self._kwargs,
                )

            if verbose:
                sample_indices = tqdm(sample_indices)

            # Iterate through all the futures; if not None (i.e. we processed
            # the just sample now), extract it. Otherwise it was pre-computed
            for i, n in enumerate(sample_indices):
                if selected_futures[i] is not None:
                    selected_samples[i] = selected_futures[i].result()

                    # Delete the future object to release its memory
                    selected_futures[i] = None

                    # If we processed this sample and we use "save_cache",
                    # cache the result in self.voxels
                    if self.save_cache:
                        self.processed[n] = selected_samples[i]

        return selected_samples


    def accumulate(
        self,
        sample_indices = ...,
        op = operator.add,
        executor = ThreadPoolExecutor,
        max_workers = None,
        verbose = True,
    ):
        '''Accumulate all selected processed samples onto the same object using
        the operator `op`.

        For example, this method can be used to voxellise multiple samples of
        lines into the same `Voxels` class. The computation is done in parallel
        and uses the least amount of memory possible.

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples to process. An `int` signifies
            the sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all samples).

        verbose : bool, default True
            Show extra information as the processing runs.

        Returns
        -------
        `type(function(samples[0]))`
            The processed object onto which all processed samples were
            superimposed.

        '''

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis.
        if sample_indices is Ellipsis:
            sample_indices = np.arange(len(self.samples))
        elif not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        # Voxellise each selected sample
        with executor(max_workers = max_workers) as exe:
            # Use pre-computed samples and process the other samples
            selected_futures = [None for _ in range(len(sample_indices))]

            for i, n in enumerate(sample_indices):
                # Optimisation: if this sample was already processed, reuse it
                if self._voxels[n] is not None:
                    continue

                # Otherwise, process the sample asynchronously
                selected_futures[i] = exe.submit(
                    self.function,
                    self.samples[n],
                    *self._args,
                    **self._kwargs,
                )

            if verbose:
                sample_indices = tqdm(sample_indices)

            # Iterate through all the futures; if not None (i.e. we voxellised
            # the sample now), get the result. Otherwise it was pre-computed
            if selected_futures[0] is None:
                superimposed = self.processed[0]
            else:
                superimposed = selected_futures[0].result()
                selected_futures[0] = None

            for i in range(1, len(sample_indices)):
                if selected_futures[i] is not None:
                    superimposed = op(
                        superimposed,
                        selected_futures[i].result(),
                    )

                    # Delete future object to release its memory
                    selected_futures[i] = None
                else:
                    superimposed = op(
                        superimposed,
                        self.processed[sample_indices[i]],
                    )

        return superimposed


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, key):
        # For accessing voxels using subscript notation
        key = int(key)

        # Allow negative indices
        while key < 0:
            key += len(self.processed)

        if key >= len(self.processed):
            raise IndexError(textwrap.fill((
                f"The index `{key}` was out of range. There are "
                f"{len(self.samples)} samples to be processed, "
                "indexed from 0."
            )))

        # If the sample was already processed and cached, return it directly
        if self.processed[key] is not None:
            return self.processed[key]

        # Otherwise process it
        p = self.function(
            self.samples[key],      # _memoryview_safe(self.samples[key])
            *self._args,
            **self._kwargs,
        )

        if self.save_cache:
            self.processed[key] = p

        return p


    def __iter__(self):
        # Allow iteration of the class - `for sample in class_instance:`
        return self


    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration

        self._index += 1
        return self[self._index - 1]
