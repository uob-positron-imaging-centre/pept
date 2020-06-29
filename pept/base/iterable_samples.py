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


# File   : iterable_samples.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 09.04.2020


from abc import ABC, abstractmethod


class IterableSamples(ABC):
    '''An abstract class for iterating through an array (or array-like) in
    samples with potential overlap.

    A class that inherits from ``IterableSamples`` must implement two
    properties:

    ``data_samples``
        The underlying data (e.g. a numpy array) that supports slicing.

    ``data_length``
        The number of items in ``data_samples`` (e.g. for a 2D numpy array,
        this is the number of rows).

    Given an implementor's properties ``data_samples`` (e.g. a numpy array) and
    ``data_length``, this class can yield samples of the data of an adaptive
    ``sample_size`` and ``overlap``, without requiring additional storage.

    The underlying data can be accessed using both indexing (``data[0]``) and
    iteration (``for sample in data: ...``).

    Particular cases:
        1. If sample_size == 0, all data_samples is returned as one single
           sample.
        2. If overlap >= sample_size, an error is raised.
        3. If overlap < 0, lines are skipped between samples.

    Attributes
    ----------
    data_samples : iterable that supports slicing
        An iterable (e.g. numpy array) that supports slicing syntax (data[5:7])
        storing the data that will be iterated over in samples.
    data_length : int
        The number of elements in `data_samples`. For a numpy.ndarray, that
        corresponds to len(`line_data`).
    sample_size : int
        An `int`` that defines the number of items that should be returned in
        a single sample when iterating over `data_samples`. A `sample_size` of
        0 yields all the data as one single sample.
    overlap : int, optional
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `data_samples`. An overlap of 0
        implies consecutive samples, while an overlap of (`sample_size` - 1)
        means incrementing the samples by one. A negative overlap implies
        skipping values between samples. An error is raised if `overlap` is
        larger than or equal to `sample_size`.
    number_of_samples : int
        An `int` that corresponds to the number of samples that can be
        accessed from the class. It takes `overlap` into consideration.

    Methods
    -------
    sample(n)
        Get sample number n (indexed from 0).

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
        has to be smaller than `sample_size`. Note that it can also be
        negative.

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.

    '''

    def __init__(self, sample_size, overlap):
        '''``IterableSamples`` class constructor.

        Parameters
        ----------
        sample_size : int
            An `int`` that defines the number of items that should be returned
            in a single sample when iterating over `data_samples`. A
            `sample_size` of 0 yields all the data as one single sample.
        overlap : int, optional
            An `int` that defines the overlap between two consecutive samples
            that are returned when iterating over `data_samples`. An overlap of
            0 implies consecutive samples, while an overlap of
            (`sample_size` - 1) means incrementing the samples by one. A
            negative overlap implies skipping values between samples. An error
            is raised if `overlap` is larger than or equal to `sample_size`.

        '''

        sample_size = int(sample_size)
        overlap = int(overlap)

        if sample_size < 0:
            raise ValueError((
                f"\n[ERROR]: sample_size = {sample_size} must be positive "
                "(>= 0).\n"
            ))

        if sample_size != 0 and overlap >= sample_size:
            raise ValueError((
                f"\n[ERROR]: overlap = {overlap} must be smaller than "
                f"sample_size = {sample_size}.\n"
            ))

        self._index = 0
        self._sample_size = sample_size
        self._overlap = overlap


    @property
    @abstractmethod
    def data_samples(self):
        '''The data encapsulated by the class the will be iterated over in
        samples with overlap.

        Must be implemented by a subclass.

        '''

        pass


    @property
    @abstractmethod
    def data_length(self):
        '''The number of items in `data_samples` (e.g. for a 2D numpy array,
        this corresponds to the number of rows).

        Must be implemented by a subclass.

        '''

        pass



    @property
    def sample_size(self):
        '''The number of items in one sample returned by the class.

        Returns
        -------
        int
            The sample size (number of lines) in one sample returned by
            the class.

        '''

        return self._sample_size


    @sample_size.setter
    def sample_size(self, sample_size):
        '''Change `sample_size` without instantiating a new object.

        It also resets the inner index of the class.

        Parameters
        ----------
        sample_size : int
            The new sample size. It has to be larger than `overlap`, unless it
            is 0 (in which case all `point_data` will be returned as one
            sample, ignoring the overlap).

        Raises
        ------
        ValueError
            If `overlap` >= `sample_size`. Overlap has to be smaller than
            `sample_size`, unless `sample_size` is 0.

        '''

        sample_size = int(sample_size)

        if sample_size < 0:
            raise ValueError((
                f"\n[ERROR]: sample_size = {sample_size} must be positive "
                "(>= 0). \n"
            ))

        if sample_size != 0 and self._overlap >= sample_size:
            raise ValueError((
                f"\n[ERROR]: overlap = {self._overlap} must be smaller than "
                f"sample_size = {sample_size}.\n"
            ))

        self._index = 0
        self._sample_size = sample_size


    @property
    def overlap(self):
        '''The overlap between every two samples returned by the class.

        Returns
        -------
        int
            The overlap (number of items) between every two samples returned by
            the class.

        '''

        return self._overlap


    @overlap.setter
    def overlap(self, overlap):
        '''Change `overlap` without instantiating a new object.

        It also resets the inner index of the class.

        Parameters
        ----------
        overlap : int
            The new overlap. It must be smaller than `sample_size`, unless
            `sample_size` is 0 (in which case all `point_data` will be returned
            as one sample and so overlap is ignored).

        Raises
        ------
        ValueError
            If `overlap` >= `sample_size`. `overlap` must be smaller than
            `sample_size`, unless `sample_size` is 0. Note that `overlap` can
            also be negative, in which case items are skipped between samples.

        '''

        overlap = int(overlap)

        if self._sample_size != 0 and overlap >= self._sample_size:
            raise ValueError((
                f"\n[ERROR]: overlap = {overlap} must be smaller than "
                f"sample_size = {self._sample_size}.\n"
            ))

        self._index = 0
        self._overlap = overlap


    @property
    def number_of_samples(self):
        '''The number of samples, considering overlap.

        If `sample_size == 0`, all data is returned as a single sample, and so
        `number_of_samples` is 1. Otherwise, it checks the number of samples
        every time it is called, taking `overlap` into consideration.

        Returns
        -------
        int
            The number of samples, taking `overlap` into consideration.

        '''

        # If self.sample_size == 0, all data is returned as a single sample.
        if self._sample_size == 0:
            return 1

        # If self.sample_size != 0, check that there is at least one sample.
        if self.data_length >= self._sample_size:
            return (self.data_length - self._sample_size) // \
                (self.sample_size - self.overlap) + 1
        else:
            return 0


    def sample(self, n):
        '''Get sample number n (indexed from 0, i.e. `n >= 0`)

        Returns the items from `data_samples` included in sample number `n`.
        Samples are numbered starting from 0.

        Parameters
        ----------
        n : int
            The number of the sample required. Note that 1 <= n <=
            number_of_samples.

        Returns
        -------
        items from data_samples
            The items returned from `point_data` using slicing, included in
            sample `n`.

        Raises
        ------
        IndexError
            If `sample_size == 0`, all data is returned as one single sample.
            Raised if `n` is not 1.
        IndexError
            If `n > number_of_samples` or `n <= 0`.

        '''

        if self._sample_size == 0:
            if n == 0:
                return self.data_samples
            else:
                raise IndexError((
                    "\n[ERROR]: Tried to access a non-existent sample "
                    f"(samples indexed from 0): asked for sample {n}, when "
                    "there is only 1 sample (because `sample_size` == 0).\n"
                ))
        elif (n >= self.number_of_samples) or n < 0:
            raise IndexError((
                "\n[ERROR]: Tried to access a non-existent sample (samples "
                f"are indexed from 0): asked for sample {n}, when there are "
                f"{self.number_of_samples} samples.\n"
            ))

        start_index = n * (self._sample_size - self._overlap)
        return self.data_samples[start_index:(start_index + self._sample_size)]


    def __len__(self):
        # Defined so that len(class_instance) returns the number of samples.
        return self.number_of_samples


    def __getitem__(self, key):
        # Defined so that samples can be accessed as class_instance[0]
        while key < 0:
            key += self.number_of_samples

        return self.sample(key)


    def __iter__(self):
        # Defined so the class can be iterated as
        # `for sample in class_instance: ...`
        return self


    def __next__(self):
        # sample_size = 0 => return all data
        if self._sample_size == 0:
            self._sample_size = -1
            return self.data_samples
        # Use -1 as a flag
        if self._sample_size == -1:
            self._sample_size = 0
            raise StopIteration

        # sample_size > 0 => return slices
        if self._index != 0:
            self._index = self._index + self._sample_size - self._overlap
        else:
            self._index = self._index + self._sample_size

        if self._index > self.data_length:
            self._index = 0
            raise StopIteration

        return self.data_samples[(self._index - self._sample_size):self._index]


