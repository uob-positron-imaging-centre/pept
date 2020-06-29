#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : aggregate.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 10.06.2020


import numpy as np


def group_by_column(data_array, column_to_separate):
    '''Group the rows in a 2D `data_array` based on the unique values in a
    given `column_to_separate`, returning the groups as a list of numpy arrays.

    Parameters
    ----------
    data_array : (M, N) numpy.ndarray
        A generic 2D numpy array-like (will be converted using numpy.asarray).
    column_to_separate : int
        The column index in `data_array` from which the unique values will be
        used for grouping.

    Returns
    -------
    groups : list of numpy.ndarray
        A list whose elements are 2D numpy arrays - these are sub-arrays from
        `data_array` for which the entries in the column `column_to_separate`
        are the same.

    Raises
    ------
    ValueError
        If data_array does not have exactly 2 dimensions.

    Examples
    --------
    Separate a 6x3 numpy array based on the last column:

    >>> x = np.array([
    >>>     [1, 2, 1],
    >>>     [5, 3, 1],
    >>>     [1, 1, 2],
    >>>     [5, 2, 1],
    >>>     [2, 4, 2]
    >>> ])
    >>> x_sep = pept.utilities.group_by_column(x)
    >>> x_sep
    >>> [array([[1, 2, 1],
    >>>         [5, 3, 1],
    >>>         [5, 2, 1]]),
    >>>  array([[1, 1, 2],
    >>>         [2, 4, 2]])]

    '''

    data_array = np.asarray(data_array)
    if data_array.ndim != 2:
        raise ValueError((
            "\n[ERROR]: `data_array` should have exactly 2 dimensions. "
            f"Received {data_array} with {data_array.ndim} dimensions.\n"
        ))

    data_col = data_array[:, column_to_separate]
    labels = np.unique(data_col)

    groups = [data_array[data_col == label] for label in labels]
    return groups


