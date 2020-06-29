#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : parallel_map.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 03.02.2020


import  numpy               as      np
from    multiprocessing     import  Pool


def parallel_map_file(
    func,           # Called as func(data_chunk, chunk_number, *args, **kwargs)
    fname,          # File that will be supplied to numpy.loadtxt
    start,          # Start line
    end,            # End line
    chunksize,      # Number of lines per chunk
    *args,
    dtype = float,
    processes = None,
    callback = lambda x: None,
    error_callback = lambda x: None,
    **kwargs
):
    '''Utility for parallelising (read CSV chunk -> process chunk) workflows.

    This function reads individual chunks of data from a CSV-formatted file,
    then asynchronously sends them as numpy arrays to an arbitrary function
    `func` for processing. In effect, it reads a file in one main thread and
    processes it in separate threads.

    This is especially useful when dealing with very large files (like we do in
    PEPT...) that you'd like to process in batches, in parallel.

    Parameters
    ----------
    func : callable
        The function that will be called with each chunk of data, the chunk
        number, the other positional arguments `*args` and keyword arguments
        `**kwargs`: `func(data_chunk, chunk_number, *args, **kwargs)`.
        `data_chunk` is a numpy array returned by `numpy.loadtxt` and
        `chunk_number` is an int. `func` must be picklable for sending to
        other threads.
    fname : file, str, or pathlib.Path
        The file, filename, or generator that numpy.loadtxt will be supplied
        with.
    start : int
        The starting line number that the chunks will be read from.
    end : int
        The ending line number that the chunks will be read from. This is
        exclusive.
    chunksize : int
        The number of lines that will be read for each chunk.
    *args : additional positional arguments
        Additional positional arguments that will be supplied to `func`.
    dtype : type
        The data type of the numpy array that is returned by numpy.loadtxt. The
        default is `float`.
    processes : int
        The maximum number of threads that will be used for calling `func`. If
        left to the default `None`, then the number returned by `os.cpu_count()`
        will be used.
    callback : callable
        When the result from a `func` call becomes ready callback is applied to
        it, that is unless the call failed, in which case the error_callback is
        applied instead.
    error_callback : callable
        If the target function `func` fails, then the error_callback is called
        with the exception instance.
    **kwargs : additional keybord arguments
        Additional keyword arguments that will be supplied to `func`.

    Returns
    -------
    list
        A Python list of the `func` call returns. The results are not
        necessarily in order, though this can be verified by using the chunk
        number that is supplied to each call to `func`. If `func` does not
        return anything, it will simply be a list of `None`s.

    Notes
    -----
    This function uses `numpy.loadtxt` to read chunks of data and
    `multiprocessing.Pool.apply_async` to call `func` asynchronously.

    As the calls to `func` happen in different threads, all the usual parallel
    processing issues apply. For example, `func` should not save data to the
    same file, as it will overwrite results from different threads and may
    become corrupt - however, there is a workaround for this particular case:
    because the chunk numbers are guaranteed to be unique, any data can be
    saved to a file whose name includes this chunk number, making it unique.

    Examples
    --------
    For a random file-like CSV data object:

    >>> import io
    >>> flike = io.StringIO("1,2,3\\n4,5,6\\n7,8,9")
    >>> def func(data, chunk_number):
    >>>     return (data, chunk_number)
    >>> results = parallel_map_file(func, flike, 0, 3, 1)
    >>> print(results)
    >>> [ ([1, 2, 3], 0), ([4, 5, 6], 1), ([7, 8, 9], 2) ]

    '''

    nchunks = int((end - start) / chunksize)

    with Pool(processes = processes) as pool:
        results = []
        for i in range(nchunks):
            data = np.loadtxt(
                fname,
                skiprows = start + i * chunksize,
                max_rows = chunksize,
                dtype = dtype
            )
            worker = pool.apply_async(
                func,
                (data, i, *args),
                kwargs,
                callback,
                error_callback
            )
            results.append(worker)

        results = [r.get() for r in results]

    return results




