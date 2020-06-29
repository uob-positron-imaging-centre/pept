#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : read_csv.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 14.04.2020


import pandas as pd
import csv


def number_of_lines(filepath_or_buffer):
    '''Return the number of lines (or rows) in a file.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Path to the file.

    Returns
    -------
    int
        The number of lines in the file pointed at by `filepath_or_buffer`.
    '''

    with open(filepath_or_buffer) as f:
        file_lines = sum(1 for line in f)

    return file_lines


def read_csv(
    filepath_or_buffer,         # Essential
    skiprows = None,            # Important
    nrows = None,               # Important
    dtype = float,              # Medium Importance
    sep = "\s+",                # Extra parameters
    engine = "c",               #       |
    na_filter = False,          #       |
    quoting = csv.QUOTE_NONE,   #       |
    memory_map = True,          #       -
    **kwargs                    # Extra keyword arguments to pandas.read_csv
):
    '''Read a given number of lines from a file and return a numpy array of the
    values.

    This is a convenience function that's simply a proxy to `pandas.read_csv`,
    configured with default parameters for fast reading and parsing of usual
    PEPT data.

    Most importantly, it reads from a **space-separated values** file at
    `filepath_or_buffer`, optionally skipping `skiprows` lines and reading in
    `nrows` lines. It returns a `numpy.ndarray` with `float` values.

    The parameters below are sent to `pandas.read_csv` with no further parsing.
    The descriptions below are taken from the `pandas` documentation.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: file://localhost/path/to/table.csv. If
        you want to pass in a path object, pandas accepts any `os.PathLike`. By
        file-like object, we refer to objects with a `read()` method, such as a
        file handler (e.g. via builtin `open` function) or `StringIO`.
    skiprows : list-like, int or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int) at
        the start of the file.
    nrows : int, optional
        Number of rows of file to read. Useful for reading pieces of large
        files.
    dtype : Type name, default `float`
        Data type for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32,
        ‘c’: ‘Int64’}.
    sep : str, default `"\s+"`
        Delimiter to use. Separators longer than 1 character and different from
        '\s+' will be interpreted as regular expressions and will also force
        the use of the Python parsing engine.
    engine : {‘c’, ‘python’}, default "c"
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    na_filter : bool, default `True`
        Detect missing value markers (empty strings and the value of
        na_values). In data without any NAs, passing na_filter=False can
        improve the performance of reading a large file.
    quoting : int or csv.QUOTE_* instance, default `csv.QUOTE_NONE`
        Control field quoting behavior per csv.QUOTE_* constants. Use one of
        QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or
        QUOTE_NONE (3).
    memory_map : bool, default True
        If a filepath is provided for filepath_or_buffer, map the file object
        directly onto memory and access the data directly from there. Using
        this option can improve performance because there is no longer any I/O
        overhead.
    kwargs : optional
        Extra keyword arguments that will be passed to `pandas.read_csv`.
    '''

    data = pd.read_csv(
        filepath_or_buffer,
        skiprows = skiprows,
        nrows = nrows,
        dtype = dtype,
        sep = sep,
        engine = engine,
        na_filter = na_filter,
        quoting = quoting,
        memory_map = memory_map,
        **kwargs
    )

    data_array = data.to_numpy(copy = True)
    del data

    return data_array


def read_csv_chunks(
    filepath_or_buffer,
    chunksize,
    skiprows = None,
    nrows = None,
    dtype = float,
    sep = "\s+",
    engine = "c",
    na_filter = False,
    quoting = csv.QUOTE_NONE,
    memory_map = True,
    **kwargs
):
    '''Read chunks of data from a file lazily, returning numpy arrays of the
    values.

    This function returns a generator - an object that can be iterated over
    once, creating data on-demand. This means that chunks of data will be
    read only when being accessed, making it a more efficient alternative to
    `read_csv` for large files (> 1.000.000 lines).

    A more convenient and feature-complete alternative is
    `pept.utilities.ChunkReader` which is more reusable and can access
    out-of-order chunks using subscript notation (i.e. data[0]).

    This is a convenience function that's simply a proxy to `pandas.read_csv`,
    configured with default parameters for fast reading and parsing of usual
    PEPT data.

    Most importantly, it lazily read chunks of size `chunksize` from a
    **space-separated values** file at `filepath_or_buffer`, optionally
    skipping `skiprows` lines and reading in `nrows` lines. It returns
    `numpy.ndarray`s with `float` values.

    The parameters below are sent to `pandas.read_csv` with no further parsing.
    The descriptions below are taken from the `pandas` documentation.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: file://localhost/path/to/table.csv. If
        you want to pass in a path object, pandas accepts any `os.PathLike`. By
        file-like object, we refer to objects with a `read()` method, such as a
        file handler (e.g. via builtin `open` function) or `StringIO`.
    chunksize : int
        Number of lines read in a chunk of data. Return TextFileReader object
        for iteration.
    skiprows : list-like, int or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int) at
        the start of the file.
    nrows : int, optional
        Number of rows of file to read. Useful for reading pieces of large
        files.
    dtype : Type name, default `float`
        Data type for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32,
        ‘c’: ‘Int64’}.
    sep : str, default `"\s+"`
        Delimiter to use. Separators longer than 1 character and different from
        '\s+' will be interpreted as regular expressions and will also force
        the use of the Python parsing engine.
    engine : {‘c’, ‘python’}, default "c"
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    na_filter : bool, default `True`
        Detect missing value markers (empty strings and the value of
        na_values). In data without any NAs, passing na_filter=False can
        improve the performance of reading a large file.
    quoting : int or csv.QUOTE_* instance, default `csv.QUOTE_NONE`
        Control field quoting behavior per csv.QUOTE_* constants. Use one of
        QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or
        QUOTE_NONE (3).
    memory_map : bool, default True
        If a filepath is provided for filepath_or_buffer, map the file object
        directly onto memory and access the data directly from there. Using
        this option can improve performance because there is no longer any I/O
        overhead.
    kwargs : optional
        Extra keyword arguments that will be passed to `pandas.read_csv`.
    '''

    reader = pd.read_csv(
        filepath_or_buffer,
        chunksize = chunksize,
        skiprows = skiprows,
        nrows = nrows,
        dtype = dtype,
        sep = sep,
        engine = engine,
        na_filter = na_filter,
        quoting = quoting,
        memory_map = memory_map,
        **kwargs
    )

    for chunk in reader:
        yield chunk.values




class ChunkReader:
    '''Class for fast, on-demand reading / parsing and iteration over chunks of
    data from CSV files.

    This is an abstraction above `pandas.read_csv` for easy and fast iteration
    over chunks of data from a CSV file. The chunks can be accessed using
    normal iteration (`for chunk in reader: ...`) and subscripting
    (`reader[0]`).

    The chunks are read lazily, only upon access. It is therefore a more
    efficient alternative to `read_csv` for large files (> 1.000.000 lines).
    For convenience, this class configures some default parameters for
    `pandas.read_csv` for fast reading and parsing of usual PEPT data.

    Most importantly, it reads chunks containing `chunksize` lines from a
    **space-separated values** file at `filepath_or_buffer`, optionally
    skipping `skiprows` lines and reading in at most `nrows` lines. It returns
    `numpy.ndarray`s with `float` values.

    Attributes
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a
        host is expected. A local file could be
        file://localhost/path/to/table.csv. If you want to pass in a path
        object, pandas accepts any `os.PathLike`. By file-like object, we
        refer to objects with a `read()` method, such as a file handler
        (e.g. via builtin `open` function) or `StringIO`.
    number_of_chunks : int
        The number of chunks (also returned when using the `len` method),
        taking into account the lines skipped (`skiprows`), the number of lines
        in the file (`file_lines`) and the maximum number of lines to be read
        (`nrows`).
    file_lines : int
        The number of lines in the file pointed at by `filepath_or_buffer`.
    chunksize : int
        The number of lines in a chunk of data.
    skiprows : int
        The number of lines to be skipped at the beginning of the file.
    nrows : int
        The maximum number of lines to be read. Only has an effect if it is
        less than `file_lines` - `skiprows`. For example, if a file has
        10 lines and `skiprows` = 5 and `chunksize` = 5, even if `nrows` were
        to be 20, the `number_of_chunks` should still be 1.

    Raises
    ------
    IndexError
        Upon access to a non-existent chunk using subscript notation
        (i.e. `data[100]` when there are 50 chunks).

    Examples
    --------
    Say "data.csv" contains 1_000_000 lines of data. Read chunks of 10_000
    lines as a time, skipping the first 100_000:

    >>> from pept.utilities import ChunkReader
    >>> chunks = ChunkReader("data.csv", 10_000, skiprows = 100_000)
    >>> len(chunks)         # 90 chunks
    >>> chunks.file_lines   # 1_000_000

    Normal iteration:

    >>> for chunk in chunks:
    >>>     ... # neat operations

    Access a single chunk using subscripting:

    >>> chunks[0]   # First chunk
    >>> chunks[-1]  # Last chunk
    >>> chunks[100] # IndexError

    See Also
    --------
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    '''

    def __init__(
        self,
        filepath_or_buffer,
        chunksize,
        skiprows = None,
        nrows = None,
        dtype = float,
        sep = "\s+",
        engine = "c",
        na_filter = False,
        quoting = csv.QUOTE_NONE,
        memory_map = True,
        **kwargs
    ):
        '''ChunkReader class constructor.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            Any valid string path is acceptable. The string could be a URL.
            Valid URL schemes include http, ftp, s3, and file. For file URLs, a
            host is expected. A local file could be
            file://localhost/path/to/table.csv. If you want to pass in a path
            object, pandas accepts any `os.PathLike`. By file-like object, we
            refer to objects with a `read()` method, such as a file handler
            (e.g. via builtin `open` function) or `StringIO`.
        chunksize : int
            Number of lines read in a chunk of data.
        skiprows : list-like, int or callable, optional
            Line numbers to skip (0-indexed) or number of lines to skip (int)
            at the start of the file.
        nrows : int, optional
            Number of rows of file to read. Useful for reading pieces of large
            files.
        dtype : Type name, default `float`
            Data type for data or columns. E.g. {‘a’: np.float64,
            ‘b’: np.int32, ‘c’: ‘Int64’}.
        sep : str, default `"\s+"`
            Delimiter to use. Separators longer than 1 character and different
            from '\s+' will be interpreted as regular expressions and will also
            force the use of the Python parsing engine.
        engine : {‘c’, ‘python’}, default "c"
            Parser engine to use. The C engine is faster while the python
            engine is currently more feature-complete.
        na_filter : bool, default `True`
            Detect missing value markers (empty strings and the value of
            na_values). In data without any NAs, passing na_filter=False can
            improve the performance of reading a large file.
        quoting : int or csv.QUOTE_* instance, default `csv.QUOTE_NONE`
            Control field quoting behavior per csv.QUOTE_* constants. Use one
            of QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or
            QUOTE_NONE (3).
        memory_map : bool, default True
            If a filepath is provided for filepath_or_buffer, map the file
            object directly onto memory and access the data directly from
            there. Using this option can improve performance because there is
            no longer any I/O overhead.
        kwargs : optional
            Extra keyword arguments that will be passed to `pandas.read_csv`.

        Raises
        ------
        EOFError : End Of File Error
            If `skiprows` >= `number_of_lines`.
        '''

        self.filepath_or_buffer = filepath_or_buffer
        self._chunksize = chunksize

        self._file_lines = number_of_lines(filepath_or_buffer)

        if skiprows is None:
            self._skiprows = 0
        elif skiprows >= self.file_lines:
            raise EOFError((
                f"\n[ERROR]: Tried to skip `skiprows` = {skiprows} lines "
                f"when there are `file_lines` = {self._file_lines} lines in "
                "the data file.\n"
            ))
        else:
            self._skiprows = skiprows

        # If undefined, set `nrows` to the maximum number of lines that can be
        # read from the file; that is `file_lines` - `skiprows`
        if nrows is None:
            self._nrows = self._file_lines - self._skiprows
        else:
            self._nrows = nrows

        self.dtype = dtype
        self.sep = sep
        self.engine = engine
        self.na_filter = na_filter
        self.quoting = quoting
        self.memory_map = memory_map
        self.kwargs = kwargs

        # The number of chunks is (the number of lines that can be read from
        # the file OR the set number of rows to be read, whichever's smaller)
        # divided by the chunksize.
        self._number_of_chunks = int(
            min(self._nrows, self._file_lines - self._skiprows) / \
            self._chunksize
        )
        self._index = 0


    @property
    def number_of_chunks(self):
        return self._number_of_chunks


    @property
    def file_lines(self):
        return self._file_lines


    @property
    def chunksize(self):
        return self._chunksize


    @chunksize.setter
    def chunksize(self, new_chunksize):
        self._chunksize = new_chunksize

        # Recalculate the number of chunks and reset the inner index
        self._number_of_chunks = int(
            min(self._nrows, self._file_lines - self._skiprows) / \
            self._chunksize
        )
        self._index = 0


    @property
    def skiprows(self):
        return self._skiprows


    @skiprows.setter
    def skiprows(self, new_skiprows):
        if new_skiprows is None:
            self._skiprows = 0
        elif new_skiprows >= self._file_lines:
            raise EOFError((
                f"\n[ERROR]: Tried to skip `skiprows` = {new_skiprows} lines "
                f"when there are `file_lines` = {self._file_lines} lines in "
                "the data file.\n"
            ))
        else:
            self._skiprows = new_skiprows

        # Recalculate the number of chunks and reset the inner index
        self._number_of_chunks = int(
            min(self._nrows, self._file_lines - self._skiprows) / \
            self._chunksize
        )
        self._index = 0


    @property
    def nrows(self):
        return self._nrows


    @nrows.setter
    def nrows(self, new_nrows):
        if new_nrows is None:
            self._nrows = self._file_lines - self._skiprows
        else:
            self._nrows = new_nrows

        # Recalculate the number of chunks and reset the inner index
        self._number_of_chunks = int(
            min(self._nrows, self._file_lines - self._skiprows) / \
            self._chunksize
        )
        self._index = 0


    def __len__(self):
        return self._number_of_chunks


    def __iter__(self):
        return self


    def __next__(self):
        if self._index >= self._number_of_chunks:
            self._index = 0
            raise StopIteration

        data = pd.read_csv(
            self.filepath_or_buffer,
            skiprows = self._skiprows + self._index * self._chunksize,
            nrows = self._chunksize,
            dtype = self.dtype,
            sep = self.sep,
            engine = self.engine,
            na_filter = self.na_filter,
            quoting = self.quoting,
            memory_map = self.memory_map,
            **self.kwargs
        )

        data_array = data.to_numpy(copy = True)
        del data

        self._index = self._index + 1

        return data_array


    def __getitem__(self, key):
        if key >= self._number_of_chunks:
            raise IndexError((
                f"\n[ERROR]: Tried to read the data chunk at index {key} when "
                f"there are {self._number_of_chunks} chunks (indexed from "
                "0).\n"
            ))

        # Allow negative indices
        while key < 0:
            key += self._number_of_chunks

        data = pd.read_csv(
            self.filepath_or_buffer,
            skiprows = self._skiprows + key * self._chunksize,
            nrows = self._chunksize,
            dtype = self.dtype,
            sep = self.sep,
            engine = self.engine,
            na_filter = self.na_filter,
            quoting = self.quoting,
            memory_map = self.memory_map,
            **self.kwargs
        )

        data_array = data.to_numpy(copy = True)
        del data

        return data_array


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"filepath_or_buffer = {self.filepath_or_buffer}\n"
            f"file_lines =         {self.file_lines}\n\n"
            f"skiprows =           {self.skiprows}\n"
            f"nrows =              {self.nrows}\n\n"
            f"chunksize =          {self.chunksize}\n"
            f"number_of_chunks =   {self.number_of_chunks}"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPL
        docstr = (
            "Class instance that inherits from `pept.utilities.ChunkReader`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n"
            "----------\n"
            f"{self.__str__()}\n"
        )

        return docstr


