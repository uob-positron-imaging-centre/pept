#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utilities.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 08.08.2021


import textwrap


def check_homogeneous_types(iterable):
    if len(iterable) == 0:
        return

    base_type = type(iterable[0])
    for i, e in enumerate(iterable):
        if not isinstance(e, base_type):
            raise TypeError(textwrap.fill((
                "The input iterable must have homogeneous types. The first "
                f"element was of type `{base_type}`, but the element at index "
                f"{i} was of type `{type(e)}`."
            )))


def memoryview_safe(x):
    """Make array safe to run in a Cython memoryview-based kernel. These
    kernels typically break down with the error ``ValueError: buffer source
    array is read-only`` when running in dask distributed or joblib.

    Taken from `https://github.com/dask/distributed/issues/1978`.
    """
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x
