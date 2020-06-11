#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2016 Thomas Baruchel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Allow to use tail-call optimized functions in Python code (for
tail-recursion or continuation-passing style).
"""

# This script was taken by Andrei Leonard Nicusan in 2020 from Thomas
# Baruchel's GitHub repository at:
# https://github.com/baruchel/tco
#
# At the README.md's suggestion, the whole library (consisting of a single
# __init__.py file) was copied into this file for ease of use, avoiding the
# addition of another dependency to the pept package.
#
# I express my thanks to Thomas Baruchel for this beautifully minimalistic
# implementation of ad hoc tail call optimisation for Python.

__version__ = '1.2.1'

class _TailCall(Exception):
    def __init__(self, f, args, uid):
        self.func, self.args, self.uid, self.follow = f.func, args, uid, id(f)

def _tailCallback(f, uid):
    """
    This is the "callable" version of the continuation, which sould only
    be accessible from the inside of the function to be continued. An
    attribute called "C" can be used in order to get back the public
    version of the continuation (for passing the continuation to another
    function).
    """
    def t(*args):
        raise _TailCall(f, args, uid)
    t.C = f
    return t

class _TailCallWrapper():
    """
    Wrapper for tail-called optimized functions embedding their
    continuations. Such functions are ready to be evaluated with
    their arguments.
    This is a private class and should never be accessed directly.
    Functions should be created by using the C() class first.
    """
    def __init__(self, func, k):
        self.func = func( _tailCallback(self, id(self)),
                          *map( lambda c: _tailCallback(c, id(self)), k) )
    def __call__(self, *args):
        f, expect = self.func, id(self)
        while True:
            try:
                return f(*args)
            except _TailCall as e:
                if e.uid == expect:
                    f, args, expect = e.func, e.args, e.follow
                else:
                    raise e

class C():
    """
    Main wrapper for tail-call optimized functions.
    """
    def __init__(self, func):
        self.func = func
    def __call__(self, *k):
        return _TailCallWrapper(self.func, k)

def with_continuations(**c):
    """
    A decorator for defining tail-call optimized functions.
    Example
    -------
        @with_continuations()
        def factorial(n, k, self=None):
            return self(n-1, k*n) if n > 1 else k

        @with_continuations()
        def identity(x, self=None):
            return x

        @with_continuations(out=identity)
        def factorial2(n, k, self=None, out=None):
            return self(n-1, k*n) if n > 1 else out(k)
        print(factorial(7,1))
        print(factorial2(7,1))
    """
    if len(c): keys, k = zip(*c.items())
    else: keys, k = tuple([]), tuple([])
    def d(f):
        return C(
            lambda kself, *conts:
                lambda *args:
                    f(*args, self=kself, **dict(zip(keys, conts)))) (*k)
    return d
