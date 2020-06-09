#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_base_classes.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.08.2019

#from    nose    import  with_setup

from    nose.tools  import  raises

import  pept
import  numpy   as      np


def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("setup_module before anything in this file")

def teardown_module(module):
    print ("teardown_module after everything in this file")


class TestLineData:

    # Run when class is created (before running any method)
    @classmethod
    def setup_class(cls):
        cls.good_data = np.arange(2800).reshape(400, 7)
        cls.bad_data = np.arange(3500).reshape(5, 100, 7)


    # Run when class is destroyed (after running all methods)
    @classmethod
    def teardown_class(cls):
        pass


    # Run before each class method
    def setup(self):
        pass


    # Run after each class method
    def teardown(self):
        pass


    def test_good_data(self):
        samples = pept.LineData(self.good_data, sample_size = 200, overlap = 10, verbose = False)
        # Test private attributes
        assert samples._index == 0, "_index was not set to 0"
        assert samples._overlap == 10, "_overlap was not set correctly"
        assert samples._sample_size == 200, "_sample_size was not set correctly"
        assert np.array_equal(samples._lines, self.good_data) == True
        assert samples._lines.flags['C_CONTIGUOUS'] == True, "_lines is not C-contiguous"
        assert samples._number_of_lines == len(samples._lines), "_number_of_lines was not set correctly"

        # Test properties
        assert np.array_equal(samples.lines, samples._lines) == True
        assert samples.sample_size == samples._sample_size
        assert samples.overlap == samples._overlap
        assert samples.number_of_samples == 2, "number of samples was not calculated correctly"
        assert samples.number_of_lines == samples._number_of_lines

        # Test property setters
        samples.sample_size = 300
        assert samples.sample_size == 300
        assert samples._index == 0
        samples.sample_size = 200

        samples.overlap = 50
        assert samples.overlap == 50
        assert samples._index == 0
        samples.overlap = 10


    @raises(ValueError)
    def test_error_sample_size(self):
        samples = pept.LineData(self.good_data, sample_size = 200, overlap = 100, verbose = False)
        # Should not be able to set sample_size <= overlap
        samples.sample_size = 100


    @raises(ValueError)
    def test_error_overlap(self):
        samples = pept.LineData(self.good_data, sample_size = 200, overlap = 100, verbose = False)
        # Should not be able to set sample_size <= overlap
        samples.overlap = 200


    def test_sample_n(self):
        pass


