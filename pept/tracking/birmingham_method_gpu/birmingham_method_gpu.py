#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
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


# File   : birmingham_method_gpu.py
# License: GNU v3.0
# Author : Dominik Werner
# Date   : 29.12.2022


import  numpy                           as      np
import  os
import  pept
from    pept.base       import   Reducer

import  time
import  textwrap



class BirminghamMethodGPU(Reducer):
    '''The Birmingham Method is an efficient, analytical technique for tracking
    tracers using the LoRs from PEPT data.

    One method is provided: `fit` to track all the samples
    encapsulated in a `pept.LineData` class *in parallel* on GPU.

    For the given `dataset` of LoRs (a pept.LineData), this function minimises
    the distance between all of the LoRs for each `sample`, rejecting a fraction of lines that
    lie furthest away from the calculated distance. The process is repeated
    iteratively until a specified fraction (`fopt`) of the original subset of
    LORs remains.

    This class is a wrapper around the `birmingham_method` subroutine
    (implemented in CUDA). It can return `PointData` classes which can be easily manipulated and
    visualised.

    Attributes
    ----------
    fopt : float
        Floating-point number between 0 and 1, representing the target fraction
        of LoRs in a sample used to locate a tracer.

    memory_usage : float
        The percentage of GPU memory to be used by the algorithm. The value between 0 and 1.
        The algorithm will (over) estimate the amount of memory needed for the computation and
        will allocate the arrays accordingly.  Memory will determine the amount of samples that
        can be processed in parallel and therfore the amount of batches that will be processed.
        The default value is 1.0, which means that the algorithm will use all the available memory.

    threads_per_block : int
        The number of threads per block. The default value is 1024. Depending on the compute
        capability of the GPU, the maximum number of threads per block can be 256, 512 or 1024.



    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    pept.scanners.ParallelScreens : Initialise a `pept.LineData` instance from
                                    parallel screens PEPT detectors.

    Examples
    --------
    A typical workflow would involve reading LoRs from a file, instantiating a
    `BirminghamMethodGPU` class, tracking the tracer locations from the LoRs, and
    plotting them.

    >>> import pept
    >>> from pept.tracking.BirminghamMethodGPU

    >>> lors = pept.LineData(...)   # set sample_size and overlap appropriately
    >>> bham = BirminghamMethodGPU()
    >>> locations = bham.fit(lors)  # this is a `pept.PointData` instance

    >>> grapher = PlotlyGrapher()
    >>> grapher.add_points(locations)
    >>> grapher.show()
    '''

    def __init__(self, fopt = 0.5, memory_usage = 1.0, threads_per_block = 1024):
        '''`BirminghamMethodGPU` class constructor.

        fopt : float, default 0.5
            Float number between 0 and 1, representing the fraction of
            remaining LORs in a sample used to locate the particle.

        memory_usage : float
            The percentage of GPU memory to be used by the algorithm. The value between 0 and 1.
            The algorithm will (over) estimate the amount of memory needed for the computation and
            will allocate the arrays accordingly.  Memory will determine the amount of samples that
            can be processed in parallel and therfore the amount of batches that will be processed.
            The default value is 1.0, which means that the algorithm will use all the available memory.

        threads_per_block : int
            The number of threads per block. The default value is 1024. Depending on the compute
            capability of the GPU, the maximum number of threads per block can be 256, 512 or 1024.


        '''
        # First imort all necessery stuff
        # gpu imports
        # make imports global. There may be a better solution for this
        global  cuda, SourceModule, DynamicSourceModule
        try:
            import  pycuda.autoinit
            import  pycuda.driver                   as      cuda
            from    pycuda.compiler                 import  SourceModule, DynamicSourceModule
        except ModuleNotFoundError as e:
            print(e)
            raise ModuleNotFoundError(
                "pyCUDA not found. Please install pyCUDA to use the GPU version of the Birmingham Method")


        # Use @fopt.setter (below) to do the relevant type-checking when
        # setting fopt (self._fopt is the internal attribute, that we only
        # access through the getter and setter of the self.fopt property).
        self.fopt = float(fopt)
        # initiate GPU first check if GPU is available
        self._gpu_available = cuda.Device.count() > 0
        if not self._gpu_available:
            raise Exception(
                "No Nvidia GPU available. To run the Birmingham \
                Method on GPU you need to have a Nvidia GPU, please check \
                your GPU drivers. If you do not have a Nvidia GPU, please \
                use the CPU version of the Birmingham Method.")
        # initiate GPU function from file located in the same folder
        # find current folder of this file

        current_folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_folder, "birmingham_method.cu"), "r") as f:
            birmingham_method = f.read()
        self._mod = SourceModule(birmingham_method)  # Dynamic , options=["-g"]
        self._birmingham_method = self._mod.get_function("birmingham_method")

        self.threads_per_block = threads_per_block
        self.memory_usage = memory_usage  # in percent



    def fit(
        self,
        lines,
        verbose = True
    ):
        '''Use the Birmingham method on GPU to track a tracer location from a numpy
        array (i.e. one sample) of LoRs.

        For the given `sample` of LoRs (a numpy.ndarray), this function
        minimises the distance between all of the LoRs, rejecting a fraction of
        lines that lie furthest away from the calculated distance. The process
        is repeated iteratively until a specified fraction (`fopt`) of the
        original subset of LORs remains.

        Parameters
        ----------
        lines : pept.LineData
            The collection of LORs that will be clustered.

        verbose : bool, default True
            Print extra information like timings or memory usage.

        Returns
        -------
        locations : pept.PointData
            The tracked locations found with the rows: [t, x, y, z, error]


        Raises
        ------
        TypeError
            if lines is not an instance of `pept.LineData` (or any class
            inheriting from it).
        '''

        if verbose:
            start = time.time()

        if not isinstance(lines, pept.LineData):
            raise TypeError((
                "\n[ERROR]: lines should be an instance of "
                "`pept.LineData` (or any class inheriting from it). Received "
                f"{type(lines)}.\n"
            ))


        nsamples = len(lines)
        nlors_per_sample = lines.sample_size
        overlap = lines.overlap
        ncols = lines[0].lines.shape[1]


        ############ GPU algorithm ############
        # We need to make sure that the GPU memory is not full
        # find out how many samples we can fit in the GPU memory
        # and then split the samples into batches
        (free, _) = cuda.mem_get_info()
        # memory used for calculations:
        # around 30 arrays of that size will be allocated, we calculate 5 extra arrays
        # to not run out of memory
        mem_for_data = 35 * \
            np.empty(nlors_per_sample * nsamples,
                     dtype=np.float32).nbytes * self.memory_usage
        max_samples = int(np.floor(free / (mem_for_data) * nsamples))
        if max_samples > nsamples:
            max_samples = nsamples
        gpu_batches = int(np.ceil(nsamples / max_samples))


        ##### MEMORY ALLLOCATION #####
        # allocate memory for all arrays needed, even intermediate ones
        # set all arrays to zero
        zero_array = np.zeros(
            nlors_per_sample * max_samples, dtype=np.float32)
        size_of_arrays = zero_array.nbytes
        # int(nlors * max_samples * np.float32(1.).nbytes)
        ttt = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(ttt, zero_array)
        xx1 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(xx1, zero_array)
        xx2 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(xx2, zero_array)
        yy1 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(yy1, zero_array)
        yy2 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(yy2, zero_array)
        zz1 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(zz1, zero_array)
        zz2 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(zz2, zero_array)

        x12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(x12, zero_array)
        y12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(y12, zero_array)
        z12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(z12, zero_array)

        r12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(r12, zero_array)
        q12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(q12, zero_array)
        p12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(p12, zero_array)

        a12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(a12, zero_array)
        b12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(b12, zero_array)
        c12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(c12, zero_array)
        d12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(d12, zero_array)
        e12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(e12, zero_array)
        f12 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(f12, zero_array)

        r2 = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(r2, zero_array)
        dev = cuda.mem_alloc(size_of_arrays)
        cuda.memcpy_htod(dev, zero_array)
        used = cuda.mem_alloc(
            int(nlors_per_sample * max_samples * np.int32(1.).nbytes)
        )

        # the result array has 5*nsamples elements
        result = np.zeros(int(5 * max_samples), dtype=np.float32)
        result_gpu = cuda.mem_alloc(result.nbytes)
        cuda.memcpy_htod(result_gpu, result)

        end_locations = np.zeros(
            5 * nsamples, dtype=np.float32).reshape(nsamples, 5)

        (free2, _) = cuda.mem_get_info()
        if verbose:
            string = \
                f"Allocated {(free - free2)/1_000:.2f} kb on GPU in {time.time() - start:.2f} seconds" \
                if (free - free2) / 1_000 < 1000 else \
                f"Allocated {(free - free2)/1_000_000:.2f} mb on GPU in {time.time() - start:.2f} seconds" \
                if (free - free2) / 1_000_000 < 1000 else \
                f"Allocated {(free - free2)/1_000_000_000:.2f} gb on GPU in {time.time() - start:.2f} seconds"
            print(string)

        result_index_start = 0
        result_index_end = 0
        # for loop over all batches to calculate the end locations
        for i in range(gpu_batches):
            if verbose:
                start_batch = time.time()
                print("GPU batch: ", i + 1, " of ", gpu_batches, end='\r')

            # make the array containing all lors for this batch
            index_start = i * max_samples * ncols * \
                (lines.sample_size - lines.overlap)
            index_end = (i + 1) * max_samples * ncols * \
                (lines.sample_size - lines.overlap) + 1
            if index_end > len(lines.lines.flatten()):
                max_samples = nsamples - i * max_samples
                index_end = index_start + max_samples * ncols * \
                    (lines.sample_size - lines.overlap) + 1
            lor_batch = np.asarray(lines.lines.flatten(
            )[index_start:index_end], order='C', dtype=np.float32)
            # allocate memory on gpu and transfer data
            gpu_lor_batch = cuda.mem_alloc(lor_batch.nbytes)
            cuda.memcpy_htod(gpu_lor_batch, lor_batch)
            # figure out how many threads and blocks we need
            max_threads_per_block = self.threads_per_block
            num_blocks = (max_samples + max_threads_per_block -
                          1) // max_threads_per_block
            threads_per_block = max_threads_per_block
            # run the GPU kernel
            self._birmingham_method(
                gpu_lor_batch,
                np.int32(nlors_per_sample),
                np.int32(ncols),
                np.int32(overlap),
                np.int32(max_samples),
                np.float32(self.fopt),
                result_gpu,
                used,
                ttt,
                xx1,
                xx2,
                yy1,
                yy2,
                zz1,
                zz2,
                x12,
                y12,
                z12,
                r12,
                q12,
                p12,
                a12,
                b12,
                c12,
                d12,
                e12,
                f12,
                r2,
                dev,
                block=(threads_per_block, 1, 1),
                grid=(num_blocks, 1),
            )
            # copy the results back to the CPU
            cuda.memcpy_dtoh(result, result_gpu)
            gpu_lor_batch.free()
            # reshape the results
            small_result = result[0:max_samples * 5].copy()
            small_result = small_result.reshape((max_samples, 5))
            # append the results to the output
            result_index_end = result_index_start + max_samples
            end_locations[result_index_start:result_index_end,
                          :] = small_result
            result_index_start += max_samples
            if verbose:
                print("GPU batch: ", i + 1, " of ", gpu_batches,
                      f" finished in {time.time() - start_batch:.2f} seconds")

        # free the GPU memory
        ttt.free()
        xx1.free()
        xx2.free()
        yy1.free()
        yy2.free()
        zz1.free()
        zz2.free()

        x12.free()
        y12.free()
        z12.free()

        r12.free()
        q12.free()
        p12.free()

        a12.free()
        b12.free()
        c12.free()
        d12.free()
        e12.free()
        f12.free()

        r2.free()

        dev.free()
        used.free()
        result_gpu.free()

        if len(end_locations) != 0:
            locations = pept.PointData(
                np.vstack(end_locations),
                columns = ["t", "x", "y", "z", "error"],
                sample_size = 0,
                overlap = 0,
                verbose = False
            )

        if verbose:
            end = time.time()
            print("\nProcessed samples in {} seconds.\n".format(end - start))

        return locations
