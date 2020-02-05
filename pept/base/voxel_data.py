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


# File   : voxel_data.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 07.01.2020


import  time
import  numpy                   as      np

import  plotly.graph_objects    as      go

import  matplotlib
import  matplotlib.pyplot       as      plt
from    matplotlib.colors       import  Normalize
from    mpl_toolkits.mplot3d    import  Axes3D

from    pept.utilities.traverse import  traverse3d


class VoxelData:


    def __init__(
        self,
        line_data,
        volume_limits = [500., 500., 500.],
        number_of_voxels = [10, 10, 10],
        traverse = True,
        verbose = False
    ):

        if verbose:
            start = time.time()

        # If `line_data` is not C-contiguous, create a C-contiguous copy
        self._line_data = np.asarray(line_data, order = 'C', dtype = float)
        # Check that line_data has shape (N, 7)
        if self._line_data.ndim != 2 or self._line_data.shape[1] != 7:
            raise ValueError('\n[ERROR]: line_data should have dimensions (N, 7). Received {}\n'.format(self._line_data.shape))

        self._number_of_lines = len(self._line_data)

        # If `volume_limits` is not C-contiguous, create a C-contiguous copy
        self._volume_limits = np.asarray(volume_limits, dtype = float, order = "C")
        # Check that volume_limits has shape (3,)
        if self._volume_limits.ndim != 1 or self._volume_limits.shape[0] != 3:
            raise ValueError("\n[ERROR]: volume_limits should have dimensions (3,). Received {}\n".format(self._volume_limits.shape))

        # If `number_of_voxels` is not C-contiguous, create a C-contiguous copy
        self._number_of_voxels = np.asarray(number_of_voxels, dtype = int, order = "C")
        # Check that number_of_voxels has shape(3,)
        if self._number_of_voxels.ndim != 1 or self._number_of_voxels.shape[0] != 3:
            raise ValueError("\n[ERROR]: number_of_voxels should have dimensions (3,). Received {}\n".format(self._number_of_voxels.shape))

        self._voxel_sizes = self._volume_limits / self._number_of_voxels

        # If, for dimension x, there are 5 voxels between coordinates 0
        # and 5, then the delimiting grid is [0, 1, 2, 3, 4, 5].
        self._voxel_grid = [np.linspace(0, self._volume_limits[i], self._number_of_voxels[i] + 1) for i in range(3)]

        # All access to voxel_positions will be done directly through the inner
        # class _VoxelPositions, so no need for a private property here
        self.voxel_positions = self._VoxelPositions(self._volume_limits, self._number_of_voxels)
        self._voxel_data = np.zeros(self._number_of_voxels, dtype = int)

        if traverse:
            if verbose:
                start_traverse = time.time()

            if traverse == True:
                self.traverse()
            else:
                self.traverse(traverse)

            if verbose:
                end_traverse = time.time()

        if verbose:
            end = time.time()
            print("Initialising the instance of VoxelData took {} seconds.\n".format(end - start))
            if traverse:
                print("Traversing all voxels took {} seconds.\n".format(end_traverse - start_traverse))


    class _VoxelPositions:

        def __init__(self, volume_limits, number_of_voxels):

            self.volume_limits = np.asarray(volume_limits, dtype = float, order = "C")
            self.number_of_voxels = np.asarray(number_of_voxels, dtype = int, order = "C")
            self.voxel_sizes = self.volume_limits / self.number_of_voxels

            self._index = 0


        def at(self, ix, iy, iz):
            # Evaluate the position of the voxel (the centre of it) at indices
            # [ix, iy, iz]

            indices = np.array([ix, iy, iz], dtype = int)

            if (indices >= self.number_of_voxels).any() or (indices < 0).any():
                raise IndexError("[ERROR]: Each of the [ix, iy, iz] indices must be between 0 and the corresponding `number_of_voxels`.")

            return self._at(indices)


        def _at(self, indices):
            # Unchecked!
            return self.voxel_sizes * (0.5 + indices)


        def at_corner(self, ix, iy, iz):
            # Evaluate the position of the voxel (the corner of it) at indices
            # [ix, iy, iz]

            indices = np.array([ix, iy, iz], dtype = int)

            if (indices >= self.number_of_voxels).any() or (indices < 0).any():
                raise IndexError("[ERROR]: Each of the [ix, iy, iz] indices must be between 0 and the corresponding `number_of_voxels`.")

            return self._at_corner(indices)


        def _at_corner(self, indices):
            # Unchecked!
            return self.voxel_sizes * indices


        def all(self):

            positions = []
            for i in range(self.number_of_voxels[0]):
                for j in range(self.number_of_voxels[1]):
                    for k in range(self.number_of_voxels[2]):
                        positions.append(self._at(np.array([i, j, k])))

            return np.array(positions)


        def __len__(self):
            return self.number_of_voxels[0]


        def __getitem__(self, key):

            if not isinstance(key, tuple):
                key = (key,)

            if len(key) > 3:
                raise ValueError("[ERROR]: The accessor takes maximum 3 indices, {} were given.".format(len(key)))

            # Calculate the starting and ending indices and the step for the
            # [x, y, z] coordinates of all the elements that are accessed.
            # The default (:, :, :) is the whole range.
            start = [0, 0, 0]
            stop = list(self.number_of_voxels)
            step = [1, 1, 1]

            # The ranges of data selection for each dimension. Default is a
            # range, but can be an explicit list too (e.g. select elements
            # [1,2,5]).
            xyz_ranges = [range(stop[i]) for i in range(3)]

            # Handles negative indices for each of the 3 dimensions.
            def make_positive(index, dimension):
                while index < 0:
                    index += self.number_of_voxels[dimension]
                return index

            # Interpret each key
            for i in range(len(key)):
                # If key[i] is an int, only access the elements at that index,
                # equivalent to range(key[i], key[i] + 1, 1).
                if isinstance(key[i], (int, np.integer)):
                    if key[i] >= self.number_of_voxels[i]:
                        raise IndexError("[ERROR]: Tried to access voxel number {} (indexed from 0), when there are {} voxels for dimension {}.".format(key[i], self.number_of_voxels[i], i))

                    index = make_positive(key[i], i)
                    start[i] = index
                    stop[i] = index + 1

                    xyz_ranges[i] = range(start[i], stop[i], step[i])

                # Interpret the possible slices (1:5, ::-1, etc.).
                elif isinstance(key[i], slice):
                    # First interpret the step for the ::-1 corner case.
                    if key[i].step is not None:
                        if not isinstance(key[i].step, (int, np.integer)):
                            raise TypeError("Slice step must be an int. Received {}.".format(type(key[i].step)))
                        if key[i].step == 0:
                            raise ValueError("Slice step cannot be zero.")
                        elif key[i].step < 0:
                            # If the step is negative, the default start and
                            # stop become (max_index - 1) and -1, such that
                            # ::-1 works.
                            start[i] = self.number_of_voxels[i] - 1
                            stop[i] = -1
                            step[i] = key[i].step
                        else:
                            step[i] = key[i].step

                    if key[i].start is not None:
                        if not isinstance(key[i].start, (int, np.integer)):
                            raise TypeError("Slice start must be an int. Received {}.".format(type(key[i].start)))
                        # Corner case: x = [1,2,3] => x[5:10] == []
                        start[i] = min(make_positive(key[i].start, i), self.number_of_voxels[i])

                    if key[i].stop is not None:
                        if not isinstance(key[i].stop, (int, np.integer)):
                            raise TypeError("Slice stop must be an int. Received {}.".format(type(key[i].stop)))
                        # Corner case: x = [1,2,3] => x[5:10] == []
                        stop[i] = min(make_positive(key[i].stop, i), self.number_of_voxels[i])

                    xyz_ranges[i] = range(start[i], stop[i], step[i])

                # Interpret iterable sequence of selected elements
                elif hasattr(key[i], "__iter__"):
                    xyz_ranges[i] = np.asarray(key[i], dtype = int)

                else:
                    raise TypeError("Indices must be either `int`, `slice` or iterable of `int`s. Received {}.".format(type(key[i])))

            positions = []
            # Iterate through all the elements that need to be accessed
            for x in xyz_ranges[0]:
                for y in xyz_ranges[1]:
                    for z in xyz_ranges[2]:
                        positions.append(self._at(np.array([x, y, z])))

            if len(positions) == 1:
                return positions[0]
            else:
                return np.array(positions)


        def __iter__(self):
            return self


        def __next__(self):
            if self._index >= len(self):
                self._index = 0
                raise StopIteration

            self._index += 1
            return self[self._index - 1]


    @property
    def line_data(self):
        return self._line_data


    @property
    def number_of_lines(self):
        return self._number_of_lines


    @property
    def volume_limits(self):
        return self._volume_limits


    @volume_limits.setter
    def volume_limits(self, volume_limits):
        # If `volume_limits` is not C-contiguous, create a C-contiguous copy
        self._volume_limits = np.asarray(volume_limits, dtype = float, order = "C")
        # Check that volume_limits has shape (3,)
        if self._volume_limits.ndim != 1 or self._volume_limits.shape[0] != 3:
            raise ValueError("\n[ERROR]: volume_limits should have dimensions (3,). Received {}\n".format(self._volume_limits.shape))

        self._voxel_sizes = self._volume_limits / self._number_of_voxels

        # If, for dimension x, there are 5 voxels between coordinates 0
        # and 5, then the delimiting grid is [0, 1, 2, 3, 4, 5].
        self._voxel_grid = [np.linspace(0, self._volume_limits[i], self._number_of_voxels[i] + 1) for i in range(3)]

        # All access to voxel_positions will be done directly through the inner
        # class _VoxelPositions, so no need for a private property here
        self.voxel_positions = self._VoxelPositions(self._volume_limits, self._number_of_voxels)
        self._voxel_data = np.zeros(self._number_of_voxels, dtype = int)


    @property
    def number_of_voxels(self):
        return self._number_of_voxels


    @number_of_voxels.setter
    def number_of_voxels(self, number_of_voxels):
        # If `number_of_voxels` is not C-contiguous, create a C-contiguous copy
        self._number_of_voxels = np.asarray(number_of_voxels, dtype = int, order = "C")
        # Check that number_of_voxels has shape(3,)
        if self._number_of_voxels.ndim != 1 or self._number_of_voxels.shape[0] != 3:
            raise ValueError("\n[ERROR]: number_of_voxels should have dimensions (3,). Received {}\n".format(self._number_of_voxels.shape))

        self._voxel_sizes = self._volume_limits / self._number_of_voxels

        # If, for dimension x, there are 5 voxels between coordinates 0
        # and 5, then the delimiting grid is [0, 1, 2, 3, 4, 5].
        self._voxel_grid = [np.linspace(0, self._volume_limits[i], self._number_of_voxels[i] + 1) for i in range(3)]

        # All access to voxel_positions will be done directly through the inner
        # class _VoxelPositions, so no need for a private property here
        self.voxel_positions = self._VoxelPositions(self._volume_limits, self._number_of_voxels)
        self._voxel_data = np.zeros(self._number_of_voxels, dtype = int)


    @property
    def voxel_sizes(self):
        return self._voxel_sizes


    @property
    def voxel_grid(self):
        return self._voxel_grid


    @property
    def voxel_data(self):
        return self._voxel_data


    def traverse_python(self, lor_indices = None):
        # Adapted from "A Fast Voxel Traversal Algorithm for Ray Tracing" by
        # John Amanatides and Andrew Woo.

        # Traverse voxels for all LoRs by default
        if lor_indices is None:
            lor_indices = range(self._number_of_lines)

        if not hasattr(lor_indices, "__iter__"):
            raise TypeError("[ERROR]: The `lor_indices` parameter must be iterable.")

        # The adapted grid traversal algorithm
        for li in lor_indices:
            # Define a line as L(t) = U + t V
            # If an LoR is defined as two points P1 and P2, then
            # U = P1 and V = P2 - P1
            p1 = self._line_data[li, 1:4]
            p2 = self._line_data[li, 4:7]
            u = p1
            v = p2 - p1

            ##############################################################
            # Initialisation stage

            # The step [sx, sy, sz] defines the sense of the LoR.
            # If V[0] is positive, then sx = 1
            # If V[0] is negative, then sx = -1
            step = np.array([1, 1, 1], dtype = int)
            for i, c in enumerate(v):
                if c < 0:
                    step[i] = -1

            # The current voxel indices [ix, iy, iz] that the line passes
            # through.
            voxel_index = np.array([0, 0, 0], dtype = int)

            # The value of t at which the line passes through to the next
            # voxel, for each dimension.
            t_next = np.array([0., 0., 0.], dtype = float)

            # Find the initial voxel that the line starts from, for each
            # dimension.
            for i in range(len(voxel_index)):
                # If, for dimension x, there are 5 voxels between coordinates 0
                # and 5, then the delimiting grid is [0, 1, 2, 3, 4, 5].
                # If the line starts at 1.5, then it is part of the voxel at
                # index 1.
                voxel_index[i] = np.searchsorted(self._voxel_grid[i], u[i], side = "right") - 1

                # If the line is going "up", the next voxel is the next one
                if v[i] >= 0:
                    offset = 1
                # If the line is going "down", the next voxel is the current one
                else:
                    offset = 0
                t_next[i] = (self._voxel_grid[i][voxel_index[i] + offset] - u[i]) / v[i]

            # delta_t indicates how far along the ray we must move (in units of
            # t) for each component to be equal to the size of the voxel in
            # that dimension.
            delta_t = np.abs(self._voxel_sizes / v)

            ###############################################################
            # Incremental traversal stage

            # Loop until we reach the last voxel in space
            while (voxel_index < self._number_of_voxels).all() and (voxel_index >= 0).all():

                self._voxel_data[tuple(voxel_index)] += 1

                # If p2 is fully bounded by the voxel, stop the algorithm
                if ((self.voxel_positions._at_corner(voxel_index) < p2).all() and
                    (self.voxel_positions._at_corner(voxel_index + 1) > p2).all()):
                    break

                # The dimension of the minimum t that makes the line pass
                # through to the next voxel
                min_i = t_next.argmin()
                t_next[min_i] += delta_t[min_i]
                voxel_index[min_i] += step[min_i]


    def traverse(self, lor_indices = None):
        # Traverse all intersecting voxels for selected LoRs.

        # Traverse voxels for all LoRs by default
        if lor_indices is None:
            lor_indices = range(self._number_of_lines)

        if not hasattr(lor_indices, "__iter__"):
            raise TypeError("[ERROR]: The `lor_indices` parameter must be iterable.")

        traverse3d(
            self._voxel_data,
            self._line_data[lor_indices],
            self._voxel_grid[0],
            self._voxel_grid[1],
            self._voxel_grid[2]
        )


    def indices(self, coords):
        # Find the voxel indices for a point at `coords`
        coords = np.asarray(coords, dtype = float)
        if coords.ndim != 1 or coords.shape[0] != 3:
            raise ValueError("The `coords` parameter must have shape (3,). Received {}.".format(coords))

        indices = np.array([0, 0, 0], dtype = int)
        for i in range(3):
            indices[i] = np.searchsorted(self._voxel_grid[i], coords[i], side = "right") - 1

        return indices


    def cube_trace(self, index, opacity = 0.4, color = None, colorbar = False):
        # For a small number of cubes

        index = np.asarray(index, dtype = int)
        xyz = self.voxel_positions.at_corner(*index)

        x = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * self._voxel_sizes[0]
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * self._voxel_sizes[1]
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * self._voxel_sizes[2]
        i = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
        j = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
        k = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])

        cube = dict(
            x =  x + xyz[0],
            y =  y + xyz[1],
            z =  z + xyz[2],
            i =  i,
            j =  j,
            k =  k,
            opacity = opacity,
            color = color
        )

        if colorbar:
            cmap = matplotlib.cm.get_cmap("magma")
            c = cmap(self._voxel_data[tuple(index)] / (self._voxel_data.max() or 1))
            cube.update(
                color = "rgb({},{},{})".format(c[0], c[1], c[2])
            )

        return go.Mesh3d(cube)


    def cubes_traces(
        self,
        condition = lambda voxel_data: voxel_data > 0,
        opacity = 0.4,
        color = None,
        colorbar = False
    ):
        # For a small number of cubes

        indices = np.argwhere(condition(self._voxel_data))
        traces = [self.cube_trace(i, opacity = opacity, color = color, colorbar = colorbar) for i in indices]

        return traces


    def voxels_trace(
        self,
        condition = lambda voxel_data: voxel_data > 0,
        size = 4,
        opacity = 0.4,
        color = None,
        colorbar = False
    ):
        # For a large number of cubes

        filtered_indices = np.argwhere(condition(self._voxel_data))
        positions = self.voxel_positions._at(filtered_indices)

        marker = dict(
            size = size,
            color = color,
            symbol = "square"
        )

        if colorbar:
            cvalues = [self._voxel_data[tuple(t)] for t in filtered_indices]
            marker.update(colorscale = "Magma", color = cvalues)

        voxels = dict(
            x = positions[:, 0],
            y = positions[:, 1],
            z = positions[:, 2],
            opacity = opacity,
            mode = "markers",
            marker = marker
        )

        return go.Scatter3d(voxels)


    def heatmap_trace(
        self,
        ix = None,
        iy = None,
        iz = None,
        width = 0
    ):

        if ix is not None:
            x = self._voxel_grid[1]
            y = self._voxel_grid[2]
            z = self._voxel_data[ix, :, :]

            for i in range(1, width + 1):
                z = z + self._voxel_data[ix + i, :, :]
                z = z + self._voxel_data[ix - i, :, :]

        elif iy is not None:
            x = self._voxel_grid[0]
            y = self._voxel_grid[2]
            z = self._voxel_data[:, iy, :]

            for i in range(1, width + 1):
                z = z + self._voxel_data[:, iy + i, :]
                z = z + self._voxel_data[:, iy - i, :]

        elif iz is not None:
            x = self._voxel_grid[0]
            y = self._voxel_grid[1]
            z = self._voxel_data[:, :, iz]

            for i in range(1, width + 1):
                z = z + self._voxel_data[:, :, iz + i]
                z = z + self._voxel_data[:, :, iz - i]

        else:
            raise ValueError("[ERROR]: One of the `ix`, `iy`, `iz` slice indices must be provided.")

        heatmap = dict(
            x = x,
            y = y,
            z = z,
            colorscale = "Magma",
            transpose = True
        )

        return go.Heatmap(heatmap)


    def __str__(self):
        # Shown when calling print(class)
        docstr = ""

        docstr += "number_of_lines =   {}\n\n".format(self.number_of_lines)
        docstr += "volume_limits =     {}\n".format(self.volume_limits)
        docstr += "number_of_voxels =  {}\n".format(self.number_of_voxels)
        docstr += "voxel_sizes =       {}\n\n".format(self.voxel_sizes)

        docstr += "line_data = \n"
        docstr += self._line_data.__str__()

        docstr += "\n\nvoxel_data = \n"
        docstr += self._voxel_data.__str__()

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR

        docstr = "Class instance that inherits from `pept.VoxelData`.\n\n" + self.__str__() + "\n\n"

        return docstr


