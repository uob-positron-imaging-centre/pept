#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019 Andrei Leonard Nicusan
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


# File   : modular_camera.py
# License: License: GNU v3.0
# Author : Sam Manger <s.manger@bham.ac.uk>
# Date   : 20.08.2019


import  time
import  numpy   as      np
from    pept    import  LineData
from 	.extensions.get_pept_event import get_pept_LOR


class ModularCamera(LineData):
	'''A subclass of `LineData` that reads PEPT data from the modular camera DAQ.

	Provides the same functionality as the `LineData` class while
	initialising `line_data` from a given file. This is a helper class
	for **PEPT using the modular camera**.

	Can read data from a `.da_1` file or equivalent. The file must contain
	the standard datawords from the modular camera output. This will then
	be automatically transformed into the standard `_line_data` format
	with every row being `[time, x1, y1, z1, x2, y2, z2]`, where the geometry
	is derived from the C-extension. The current useable geometry is a square
	layout with 4 stacks for 4 modules, separated by 250 mm.

	Parameters
	----------
	dataFile : str
		A string with the (absolute or relative) path to the data file
		from which the PEPT data will be read. It should include the
		full file name, along with the extension (.da_1)
	sample_size : int, optional
		An `int`` that defines the number of lines that should be
		returned when iterating over `_line_data`. A `sample_size` of 0
		yields all the data as one single sample. (Default is 200)
	overlap : int, optional
		An `int` that defines the overlap between two consecutive
		samples that are returned when iterating over `_line_data`.
		An overlap of 0 means consecutive samples, while an overlap
		of (`sample_size` - 1) means incrementing the samples by one.
		A negative overlap means skipping values between samples. An
		error is raised if `overlap` is larger than or equal to
		`sample_size`. (Default is 0)
	filtered : list, optional
		A list of 'int's of module pair numbers that will be filtered
		from the PEPT data. For use when noisy data affects tracking.
		(Default is [])
	verbose : bool, optional
		An option that enables printing the time taken for the
		initialisation of an instance of the class. Useful when
		reading large files (10gb files for PEPT data is not unheard
		of). (Default is True)

	Attributes
	----------
	_line_data : (N, 7) numpy.ndarray
		An (N, 7) numpy array that stores the PEPT LoRs as time and
		cartesian (3D) coordinates of two points defining a line, **in mm**.
		Each row is then `[time, x1, y1, z1, x2, y2, z2]`.
	sample_size : int
		An `int` that defines the number of lines that should be
		returned when iterating over `_line_data`. (Default is 200)
	overlap : int
		An `int` that defines the overlap between two consecutive
		samples that are returned when iterating over `_line_data`.
		An overlap of 0 means consecutive samples, while an overlap
		of (`sample_size` - 1) means incrementing the samples by one.
		A negative overlap means skipping values between samples. It
		has to be smaller than `sample_size`. (Default is 0)
	numberOfLines : int
		An `int` that corresponds to len(`_line_data`), or the number of
		LoRs stored by `_line_data`.

	Raises
	------
	ValueError
		If `overlap` >= `sample_size`. Overlap has to be smaller than
		`sample_size`. Note that it can also be negative.
	ValueError
		If the data file does not have (N, 7) shape.

	Notes
	-----
	The class saves `line_data` as a **contiguous** numpy array for
	efficient access in C functions. It should not be changed after
	instantiating the class.

	'''

	def __init__(self,
				 dataFile,
				 sample_size = 200,
				 overlap = 0,
				 filtered =[],
				 verbose = True):

		if verbose:
			start = time.time()

		if sample_size != 0 and overlap >= sample_size:
			raise ValueError('\n[ERROR]: overlap = {} must be smaller than sample_size = {}\n'.format(overlap, sample_size))

		self._index = 0
		self._sample_size = sample_size
		self._overlap = overlap

		self.dataFile = dataFile
		self.filtered = []

		x = 10

		header_buffer_size = 1000

		self.n_events = 0

		if sample_size == 0:
			sample_size = int( (os.path.getsize(self.dataFile) - header_buffer_size) /4 ) # Set the sample size to the expected number of events
			self.overlap = -1


		# Modular camera data reader requires 'itag' for timing. We will drop this column at the end of initialisation
		# Row: [itag, itime, X1, Y1, Z1, X2, Y2, Z2]

		self._line_data = np.zeros([sample_size, 8])

		with open(self.dataFile,"rb") as f:

			# Skip over the header and handshake word

			f.seek(header_buffer_size)

			word = f.read(4)

			if word.hex() == 'cefacefa':
				# Skip two words
				word = f.read(4)
				word = f.read(4)

			itime = 0
			itag = 0

			BufTime = 0
			nBuf = 0

			while word != b'' and (self.n_events < sample_size):

				word = f.read(4)

				if word.hex() == 'cefacefa': # Handshake word
					# Skip two words
					word = f.read(4)
					word = f.read(4)

				if word!=b'':
					word = int.from_bytes(word, "little")

					self._line_data[self.n_events,:] = get_pept_LOR(word,itag,itime) # C function

					itag  = self._line_data[self.n_events,1]
					itime = self._line_data[self.n_events,1]

					self.n_events = self.n_events + 1

					if (self.n_events % x) == 0:
						print("Got ", self.n_events,"\n")
						x = x * 10

		# Remove 'zero' lines
		self._line_data = self._line_data[np.all(self._line_data,axis=1)]

		# Drop itag column
		self._line_data = np.delete(self._line_data, 0, axis=1)

		self._number_of_lines = len(self._line_data)

		if verbose:
			end = time.time()
			print("Initialising the PEPT data took {} seconds\n".format(end - start))
