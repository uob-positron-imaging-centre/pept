#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019 Sam Manger
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


import matplotlib.pyplot as plt
import numpy as np
import pept
import os
import matplotlib.lines as mlines
from    joblib                                  import      Parallel,       delayed
from    tqdm                                    import      tqdm
import time

from .extensions.birmingham_method import birmingham_method

class BirminghamMethod:
	'''

	The Birmingham Method is an analytical technique for tracking particles using PEPT data from LORs.

	'''

	def __init__(self,
				 fopt = 0.5):
		self.fopt = fopt
		print("Initialised BirminghamMethod")

	def track_sample(self,
		sample,
		fopt = None,
		as_array = True,
		verbose = False):

		'''The Birmingham Method for particle tracking. For the given pept.LineData, the method 
		takes a sample of LORs (defined by sample_size) and minimises the distance between all 
		of the LORs, rejecting a fraction of lines that lie furthest away from the calculated 
		distance. The process is repeated iteratively until a specified fraction (fopt) of the
		original subset of LORs remains.

		Parameters
		----------
		sample : (N, M=7) numpy.ndarray
			The sample of LORs that will be clustered. Each LOR is expressed as a row
			and is formatted as `[time, x1, y1, z1, x2, y2, z2]`.
		fopt   : float, optional
			Fraction of remaining LORs in a sample used to locate the particle
		as_array : bool, optional
			If set to True, the tracked locations are
			returned as numpy arrays. If set to False, they are returned inside
			instances of `pept.PointData`.
		verbose : bool, optional
			Provide extra information when computing the cutpoints: time the operation
			and show a progress bar. The default is `False`.

		Returns
		-------
		locations : numpy.ndarray or pept.PointData
			The tracked locations found
		used	  : numpy.ndarray
			If as_array is true, then an index of the LORs used to 
			locate the particle is returned
			[ used for multi-particle tracking, not implemented yet]

		Raises
		------
		TypeError
			If `sample` is not a numpy array of shape (N, M), where M = 7.

		'''

		if verbose:
			start = time.process_time()

		if fopt == None:
			fopt = self.fopt

		# sample row: [time, x1, y1, z1, x2, y2, z2]
		if sample.ndim != 2 or sample.shape[1] != 7:
			raise TypeError('\n[ERROR]: sample should have two dimensions (M, N), where N = 7. Received {}\n'.format(sample.shape))
	
	
		location, used = birmingham_method(sample, fopt)

		if verbose:
			end = time.process_time()
			print("Tracking one location with %i LORs took %.3f seconds" % (sample.shape[0], end-start))
	
		if not as_array:
			location = pept.PointData(location,
									 sample_size = 0,
									 overlap = 0,
									 verbose = False)
			return location
		else:
			return location


	def track(self,
		LORs,
		fopt = None,
		err_max = 10,
		verbose = False):

		'''Fit lines of response (an instance of 'LineData') and return the tracked locations.

		Parameters
		----------
		LORs : (N, M=7) numpy.ndarray
			The sample of LORs that will be clustered. Each LOR is expressed as a row
			and is formatted as `[time, x1, y1, z1, x2, y2, z2]`.
		fopt   : float, optional
			Fraction of remaining LORs in a sample used to locate the particle
		err_max : float, default = 10
			The maximum error allowed to return a 'valid' tracked location
		as_array : bool, optional
			If set to True, the tracked locations are
			returned as numpy arrays. If set to False, they are returned inside
			instances of `pept.PointData`.
		verbose : bool, optional
			Provide extra information when computing the cutpoints: time the operation
			and show a progress bar. The default is `False`.

		Returns
		-------
		locations : pept.PointData
			The tracked locations found

		Raises
		------
		Exception
			If 'LORs' is not an instance of 'pept.LineData'

		'''

		if verbose:
			start = time.process_time()

		if not isinstance(LORs, pept.LineData):
			raise Exception('[ERROR]: LORs should be an instance of pept.LineData (or any class inheriting from it)')

		# Calculate all samples in 'LORs' in parallel using joblib
		# Collect all outputs as a list. If verbose, show progress bar with
		# tqdm
		if verbose:
			data_list = Parallel(n_jobs = -1)(delayed(self.track_sample)(sample,
												fopt = fopt,
												as_array = True) for sample in tqdm(LORs))
		else:
			data_list = Parallel(n_jobs = -1)(delayed(self.track_sample)(sample,
												fopt = fopt,
												as_array = True) for sample in LORs)

		# Access joblib.Parallel output as list comprehensions

		# return data_list

		# # Remove LORs with error above max
		data_list = np.vstack(data_list)
		cond = np.where(data_list[:,4] > err_max)
		data_list = np.delete(data_list, cond, axis=0) 

		locations = pept.PointData(data_list,
									 sample_size = 0,
									 overlap = 0,
									 verbose = False)



		if verbose:
			end = time.process_time()
			print("\nTracking locations took {} seconds\n".format(end - start))

		return locations

	def find_fopt(self, static_line_data, show_graph=True, sample_size=250, fopt=np.arange(0.05,0.90,0.05), verbose = True):
	
		''' Function returns the optimal fopt parameter for the given dataset

		Parameters
		----------
		static_line_data : instance of pept.LineData
			LORs corresponding to a static particle. The LORs are tracked using the Birmingham Method through a range of fopt
			values in order to determine the 'best' fopt value.

		Returns
		-------
		self.f_opt	: float
			A value between 0 and 1 corresponding to the best fopt value for the given dataset. The best value is chosen by
			minimising the standard deviation of tracked positions over the static dataset. The value may be overwritten using
			a setter on this class.

		Raises
		------
		Exception
			If 'static_line_data' is not an instance of 'pept.LineData'

		'''

		# Check line_data is an instance (or a subclass!) of pept.LineData
		if not isinstance(static_line_data, pept.LineData):
			raise Exception('[ERROR]: line_data should be an instance of pept.LineData')

		self._static_data = static_line_data

		self._static_data.sample_size = sample_size

		std_dev_min = 999
		f_best = 0

		fig, ax = plt.subplots(1,1,figsize=(5,5))

		for f in fopt:
			# print(f)
			locations = []
			for n in range(1,self._static_data.number_of_samples):
				LORs = self._static_data.sample_n(n)
				location, used = birmingham_method(LORs, f)
				locations.append(location)

			locations = np.array(locations)

			x = locations[:,1]
			y = locations[:,2]
			z = locations[:,3]

			err = locations[:,4]
			
			std_dev = np.sqrt(x.std()**2 + y.std()**2 + z.std()**2)
			# ms = (std_dev**2) * (200/(15**2))

			if verbose:
				print("\nMean positions for fopt = %.2f are: " % f)
				print("x = %.2f mm +/- %.2f \t  y = %.2f mm +/- %.2f \t z = %.2f mm +/- %.2f" % (x.mean(), x.std(), y.mean(), y.std(), z.mean(), z.std()))
				print("So the mean error is: %.2f mm" % std_dev)
				print("The precision of the measurements is: %.2f mm\n" % err.mean())

			accuracy = ax.plot(f,std_dev, c = 'r', marker='o',label="Accuracy",linestyle='')
			precision = ax.plot(f,err.mean(), c = 'b', marker='s', label="Precision",linestyle='')

			if std_dev < std_dev_min:
				std_dev_min = std_dev
				f_best = f

		ax.axvline(f_best,0,50)

		ax.set_xlabel('fopt')
		ax.set_ylabel('mm')

		ax.set_ylim([0,25])

		red = mlines.Line2D([],[], color='red', marker='o', linestyle='', label = "Accuracy")
		blue = mlines.Line2D([],[], color='blue', marker='s', linestyle='', label = "Precision")

		plt.legend(handles=[red,blue])

		self._fopt = f_best

	@property
	def fopt(self):
		'''The fraction of LORs used to locate a particle

		fopt : instance of pept.LineData
		'''	
		return self._fopt

	@fopt.setter
	def fopt(self, new_fopt):
		'''The fraction of LORs used to locate a particle

		Parameters
		----------

		fopt : float between 0 and 1
			The fraction of original LORs used to produce a final location for a given number of LORs

		Raises
		------
		ValueError
			If 'fopt' is set to be less than 0 or greater than 1.
		
		'''

		if (new_fopt > 1) or (new_fopt < 0):
			raise ValueError('[ERROR]: fopt should be set between 0 and 1')

		self._fopt = new_fopt


	#def track(self, line_data = None, sample_no=None, sample_size=250, fopt=None, as_array=False):
		'''The main 'tracking' method in the Birmingham method. For the given pept.LineData, the
		method takes a sample of LORs (defined by sample_size) and minimises the distance between
		all of the LORs, rejecting a fraction of lines that lie furthest away from the calculated 
		distance. The process is repeated iteratively until a specified fraction (fopt) of the
		original subset of LORs remains. The method returns a set of locations either as an array
		or as an instance of pept.PointData and an array of integers corresponding to the lines
		that were used to locate the particle.

		Parameters
		----------

		sample_no:
			The subset of LORs used to determine a location

		sample_size: int
			The number of LORs used to determine a location

		fopt : float between 0 and 1
			The fraction of original LORs used to produce a final location for a given number of LORs

		as_array: bool, default False
			If true, return the tracked location as a numpy array. Else, return the location as
			a point data object.

		Returns
		-------

		locations:

		used:

		'''	
		'''	
		if fopt != None:
			self._fopt = fopt

		self._line_data.sample_size = sample_size
		self._line_data.sample_size = sample_size
		
		locations = []

		if sample_no == None:
			for LORs in self._line_data:
				location, used = birmingham_method(LORs, self.fopt)
				locations.append(location)
		else:
			LORs = self._line_data.sample_n(sample_no)
			locations, used = birmingham_method(LORs, self.fopt)

		locations = np.vstack(locations)

		self._line_data
		
		if not as_array:
			locations = pept.PointData(locations,
									 sample_size = 0,
									 overlap = 0,
									 verbose = False)
		# else:
			# locations = np.(locations)

		return locations
		'''