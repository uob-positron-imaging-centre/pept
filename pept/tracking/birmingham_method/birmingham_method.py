import matplotlib.pyplot as plt
import numpy as np
import pept
import os
import matplotlib.lines as mlines

from .extensions.birmingham_method import birmingham_method

class BirminghamMethod():
	'''

	The Birmingham Method is an analytical technique for tracking particles using PEPT data from LORs.

	'''

	def __init__(self):
		print("Initialised BirminghamMethod")
		self._line_data = None

		
	@property
	def line_data(self):
		'''The LoRs for which the cutpoints are computed.

		line_data : instance of pept.LineData

		'''

		return self._line_data


	@line_data.setter
	def line_data(self, new_line_data):
		''' The LoRs for which the cutpoints are computed.

		Parameters
		----------
		line_data : instance of pept.LineData
			The LoRs to be tracked using the Birmingham Method. It is required to be an
			instance of `pept.LineData`.

		Raises
		------
		Exception
			If `line_data` is not an instance of `pept.LineData`.

		'''

		# Check line_data is an instance (or a subclass!) of pept.LineData
		if not isinstance(new_line_data, pept.LineData):
			raise Exception('[ERROR]: line_data should be an instance of pept.LineData')

		self._line_data = new_line_data

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


	def track(self, line_data = None, sample_no=None, sample_size=250, fopt=None, as_array=False):
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