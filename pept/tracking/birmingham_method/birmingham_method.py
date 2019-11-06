import matplotlib.pyplot as plt
import numpy as np
import pept
import os

from .extensions.birmingham_method import birmingham_method

class BirminghamMethod():
	def __init__(self, LineData):
		print("Initialising BirminghamMethod")
		
		self.LineData = LineData

	def set_fopt(self, static_fname, show_graph=True, sample_size = 250, fopt=np.arange(0.05,0.90,0.05), verbose = True):
	
		'''
		Function returns the optimal fopt parameter for the given dataset

		'''

		self._static_data = pept.scanners.ModularCamera(static_fname,1000)
		
		self._static_data.sample_size = sample_size

		print(self._static_data.number_of_samples)

		std_dev_min = 999
		f_best = 0

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

			r2 = (x.mean()**2 + (y.mean()-190)**2 + z.mean()**2)**0.5

			err = locations[:,4]
			std_dev = np.sqrt(x.std()**2 + y.std()**2 + z.std()**2)
			# ms = (std_dev**2) * (200/(15**2))

			if verbose:
				print("\nMean positions for fopt = %.2f are: " % f)
				print("x = %.2f mm +/- %.2f \t  y = %.2f mm +/- %.2f \t z = %.2f mm +/- %.2f" % (x.mean(), x.std(), y.mean(), y.std(), z.mean(), z.std()))
				print("So the mean error is: %.2f mm" % std_dev)
				print("The precision of the measurements is: %.2f mm\n" % err.mean())

			plt.plot(f,std_dev, c = 'r', marker='o')
			plt.plot(f,err.mean(), c = 'b', marker='s')

			if std_dev < std_dev_min:
				std_dev_min = std_dev
				f_best = f

		plt.axvline(f_best,0,50)







	def track(self):
		fopt = 0.3
		LORs = self.LineData.line_data
		location, used = birmingham_method(LORs, fopt)
		fig, ax = self.LineData.plot_all_lines(color='k',alpha=0.1)
		ax.scatter(location[1],location[2],location[3],c='r')
