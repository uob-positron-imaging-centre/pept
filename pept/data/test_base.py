import  matplotlib.pyplot       as  plt
from    matplotlib.colors       import Normalize
from 	mpl_toolkits.mplot3d 	import Axes3D

point_data = np.random.randint(low=1, high=100, size=(100,5))

'''Plot all points using matplotlib

Given a **mpl_toolkits.mplot3d.Axes3D** axis, plots all points on it.

Parameters
----------
ax : mpl_toolkits.mplot3D.Axes3D object
	The 3D matplotlib-based axis for plotting.

Note
----
Plotting all points in the case of large LoR arrays is *very*
computationally intensive. For large arrays (> 10000), plotting
individual samples using `plot_points_sample_n` is recommended.

'''
# Sam -
# point_data row: [time, x, y, z, etc]
# want to plot scatter 3d for [x, y, z] while also
# colour-coding the last column

# if ax == None:
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

# Scatter x, y, z, [color]

x = point_data[:, 1],
y = point_data[:, 2],
z = point_data[:, 3],

color = point_data[:, -1],

cmap = plt.cm.magma
color_array = cmap(color)

ax.scatter(x,y,z,c=color_array[0])
