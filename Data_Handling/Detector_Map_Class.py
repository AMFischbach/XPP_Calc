# Author: Trey Fischbach, Date Created: Aug 4, 2023, Date Last Modified: Aug 18, 2023

import math
import numpy as np
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm
from matplotlib.colors import Normalize

from XPP_Calc.XPP_Motor_Pos_Class import XPP_Motor_Pos 

class Detector_Map():
	"""
	A class meant to be a data type for all detector maps.
	These can be maps of any kind, xyz, hkl, or gamma, delta.
	This class also contains methods for plotting these objects.
	"""
	def __init__(self, vector_map, boundary=None, intensity_map=None, motor_pos=None):
		"""
		Initalize the reciprocal map object.
		"""
		# 2D Numpy matrix where each element is a numpy array containing vectors
		# corresponding to each detector pixel
		self.vector_map = vector_map
		
		# Maximum and minimum of all h,k,l values in a reciprocal map
		# ex: [[hmin, hmax], [kmin, kmax], [lmin, lmax]]
		self.boundary = boundary

		# 2D Numpy matrix where each element is the intensity value recorded at
		# that pixel on the detector
		self.intensity_map = intensity_map

		# XPP_Motor_Pos object
		self.motor_pos = motor_pos

		# If we are given an intensity map we check to see if its dimensions match the vector_map
		if intensity_map is not None:
			if vector_map.shape != intensity_map.shape:
				print("WARNING INTENSITY MAP SHAPE DOES NOT MATCH VECTOR MAP")
	
	def _title_plot(self, ax, title, axisLabels):
		"""
		A helper function that titles the axis and the plot for 3D plots
		"""
		ax.set_xlabel(axisLabels[0])
		ax.set_ylabel(axisLabels[1])
		ax.set_zlabel(axisLabels[2])
		ax.set_title(title)

	def _make_axis_equal(self, ax):
		"""
		A helper function that make all axis equal in a 3D plot
		"""
		# Make the axis equal
		x_limits = ax.get_xlim3d()
		y_limits = ax.get_ylim3d()
		z_limits = ax.get_zlim3d()

		x_range = abs(x_limits[1] - x_limits[0])
		x_middle = np.mean(x_limits)
		y_range = abs(y_limits[1] - y_limits[0])
		y_middle = np.mean(y_limits)
		z_range = abs(z_limits[1] - z_limits[0])
		z_middle = np.mean(z_limits)

		# The plot bounding box is a sphere in the sense of the infinity
		# norm, hence I call half the max range the plot radius.
		plot_radius = 0.5*max([x_range, y_range, z_range])

		ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
		ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
		ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

	
	def get_vector_map_magnitude(self):
		"""
		A weird function written for Apurva which returns the magnitude of each
		3D vector in a 2D matrix.
		"""
		# Get dimension of the matrix
		row_num, col_num = self.vector_map.shape

		# Initalize the new matrix containing the magnitude of each vector
		magnitudes = np.zeros((row_num, col_num))

		# Loop through the matrix and compute the magnitude, adding it to magnitudes
		for row in range(row_num):
			for col in range(col_num):
				magnitudes[row, col] = math.sqrt((self.vector_map[row,col])[0]**2 \
								+ (self.vector_map[row,col])[1]**2\
								+ (self.vector_map[row,col])[2]**2)
		
		return magnitudes

	def plot_detector_map_2D(self, axis1_function, axis2_function, title="", xlabel="", ylabel="", colorbarLabel="", axisEqual=False):		
		"""
		Plots a given detector_map object on a 2D plot.
		INPUTS - axis1_function/axis2_function: the axis that each point's intensity will be mapped to
			 For example let's say we have x,y,z for each point and then some intensity values
			 contained in the intensity_map.
			 If we make axis1_function = lambda xyz: xyz[0] and axis2_function = lambda xyz: xyz[1],
			 we will plot all the points intensity in the x,y plane.
		"""
		# Extract detector dimensions
		row_num, col_num = self.intensity_map.shape

		# Initalize X and Y values
		x_grid = np.empty((row_num, col_num))
		y_grid = np.empty((row_num, col_num))

		# For each value in the map detirmine the x,y and intensity value
		for row in range(row_num):
			for col in range(col_num):
				x_grid[row, col] = axis1_function(self.vector_map[row, col])
				y_grid[row, col] = axis2_function(self.vector_map[row, col])

		# Plot it!
		plt.pcolormesh(x_grid, y_grid, self.intensity_map, cmap='viridis')

		# Title axis, figure, and colorbar
		plt.colorbar(label=colorbarLabel)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)

		# Add contour lines
		plt.contour(x_grid, y_grid, self.intensity_map, colors="black") # Adds some nice contours

		# Make the axes equal if the user wants
		if axisEqual:
			ax = plt.gca()
			ax.set_aspect('equal', adjustable='box')

		# Display the plot
		plt.show()

	def plot_intensity_map_2D(self, title="", xlabel="", ylabel="", colorbarLabel="", contour_map=False, contours=False, color="blue", newFigure=True):		

		"""
		Plots the 2D intensity matrix of the Detector_Map object.

		countour_map indicates if we plot only a contour map
		contours indicates if we plot contours over the intensity_map image
		"""
		row_num, col_num = self.intensity_map.shape

		# Plot the values
		x_grid = np.arange(col_num)
		y_grid = np.arange(row_num)
		X, Y = np.meshgrid(x_grid, y_grid)
	
		# If we are plotting contour maps
		if contour_map:
			contour_plot = plt.contour(X, Y, self.intensity_map, levels=10, colors=color)
			plt.clabel(contour_plot, inline=True, fontsize=8)

		# If we are plotting color gradient
		else:
			plt.pcolormesh(X, Y, self.intensity_map, cmap="viridis")
			if contours:
				plt.contour(X, Y, self.intensity_map, levels=10, colors="black")

		if newFigure:
			# Title axis, figure, and colorbar
			if not contour_map:
				plt.colorbar(label=colorbarLabel)

			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)

			# Make the axes equal
			ax = plt.gca()
			ax.set_aspect('equal', adjustable='box')
			plt.show()

	def plot_reciprocal_map_2D(self, title="", xlabel="", ylabel="", pixel_point=None, just_contour=False):
		"""
		Plots the detector image (intentisty map) with contour lines corresponding to
		h,k,l superimposed.

		pixel_point can be a specified detector pixel where a black point will be plotted
		as a list of two numbers [row, col]

		just_contour means that the intensity map won't be plotted
		"""
		# Initalize vector maps for h,k,l
		vector_map_h = np.empty(self.vector_map.shape)
		vector_map_k = np.empty(self.vector_map.shape)
		vector_map_l = np.empty(self.vector_map.shape)

		# Population vector maps
		row_num, col_num = self.vector_map.shape
		for row in range(row_num):
			for col in range(col_num):
				vector_map_h[row, col] = (self.vector_map[row, col])[0]
				vector_map_k[row, col] = (self.vector_map[row, col])[1]
				vector_map_l[row, col] = (self.vector_map[row, col])[2]
		
		# Make a new figure
		plt.figure()
		
		# Plot the intensity map (detector image)
		if not just_contour:
			self.plot_intensity_map_2D(newFigure=False)
		
		# Plot each of the countour maps
		x_grid = np.arange(self.intensity_map.shape[1])
		y_grid = np.arange(self.intensity_map.shape[0])
		X, Y = np.meshgrid(x_grid, y_grid)

		contour_map_h = plt.contour(X, Y, vector_map_h, levels=10, colors="red")
		plt.clabel(contour_map_h, inline=True, fontsize=8)
		contour_map_k = plt.contour(X, Y, vector_map_k, levels=10, colors="blue")
		plt.clabel(contour_map_h, inline=True, fontsize=8)
		contour_map_l = plt.contour(X, Y, vector_map_l, levels=10, colors="green")
		plt.clabel(contour_map_h, inline=True, fontsize=8)

		# Plot the desired point
		if pixel_point is not None:
			plt.scatter(pixel_point[0], pixel_point[1], color="black")

		# Label the axes and the plot
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		
		# Make the axes equal
		ax = plt.gca()

		# Plot the figure
		plt.show()

	def plot_vector_map_3D(self, surface=False, outline=False, ax=None, color="red", title="", xlabel="", ylabel="", colorbarLabel=""):		
		"""
		Plots the vector position of each pixel in a 3D scatter plot.
		INPUTS - surface: boolean, if True then we make a surface map, 
					   if False then we make a scatter plot.
			 outline: boolen, if True then we only display the outline
			 		  of the vector map, useful when the vector
					  map is too large to plot.
		"""
		# If we are not given an axis to plot to we just create a new figure
		if ax is None:
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			
			# Plot the origin
			ax.scatter3D(0,0,0,c="black")
			
			newFigure = True
		else:
			newFigure = False

				
		# Get the dimensions of the vector map
		row_num, col_num = self.vector_map.shape
	
		# If we are plotting a surface
		if surface:

			# If we are only plotting the outline
			if outline:
				x_vals = []
				y_vals = []
				z_vals = []
				
				# Get the indices of the points we want to plot
				points = []
				points.append(self.vector_map[0,0])
				points.append(self.vector_map[0,self.vector_map.shape[1]-1])
				points.append(self.vector_map[self.vector_map.shape[0]-1,0])
				points.append(self.vector_map[self.vector_map.shape[0]-1, self.vector_map.shape[1]-1])
				points.append(self.vector_map[math.floor((self.vector_map.shape[0]-1)/2), math.floor((self.vector_map.shape[1]-1)/2)])

				# Populate x_vals, y_vals, and z_vals
				for point in points:
					x_vals.append(point[0])
					y_vals.append(point[1])
					z_vals.append(point[2])
	
			# If we are plotting all points	as a surface	
			else:
				x_vals = np.empty(self.vector_map.size)
				y_vals = np.empty(self.vector_map.size)
				z_vals = np.empty(self.vector_map.size)

				index = 0
				for row in range(row_num):
					for col in range(col_num):
						x_vals[index] = (self.vector_map[row, col])[0]
						y_vals[index] = (self.vector_map[row, col])[1]
						z_vals[index] = (self.vector_map[row, col])[2]
						index += 1
			
			wireframe = ax.plot_trisurf(x_vals, y_vals, z_vals, color=color, alpha=0.7)

		# If we are plotting points (there is no outline option here)
		else:

			# Plot each vector as a point
			for row in range(row_num):
				for col in range(col_num):
					x = (self.vector_map[row, col])[0]
					y = (self.vector_map[row, col])[1]
					z = (self.vector_map[row, col])[2]
					ax.scatter3D(x,y,z,color=color)

		# Only if this is a new figure do we title or adjust any formatting
		if newFigure:
			# Title everything, make axis each, show plot
			self._title_plot(ax, title, axisLabels)
			self._make_axis_equal(ax)
			plt.show()


