# Author: Trey Fischbach, Date Created: Aug 4, 2023, Date Last Modified: Aug 4, 2023

import math
import numpy as np
import copy
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from Data_Handling.Mesh_Class import Mesh
from Data_Handling.Detector_Map_Class import Detector_Map

class Reciprocal_Volume():
	"""
	A class meant to be a data type for reciprocal volumes.
	"""
	def __init__(self, detector_maps, cell_num, axes_names = ["","",""], save_filepath = ""):
		"""
		Initalize the object attributes
		"""
		# A populate the mesh object
		self.mesh = Mesh(detector_maps, cell_num=cell_num)

		# A list of all the detector_maps used to construct the 3D volume
		self.detector_maps = detector_maps

		# The title of each axis
		self.axes_names = axes_names

		# The folder path where any data will be stored
		self.save_filepath = save_filepath

		# Create interpolating function
		self.interpolating_function = self.make_interpolating_function()


	def make_interpolating_function(self):
		"""
		Generates the interpolating function for the 3D volume. This allows us
		to give ANY point inside the mesh and have a intensity value returned.
		"""
		# First shift the mesh to be at the center of each cuboid
		x_centers = self.mesh.coordinates[0] + 0.5*self.mesh.spacing[0]*np.ones(self.mesh.coordinates[0].size)
		y_centers = self.mesh.coordinates[1] + 0.5*self.mesh.spacing[1]*np.ones(self.mesh.coordinates[1].size)
		z_centers = self.mesh.coordinates[2] + 0.5*self.mesh.spacing[2]*np.ones(self.mesh.coordinates[2].size)

		# Create the interpolating function
		interpolating_function = RegularGridInterpolator((x_centers, y_centers, z_centers), self.mesh.values)

		return interpolating_function

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
		#ax.scatter(axis1_values, axis2_values, c=intensity_values)

	def plot_slice(self, indices, point=False, title="", xlabel="", ylabel="", ax=None, saveFig=False):
		"""
		Generates and plots a slice of the 3D volume.
		INPUTS - indices: the vector normal to the slicing plane
			 point: a point that lies somewhere inside the data cube
			 on the plane.
		"""
		# If we aren't given an axis to plot to we make a new figure
		if ax is None:
			fig, ax = plt.subplots()
			newFigure = True
		else:
			newFigure = False

		# Create and populate the slice from the 3D volume
		v1, v2, X, Y, slice_intensities = self.get_slice(indices, point)						

		round_down = lambda x: math.floor(x*100)/100

		# We compute what the x and y axis are from the v1 and v2
		axis1 = "( " + str(round_down(v1[0])) + self.axes_names[0] + ", " +\
			       str(round_down(v1[1])) + self.axes_names[1] + ", " +\
			       str(round_down(v1[2])) + self.axes_names[2] + ")"
		axis2 = "( " + str(round_down(v2[0])) + self.axes_names[0] + ", " +\
			       str(round_down(v2[1])) + self.axes_names[1] + ", " +\
			       str(round_down(v2[2])) + self.axes_names[2] + ")"

		# Plot the slice
		plt.pcolormesh(X,Y, slice_intensities, shading="auto",cmap="viridis")

		if newFigure:
			plt.xlabel(axis1)
			plt.ylabel(axis2)
			plt.title(title)
			
			if saveFig:
				print(self.save_filepath + title+".png")
				plt.savefig(self.save_filepath + title+".png")
			else:
				plt.show()


	def get_slice(self, indices, point=False):
		"""
		Produces a 2D slice of the 3D volume.
		"""
		# If we are not given a point then we take the center of the
		# data prism as a point in the plane
		if not point:
			point = np.empty(3)
			for i in range(3):
				point[i] = (self.mesh.coordinates[i][-1] - self.mesh.coordinates[i][0])*0.5\
					+ self.mesh.coordinates[i][0] + self.mesh.spacing[i]*0.5

		else:
			# If the given point is not a numpy array we convert it to be one
			if not isinstance(point, np.ndarray):
				point = np.array(point)

		# Compute D in the plane equation and initalize A,B,C for clairity
		A = indices[0]
		B = indices[1]
		C = indices[2]
		D = -(A*point[0] + B*point[1] + C*point[2])

		# Find the intersection points of the plane and the cube
		intersection_points = []

		# Function that computes the point on the line (formed by two vertices) 
		# which intersects with the slicing plane
		def get_intersection_point(v1, v2):
			numerator = -(A*v1[0] + B*v1[1] + C*v1[2] + D)
			denominator = (A*(v2[0]-v1[0]) + B*(v2[1]-v1[1]) + C*(v2[2]-v1[2]))
			
			# If denominator is zero then there is no point
			if denominator == 0:
				return None
			
			# If the numerator is zero then the line is fully contained in the plane 
			# In this case we return both vertices
			if numerator == 0:
				return np.array([v1, v2])
			
			# Compute the point based upon value of t
			t = numerator/denominator
			intersection_point = np.array([v1[0] + t*(v2[0]-v1[0]),\
					  v1[1] + t*(v2[1]-v1[1]),\
					  v1[2] + t*(v2[2]-v1[2])])
		
			# Compute distances (to check if point is inbetween the vertices)
			edge_length = np.linalg.norm(v2-v1)
			distance1 = np.linalg.norm(intersection_point-v1)
			distance2 = np.linalg.norm(intersection_point-v2)

			# If the intersection point is between two the vertices
			if distance1 < edge_length and distance2 < edge_length:
				return intersection_point

			# If the point not on the edge
			else:
				return None

		# All points on edges of the prism where the slicing plane intercepts
		intersection_points = []

		# Loop through every single edge in the data prism
		for edge in self.mesh.edges:
			# If it exits get the point on the edge where the plane intercept
			intersection_point = get_intersection_point(edge[0], edge[1])
			
			# If there is a point, we add it to the list
			if intersection_point is not None:
				# Sometimes if the plane fully contains an edge, point actually contains 2 points
				# In this case we add both points to the intersection points list
				if intersection_point.size == 6: 
					intersection_points.append(intersection_point[0])
					intersection_points.append(intersection_point[1])
				
				else:
					intersection_points.append(intersection_point)

		# There's a very slight edge case in which the slice only slices through a single vertex
		# This is bad, so we need to check here for repeated points
		if len(intersection_points) == 3:
			if intersection_points[0] == intersection_points[1] and intersection_points[0] == intersection_points[2]:
				print("slice only contains a vertex of the data prism")
				return False

		# Now that we have the intersection points we create the mesh
			
		# Find v1 and v2, two orthonormal vectors in the slicing plane
		# This is done by starting off with a seed vector v0
		# We pick v0 so that it results in a nice v1 and v2, the math is too much to explain here
		if indices[2] == 0:
			v0 = np.cross(indices, np.array([0,0,1]))
		else:
			v0 = np.cross(indices, np.array([1,0,0]))

		# Compute v1 and v2
		v1 = np.cross(indices, v0)
		v2 = np.cross(indices, v1)

		# Normalize v1 and v2
		normalize = lambda v: (1/np.linalg.norm(v))*v	
		v1 = normalize(v1)
		v2 = normalize(v2)

		# If all the elements of v1 or v2 are not positive, we flip the sign
		if v1[0] <= 0 and v1[1] <= 0 and v1[2] <= 0:
			v1 = -1*v1
		if v2[0] <= 0 and v2[1] <= 0 and v2[2] <= 0:
			v2 = -1*v2

		# TRANSFER POINTS FROM 3D to 2D
		to2D = lambda xyz: np.array([np.dot(xyz,v1), np.dot(xyz,v2)])
		
		# Transfer all intersection points into 2D
		intersection_points2D = []
		for intersection_point in intersection_points:
			intersection_points2D.append(to2D(intersection_point))
		
		# List the x values and y values separately
		xvals = [ipoint[0] for ipoint in intersection_points2D]
		yvals = [ipoint[1] for ipoint in intersection_points2D]
		
		# Get the boundaries of the 2D mesh
		gridBoundaries = [[min(xvals), max(xvals)], [min(yvals), max(yvals)]]
	
		# Get the step size for both x and y of the 2D slice
		x_step = abs(0.5*np.dot(v1,self.mesh.spacing))
		y_step = abs(0.5*np.dot(v2,self.mesh.spacing))
		
		# The x and y coordinates of the 2D mesh
		x_coordinates = np.arange(gridBoundaries[0][0], gridBoundaries[0][1]+x_step, x_step)
		y_coordinates = np.arange(gridBoundaries[1][0], gridBoundaries[1][1]+y_step, y_step)

		# Make a 2D mesh
		X, Y = np.meshgrid(x_coordinates, y_coordinates)
		
		# Convert each point in the 2D mesh to 3D and find out its corresponding value
		# in the data prism
		slice_intensities = np.zeros(X.shape)
		
		# TRANSFER POINTS FROM 2D to 3D
		to3D = lambda xy: xy[0]*v1 + xy[1]*v2
	
		# Compute the intensity of each point on the mesh	
		for row in tqdm(range(X.shape[0]), desc="taking slice: "+str(indices), ncols=100, leave=False):
			for col in range(X.shape[1]):
				mesh_point3D = to3D([X[row, col], Y[row, col]])
				
				# Some of these mesh points will be outside the prism. If this is the case
				# an error will be thrown.
				try:
					slice_intensities[row, col] = self.interpolating_function(mesh_point3D)

				except ValueError:
					# For all mesh points outside the data prism their intensity value is set to NAN
					slice_intensities[row, col] = np.nan
		
		return v1, v2, X, Y, slice_intensities

	def plot_reciprocal_space_coverage(self):
		"""
		Plots in 3D the area of reciprocal space that all the detector maps cover
		along with the mesh boundary. This is helpful for visulizing what areas of
		reciprocal space we have actually touched, as well as what areas of the mesh
		are unpopulated.
		"""
		# For each detector map get the hkl value for the corner of the detector
		# along with the center of the detector. We will plot these surfaces together
		# along with the mesh outline.
			
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")

		# The colors of each detector
		colors = plt.cm.viridis(np.linspace(0, 1, len(self.detector_maps)))
	
		# Plot each detector
		for i, detector_map in enumerate(self.detector_maps):
			detector_map.plot_vector_map_3D(surface=True, outline=True, color=colors[i], ax=ax)
	
		# Plots the outline of the mesh
		self.plot_mesh_outline(ax)

		# Title the plot and axes
		self._title_plot(ax, "All Detector Maps Inside the Mesh Volume", self.axes_names)

		# Show the plot
		plt.show()

	def get_slice_old(self, indices, point=False, ax=None):
		"""
		Produces a 2D slice of the 3D volume.
		"""
		# If we are not given a point then we take the center of the
		# data prism as a point in the plane
		if not point:
			point = np.empty(3)
			for i in range(3):
				point[i] = (self.mesh.coordinates[i][-1] - self.mesh.coordinates[i][0])*0.5\
					+ self.mesh.coordinates[i][0] + self.mesh.spacing[i]*0.5

		else:
			# If the given point is not a numpy array we convert it to be one
			if not isinstance(point, np.ndarray):
				point = np.array(point)

		# Find v1 and v2, two orthonormal vectors in the plane
		# This is done by starting off with a seed vector v0
		# We pick v0 so that it results in a nice v1 and v2, the math is too much to explain here
		if indices[2] == 0:
			v0 = np.cross(indices, np.array([0,0,1]))
		else:
			v0 = np.cross(indices, np.array([1,0,0]))

		# Compute v1 and v2
		v1 = np.cross(indices, v0)
		v2 = np.cross(indices, v1)

		# Normalize v1 and v2
		normalize = lambda v: (1/np.linalg.norm(v))*v	
		v1 = normalize(v1)
		v2 = normalize(v2)

		# If all the elements of v1 or v2 are negative, we flip the sign
		if v1[0] < 0 and v1[1] < 0 and v1[2] < 0:
			v1 = -1*v1
		if v2[0] < 0 and v2[1] < 0 and v2[2] < 0:
			v2 = -1*v2

		# Find the step vectors in the direction of v1 and v2 for the generation of the slice points
		v1_step = 0.1*v1*self.mesh.spacing
		v2_step = 0.1*v2*self.mesh.spacing

		# Loop through all points on the plane until we hit a boundary
		# The stored axis1 and axis2 values for the plane along with intensity values
		# The axis1 unit vector is v1 and axis2 unit vector is v2
		axis1_values = []
		axis2_values = []
		intensity_values = []

		xvalues = []
		yvalues = []
		zvalues = []

		# c1 and c2 are constants which are multiplied against step_sizes to move in v1 or v2 directions
		# Turns c1 and c2 into xyz
		get_xyz = lambda c1, c2: c1*v1_step + c2*v2_step + point 

		# Gets the magnitude for a point for each axis
		get_magnitude1 = lambda c1: np.linalg.norm((point + c1*v1_step)*v1)
		get_magnitude2 = lambda c2: np.linalg.norm((point + c2*v2_step)*v2)

		# Add the center point to the intensity values
		xyz = get_xyz(0, 0)

		intensity_values.append(self.interpolating_function(xyz))
		axis1_values.append(get_magnitude1(0))
		axis2_values.append(get_magnitude2(0))

		# The number of times we have hit the boundary moving the point along the v2 direction
		# If this number goes to 2, then the slice has been populated
		v2_boundary_hits = 0
		c2 = 0
		while v2_boundary_hits < 2:
			# The number of times we have hit the boundary moving the point along the v1 direction
			# for this v2 value
			# If this number goes to 2, then the line has been populated 
			v1_boundary_hits = 0
			c1 = 0
			while v1_boundary_hits < 2:
				try:
					# We move in increasing v1
					if v1_boundary_hits == 0:
						c1 += 1
					# We move in decreasing v1
					elif v1_boundary_hits == 1:
						c1 -= 1
					# Add the new value to the list	
					intensity_values.append(self.interpolating_function(get_xyz(c1, c2)))
					axis1_values.append(get_magnitude1(c1))
					axis2_values.append(get_magnitude2(c2))
					
					xyz = get_xyz(c1, c2)
					xvalues.append(xyz[0])
					yvalues.append(xyz[1])
					zvalues.append(xyz[2])

				# If we hit a boundary
				except ValueError:
					# Add to the v1 boundary hits counter and resets c1
					v1_boundary_hits += 1
					c1 = 0
			
			
			# Once we hit the v1 boundary twice, we move to the next v2 value
			# We add the new value to the list 
			try:
				if v2_boundary_hits == 0:
					c2 += 1
				
				elif v2_boundary_hits == 1:
					c2 -= 1
				
				intensity_values.append(self.interpolating_function(get_xyz(c1, c2)))
				axis1_values.append(get_magnitude1(c1))
				axis2_values.append(get_magnitude2(c2))
			
				xyz = get_xyz(c1, c2)
				xvalues.append(xyz[0])
				yvalues.append(xyz[1])
				zvalues.append(xyz[2])

			# If we hit a boundary
			except ValueError:
				# Add to the v2 boundary hits counter and resets c2
				v2_boundary_hits += 1
				c2 = 0

		ax.plot_trisurf(xvalues, yvalues, zvalues, color="black", alpha=0.8)	
		return axis1_values, axis2_values, intensity_values


	def plot_3D_volume(self):
		"""
		Plots the Reciprocal_Volume in a 3D space
		INPUTS - 3D_volume: a 3D numpy array filled with floats
			 mesh.coordinates: (x_coords, y_coords, z_coords) where each is a 1D numpy array
			 		   corresponding to the x,y,or z position of any volume element.
					   Like a "key" to understanding where the 3D_volume is.
		"""
		get_cuboid_dim = lambda center, w: [center - w, center + w, center + w, center - w,\
						    center - w, center - w, center + w, center + w,\
						    center - w, center - w]
		
		xw = self.mesh.spacing[0]
		yw = self.mesh.spacing[1]
		zw = self.mesh.spacing[2]

		fig = go.Figure()

		for i in range(mesh.values.shape[0]):
			for j in range(mesh.values.shape[1]):
				for k in range(mesh.values.shape[2]):
					x_center = self.mesh.coordinates[0][i]
					y_center = self.mesh.coordinates[1][j]
					z_center = self.mesh.coordinates[2][k]
					color = self.mesh.values[i,j,k]

					cuboid = go.Mesh3d(
							x = get_cuboid_dim(x_center, xw),
							y = get_cuboid_dim(y_center, yw),
							z = get_cuboid_dim(z_center, zw),
							i = [0,0,1,1,0,2,2,3,3,2],
							j = [1,2,2,3,1,3,4,5,5,4],
							k = [2,3,0,1,2,5,6,7,7,6],
							intensity = (100,100,100),
							#intensity = 100*(color,color,color),
							showscale=True,
							opacity=0.5
					)

					fig.add_trace(cuboid)

		fig.update_layout(scene=dict(aspectmode="cube"))

		fig.show()
	
	def plot_mesh(self, ax=None, colored=False, title="", axisLabels=["","",""]):
		"""
		Plots the mesh of the reciprocal_volume as a scatter plot of points
		"""
		# If we are not given an axis to plot to we just create a new figure
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection="3d")
			newFigure = True
		else:
			newFigure = False

		half_xw = self.mesh.spacing[0]*0.5
		half_yw = self.mesh.spacing[1]*0.5
		half_zw = self.mesh.spacing[2]*0.5

		x_values = self.mesh.coordinates[0]
		y_values = self.mesh.coordinates[1]
		z_values = self.mesh.coordinates[2]

		# If we colored is True, then we make the color of each vector 
		# equal to its corresponding intensity value in mesh.values 
		if colored:
			colormap = plt.cm.viridis

			# Copy the mesh to get colorscale
			values_copy = copy.deepcopy(self.mesh.values)

			# Reshape it and normalize the intensity for coloring
			reshaped_mesh = np.array(list(values_copy.flat))
			min_val = np.min(reshaped_mesh)
			max_val = np.max(reshaped_mesh)
			norm = Normalize(min_val, max_val)

			# Get the color of every point
			colors = np.array([[colormap(norm(value))] for value in reshaped_mesh])
			
			# Reshape the color array to be a 3D matrix with RGBA values
			colors = colors.reshape(self.mesh.values.shape[0],\
						self.mesh.values.shape[1],\
						self.mesh.values.shape[2], 4)

			for xindex, xval in enumerate(x_values):
				for yindex, yval in enumerate(y_values):
					for zindex, zval in enumerate(z_values):
						# Get the color
						color = [color_element for color_element in colors[xindex, yindex, zindex, :]]

						# Plot the point
						ax.scatter(xval+half_xw, yval+half_yw, zval+half_zw, color = color)
		else:
			for xval in x_values:
				for yval in y_values:
					for zval in z_values:
						ax.scatter(xval+half_xw, yval+half_yw, zval+half_zw,c="red")

		# Only if this is a new figure do we title or adjust any formatting
		if newFigure:
			# Title everything, make axis each, show plot
			self._title_plot(ax, title, axisLabels)
			self._make_axis_equal(ax)
			plt.show()

	def plot_mesh_outline(self, ax):
		"""
		Plots the outline of the mesh as a frame
		"""
		# If we are not given an axis to plot to we just create a new figure
		if ax is None:
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			newFigure = True
		else:
			newFigure = False
		
		# Plot each edge
		for edge in self.mesh.edges:
			endpoint1 = edge[0]
			endpoint2 = edge[1]

			x_coords = [endpoint1[0], endpoint2[0]]
			y_coords = [endpoint1[1], endpoint2[1]]
			z_coords = [endpoint1[2], endpoint2[2]]
		
			ax.plot(x_coords, y_coords, z_coords, color="black")
	
		if newFigure:
			self._make_axis_equal(ax)
			plt.show()



