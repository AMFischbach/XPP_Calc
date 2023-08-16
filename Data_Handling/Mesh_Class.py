# Author: Trey Fischbach, Date Created: Aug 11, 2023, Date Last Modified: Aug 11, 2023

import math
import numpy as np
import copy
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from Data_Handling.Detector_Map_Class import Detector_Map

class Mesh():
	"""
	A class that stores mesh and computes mesh attributes for 3D volumes.
	"""
	def __init__(self, detector_maps, mesh_granularity=None, cell_num=None):
		"""
		Initalize and compute mesh attributes from the given list of Detector_Map 
		INPUTS - detector_maps: an array containning detector_map objects
			 intensities: an array containning 2D numpy arrays corresponding to detector
			 images for each map.
		 	 mesh_granularity: the cubic volume of each mesh element in inverse angstroms
		"""
		# First we need to determine the spacing and boundary of the mesh
		self.boundaries, self.spacing = self.get_mesh_attributes(detector_maps, mesh_granularity, cell_num)

		# With these mesh attributes create the mesh
		self.coordinates = self.create_mesh(self.boundaries, self.spacing)

		# Populate the mesh to create 3D volume
		self.values = self.populate_mesh(detector_maps, self.coordinates, self.spacing)

		# Compute faces, edges, and vertices, these are helpful when slicing or plotting the boundary of the data prism
		self.vertices = self.get_vertices(self.coordinates)
		self.edges = self.get_edges(self.vertices)
		self.faces = self.get_faces(self.vertices)

		# Compute the number of bins in each direction
		self.bins = [self.coordinates[0].size, self.coordinates[1].size, self.coordinates[2].size]

	def __str__(self):
		return \
		f"""
		boundaries: {self.boundaries}
		bins: {self.bins}"""

	def get_mesh_attributes(self, detector_maps, mesh_granularity=None, cell_num=None):
		"""
		Determines the spacing and boundary of the mesh given the list of Detector_Map objects.
		"""
		# We do this by looping through all the maximum and minimum hkl values for each reciprocal map
		# and finding the absolute max and absolute min (over all maps) for each h, k, and l.
		hvalues = []
		kvalues = []
		lvalues = []

		for detector_map in detector_maps:
			hvalues.append(detector_map.boundary[0][0])
			hvalues.append(detector_map.boundary[0][1])
			kvalues.append(detector_map.boundary[1][0])
			kvalues.append(detector_map.boundary[1][1])
			lvalues.append(detector_map.boundary[2][0])
			lvalues.append(detector_map.boundary[2][1])

		# The maximum and minimum h,k,l values found in all the maps
		# These will be the boundaries of the mesh
		meshBoundaries = [[min(hvalues), max(hvalues)], [min(kvalues), max(kvalues)], [min(lvalues), max(lvalues)]]
		
		# Now we find the spacing of the mesh in all 3 dimensions
		# The sum of all the max - min values (think about the sum of all three lattice parameters for a cubic lattice)
		totalSum = 0
		for index in range(3):
			totalSum += meshBoundaries[index][1] - meshBoundaries[index][0]

		# These ratios indicate what percent of the mesh's volume is in each dimension
		# The ratios between the 3 bin numbers that we want
		hratio = (meshBoundaries[0][1] - meshBoundaries[0][0])/totalSum
		kratio = (meshBoundaries[1][1] - meshBoundaries[1][0])/totalSum
		lratio = (meshBoundaries[2][1] - meshBoundaries[2][0])/totalSum

		# We compute the spacing of the mesh unit cell in each dimension using different techniques depending upon the user input
		if mesh_granularity is not None:
			# The average length of a unit cell
			perimeter = 3*mesh_granularity**(1/3)

			# Compute the length of the unit cell on each axis
			hw = perimeter*hratio
			kw = perimeter*kratio
			lw = perimeter*lratio

		elif cell_num is not None:
			# Since we add addtional bins at the end, we have to adjust this cell number
			cell_num = ((cell_num**(1/3))-1)**3

			# The average length of a unit cell multiplied by 3
			# Just like the sum of the lattice parameters
			perimeter = 3*(cell_num**(1/3))

			# These whole numbers indicate how many mesh unit cells should fit along each dimension
			# The process of computing them is actually a bit complicated
			
			# We approximately get the bin numbers
			h_approx = hratio*perimeter
			k_approx = kratio*perimeter
			l_approx = lratio*perimeter
			
			# We compute a scaling factos and scale the approximations
			scaling_factor = (cell_num/(h_approx*k_approx*l_approx))**(1/3)
			h_approx = h_approx*scaling_factor
			k_approx = k_approx*scaling_factor
			l_approx = l_approx*scaling_factor

			# Now we commpute scaling error
			scaling_error = cell_num - (h_approx*k_approx*l_approx)

			# Adjust for scaling error and turn into integers
			hnum = math.floor(h_approx + h_approx*(scaling_error/(h_approx+k_approx+l_approx)))
			knum = math.floor(k_approx + k_approx*(scaling_error/(h_approx+k_approx+l_approx)))
			lnum = math.floor(l_approx + l_approx*(scaling_error/(h_approx+k_approx+l_approx)))

			# If the mesh is only on cell wide then we find the width differently
			if hnum == 1:
				hnum = 2
			if knum == 1:
				knum = 2
			if lnum == 1:
				lnum = 2
			
			# Compute the length of the unit cell on each axis
			hw = (meshBoundaries[0][1] - meshBoundaries[0][0])/(hnum-1)
			kw = (meshBoundaries[1][1] - meshBoundaries[1][0])/(knum-1)
			lw = (meshBoundaries[2][1] - meshBoundaries[2][0])/(lnum-1)

		else:
			print("need to specify either mesh_granularity or cell number")
			return False

		# Now we need to add two more unit cells on each axis
		# This is to make sure that each hkl point always borders eight cells of the mesh
		meshBoundaries[0][0] = meshBoundaries[0][0] - hw
		meshBoundaries[0][1] = meshBoundaries[0][1] + hw
		meshBoundaries[1][0] = meshBoundaries[1][0] - kw
		meshBoundaries[1][1] = meshBoundaries[1][1] + kw
		meshBoundaries[2][0] = meshBoundaries[2][0] - lw
		meshBoundaries[2][1] = meshBoundaries[2][1] + lw

		# Convert into numpy array
		spacing = np.array((hw, kw, lw))

		# Now we return the mesh boundaries and effective mesh unit cell
		return meshBoundaries, spacing

	def create_mesh(self, meshBoundaries, spacing):
		"""
		Create a 3D numpy array corresponding the bottom most corner of each cuboid.
		"""
		coordinates = np.empty(3, dtype=tuple)
		for index in range(3):
			newAxis = np.arange(meshBoundaries[index][0],\
					     meshBoundaries[index][1],\
					     spacing[index])
			coordinates[index] = newAxis

		return coordinates

	def get_vertices(self, coordinates):
		"""
		Given the coordinates of the mesh, we create the eight vertices of the mesh.
		"""
		# Define the min and max for clarity
		xmin = coordinates[0][0]
		xmax = coordinates[0][-1] + self.spacing[0]
		ymin = coordinates[1][0]
		ymax = coordinates[1][-1] + self.spacing[1]
		zmin = coordinates[2][0]
		zmax = coordinates[2][-1] + self.spacing[2]

		# Initalize the vertices list
		vertices = np.empty(8, dtype=tuple)
		
		# Populate the vertices list
		vertices[0] = np.array([xmax, ymax, zmin])
		vertices[1] = np.array([xmin, ymax, zmin])
		vertices[2] = np.array([xmax, ymax, zmax])
		vertices[3] = np.array([xmin, ymax, zmax])
		vertices[4] = np.array([xmax, ymin, zmin])
		vertices[5] = np.array([xmin, ymin, zmin])
		vertices[6] = np.array([xmax, ymin, zmax])
		vertices[7] = np.array([xmin, ymin, zmax])

		return vertices

	def get_edges(self, vertices):
		"""
		Gets the edges of the data prism
		"""
		# Initalize the edges list
		edges = np.empty(12, dtype=tuple)

		# Populate the edges list
		edges[0] = np.array([vertices[0], vertices[1]])
		edges[1] = np.array([vertices[1], vertices[5]])
		edges[2] = np.array([vertices[5], vertices[4]])
		edges[3] = np.array([vertices[4], vertices[0]])
		edges[4] = np.array([vertices[0], vertices[2]])
		edges[5] = np.array([vertices[1], vertices[3]])
		edges[6] = np.array([vertices[5], vertices[7]])
		edges[7] = np.array([vertices[4], vertices[6]])
		edges[8] = np.array([vertices[2], vertices[3]])
		edges[9] = np.array([vertices[3], vertices[7]])
		edges[10] = np.array([vertices[7], vertices[6]])
		edges[11] = np.array([vertices[6], vertices[2]])

		return edges

	def get_faces(self, vertices):
		"""
		Given the vertices defining the boundary of the mesh, we construct the faces 
		"""
		# Initalize the faces matrix
		faces = np.empty(6, dtype=tuple)

		# Populate the faces matrix
		faces[0] = np.array([vertices[0], vertices[1], vertices[2], vertices[3]])
		faces[1] = np.array([vertices[1], vertices[5], vertices[3], vertices[7]])
		faces[2] = np.array([vertices[5], vertices[4], vertices[7], vertices[6]])
		faces[3] = np.array([vertices[4], vertices[0], vertices[6], vertices[2]])
		faces[4] = np.array([vertices[4], vertices[5], vertices[0], vertices[1]])
		faces[5] = np.array([vertices[2], vertices[3], vertices[6], vertices[7]])

		return faces

	
	def populate_mesh(self, detector_maps, coordinates, spacing):
		"""
		Populates a given mesh using a list of detector_map objects
		INPUTS - detector_maps: list of detector_map objects to populate mesh
			 coordinates: np(x_values, y_values, z_values) where each is a 1D np array
			 spacing: np(xspacing, yspacing, zspacing) where each is a float
				       indicating the spacing betwen each mesh element along
				       each axis.
		"""
		# Get the mesh translation value, allowing all points to be put
		# inside the mesh frame of reference
		mesh_translation = np.array([coordinates[0][0], coordinates[1][0],\
				coordinates[2][0]])

		# Initalize the spacing values
		xw, yw, zw = spacing

		# Initalize the unpopulated mesh
		mesh = np.zeros((coordinates[0].size, coordinates[1].size, coordinates[2].size))

		# Initalize the mesh that keeps track of how many times each cuboid has been populated
		# This mesh makes it possible to normalize the data so there is not over representation
		population_counter = np.zeros((coordinates[0].size, coordinates[1].size, coordinates[2].size))
 
		# Store these value for speed
		sqrt3inv = 1/math.sqrt(3)
		inv_spacing = np.array([1/xw, 1/yw, 1/zw])

		# Loop through each detector_map
		for detector_map in tqdm(detector_maps, desc="populating mesh", ncols=100, leave=True):

			# Get the intensity_map (detector data)
			intensity_map = detector_map.intensity_map

			# Get the vector_map (detector pixel location)
			vector_map = detector_map.vector_map

			# Get the dimensions of the vector map
			row_num, col_num = vector_map.shape

			# Loop through each vector in the vector map and populate the mesh
			for row in range(row_num):
				for col in range(col_num):
					# Put the vector into the mesh frame of reference
					vector = vector_map[row, col] - mesh_translation	

					# Get the intensity value of this vector
					intensity = intensity_map[row, col]

					# Find the indices of the eight mesh volumes that will be populated
					indices = np.empty(8, dtype = tuple)
					indices[0] = np.array((math.floor(vector[0]/xw+0.5)-1,\
						      math.floor(vector[1]/yw+0.5)-1,\
						      math.floor(vector[2]/zw+0.5)-1))

					indices[1] = np.array((indices[0][0], indices[0][1]+1, indices[0][2]))
					indices[2] = np.array((indices[0][0], indices[0][1], indices[0][2]+1))
					indices[3] = np.array((indices[0][0], indices[0][1]+1, indices[0][2]+1))
					indices[4] = np.array((indices[0][0]+1, indices[0][1], indices[0][2]))
					indices[5] = np.array((indices[0][0]+1, indices[0][1]+1, indices[0][2]))
					indices[6] = np.array((indices[0][0]+1, indices[0][1], indices[0][2]+1))
					indices[7] = np.array((indices[0][0]+1, indices[0][1]+1, indices[0][2]+1))

					# Populate each cuboid
					for index in indices:
						# Get position of cuboid, (+0.5,to go to the center of the cuboid)
						cubePos = index*spacing + 0.5*spacing

						# The contribution of each element
						contribution = np.empty(3)

						# For each element in vector (think each h,k,l)
						for i, element in enumerate(vector):
							if element < cubePos[i]: 
								contribution[i] = (inv_spacing[i])*\
										(spacing[i]-(cubePos[i]-element))
							else:
								contribution[i] = (-inv_spacing[i])*\
										(element - cubePos[i]) + 1
						# Add one to the population counter
						population_counter[index[0], index[1], index[2]] += 1
						
						# The number of times this cuboid has been added to
						contribution_num = population_counter[index[0], index[1], index[2]]

						# Assign the contribution to the mesh (average it with the other values that have already)
						# be assigned
						contribution = intensity*sqrt3inv*np.linalg.norm(contribution)
						mesh[index[0], index[1], index[2]] = (mesh[index[0], index[1], index[2]] * (contribution_num-1) + contribution)/(contribution_num)
		return mesh
	
		
