# Author: Trey Fischbach, Date Created: Jul 20, 2023, Date Last Modified: Jul 20, 2023
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

from .. import Enums 
from .. import XPP_Motor_Pos_Class 
from mpl_toolkits import mplot3d
from tqdm import tqdm


class XPP_Detector():
	"""
	A detector class that emulates XPP's robot arm detector
	"""

	def __init__(self, detecPos, detecOrientation, gammaObj, deltaObj, detectorAttributes, wavelength, name, xyzFormat):
		"""
		Define and initialize XPP detector. PixelSize is in microns.
		____INPUTS____
		detecPos: incidentAxisPos, horizontalAxisPos, virticalAxis,Pos 
		detecOrientation: Euler angles at which detector is orientated
		gammaObj and deltaObj: the component ophyd object from the diffractometer class of gamma and delta
		detectorAttributes: (pixelWidthNum, pixelHeightNum, pixel_size, offsets)
		wavelength: wavelength of incident light in Angstroms
		name: the name of the detector, commonly the detector type
		"""
		# The detector position and euler angles are just floats or None
		self.incidentAxisPos = None
		self.horizontalAxisPos = None
		self.virticalAxisPos = None
		self.alpha = None
		self.beta = None
		self.gamma = None

		# Gamma and delta's cpts which are handed down from XPP_Diffractometer Class
		self.gammaObj = gammaObj
		self.deltaObj = deltaObj

		# Set detector attributes
		self.pixelWidthNum = detectorAttributes[0]
		self.pixelHeightNum = detectorAttributes[1]
		self.pixel_size = detectorAttributes[2]*(10**-3)

		# Initalize offset values for detector motors
		self.offsets = detectorAttributes[3]
		
		# Wavelength of the expirement, used in the pureQ calcs
		self.wavelength = wavelength

		# Name of the detector
		self.name = name

		# List of banned gamma and delta ranges in the format:
		# [[[gammaMin, gammaMax],[deltaMin,deltaMax]], ... , [][]]
		self.banned_regions = []

		# xyz format
		self.xyzFormat = xyzFormat

		# The cropped area of the detector. Used for image processing.
		# [[xstart, xstop], [ystart, ystop]]
		self.crop_boundary = [[0,self.pixelWidthNum-1],[0,self.pixelHeightNum-1]]

		# Initialize the detector angle values of gamma and delta to reflect
		# the inital position of the detector
		self.move(detecPos, detecOrientation)
		
	def _get_detec_motor(self, motorEnum):
		"""
		Given a goniometer motor enum, returns the ophyd soft positioner object and the name as a string.
		"""
		if motorEnum == Detector.incidentAxisPos:
			return self.incidentAxisPos, "incidentAxisPos"
		elif motorEnum == Detector.horizontalAxisPos:
			return self.horizontalAxisPos, "horizontalAxisPos"
		elif motorEnum == Detector.virticalAxisPos:
			return self.virticalAxisPos, "virticalAxisPos"
		elif motorEnum == Detector.detecAlpha:
			return self.alpha, "detecAlpha"
		elif motorEnum == Detector.detecBeta:
			return self.beta, "detecBeta"
		elif motorEnum == Detector.detecGamma:
			return self.gamma, "detecGamma"
		# If the motor belongs to the goniometer or is invalid
		else:
			return None, ""

	def _user_to_standard_axis(self, userTriplet):
		"""
		Because of the insanity at XPP, the user may input data in the format:
		(horizontalAxisPos, virticalAxisPos, incidentAxisPos). 
		All computations in this class are done using the standard and LOGICAL format:
		(incidentAxisPos, horizontalAxisPos, virticalAxisPos).
		"""
		# Initalize the detector triplet in the standard format
		standardTriplet = [0,0,0]

		# Arrange the axis to standard format
		for index in range(3):
			if self.xyzFormat[index] == Detector.incidentAxisPos:
				standardTriplet[0] = userTriplet[index]
			elif self.xyzFormat[index] == Detector.horizontalAxisPos:
				standardTriplet[1] = userTriplet[index]
			elif self.xyzFormat[index] == Detector.virticalAxisPos:
				standardTriplet[2] = userTriplet[index]

		return standardTriplet

	def _standard_to_user_axis(self, standardTriplet):
		"""
		Opposite of _user_to_standard_axis. Takes a standard axis order and converts into the user preference.
		"""
		# Initalize the detector triplet in the user format
		userTriplet = [0,0,0]

		# Arrange the axis to user format
		for index in range(3):
			if self.xyzFormat[index] == Detector.incidentAxisPos:
				userTriplet[index] = standardTriplet[0]
			elif self.xyzFormat[index] == Detector.horizontalAxisPos:
				userTriplet[index] = standardTriplet[1]
			elif self.xyzFormat[index] == Detector.virticalAxisPos:
				userTriplet[index] = standardTriplet[2]

		return userTriplet
	
	def _get_actual_detec_pos(self, detecPos=None):
		"""
		Given a detector position in the detector's frame of reference, applies the neccesary offsets
		to move into the goniometer's frame of reference.
		"""
		# If not given a detecPos, then we use the current detector pos
		if detecPos is None:
			detecPos = (self.incidentAxisPos, self.horizontalAxisPos, self.virticalAxisPos)

		# If we are given a detecPos, then we need to rearrange the axis to match the user's input
		else:
			detecPos = self._user_to_standard_axis(detecPos)

		# Convert into goniometer frame
		incidentAxisPos = detecPos[0] - self.offsets["incidentAxisPos"]
		horizontalAxisPos = detecPos[1] - self.offsets["horizontalAxisPos"]
		virticalAxisPos = detecPos[2] - self.offsets["virticalAxisPos"]

		return (incidentAxisPos,horizontalAxisPos,virticalAxisPos)

	def _get_actual_detec_orien(self, detecOrientation=None):
		"""
		Given a detector orientation, applies the neccesary offsets.
		"""
		# If not given a detecOrientation, then we use the current detector pos
		if detecOrientation is None:
			# If the angles are not defined, then we return none
			if self.alpha is None or self.beta is None or self.gamma is None:
				return (None, None, None)
			else:
				detecOrientation = (self.alpha, self.beta, self.gamma)
			
		# If we are given a detecOrientation, then we need to rearrange the axis to match the user's input
		else:
			detecOrientation = self._user_to_standard_axis(detecOrientation)

		# Convert into goniometer frame
		alpha = detecOrientation[0] - self.offsets["detecAlpha"]
		beta = detecOrientation[1] - self.offsets["detecBeta"]
		gamma = detecOrientation[2] - self.offsets["detecGamma"]

		return (alpha,beta,gamma)

	def set_offset(self, motorEnum, offset): 
		"""
		Sets an offset for a given detector motor. Useful for aligning the detector robot are axis with the
		goniometer center.
		"""
		# Get the motor object and name
		motor, motorName = self._get_detec_motor(motorEnum)

		# If the motor belongs to the detector set the offset
		if motorName:
			self.offsets[motorName] = offset
			return True 
		else:
			print("failed to set offset")
			return False

	def set_crop_boundary(self, new_crop_boundary=False):
		"""
		Sets a new crop bounary for the detector. We pass it through this method to check that the 
		boundary is actually within the detector.
		"""
		# If a new crop boundary is not given to us, we set the crop_boundary back to default
		if new_crop_boundary is False:
			default_crop_boundary = [[0,self.pixelWidthNum-1],[0,self.pixelHeightNum-1]]
			self.crop_boundary = default_crop_boundary
			return True

		# Check to see that the crop window is inside the detector
		width_min_condition = new_crop_boundary[0][0] >= 0 and new_crop_boundary[0][0] < self.pixelWidthNum-1
		height_min_condition = new_crop_boundary[1][0] >= 0 and new_crop_boundary[1][0] < self.pixelHeightNum-1
		width_max_condition = new_crop_boundary[0][1] > 0 and new_crop_boundary[0][1] <= self.pixelWidthNum-1
		height_max_condition = new_crop_boundary[1][1] > 0 and new_crop_boundary[1][1] <= self.pixelHeightNum-1

		if width_min_condition and height_min_condition and width_max_condition and height_max_condition:
			self.crop_boundary = new_crop_boundary
			return True

		else:
			print("invalid range")
			return False

	def move(self, detecPos, detecOrientation = None):
		"""
		Moves the detector to a new incidentAxisPos,horizontalAxisPos,z position and changes the Euler angles.
		"""
		# Get the equivalent gamma and delta values
		gamma, delta = self.detector_to_angle(detecPos)

		# Check to make sure that there are no existing limits prohibiting this move
		for banned_region in self.banned_regions:
			if self.in_range(gamma, delta, banned_region):
				print("CANNOT MOVE TO THIS POSITION DUE TO LIMIT:", banned_region)
				return None

		# If there are not:
		# Update the diffractometer angles
		self.gammaObj.set(gamma)
		self.deltaObj.set(delta)
		
		# Update the X,Y,Z positions, NO OFFSET, but we still have to rearrange the order to be standard 
		detecPos = self._user_to_standard_axis(detecPos)
		self.incidentAxisPos = detecPos[0]
		self.horizontalAxisPos = detecPos[1]
		self.virticalAxisPos = detecPos[2]

		# Update the Euler angles
		# If no angles are given, then the detecOrientation is set to None
		if detecOrientation is None:
			self.alpha = None
			self.beta = None
			self.gamma = None
		else:
			# Update the alpha, beta, and gamma NO OFFSET, but we still have to rearrange the order to be standard
			detecOrientation = self._user_to_standard_axis(detecOrientation)
			self.alpha = detecOrientation[0]
			self.beta = detecOrientation[1]
			self.gamma = detecOrientation[2]

	def detector_to_angle(self, detecPos=None, standardFormat=False):
		"""
		Given detector position in x,y,z, we calculate the corresponding gamma and delta angles
		accordingly. This allows us to convert xpp's robot arm movements to what would be the gamma
		and delta angle movement of a typical 6 circle diffractometer.

		Returns (gamma, delta)
		"""		
		# Get the true detector position
		if detecPos is None:
			detecPos = self._get_actual_detec_pos()
		else:
			# If the input is not already in the standard format (offsets not added and not in the correct order)
			if not standardFormat:
				detecPos = self._get_actual_detec_pos(detecPos)


		# Convert to goniometer frame of reference
		incidentAxisPos = detecPos[0]
		horizontalAxisPos = detecPos[1]
		virticalAxisPos = detecPos[2]

		# FIRST GAMMA IS COMPUTED
		# There may be the case in which the detector is on the z axis, in this case
		# the gamma angle is inconsequential, so we leave it be
		d = math.sqrt(incidentAxisPos**2+horizontalAxisPos**2)
		if d == 0:
			gamma = math.radians(self.gammaObj.position)
			# Gamma may not exist yet if it is being initialized
			if gamma is None:
				gamma = 0

		# There also is the case in which gamma is in quadrent III or IV (in x, y plane)
		# Here the angle just has to be shifted due to loss of sensitivity to negative x and y via d
		else:
			gamma = math.acos((incidentAxisPos)/d)
			if horizontalAxisPos < 0:
				gamma = -gamma 

		# NEXT DELTA IS COMPUTED	
		r = math.sqrt(incidentAxisPos**2+horizontalAxisPos**2+virticalAxisPos**2)

		# There may be the case that we are on the origin. This sometimes happens when you update all three
		# detector positions in a row and somewhere in the sequence of updating the detector positions x,y,z
		# all become zero.

		if r == 0:
			delta = math.radians(self.deltaObj.position)
			# Delta may not exist yet if it is being initialized
			if delta is None:
				delta = 0
		else:
			delta = math.acos(d/r) # ONLY WORKS FOR +z
			if virticalAxisPos < 0:
				delta = -delta

		# There is a 90 degree offset bc the math was done assuming that gamma = 0 when x,y,z = [0,1,0]
		gamma = math.degrees(gamma)
		delta = math.degrees(delta)

		return gamma, delta

	def angle_to_detector(self, gamma, delta, r=None):
		"""
		Given gamma and delta angles in degrees, converts to a possible x,y,z detector position while maintaining
		the current distance of the detector from the detector.
		"""
		currentX, currentY, currentZ = self._get_actual_detec_pos()

		# If we are not given an r, we set r to be the current distance of the detector from the origin
		if r is None:
			r = math.sqrt(currentX**2+currentY**2+currentZ**2)

		# First convert gamma and delta to radians
		gamma = math.radians(gamma)
		delta = math.radians(delta)

		# Compute simple polar to cartesian and add back offsets
		incidentAxisPos = r*math.cos(delta)*math.cos(gamma) + self.offsets["incidentAxisPos"]
		horizontalAxisPos = r*math.cos(delta)*math.sin(gamma) + self.offsets["horizontalAxisPos"]
		virticalAxisPos = r*math.sin(delta) + self.offsets["virticalAxisPos"]

		# Put the axis in the user defined format
		detecPos = self._standard_to_user_axis((incidentAxisPos, horizontalAxisPos, virticalAxisPos))

		return detecPos

	def create_from_xyzMatrix(self, detecPos=None, detecOrientation=None, furtherTransformation=None):
		"""
		Using detector position and orientation we create a 2D matrix of x,y,z values
		corresponding to each detector pixel. We can also optionally pass a function which
		will apply various transformations to the xyzMatrix. This can be used to compute hkl,
		gamma and delta, Q values, or anything else! The reason for structuring the code this 
		way is to streamline the process of computing these matrices while maintaing speed.
		"""
		# Get the actual detector position
		detecPos = self._get_actual_detec_pos(detecPos)
		
		# Get the actual Euler Angles
		trueAlpha, trueBeta, trueGamma = self._get_actual_detec_orien(detecOrientation)

		# If no further transformation is given, then just return back the original values
		if furtherTransformation is None:
			furtherTransformation = lambda xyz: xyz

		# The dimensions of the cropped region of the detector which we are transforming
		cropped_width = self.crop_boundary[0][1] - self.crop_boundary[0][0] + 1
		cropped_height = self.crop_boundary[1][1] - self.crop_boundary[1][0] + 1

		# The x and y coordinate of the top left pixel in the cropped region
		incidentAxisPos00 = -(((self.pixelWidthNum-1)/2) - self.crop_boundary[0][0])*self.pixel_size
		horizontalAxisPos00 = (((self.pixelHeightNum-1)/2) - self.crop_boundary[1][0])*self.pixel_size
		
		# The x and y coordinate of the bottom right pixel in the cropped_region
		incidentAxisPosNN= -(((self.pixelWidthNum-1)/2) - self.crop_boundary[0][1])*self.pixel_size
		horizontalAxisPosNN = (((self.pixelHeightNum-1)/2) - self.crop_boundary[1][1])*self.pixel_size

		# A list of x values corresponding to each column and y values to each row 
		incidentAxisVals = np.linspace(incidentAxisPos00, incidentAxisPosNN, cropped_width)
		horizontalAxisVals = (np.linspace(horizontalAxisPos00, horizontalAxisPosNN, cropped_height))[::-1]
	
		# initalize x,y,z values for each pixel in a 2x2 matrix corresponding to the detector
		xyzMatrix = np.zeros((cropped_height, cropped_width), dtype=object)
		
		# In the case that no Euler angles are specified, we compute the rotation matrix
		# assumming the normal of the detector faces the origin
		if trueAlpha is None:
			rotationM = self.get_rotation_matrix(detecPos, True)
		else:

			# If Euler angles are specified, we compute the rotation matrix here
			# Define the rotation matrix by forming 3 separete matrices corresponding to each euler angle
			cosAlpha = math.cos(math.radians(trueAlpha))
			sinAlpha = math.sin(math.radians(trueAlpha))
			cosBeta = math.cos(math.radians(trueBeta))
			sinBeta = math.sin(math.radians(trueBeta))
			cosGamma = math.cos(math.radians(trueGamma))
			sinGamma = math.sin(math.radians(trueGamma))
				
			alphaM = np.array([[1, 0, 0], [0, cosAlpha, -sinAlpha], [0, sinAlpha, cosAlpha]])
			betaM = np.array([[cosBeta, 0, sinBeta], [0, 1, 0], [-sinBeta, 0, cosBeta]])
			gammaM = np.array([[cosGamma, -sinGamma, 0], [sinGamma, cosGamma, 0],[0, 0, 1]])

			rotationM = np.matmul(gammaM, np.matmul(betaM, alphaM))

		# Define the translation vector (center of the detector in incidentAxisPos,horizontalAxisPos,z)
		translationVector = np.array(detecPos)

		# Go through each detector pixel and compute xyz vector
		for row in tqdm(range(horizontalAxisVals.size), desc="finding detector realspace positions", ncols=100, leave=False):
			for col in range(incidentAxisVals.size):
				# 1.) Assume detector is lying in incidentAxis,horitonalAxis plane with virtical = 0
				#     and the pixel size facing upwards.
				#     The center of the detector is on the origin with the (0,0) pixel in quadrent II.
				#     Compute the incidentAxisPos,y,z position of each pixel on the detector
				xyzMatrix[row, col] = np.array([incidentAxisVals[col], horizontalAxisVals[row], 0])
				
				# 2.) Rotate the detector about the the three axis
				xyzMatrix[row, col] = rotationM.dot(xyzMatrix[row, col])
				
				# 3.) Move the center of the detector to its incidentAxisPos,horizontalAxisPos,z position
				xyzMatrix[row, col] = np.add(xyzMatrix[row, col], translationVector)
				
				# 4.) Apply any additional desired transformations
				xyzMatrix[row, col] = furtherTransformation(xyzMatrix[row,col])
				 
		# Note that due to the final transformation this may no longer be the xyz matrix
		# For speed we do not create 2 matrices
		return xyzMatrix

	def create_GD_matrix(self, detecPos=None, detecOrientation=None):
		"""
		Creates a 2D matrix of gamma and delta pairs corresponding to each pixel on the detector.
		Uses the create_xyzMatrix method with a final specified transformation.
		"""
		# Define the final transformation to take place to each xyz pair after it is computed
		def xyz_to_gamma(xyz):
			return self.detector_to_angle(xyz, True)

		# Compute and return GDmatrix
		return self.create_from_xyzMatrix(detecPos, detecOrientation, xyz_to_gamma)

	def create_Q_matrix(self, detecPos=None, detecOrientation=None):
		"""
		Creates a 2D matrix of qx, qy, qz values corresponding to each pixel on the detector.
		Uses the create_xyzMatrix method with a final specified transformation.
		"""
		def xyz_to_Q(xyz):
			return self._standard_to_user_axis(self.get_pure_Q(xyz, True))

		return self.create_from_xyzMatrix(detecPos, detecOrientation, xyz_to_Q)

	def in_range(self, gamma, delta, banned_region):
		"""
		Checks to see if the given gamma and delta are within a limit
		limit: [[gammaMin, gammaMax][deltaMin, deltaMax]]
		"""
		# Define the min's and max's
		gammaMin = banned_region[0][0]
		gammaMax = banned_region[0][1]
		deltaMin = banned_region[1][0]
		deltaMax = banned_region[1][1]

		# check the current gamma and delta angles to make sure we are not in the affected range
		gammaCondition = (gamma < gammaMin or gamma > gammaMax)
		deltaCondition = (delta < deltaMin or delta > deltaMax)

		# If the current detector position is in the range
		return not (gammaCondition or deltaCondition)
	
	def set_banned_region(self, banned_region): 
		"""
		Limits our detector from being in a position within this range
		Note that this limit only includes positions within both the gamma and delta range
		"""
		# Get the current gamma and delta values
		currentGamma = self.gammaObj.position
		currentDelta = self.deltaObj.position

		# If the detector is currently in a position inside the desired new limit
		if self.in_range(currentGamma, currentDelta, banned_region):
			print("current detector position is in the desired limited range")
		else:
			self.banned_regions.append(banned_region)
	
	def get_pure_Q(self, xyzPos=None, standardFormat=False):
		"""
		To find what our Q value would be without a crystal we can calculate this using some xyz position.
		We assume that our beam is propagating along the +x axis (originating from x=-infinity) and that 
		our point of scattering it at the origin.
		NOTE: WE ASSUME THAT IS xyzPOS IS GIVEN, THE OFFSETS HAVE ALREADY BEEN ADDED, (so it can be
		used inside the create_from_xyzMatrix method)
		"""
		# Get the true detector position
		if xyzPos is None:
			xyzPos = self._get_actual_detec_pos()
		else:
			# If the input is not already in the standard format (offsets not added and not in the correct order)
			if not standardFormat:
				xyzPos = self._get_actual_detec_pos(xyzPos)

		incidentAxisPos = xyzPos[0]
		horizontalAxisPos = xyzPos[1]
		virticalAxisPos = xyzPos[2]

		# The magnitude of the k and k' vector are both equal to the standard 2pi/lambda
		kmag = 2*math.pi/self.wavelength
		
		# We generate the k vector
		kincident = kmag*np.array([1,0,0])
		
		# We generate the k' vector
		r = math.sqrt(incidentAxisPos**2+horizontalAxisPos**2+virticalAxisPos**2)

		# To prevent divsion by zero we deal with the special case of the origin
		if r == 0:
			kreflc = np.array([0,0,0])
		else:
			kreflc = kmag*np.array([incidentAxisPos/r, horizontalAxisPos/r, virticalAxisPos/r])

		# We subtract the two to find our Q vector
		Q = np.subtract(kincident, kreflc)
		
		return Q

	def get_rotation_matrix(self, detecPos=None, standardFormat=False):
		"""
		Given detector x,y,z, finds the rotation matrix that will roatate the detector
		so that it is pointing towards the origin. 
		Works but doesn't give the detector in the ideal position. For lots of positions
		it's all flipped around.
		"""
		# Get the true detector position
		if detecPos is None:
			detecPos = self._get_actual_detec_pos()
		else:
			# If the input is not already in the standard format (offsets not added and not in the correct order)
			if not standardFormat:
				detecPos = self._get_actual_detec_pos(detecPos)

		# So the best way to do this is do normalize the given vector, and then find
		# the rotation matrix such that the (0,0,1) is rotated to point in the direction
		# of the negative normalized vector. 
		
		# CREDIT TO STACK OVERFLOW for the derivation of the roatation matrix
		# Define 2 vector a and b, (a will be rotated onto b)
		a = np.array([0,0,1]) 
		b = np.array([detecPos[0], detecPos[1], detecPos[2]])

		# Normalize b and flip its direction
		b = -b/np.linalg.norm(b)

		# Define v to be the cross product of a and b
		v = np.cross(a,b)

		# The cosine of the angle between a and b 
		c = np.dot(a,b)

		# The skew-symmetric cross product matrix of v (cross product operator)
		SSCPM = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
		
		# Compute rotation matrix
		RM = np.identity(3) + SSCPM + (1/(1+c))*np.matmul(SSCPM, SSCPM)

		return RM

	def get_tangent_E_angles(self, detecPos):
		"""
		Failed attempt at writing this method which has become the bane of my existence
		"""
		incidentAxisPos = detecPos[0]
		horizontalAxisPos = detecPos[1]
		virticalAxisPos = detecPos[2]
		
		d = math.sqrt(incidentAxisPos**2 + horizontalAxisPos**2)
		r = math.sqrt(incidentAxisPos**2 + horizontalAxisPos**2 + virticalAxisPos**2)

		trueGamma = math.degrees(math.asin(horizontalAxisPos/d))
		trueBeta = math.degrees(math.acos(virticalAxisPos/r)) - 180
		trueAlpha = 0

		# Define the rotation matrix by forming 3 separete matrices corresponding to each euler angle
		cosAlpha = math.cos(math.radians(trueAlpha))
		sinAlpha = math.sin(math.radians(trueAlpha))
		cosBeta = math.cos(math.radians(trueBeta))
		sinBeta = math.sin(math.radians(trueBeta))
		cosGamma = math.cos(math.radians(trueGamma))
		sinGamma = math.sin(math.radians(trueGamma))
		
		alphaM = np.array([[1, 0, 0], [0, cosAlpha, -sinAlpha], [0, sinAlpha, cosAlpha]])
		betaM = np.array([[cosBeta, 0, sinBeta], [0, 1, 0], [-sinBeta, 0, cosBeta]])
		gammaM = np.array([[cosGamma, -sinGamma, 0], [sinGamma, cosGamma, 0],[0, 0, 1]])

		rotationM = np.matmul(alphaM, np.matmul(betaM, gammaM))

		return rotationM
	
	def plot(self, points=False, title=""):
		"""
		Plots a rectangular frame representing the detector and its orientation.
		This is helpful for testing that the detector is in the desired orientation.
		"""
		# Create figure
		fig = plt.figure()
		ax = plt.axes(projection='3d')

		# Plot the origin and the unit vectors
		ax.scatter3D(0,0,0,c="black")
		ax.plot([0,10],[0,0],[0,0],c="red")
		ax.plot([0,0],[0,10],[0,0],c="blue")
		ax.plot([0,0],[0,0],[0,10],c="green")
		
		# Plot the full detector
		# Keep track of the current crop boundary
		old_crop_boundary = copy.deepcopy(self.crop_boundary)
		
		# Temporarily undo the crop boundary to compute all xyz points for full detector
		self.set_crop_boundary()
		self._plot_cropped_region(ax, "blue", points)
		
		# Reinstate the crop boundary and plot cropped detector
		self.set_crop_boundary(old_crop_boundary)
		self._plot_cropped_region(ax, "red", points)

		# Make all axis equal
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

		# Title the graph
		if title == "":
			# Get the actual detector position and title the plot accordingly
			detecPos = self._get_actual_detec_pos()
			detecPos = self._standard_to_user_axis(detecPos)
			title = "detector position: (" + str(detecPos[0]) + ", " + str(detecPos[1]) + ", " + str(detecPos[2]) + ")"

		plt.title(title)
		ax.set_xlabel("Incident Beam Axis")
		ax.set_ylabel("Horizontal Beam Axis")
		ax.set_zlabel("Virtical Beam Axis")
		plt.show()

	def _plot_cropped_region(self, ax, color, points=False):
		"""
		A helper method to plot. Plots the outline/points of the currently cropped region.
		THE USER SHOULD NEVER USE THIS.
		"""
		# Generate the xyzMatrix corresponding to the xyz position of each pixel
		xyzMatrix = self.create_from_xyzMatrix()
		
		# If the user has requested that we plot all pixels as points
		if points:	
			for row in range(self.xyzMatrix.shape[0]):
				for col in range(self.xyzMatrix.shape[1]):
					# Plot the detector pixel in real space
					incidentAxisPos = (xyzMatrix[row, col])[0]
					horizontalAxisPos = (xyzMatrix[row, col])[1]
					virticalAxisPos = (xyzMatrix[row, col])[2]
					ax.scatter3D(incidentAxisPos,horizontalAxisPos,virticalAxisPos,c="green")

		topLeftIndex = (0, 0)
		topRightIndex = (0,xyzMatrix.shape[1]-1)
		bottomLeftIndex = (xyzMatrix.shape[0]-1, 0)
		bottomRightIndex = (xyzMatrix.shape[0]-1, xyzMatrix.shape[1]-1)

		# Now plot the outline of the detector along with corner to corner lines
		# Get the four points by applying transformations
		topLeft = xyzMatrix[topLeftIndex[0],topLeftIndex[1]]
		topRight = xyzMatrix[topRightIndex[0],topRightIndex[1]]
		bottomLeft = xyzMatrix[bottomLeftIndex[0],bottomLeftIndex[1]]
		bottomRight = xyzMatrix[bottomRightIndex[0],bottomRightIndex[1]]

		# Plot the lines
		ax.plot([topLeft[0], topRight[0]], [topLeft[1], topRight[1]], [topLeft[2], topRight[2]], c=color)
		ax.plot([topRight[0], bottomRight[0]], [topRight[1], bottomRight[1]], [topRight[2], bottomRight[2]], c=color)
		ax.plot([bottomRight[0], bottomLeft[0]], [bottomRight[1], bottomLeft[1]], [bottomRight[2], bottomLeft[2]], c=color)
		ax.plot([bottomLeft[0], topLeft[0]], [bottomLeft[1], topLeft[1]], [bottomLeft[2], topLeft[2]], c=color)
		ax.plot([bottomLeft[0], topRight[0]], [bottomLeft[1], topRight[1]], [bottomLeft[2], topRight[2]], c=color)
		ax.plot([bottomRight[0], topLeft[0]], [bottomRight[1], topLeft[1]], [bottomRight[2], topLeft[2]], c=color)

