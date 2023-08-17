# Author: Trey Fischbach, Date Created: Jul 11, 2023, Date Last Modified: Jul 20, 2023

# Import the necessary packages
from hkl import E6C
from hkl.util import Lattice
from ophyd import PseudoSingle, SoftPositioner
from ophyd import Component as Cpt

from XPP_Simulation.XPP_Detector_Class import XPP_Detector
from XPP_Simulation.Detector_Subclasses import * 
from XPP_Calc.Enums import *
from XPP_Calc.XPP_Motor_Pos_Class import XPP_Motor_Pos 

import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class XPP_Diffractometer(E6C):
	"""
	XPP's six circle diffractometer. Inherits all of hklpy functionallity through the parent class E6C.
	"""
	# the reciprocal axes are called: pseudo in hklpy
	h = Cpt(PseudoSingle, '', kind="hinted")
	k = Cpt(PseudoSingle, '', kind="hinted")
	l = Cpt(PseudoSingle, '', kind="hinted")

	# the motor axes are called: real in hklpy
	theta = Cpt(SoftPositioner, kind="hinted")
	swivel_x = Cpt(SoftPositioner, kind="hinted")
	swivel_z = Cpt(SoftPositioner, kind="hinted")
	phi = Cpt(SoftPositioner, kind="hinted")
	gamma = Cpt(SoftPositioner, kind="hinted")
	delta = Cpt(SoftPositioner, kind="hinted")

	def __init__(self, initPos, energy, detectorType, *args, **kwargs): 
		"""
		Initialize the diffractometer.
		
		Parameters:
		initPos (XPP_Motor_Pos Object) - the initial orientation and setup of the diffractometer NO OFFSETS
		energy (float) - the energy of the X-rays in keV
		detectorType (Enum) - indicates which detector type to use

		LENGTH UNITS ARE IN MM
		"""
		super().__init__(*args, **kwargs)

		# Rename the axis to match the XPP naming convention
		self.calc.physical_axis_names = {
				"gamma":"gamma",
				"mu":"theta",
				"chi":"swivel_z",
				"phi":"phi",
				"omega":"swivel_x",
				"delta":"delta"}

		# Set the energy
		self.calc.energy = energy

		# Dictionary of offset values
		self.offsets = {
			"gamma":0,
			"theta":0,
			"swivel_z":0,
			"phi":0,
			"swivel_x":0,
			"delta":0}

		# Create the detector obj
		self.detector = get_detector(detectorType, initPos.detecPos, initPos.detecOrientation, self.gamma, self.delta, self.calc.wavelength) 

		# Initialize all diffractor motors except the detector
		self.set_goni_angle(Goniometer.theta, initPos.goniometerPos[0])
		self.set_goni_angle(Goniometer.swivel_x, initPos.goniometerPos[1])
		self.set_goni_angle(Goniometer.swivel_z, initPos.goniometerPos[2])
		self.set_goni_angle(Goniometer.phi, initPos.goniometerPos[3])

		#self.show_constraints()

	def add_sample(self, name, lattice_parms):
		"""
		Adds a sample to the diffractometer
		Parameters:
		lattice_parms (list of float)- [a,b,c,A,B,C]
		"""
		lattice = Lattice(a = lattice_parms[0], b = lattice_parms[1],
				c = lattice_parms[2], alpha = lattice_parms[3],
				beta = lattice_parms[4], gamma = lattice_parms[5])
		self.calc.new_sample(name,lattice=lattice)

	def add_reflection(self, reflection, theta=0, swivel_x=0, swivel_z=0, phi=0, gamma=0, delta=0, detecPos=None):
		"""
		Adds a reflection within the hklpy diffractometer.
		Either takes in the six circle diffractometer inputs OR the four goniometer angles with a detector x,y,z position.
		Note that the user does not have to define the non zero angles.
		"""
		# Initalize h,k,l for clarity
		h = reflection[0]
		k = reflction[1]
		l = reflection[2]

		# Adjust goniometer motor inputs to account for any offsets 
		theta = theta - self.offsets["theta"]	
		swivel_x = swivel_x - self.offsets["swivel_x"]
		swivel_x = swivel_z - self.offsets["swivel_z"]
		phi = phi - self.offsets["phi"]

		# If given a detector position, need to compute gamma and delta
		if detecPos is not None:
			# This will automatically account for the offsets included in the given detecPos
			gamma, delta = self.detector.detector_to_angle(detecPos)			
			

		# Add the reflection
		reflection = self.calc.sample.add_reflection(
				h,k,l,
				position = self.calc.Position(
					theta = theta,
					swivel_x = swivel_x,
					swivel_z = swivel_z,
					phi = phi,
					gamma = gamma,
					delta = delta)
				)
		
		# Return the reflection
		return reflection

	def compute_UB_matrix(self, r1, r2):
		"""Given two reflections (produced by the add_reflection method), comuptes the UB matrix"""
		self.calc.sample.compute_UB(r1,r2)

	def get_XPP_Motor_Pos(self):
		"""Creates an XPP_Motor_Pos object based upon current motor position"""
		goniometerPos = []
		goniometerPos.append(self.calc["theta"].value)
		goniometerPos.append(self.calc["swivel_x"].value)
		goniometerPos.append(self.calc["swivel_z"].value)
		goniometerPos.append(self.calc["phi"].value)

		detecPos = [self.detector.incidentAxisPos, self.detector.horizontalAxisPos, self.detector.virticalAxisPos]
		detecOrientation = [self.detector.alpha, self.detector.beta, self.detector.gamma]	

		return XPP_Motor_Pos(goniometerPos, detecPos, detecOrientation)
	
	def _get_goni_motor(self, motorEnum):
		"""
		Given a goniometer motor enum, returns the ophyd soft positioner object and the name as a string
		The reason for using enums is so that the user cannot move the gamma and delta functions
		outside of the XPP_Diffractometer class. This way the user interacts only with the XPP motors. 
		"""
		# For the sample angles we simply set the component to the pos
		if motorEnum == Goniometer.theta:
			return self.theta, "theta"
		elif motorEnum == Goniometer.swivel_x:
			return self.swivel_x, "swivel_x"
		elif motorEnum == Goniometer.swivel_z:
			return self.swivel_z, "swivel_z"
		elif motorEnum == Goniometer.phi:
			return self.phi, "phi"
		# If the motor belongs to the detector or is invalid
		else:
			return None, ""
	
	def set_goni_angle(self, motorEnum, pos):
		"""
		Given a Goniometer enum and a motor position, we move the corresponding goniometer motor to the specified
		position. Returns True if the motor move was successful, False otherwise.
		"""
		# Get the motor object and name
		motor, motorName = self._get_goni_motor(motorEnum)

		# If the motor belongs to the goniometer
		if motor:
			# Get limits
			lowerLimit, upperLimit = self.calc[motorName].limits

			# If the desired position is within the allowed motor range
			if pos > lowerLimit and pos < upperLimit:
				motor.set(pos-self.offsets[motorName])
				return True

			# If not
			else:
				print("out of allowed range")
				return False
		else:
			print("invalid goniometer motor")
			return False

	def set_goni_angle_limit(self, motorEnum, limit):
		"""
		Sets a limit on the motion of a goniometer angle.
		limit: (lower limit, upper limit)
		Returns True if the motor move was successful, False otherwise.
		"""
		# Get the motor object and name
		motor, motorName = self._get_goni_motor(motorEnum)

		# If the motor belongs to the goniometer
		if motor:
			pos = self.calc[motorName].value # the current position of the motor in the engine (no offset)
			offset = self.offsets[motorName]

			# If the current motor position is within the proposed limits (assumed to contain offset)
			if pos + offset  >= limit[0] and pos + offset <= limit[1]:
				# Set the internal motor limits (no offset)
				self.calc[motorName].limits = (limit[0]-offset, limit[1]-offset)
				return True

			else:
				print("this motor currently has a position which is outside the proposed limits")
				return False
		else:
			print("invalid goniometer motor")
			return False

	def fix_goni_angle(self, motorEnum, pos):
		"""
		Moves a goniometer angle to a certain value and fixs it there.
		Returns True if fixing the motor was successful, False otherwise.
		"""
		# Get the motor object and name
		motor, motorName = self._get_goni_motor(motorEnum)

		# If the motor belongs to the goniometer
		if motor:
			# Attempt to move the motor to the fixed position, motor limits may prevent this
			if self.set_goni_angle(motorEnum, pos):
				# Retrive offsets
				offset = self.offsets[motorName]

				# Fix the motor
				self.calc[motorName].fit = False
				self.calc[motorName].value = pos - offset
				self.calc[motorName].limits = (pos - offset, pos - offset)

				return True 

			else:
				print("failed to fix goniometer motor")
				return False
		else:
			print("invalid goniometer motor")
			return False
	
	def set_goni_offset(self, motorEnum, offset):
		"""
		Add an offset to a goniometer motor to match the physical motor readouts to the hklpy computation engine.
		Returns True if the offset was successful, False otherwise.
		"""
		# Get the motor object and name
		motor, motorName = self._get_goni_motor(motorEnum)

		# If the motor belongs to the goniometer set the offset
		if motorName:
			self.offsets[motorName] = offset
			return True
		else:
			print("invalid goniometer motor")
			return False 


	def hkl_to_motor(self, hkl, r=None):
		"""
		Computes all possible motor positions that will reach the given hkl value
		The reason that r is set to None is so that in the case no r is given, the detector method can
		automatically compute r using optional parameters. Since HKLPY automatically filters out the results of the theta, swivel_x, swivel_z, and phi, we must filter our the detector limits ourselves.
		Returns a list of XPP_motor_pos objects.
		"""
		# A list of all the possible values of theta, swivel_x, swivel_z, phi, gamma, and delta that satisfy hkl
		possible_angles = self.calc.forward(hkl)

		# A list of XPP_motor_pos objects which satisfy the given hkl and motor limits 
		possible_motor_pos = []

		# Must convert all the gamma and delta values to detector x,y,z
		for angles in possible_angles:
			gamma = angles.gamma
			delta = angles.delta
			
			allowed = True

			# Check to see if these angles are in an area the detector can't move
			for banned_region in self.detector.banned_regions:

				# If the proposed detector pos is inside a banned region we move to the next
				# proposed set of angles
				if self.detector.in_range(gamma, delta, banned_region):
					allowed = False 	
					break

			if allowed:
				# If the proposed detector position is in an allowed region, compute x,y,z 
				x,y,z = self.detector.angle_to_detector(gamma, delta, r) 

				# Create a motor pos object and add it to the list of posible motor positions
				# We must add back all the offsets for the goniometer motors
				goniometerPos = []
				goniometerPos.append(angles.theta+self.offsets["theta"])
				goniometerPos.append(angles.swivel_x+self.offsets["swivel_x"])
				goniometerPos.append(angles.swivel_z+self.offsets["swivel_z"])
				goniometerPos.append(angles.phi+self.offsets["phi"])

				# Detector values already have offsets included from detector class
				detecPos = (x, y, z)
				detecOrientation = self.detector.get_tangent_E_angles(detecPos)
				
				# Add the possible motor position object to the list
				possible_motor_pos.append(XPP_Motor_Pos(goniometerPos, detecPos, detecOrientation))

		return possible_motor_pos

	def detector_to_hkl(self, GDmatrix=None):
		"""
		Creates a numpy matrix of hkl values corresponding to each pixel on the detector
		The matrix is 2D with size (pixelWidthNum x pixelHeightNum) with each entry a numpy array
		of hkl values.
		Uses the create_xyzMatrix method in the detector class with a final transformation
		"""
		if GDmatrix is None:
			# Create a matrix corresponding to the gamma and delta values of each pixel
			GDmatrix = self.detector.create_GD_matrix()
		
		# Get the size of the GDmatrix to loop through all its elements
		rowNum, columnNum = GDmatrix.shape
		
		# Predefine the four goniometer motors
		theta = self.calc["theta"].value
		swivel_x = self.calc["swivel_x"].value
		swivel_z = self.calc["swivel_z"].value
		phi = self.calc["phi"].value

		# Initalize the array storing the max and min values
		hkl_max_min = [[float("inf"), float("-inf")],[float("inf"), float("-inf")],[float("inf"), float("-inf")]]
		
		# Loop through GD_matrix to generate hkl values
		for row in tqdm(range(rowNum), desc="finding detector reciprocal space positions", ncols=100, leave=False):
			for col in range(columnNum):
				# Compute hkl and update GDmatrix
				hkl = self.inverse((theta, swivel_x, swivel_z, phi, (GDmatrix[row, col])[0], (GDmatrix[row, col])[1]))
				GDmatrix[row, col] = np.array([hkl.h, hkl.k, hkl.l])

				# Define max and mins
				if hkl.h > hkl_max_min[0][1]:
					hkl_max_min[0][1] = hkl.h
				if hkl.h < hkl_max_min[0][0]:
					hkl_max_min[0][0] = hkl.h
				if hkl.k > hkl_max_min[1][1]:
					hkl_max_min[1][1] = hkl.k
				if hkl.k < hkl_max_min[1][0]:
					hkl_max_min[1][0] = hkl.k
				if hkl.l > hkl_max_min[2][1]:
					hkl_max_min[2][1] = hkl.l
				if hkl.l < hkl_max_min[2][0]:
					hkl_max_min[2][0] = hkl.l


		# GDmatrix at this point is hkl matrix, for speed we do not create 2 separate matrices
		return GDmatrix, hkl_max_min

	def plot_hkl_magnitude(self, hkl_matrix):
		"""
		Given a 2D numpy array of hkl values corresponding to each detector pixel, plot the norm of each 
		hkl vector.
		"""
		# Get dimension of the hkl matrix
		rowNum, columnNum = hkl_matrix.shape

		# Initalize the new matrix containing the magnitude of each hkl vector
		hkl_mag = np.zeros((rowNum, columnNum))

		# Loop through the matrix and compute the magnitude, adding it to hkl_mag
		for row in range(rowNum):
			for col in range(columnNum):
				hkl_mag[row, col] = math.sqrt((hkl_matrix[row,col])[0]**2 + (hkl_matrix[row,col])[1]**2 + (hkl_matrix[row,col])[2]**2)
		
		# Plot the values
		x_coords = np.arange(columnNum)
		y_coords = np.arange(rowNum)
		X, Y = np.meshgrid(x_coords, y_coords)

		plt.pcolormesh(X, Y, hkl_mag, cmap='viridis')
		plt.colorbar(label='Magnitude of hkl')
		plt.xlabel('Column Index')
		plt.ylabel('Row Index')
		plt.title('2D Color Plot of hkl_matrix')
		plt.contour(X, Y, hkl_mag, colors="black", linewidth=0.5) # Adds some nice contours
		plt.show()

