# Author: Trey Fischbach, Date Created: Jul 11, 2023, Date Last Modified: Jul 20, 2023

class XPP_Motor_Pos():
	def __init__(self, goniometerPos, detecPos, detecOrientation):
		"""
		A simple class meant to be used as a data type for a set of xpp motor positions
		NOTE THAT THE ORDER OF ELEMENTS IN goniometerPos MATTERS
		"""
		# The three categories we are grouping motors into
		self.goniometerPos = goniometerPos
		self.detecPos = detecPos
		self.detecOrientation = detecOrientation

		# Angle Motors are the diffractometer angles we can directly control with motors
		self.theta = goniometerPos[0]
		self.swivel_x = goniometerPos[1]
		self.swivel_z = goniometerPos[2]
		self.phi = goniometerPos[3]

		# Detector position is the x,y,z position of the detector 
		self.incidentAxisPos = detecPos[0]
		self.horizontalAxisPos = detecPos[1]
		self.virticalAxisPos = detecPos[2]

		# Detec orinetation is the euler angles of the detector
		self.alpha = detecOrientation[0]
		self.beta = detecOrientation[1]
		self.gamma = detecOrientation[2]
	
	def __str__(self):
		return \
		f"""
		goniometerPos - theta: {self.theta}, swivel_x: {self.swivel_x}, swivel_z: {self.swivel_z}, phi: {self.phi}, 
		detecPos - incidentAxisPos: {self.incidentAxisPos}
			 horizontalAxisPos: {self.horizontalAxisPos}
			   virticalAxisPos: {self.virticalAxisPos}, 
		detecOrientation - alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}
		"""
