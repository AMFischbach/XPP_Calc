# Author: Trey Fischbach, Date Created: Jul 11, 2023, Date Last Modified: Jul 20, 2023

class XPP_Motor_Pos():
	def __init__(self, goniometerPos, detec_pos, detec_orien):
		"""
		A simple class meant to be used as a data type for a set of xpp motor positions
		NOTE THAT THE ORDER OF ELEMENTS IN goniometerPos MATTERS
		"""
		# The three categories we are grouping motors into
		self.goniometerPos = goniometerPos
		self.detec_pos = detec_pos
		self.detec_orien = detec_orien

		# Angle Motors are the diffractometer angles we can directly control with motors
		self.theta = goniometerPos[0]
		self.swivel_x = goniometerPos[1]
		self.swivel_z = goniometerPos[2]
		self.phi = goniometerPos[3]

		# Detector position is the x,y,z position of the detector 
		self.incidentAxisPos = detec_pos[0]
		self.horizontalAxisPos = detec_pos[1]
		self.virticalAxisPos = detec_pos[2]

		# Detec orinetation is the euler angles of the detector
		self.alpha = detec_orien[0]
		self.beta = detec_orien[1]
		self.gamma = detec_orien[2]
	
	def __str__(self):
		return \
		f"""
		goniometerPos - theta: {self.theta}, swivel_x: {self.swivel_x}, swivel_z: {self.swivel_z}, phi: {self.phi}, 
		detec_pos - incidentAxisPos: {self.incidentAxisPos}
			 horizontalAxisPos: {self.horizontalAxisPos}
			   virticalAxisPos: {self.virticalAxisPos}, 
		detec_orien - alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}
		"""
