from enum import Enum

class Goniometer(Enum):
	""" Enums for the four goniometer motors """
	theta = 1
	swivel_x = 2
	swivel_z = 3
	phi = 4

class Detector(Enum):
	""" Enums for the 6 detector degrees of freedom"""
	# Start counting at 5 to avoid accidental overlap with goniometer enumms
	incidentAxisPos = 5
	horizontalAxisPos = 6
	virticalAxisPos = 7
	detecAlpha = 8
	detecBeta = 9
	detecGamma = 10

class DetectorTypes(Enum):
	""" Enums for each detector type """
	Jungfrau = 1
	Zach = 2
	Testing = 3
	Testing2 = 4
	Testing3 = 5
