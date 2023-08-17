# Author: Trey Fischbach, Date Created: Jul 31, 2023, Date Last Modified: Jul 31, 2023

# Import the necessary files
from XPP_Simulation.XPP_Detector_Class import XPP_Detector
from XPP_Calc.Enums import *

def get_detector(detectorTypeEnum, detecPos, detecOrientation, gammaObj, deltaObj, wavelength):
	"""
	Given a detector type enum, returns a newly created instance of the 
	desired detector class.
	"""
	if detectorTypeEnum == DetectorTypes.Zach:
		return Zach_Detector(detecPos, detecOrientation, gammaObj, deltaObj, wavelength)
	elif detectorTypeEnum == DetectorTypes.Jungfrau:
		return Jungfrau_Detector(detecPos, detecOrientation, gammaObj, deltaObj, wavelength)
	elif detectorTypeEnum == DetectorTypes.Testing:
		return Position_Testing_Detector(detecPos, detecOrientation, gammaObj, deltaObj, wavelength) 
	elif detectorTypeEnum == DetectorTypes.Testing2:
		return Testing_Detector2(detecPos, detecOrientation, gammaObj, deltaObj, wavelength)
	elif detectorTypeEnum == DetectorTypes.Testing3:
		return Testing_Detector3(detecPos, detecOrientation, gammaObj, deltaObj, wavelength)
	else:
		print("invalid detector enum")
		return None

class Zach_Detector(XPP_Detector):
	"""
	The detector settings needed to analyze the test data zach gave me
	"""
	
	def __init__(self, detecPos, detecOrientation, gammaObj, deltaObj, wavelength):
		pixelWidthNum = 500
		pixelHeightNum = 200
		pixel_size = 75 # microns

		# Motor offsets
		offsets = {
			"incidentAxisPos":0,
			"horizontalAxisPos":0,
			"virticalAxisPos":0,
			"detecAlpha":0,
			"detecBeta":0,
			"detecGamma":0}

		detectorAttributes = (pixelWidthNum, pixelHeightNum, pixel_size, offsets)

		name = "Zach Detector"

		xyzFormat = [Detector.horizontalAxisPos, Detector.virticalAxisPos, Detector.incidentAxisPos] 
		
		super().__init__(detecPos, detecOrientation, gammaObj, deltaObj, detectorAttributes, wavelength, name, xyzFormat)

class Jungfrau_Detector(XPP_Detector):
	"""
	The detector settings needed to analyze the test data zach gave me
	"""
	
	def __init__(self, detecPos, detecOrientation, gammaObj, deltaObj, wavelength):
		pixelWidthNum = 1060
		pixelHeightNum = 1024
		pixel_size = 75 # microns

		# Motor offsets
		offsets = {
			"incidentAxisPos":0,
			"horizontalAxisPos":0,
			"virticalAxisPos":0,
			"detecAlpha":0,
			"detecBeta":0,
			"detecGamma":0}

		detectorAttributes = (pixelWidthNum, pixelHeightNum, pixel_size, offsets)

		name = "Jungfrau Detector"
		
		xyzFormat = [Detector.horizontalAxisPos, Detector.virticalAxisPos, Detector.incidentAxisPos] 

		super().__init__(detecPos, detecOrientation, gammaObj, deltaObj, detectorAttributes, wavelength, name, xyzFormat)

class Position_Testing_Detector(XPP_Detector):
	"""
	Allows for easy testing of the detector positioning and orientation with a few large pixels
	"""

	def __init__(self, detecPos, detecOrientation, gammaObj, deltaObj, wavelength):
		pixelWidthNum = 10
		pixelHeightNum = 10
		pixel_size = 5000 # microns

		# Motor offsets
		offsets = {
			"incidentAxisPos":0,
			"horizontalAxisPos":0,
			"virticalAxisPos":0,
			"detecAlpha":0,
			"detecBeta":0,
			"detecGamma":0}

		detectorAttributes = (pixelWidthNum, pixelHeightNum, pixel_size, offsets)

		name = "Position_Testing_Detector"
		
		xyzFormat = [Detector.horizontalAxisPos, Detector.virticalAxisPos, Detector.incidentAxisPos] 

		super().__init__(detecPos, detecOrientation, gammaObj, deltaObj, detectorAttributes, wavelength, name, xyzFormat)

class Testing_Detector2(XPP_Detector):
	"""
	Allows for easy testing of the detector positioning and orientation with a few large pixels
	"""

	def __init__(self, detecPos, detecOrientation, gammaObj, deltaObj, wavelength):
		pixelWidthNum = 6
		pixelHeightNum = 5
		pixel_size = 5000 # microns

		# Motor offsets
		offsets = {
			"incidentAxisPos":0,
			"horizontalAxisPos":0,
			"virticalAxisPos":0,
			"detecAlpha":0,
			"detecBeta":0,
			"detecGamma":0}

		detectorAttributes = (pixelWidthNum, pixelHeightNum, pixel_size, offsets)

		name = "Position_Testing_Detector"
		
		xyzFormat = [Detector.horizontalAxisPos, Detector.virticalAxisPos, Detector.incidentAxisPos] 

		super().__init__(detecPos, detecOrientation, gammaObj, deltaObj, detectorAttributes, wavelength, name, xyzFormat)


class Testing_Detector3(XPP_Detector):
	def __init__(self, detecPos, detecOrientation, gammaObj, deltaObj, wavelength):
		pixelWidthNum = 100
		pixelHeightNum = 100
		pixel_size = 100 # microns

		# Motor offsets
		offsets = {
			"incidentAxisPos":0,
			"horizontalAxisPos":0,
			"virticalAxisPos":0,
			"detecAlpha":0,
			"detecBeta":0,
			"detecGamma":0}

		detectorAttributes = (pixelWidthNum, pixelHeightNum, pixel_size, offsets)

		name = "Position_Testing_Detector"
		
		xyzFormat = [Detector.horizontalAxisPos, Detector.virticalAxisPos, Detector.incidentAxisPos] 

		super().__init__(detecPos, detecOrientation, gammaObj, deltaObj, detectorAttributes, wavelength, name, xyzFormat)


