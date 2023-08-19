# Import the XPP_Calc package (xdc stands for X-Ray Diffraction Computer or maybe XPP - Diffraction - Computer)
import XPP_Calc as xdc

# PART 1: Setting up your diffractometer and computing UB matrix

# The "" input is left over from an annoying hklpy thing and yes unfortunately you have to input a name
# (diffractometer energy, detector type) 
diffractometer = xdc.XPP_Diffractometer(8.0509, xdc.DetectorTypes.Jungfrau, "", name="diffractometer")

# Add silicon's reflection to the computation engine
a0 = 5.43
diffractometer.add_sample("silicon", [a0, a0, a0, 90, 90, 90])

# Create two reflections and compute the UB matrix

# For the first reflection we use the standard six circle diffractometer inputs
r1 = diffractometer.add_reflection((4,0,0), theta=0, swivel_x=-145.451, swivel_z=0, phi=0, gamma=0, delta=69.0966)

# For the second reflection we use the XPP x,y,z coordinates
# Since we don't have an x,y,z for silicon, we compute what one could be
detec_pos = diffractometer.detector.angle_to_detector(0,69.0966,100)
r2 = diffractometer.add_reflection((0,4,0), theta=0, swivel_x=-145.451, swivel_z=90, phi=0, detec_pos = detec_pos)


# It doesn't matter which way you compute the reflections, just whatever is easier for you!

# Now compute UB matrix
diffractometer.compute_UB_matrix(r1, r2)

# Report what has been setup
diffractometer.pa()

# PART 2: Checking your UB matrix

# Compute recpirocal space position for the first refelction
# Move the goniometer to the desired position
diffractometer.set_goni_angle(xdc.Goniometer.theta, 0)
diffractometer.set_goni_angle(xdc.Goniometer.swivel_x, -145.451)
diffractometer.set_goni_angle(xdc.Goniometer.swivel_z, 0)
diffractometer.set_goni_angle(xdc.Goniometer.phi, 0)

# Move the detector to the desired position
newDetecPos = diffractometer.detector.angle_to_detector(0,69.0966,100)
diffractometer.detector.move(newDetecPos)

# Compute the hkl value of the center of the detector
current_hkl = diffractometer.detector_center_to_hkl()
print("Current hkl")
print(current_hkl)

# Now lets go the other way to find how to configure the diffractometer to get to the reflection
possible_configurations = diffractometer.hkl_to_motor((4,0,0))
print("Possible Configurations")
for configuration in possible_configurations:
	print(configuration)

# Repeat this process for the second rotation as well to check how good the UB matrix is

# PART 3: Creating Detector Maps
"""
Detector Maps can be created at any time for any diffractometer orientation.
When making Detector_Map objects, be very careful about how the program thinks
the detector is positioned in real space. Addtionally, if you are only interested
in a region of your detector, make sure that this region is defined how you wish.
"""

# Move the detector to be on the incident beam axis with its surface normal pointing towards the origin
diffractometer.detector.move((0,0,100), (0,-90,90))

# Now crop the detector to only include the first half of the Jungfrau detector
# Note that one panel of the Jungfrau is 1024x512 pixels
diffractometer.detector.set_crop_boundary([[0,1023],[0,511]])

# Plot the detector in real space to confirm that the positioning and cropping is correct
# Figures should appear in NoMachine. If you are using termnial there may be an issue.
#diffractometer.detector.plot()

# Now that we've confirmed our detector is where we want it, lets create a detector map
# First compute the hkl values for each pixel along with the maximum and minimum hkl value
hkl_vals, max_min_vals = diffractometer.detector_area_to_hkl()

"""
Note that detector_area_to_hkl diffractometer does not create an Detector_Map object beacuase it
doesn't have the intensity map (detector image).

At this point you would insert your image from the detector. For this tutorial we will just use an intensity map.
IT IS ESSENTIAL THAT THE DIMENSIONS OF YOUR INTENSITY MAP EQUAL THE DIMENSIONS OF YOUR VECTOR MAP
"""
# Create the map without an intensity map (ideally you give the object the intensity map upon creation)
detector_map = xdc.Detector_Map(hkl_vals, max_min_vals)

# Add an intensity map to the object (which here is just the norm of each detector_vector)
detector_map.intensity_map = detector_map.get_vector_map_magnitude() 

# Now lets plot our detector map!

# Plot just the detector image
detector_map.plot_intensity_map_2D(title="Intensity Map")

# Plot the contour maps without the detector image
# Very important to note that even if the Detector_Map object hasn't been assigned an intensity_map,
# this function will still work.
detector_map.plot_reciprocal_map_2D(title="Countour Maps", just_contour=True)

# Plot the contour maps onto the detector image
detector_map.plot_reciprocal_map_2D(title="Countour Maps")

# Plot the detector map onto h and sqrt(k^2+l^2)
import math # We need this for the sqaure root

# Define what each axis will be
func1 = lambda hkl: math.sqrt(hkl[1]**2 + hkl[2]**2)
func2 = lambda hkl: hkl[0]

detector_map.plot_detector_map_2D(func1, func2, title="Intensity on h vs sqrt(k^2 + l^2)", xlabel="sqrt(k^2 + l^2)", ylabel="h")

# PART 4: Creating 3D volumes
"""
I want to preface this section with a disclaimer that I never really got to test this.
But from what little I have done it does appear to be working well.
This part of the tutorial will take about 20 minutes to run. If you want
shorter runtime, use one of the testing detectors that doesn't have a million pixels.
"""

# Lets move the detector around and create a list of detector maps
detector_maps = []
for i in range(8):
	diffractometer.detector.move((i*5,i*5,100),(0,-90,90))

	# Create a hkl detector map
	hkl_vector_map, mesh_boundary = diffractometer.detector_area_to_hkl()

	detector_map = xdc.Detector_Map(hkl_vector_map, mesh_boundary)
	
	# Generate magnitude of hkl values in each detector map
	detector_map.intensity_map = detector_map.get_vector_map_magnitude()

	detector_maps.append(detector_map)

# Construct Reciprocal_Volume object
threeD_volume = xdc.Reciprocal_Volume(detector_maps, 5000, ["h", "k", "l"])

# View the area of reciprocal space that all the maps cover
threeD_volume.plot_reciprocal_space_coverage()

# Take a slice of the 3D volume
threeD_volume.plot_slice((1,1,1), title = "(1,1,1)")

# Please email me at amf16@rice.edu if you have any questions about this code
