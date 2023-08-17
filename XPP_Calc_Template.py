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

