from setuptools import setup, find_packages

VERSION = "1.0"
DESCRIPTION = "A package that allows for helpful diffraction computations at XPP"
INSTALL_REQS = ["numpy", "matplotlib", "tqdm", "scipy"]

setup(
	name="XPP_Calc",
	version=VERSION,
	author="Trey Fischbach",
	description=DESCRIPTION,
	packages=find_packages(),
	install_requires=INSTALL_REQS,
	python_requires=">=3.9",
	url="https://github.com/AMFischbach/XPP_Diffraction_Computer"



)
