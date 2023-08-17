# XPP_Calc
A package that allows for helpful diffraction computations at XPP

INSTALLATION STEPS:
XPP_Calc relies upon hklpy to preform reciprocal space computations. Hklpy is 
not compatible with most systems and requires both UNIX and a specific version of python to run.
Becuase of this, it is strongly recomended that all XPP_Calc users install and setup their
coding environment in the following way using anaconda (assumed to already be installed)

STEP 1: Create conda environment with hklpy:

conda create -n DUMMY_ENV_NAME -c conda-forge hklpy

STEP 2: Activate the newly create conda environment:

conda activate DUMMY_ENV_NAME

STEP 3: Donwload XPP_Calc from github:

pip install git+https://github.com/AMFischbach/XPP_Calc.git

- Note that the link may change so confirm that the github link is updated. Also note that git+ is 
added to the beginning of the url and .git is added to the end.

ENJOY!
