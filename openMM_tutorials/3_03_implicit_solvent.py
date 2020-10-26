from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

"""
AMBER Implicit Solvent

* implicitSolvent (Generalized Born models)
None : No implicit solvent is used.
HCT  : Hawkins-Cramer-Truhlar GBSA model (corresponds to igb=1 in AMBER)
OBC1 : Onufriev-Bashford-Case GBSA model using the GBOBCI parameters (corresponds to igb=2 in AMBER).
OBC2 : Onufriev-Bashford-Case GBSA model using the GBOBCII parameters (corresponds to igb=5 in AMBER).
       This is the same model used by the GBSA-OBC files.
GBn  : GBn solvation model (corresponds to igb=7 in AMBER).
GBn2 : GBn2 solvation model (corresponds to igb=8 in AMBER).

"""

# create system
prmtop = AmberPrmtopFile('HP36.prmtop')
inpcrd = AmberInpcrdFile('HP36.inpcrd')

# system = prmtop.createSystem(nonbondedMethod=PME,
#                              nonbondedCutoff=1*nanometer,
#                              constraints=HBonds)

system = prmtop.createSystem(implicitSolvent=OBC2)

# specify the dielectric constants to use for the solute and solvent
system = prmtop.createSystem(implicitSolvent=OBC2,
                             soluteDielectric=1.0, solventDielectric=78.5) # default value

# the effect of a non-zero salt concentration by the Debye-Huckel screening parameter
system = prmtop.createSystem(implicitSolvent=OBC2,
                             implicitSolventKappa=1.0/nanometer)


"""
Nonbonded Interactions
"""




# ----------------------------------------
