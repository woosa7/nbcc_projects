from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

"""
Nonbonded Interactions
- force field 또는 prmtop file 로부터 system 생성시 nonbonded interactions 처리 방법을 지정

NoCutoff : No cutoff is applied.

CutoffNonPeriodic :
    Eliminate all interactions beyond a cutoff distance.

CutoffPeriodic :
    Eliminate all interactions beyond a cutoff distance.
    Each atom interacts only with the nearest periodic copy of every other atom.

Ewald :
    Periodic boundary conditions are applied.
    Compute long range Coulomb interactions.
    (This option is rarely used, since PME is much faster for all but the smallest systems.)

PME (Particle Mesh Ewald) :
    Periodic boundary conditions are applied.
    Compute long range Coulomb interactions.

LJPME :
    Periodic boundary conditions are applied.
    Compute long range interactions for both Coulomb and Lennard-Jones.
"""

prmtop = AmberPrmtopFile('HP36.prmtop')
inpcrd = AmberInpcrdFile('HP36.inpcrd')

system = prmtop.createSystem(nonbondedMethod=PME,
                             nonbondedCutoff=1*nanometer,
                             constraints=HBonds)


"""
error tolerance 지정 가능 (Ewald, PME, or LJPME)

The error tolerance is roughly equal to the fractional error in the forces
due to truncating the Ewald summation.

* default value = 0.0005
"""

system = prmtop.createSystem(nonbondedMethod=PME,
                             nonbondedCutoff=1*nanometer,
                             ewaldErrorTolerance=0.00001)

"""
switchDistance

Lennard-Jones interactions은 cutoff distance에서 명확하게 truncate 되지 않고
0으로 수렴하는 경향이 있음
이 옵션은 energy conservation을 향상시켜 준다.
"""

system = prmtop.createSystem(nonbondedMethod=PME,
                             nonbondedCutoff=1*nanometer,
                             switchDistance=0.9*nanometer)


# ---------------------------------------------------------------------------
"""
Constraints
- constrain certain bond lengths and angles
- it allows one to use a larger integration time step. (1 fs --> 2 fs)

None     : default value.
HBonds   : The lengths of all bonds that involve a hydrogen atom are constrained.
AllBonds : The lengths of all bonds are constrained.
HAngles  : The lengths of all bonds are constrained.
           In addition, all angles of the form H-X-H or H-O-X (X is an arbitrary atom) are constrained.

"""

system = prmtop.createSystem(nonbondedMethod=PME,
                             nonbondedCutoff=1*nanometer,
                             constraints=HBonds)


"""
OpenMM makes water molecules completely rigid, constraining both their bond lengths and angles.
rigidWater=False --> disable this behavior (flexible water).
reduce the integration step size, typically to about 0.5 fs.
"""

system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=None, rigidWater=False)


# ---------------------------------------------------------------------------
"""
Heavy Hydrogens
- increase the mass of hydrogen atoms.

This applies only to hydrogens that are bonded to heavy atoms,
and any mass added to the hydrogen is subtracted from the heavy atom.
This keeps their total mass constant while slowing down the fast motions of hydrogens.
When combined with constraints (constraints=AllBonds),
this allows a further increase in integration step size.
"""

system = prmtop.createSystem(hydrogenMass=4*amu)




# ---------------------------------------------------------------------------
