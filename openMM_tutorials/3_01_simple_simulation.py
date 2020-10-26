from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

sim_type = 'pdb'    # pdb, amber, gromacs
print('type of simulation :', sim_type)

# ----------------------------------------
# create system
if sim_type == 'amber':
    # files from AmberTools
    # prmtop file already contains the force field parameters
    prmtop = AmberPrmtopFile('HP36.prmtop')
    inpcrd = AmberInpcrdFile('HP36.inpcrd')
    system = prmtop.createSystem(nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer,
                                 constraints=HBonds)
elif sim_type == 'gromacs':
    # Unlike OpenMM and AMBER, which can store periodic unit cell information with the topology,
    # Gromacs only stores it with the coordinates.
    gro = GromacsGroFile('input.gro')
    top = GromacsTopFile('input.top',
                         periodicBoxVectors=gro.getPeriodicBoxVectors(),
                         # force field definition files
                         includeDir='/usr/local/gromacs/share/gromacs/top')
    system = top.createSystem(nonbondedMethod=PME,
                              nonbondedCutoff=1*nanometer,
                              constraints=HBonds)
else:
    # contains the molecular topology and atom positions.
    pdb = PDBFile('HP36_water.pdb')
    # Amber forcefield
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    # nonbondedMethod : the long range electrostatic interactions. PME : particle mesh Ewald.
    # nonbondedCutoff : cutoff for the direct space interactions.
    # constrain the length of all bonds that involve a hydrogen atom.
    # 10*angstrom == 1*nanometer
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1*nanometer,
                                     constraints=HBonds)

# ----------------------------------------
# creates the integrator to use for advancing the equations of motion (Langevin dynamics).
# the simulation temperature, the friction coefficient, and the step size.
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

# ----------------------------------------
# setup the simulation
if sim_type == 'amber':
    simulation = Simulation(prmtop.topology, system, integrator)
    simulation.context.setPositions(inpcrd.positions)
    # check to see if the inpcrd file contained box vectors.
    if inpcrd.boxVectors is not None:
        print(inpcrd.boxVectors)
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
elif sim_type == 'gromacs':
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)
else:
    simulation = Simulation(pdb.topology, system, integrator)
    # the initial atom positions for the simulation
    simulation.context.setPositions(pdb.positions)

# ----------------------------------------
# perform a local energy minimization
simulation.minimizeEnergy()

# tolerance
# energy가 수렴된 것으로 간주되는 시점을 지정. default : 10 kJ/mole
simulation.minimizeEnergy(tolerance=5*kilojoule/mole)

# maxIterations
# 지정하지 않으면 수렴할 때까지 계속 minimize 수행.
simulation.minimizeEnergy(tolerance=0.1*kilojoule/mole, maxIterations=500)

# ----------------------------------------
simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
                                              potentialEnergy=True,
                                              temperature=True))
simulation.step(10000)

# ----------------------------------------
