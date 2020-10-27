from simtk.openmm import app
from simtk import unit, openmm
# woosa7
from reporters.crdreporter import CRDReporter

# ---------------------------------------------------------------------------
# GPU set up
GPU_NUM = 0

# ---------------------------------------------------------------------------
# Create system
# GBSA model using OBC2 parameters
# default soluteDielectric = 1.0 and solventDielectic = 78.5 are adopted

prmtop = app.AmberPrmtopFile('protein.top')
inpcrd = app.AmberInpcrdFile('protein.inpcrd')
system = prmtop.createSystem(implicitSolvent=app.OBC2,
                             implicitSolventKappa=1.0/unit.nanometer,
                             constraints=app.HBonds,
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.4*unit.nanometer)

# Create simulation context

temp0    = 300.0 # in unit.kelvin
gamma_ln = 2.0   # in 1/unit.picosecond
dt       = 0.002 # in unit.picosecond

integrator = openmm.LangevinIntegrator(temp0*unit.kelvin,
                                       gamma_ln/unit.picoseconds,
                                       dt*unit.picosecond)

platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed',
              'DeviceIndex': '{}'.format(GPU_NUM)}

simulation = app.Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)

# Minimization

simulation.minimizeEnergy()

# Run simulation

md_steps = 500000000   # 1,000 ns = 1 microsecond. (10,000 structures)

crd_file = 'protein.crd'
dcd_file = 'protein.dcd'
dcd_period = 50000     # 100 ps interval (50000 * dt)

log_file = 'protein.log'
log_period = 50000     # 100 ps interval (50000 * dt)

simulation.reporters.append(CRDReporter(crd_file, dcd_period))
simulation.reporters.append(app.DCDReporter(dcd_file, dcd_period))
simulation.reporters.append(app.StateDataReporter(log_file, log_period,
                    step=True, potentialEnergy=True, kineticEnergy=True,
                    totalEnergy=True, temperature=True,
                    elapsedTime=True, remainingTime=True, totalSteps=md_steps,
                    separator=' '))

simulation.step(md_steps)

# ---------------------------------------------------------------------------
