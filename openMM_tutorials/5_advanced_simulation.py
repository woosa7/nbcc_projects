"""
Advanced Simulation
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

pdb = PDBFile('HP36_water.pdb')
forcefield = ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml')


"""
1. Simulated Annealing

ex) reduce the temperature from 300 K to 0 K in 100 increments, executing 1000 time steps at each temperature
"""

print('Simulated Annealing...')

system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer,
                                 constraints=HBonds)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))

# total step = 100 * 1000 = 100,000
for i in range(100):
    integrator.setTemperature(3*(100-i)*kelvin)
    simulation.step(1000)



"""
2. Applying an External Force to Particles

simulate a non-periodic system contained inside a spherical container with radius 2 nm.
implement the container by applying a harmonic potential to every particle:

E(r)=0              r≤2 (nm)
E(r)=100(r−2)^2     r>2 (nm)

where r is the distance of the particle from the origin.


3. Reporting Forces

OpenMM reporters store only positions, not velocities, forces, or other data.
Create reporter that outputs forces.
"""

# ---------------------------------------------
class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        # (1) The number of time steps until the next report.
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        # Whether the next report will need particle positions (2), particle velocities (3), forces (4), energies (5).
        # Whether the positions should be wrapped to the periodic box.
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(kilojoules/mole/nanometer)
        # one line per particle.
        for f in forces:
            self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))
# ---------------------------------------------

print('Non-periodic system : External Force...')

system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=CutoffNonPeriodic,
                                 nonbondedCutoff=1*nanometer,
                                 constraints=None)
force = CustomExternalForce('100*max(0, r-2)^2; r=sqrt(x*x+y*y+z*z)')
system.addForce(force)
print('Particle :', system.getNumParticles())
for i in range(system.getNumParticles()):
    force.addParticle(i, [])
integrator = LangevinIntegrator(300*kelvin, 91/picosecond, 0.002*picoseconds)


simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(ForceReporter('forces.txt', 1000))
simulation.step(10000)



"""
4. Computing potential energy from pdb
"""

import os
for file in os.listdir('structures'):
    pdb = PDBFile(os.path.join('structures', file))
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    print(file, state.getPotentialEnergy())


# ---------------------------------------------------------------------------
