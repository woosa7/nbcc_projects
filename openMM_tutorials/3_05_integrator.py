from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

"""
Integrators

1. Langevin Integrator
simulation temperature, friction coefficient, step size


2. Variable Time Step Langevin Integrator
continuously adjusts its step size to keep the integration error below a specified tolerance.

It is very useful in cases where you do not know in advance what step size will be stable.
step size 대신 integration error tolerance 설정.
step size 와 integration accuracy에 영향을 주는 parameter.
Smaller values will produce a smaller average step size.
You should try different values to find the largest one
that produces a trajectory sufficiently accurate for your purposes.


3. Verlet Integrator
used for running constant energy dynamics. The only option is the step size.

* Temperature Coupling
use a Verlet integrator, then add an Andersen thermostat to your system

"""

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)


# 3번째 인수 = integration error tolerance
integrator = VariableLangevinIntegrator(300*kelvin, 1/picosecond, 0.001)


# Temperature Coupling
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
                             constraints=HBonds)
system.addForce(AndersenThermostat(300*kelvin, 1/picosecond))
integrator = VerletIntegrator(0.002*picoseconds)


# Pressure Coupling - constant pressure
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
                             constraints=HBonds)
system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)



"""
Removing Center of Mass Motion

기본적으로 system 객체는 각 time step마다 모든 center of mass motion을 제거하여
전체 system이 시간에 따라 이동하지 않도록 함.
특별한 경우 system이 시간에 따라 이동하도록 지정할 필요가 있는 경우
removeCMMotion=False 설정.
"""

system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                                 removeCMMotion=False)

# ---------------------------------------------------------------------------
