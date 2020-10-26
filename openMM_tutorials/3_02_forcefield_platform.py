from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

'''
ForceField

* (main force field, water model)
* amber14-all.xml : a shortcut for loading several different files
  that together make up the AMBER14 force field.

* .pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages/simtk/openmm/app/data/

amber14/protein.ff14SB.xml      Protein (recommended)
amber14/DNA.OL15.xml            DNA (recommended)
amber14/RNA.OL3.xml             RNA
amber14/lipid17.xml             Lipid

* Explicit Solvent Model
amber14/tip3p.xml               Water model : tip3p, tip3pfb, tip4pew, tip4pfb

* Implicit (GBSA-OBC) Solvation Model
amber96_obc.xml, amber99_obc.xml, amber03_obc.xml, amber10_obc.xml
'''

pdb = PDBFile('HP36_water.pdb')
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer,
                                 constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)


'''
platform
'''
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0,1', 'Precision': 'double'}  # use 2 GPUs
simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)


# ----------------------------------------
simulation.minimizeEnergy()

simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
                                              potentialEnergy=True,
                                              temperature=True))
simulation.step(10000)

# ----------------------------------------
