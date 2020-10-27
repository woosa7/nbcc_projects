from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

"""
Model Building --> Modeller

시뮬레이션하기 전에 pdb 파일에 대한 전처리가 필요한 경우
- 수소 원자 추가
- explicit water simulation을 위한 물 분자 추가
- 그 외 다양한 pdb 파일의 문제 수정 등


* addHydrogens()
- For each residue, it selects the protonation state that is most common at the specified pH.
- In the case of Cysteine residues, it also checks whether the residue participates in a disulfide bond.
  (Cysteine 잔기의 경우 이황화 결합에 참여하는지 여부 확인)
- Histidine has two different protonation states that are equally likely at neutral pH.
  It therefore selects which one to use based on which will form a better hydrogen bond.
  (Histidine 잔기의 경우 수소 결합에 따라 어떤 protonation 상태를 사용할 것인지 선택함)


* addSolvent()
- create a box of solvent (water and ions) around the solute
- It also determines the charge of the solute,
  and adds enough positive or negative ions to make the system neutral.
- padding : This determines the largest size of the solute along any axis (x, y, or z).
  It then creates a cubic box of width (solute size)+2*(padding).

- solvates the system with a salt solution. By default, Na+ and Cl- ions are used.
- total ionic strength, ion 종류 지정
modeller.addSolvent(forcefield, ionicStrength=0.1*molar, positiveIon='K+', negativeIon='Cl-')


* addMembrane()
- To simulate a membrane protein
modeller.addMembrane(forcefield, lipidType='POPC', minimumPadding=1*nanometer)


* deleteWater()
- remove all water molecules from the system

"""

print('Loading...')
pdb = PDBFile('input_H_deleted.pdb')
forcefield = ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml')
modeller = Modeller(pdb.topology, pdb.positions)

print('Adding hydrogens...')
modeller.addHydrogens(forcefield, pH=7.0)  # default pH

print('Adding solvent...')
# x,y,z 축에서 가장 큰 solute size에 padding 처리
modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer)


print('Create system & minimize...')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy(maxIterations=100)


print('Saving...')
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('output.pdb', 'w'))

# ---------------------------------------------------------------------------
