
Perform the tleap as follows
(-s flag tells tleap to ignore any leaprc file it might find)

  tleap -s

with the following commands

  source leaprc.protein.ff14SB
  source leaprc.water.tip3p
  prot = loadpdb 1VII_H_deleted.pdb
  solvateBox prot TIP3PBOX 10.0 iso
  addIons prot Na+ 0
  addIons prot Cl- 0
  saveAmberParm prot HP36.top HP36_initial.crd
  quit

After this is done, do the following to create pdb for checking:

  ambpdb -p HP36.top < HP36_initial.crd > HP36_initial.pdb

