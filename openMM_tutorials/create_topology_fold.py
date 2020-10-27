import subprocess

# ---------------------------------------------------------------------------
# tLeap for foled structure
# ---------------------------------------------------------------------------
pdb_code = '1VIMC'

raw_pdb = '{}_H_deleted.pdb'.format(pdb_code)

# tleap for implicit water simulation
leap_script1 = 'leap_1.in'
with open(leap_script1, "w") as f:
    print('source leaprc.protein.ff14SB', file=f)
    print('prot = loadpdb {}'.format(raw_pdb), file=f)
    print('set default PBradii mbondi2', file=f)   # to prepare the prmtop file
    print('saveamberparm prot {0}.top {0}.inpcrd'.format(pdb_code), file=f)
    print('savepdb prot {}.pdb'.format(pdb_code), file=f)
    print('quit', file=f)

subprocess.call('tleap -s -f {}'.format(leap_script1), shell=True)

# tleap for pql file generation
leap_script2 = 'leap_2.in'
with open(leap_script2, "w") as f:
    print('source leaprc.protein.ff14SB', file=f)
    print('source leaprc.water.tip3p', file=f)
    print('prot = loadpdb {}'.format(raw_pdb), file=f)
    print('solvateBox prot TIP3PBOX 10.0 iso', file=f)      # solvate Model / Buffer size / iso
    print('addIons prot Na+ 0', file=f)
    print('addIons prot Cl- 0', file=f)
    print('set default PBradii mbondi2', file=f)
    print('saveamberparm prot {0}_water.top {0}_initial.crd'.format(pdb_code), file=f)
    print('savepdb prot {}_water.pdb'.format(pdb_code), file=f)
    print('quit', file=f)

subprocess.call('tleap -s -f {}'.format(leap_script2), shell=True)

# ---------------------------------------------------------------------------
