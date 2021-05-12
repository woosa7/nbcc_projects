import numpy as np

"""
"set_residue_info" sets a file on the residue information
to be used in the decomposition calculation

input:  BINARY_natom.dat
        BINARY_nres.dat
        protein.pql

output: residue_info_BINARY.dat
        residue_info_monomer_1.dat
        residue_info_monomer_2.dat

# natom
monomer_1:     1  6516
monomer_2:  6517  9741

# nres
monomer_1:     1   439
monomer_2:   440   648

"""

# read BINARY_natom & BINARY_nres
atom_start = []
atom_end = []
with open('BINARY_natom.dat', 'r') as f:
    lines = f.readlines()[1:]  # skip first line
    for line in lines:
        istart = int(line[12:17])
        iend   = int(line[18:23])
        atom_start.append(istart)
        atom_end.append(iend)

res_start = []
res_end = []
with open('BINARY_nres.dat', 'r') as f:
    lines = f.readlines()[1:]  # skip first line
    for line in lines:
        istart = int(line[12:17])
        iend   = int(line[18:23])
        res_start.append(istart)
        res_end.append(iend)

# read pql file
atom,atom_no,atom_name,res_name,res_no,coord_x,coord_y,coord_z = np.loadtxt('protein.pql',
            skiprows=1,
            dtype="S4, int, S4, S4, int, float, float, float",
            unpack=True)

n_atom = len(atom_no)
n_res = res_no[-1]

natm_monomer_1 = atom_end[0] - atom_start[0] + 1
natm_monomer_2 = atom_end[1] - atom_start[1] + 1
nres_monomer_1 = res_end[0] - res_start[0] + 1
nres_monomer_2 = res_end[1] - res_start[1] + 1

print('number of atoms in BINARY       = %s' % n_atom)
print('number of atoms in monomer 1    = %s' % natm_monomer_1)
print('number of atoms in monomer 2    = %s' % natm_monomer_2)
print()
print('number of residues in BINARY    = %s' % n_res)
print('number of residues in monomer 1 = %s' % nres_monomer_1)
print('number of residues in monomer 2 = %s' % nres_monomer_2)
print()

# counting residues
dict = {}
for n in res_no:
    if n in dict:
        dict[n] += 1
    else:
        dict[n] = 1

# output the results
f = open('residue_info_BINARY.dat', "w")
print('{:>4d}'.format(n_res), file=f)

f_m1 = open('residue_info_monomer_1.dat', "w")
print('{:>4d}'.format(nres_monomer_1), file=f_m1)

f_m2 = open('residue_info_monomer_2.dat', "w")
print('{:>4d}'.format(nres_monomer_2), file=f_m2)

for i in range(n_res):
    rno = i+1
    if rno == 1:
        istart = 1
        iend = dict[rno]
    else:
        istart = iend + 1
        iend = iend + dict[rno]

    # BINARY
    print('{:>4d} {:>5d} {:>5d}'.format(rno, istart, iend), file=f)

    if rno <= res_end[0]:
        # monomer_1
        print('{:>4d} {:>5d} {:>5d}'.format(rno, istart, iend), file=f_m1)
    else:
        # monomer_2 : re-numbering
        _atoms = dict[rno]
        rno2 = rno - res_end[0]

        if rno2 == 1:
            istart2 = 1
            iend2 = _atoms
        else:
            istart2 = iend2 + 1
            iend2 = iend2 + _atoms

        print('{:>4d} {:>5d} {:>5d}'.format(rno2, istart2, iend2), file=f_m2)

f.close()
f_m1.close()
f_m2.close()

# =============================================================================
