import numpy as np

"""
residue information to be used in the decomposition calculation

input file  : protein.pql
output file : residue_info.dat
"""

# read pql file
atom,atom_no,atom_name,res_name,res_no,coord_x,coord_y,coord_z = np.loadtxt('protein.pql',
            skiprows=1,
            dtype="S4, int, S4, S4, int, float, float, float",
            unpack=True)

n_atom = len(atom_no)
n_res = res_no[-1]
print('total number of atoms = %s' % n_atom)
print('total number of residues = %s' % n_res)

# counting residues
dict = {}
for n in res_no:
    if n in dict:
        dict[n] += 1
    else:
        dict[n] = 1

# write file
f = open('residue_info.dat', "w")
print('{:>4d}'.format(n_res), file=f)

for i in range(n_res):
    rno = i+1
    if i == 0:
        istart = 1
        iend = dict[rno]
    else:
        istart = iend + 1
        iend = iend + dict[rno]

    print('{:>4d} {:>5d} {:>5d}'.format(rno, istart, iend), file=f)

f.close()

# =============================================================================
