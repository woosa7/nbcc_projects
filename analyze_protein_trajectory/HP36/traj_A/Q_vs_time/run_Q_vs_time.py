import os
import sys
import glob
import numpy as np

"""
calculate the fraction of native contacts Q versus time

input:  list_files (list for pdb files to be analyzed)
        residue_info.dat
        native_contacts_atoms.dat
output: Q_vs_time.dat
"""

dat_dir = sys.argv[1]
list_pdb = sorted(glob.glob('{}/*.pdb'.format(dat_dir)))
ndata = len(list_pdb)

# residue_info.dat
l_res, l_start, l_end = np.loadtxt('residue_info.dat',
                                   skiprows=1, unpack=True, dtype=int)
n_res = len(l_res)
n_atom = l_end[-1]

# read native "atomic" contact info
iatom_contact = []
jatom_contact = []
dist_contact = []

with open('native_contacts_atoms.dat', 'r') as f:
    lines = f.readlines()
    ncontact_atoms = int(lines[0])

    for line in lines[1:]:  # skip first row
        i_atom = int(line[0:6])
        j_atom = int(line[22:27])
        dist = float(line[43:51])

        iatom_contact.append(i_atom)
        jatom_contact.append(j_atom)
        dist_contact.append(dist)

print('total number of pdb files =', ndata)
print('number of residues        =', n_res)
print('number of atoms           =', n_atom)
print('num. native atom contacts =', ncontact_atoms)
print()


# set parameters necessary for calculating Q_value
beta = 5.0
lambda_ = 1.8

# calculate Q_value
f_out = open('Q_vs_time.dat', 'w')

for k, file in enumerate(list_pdb):

    # HP36_300K_A_0000010.pdb --> 10
    itime = int(file.split('_')[-1].replace('.pdb',''))

    pdb = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                # atom_no, atom_name, res_name, res_no, chain, x, y, z
                atom = [int(line[6:11]), line[12:16],
                        line[17:20], int(line[22:26]), line[21],
                        float(line[30:38]), float(line[38:46]), float(line[46:54])]

                pdb.append(atom)

    Q_value = []
    for index_atom in range(ncontact_atoms):
        i = iatom_contact[index_atom]
        j = jatom_contact[index_atom]
        dist0 = lambda_ * dist_contact[index_atom]

        i_atom = [x for x in pdb if x[0] == i][0]
        ix = i_atom[5]
        iy = i_atom[6]
        iz = i_atom[7]

        j_atom = [x for x in pdb if x[0] == j][0]
        jx = j_atom[5]
        jy = j_atom[6]
        jz = j_atom[7]

        xx = (ix-jx)**2
        yy = (iy-jy)**2
        zz = (iz-jz)**2

        dist = np.sqrt(xx + yy + zz)
        contact = 1.0 / (1.0 + np.exp(beta*(dist-dist0)))

        Q_value.append(contact)

    print('{:>10d}    {:16.8f}'.format(itime, np.mean(Q_value)), file=f_out)

f_out.close()


# =============================================================================
