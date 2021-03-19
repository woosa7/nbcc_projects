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

# set parameters necessary for calculating Q_value
beta = 5.0
lambda_ = 1.8

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
print('number of atoms           =', n_atom)
print('number of residues        =', n_res)
print('num. native atom contacts =', ncontact_atoms)
print()

# calculating Q_value
f_out = open('Q_vs_time.dat', 'w')

for k, file in enumerate(list_pdb):

    # HP36_300K_A_0000010.pdb --> 10
    itime = int(file.split('_')[-1].replace('.pdb',''))

    pdb = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                pdb.append(line.replace('\n',''))

    Q_value = []
    for index_atom in range(ncontact_atoms):
        i = iatom_contact[index_atom]
        j = jatom_contact[index_atom]
        dist0 = lambda_ * dist_contact[index_atom]

        i_atom = [x for x in pdb if int(x[6:11]) == i][0]
        ix = float(i_atom[30:38])
        iy = float(i_atom[38:46])
        iz = float(i_atom[46:54])

        j_atom = [x for x in pdb if int(x[6:11]) == j][0]
        jx = float(j_atom[30:38])
        jy = float(j_atom[38:46])
        jz = float(j_atom[46:54])

        xx = (ix-jx)**2
        yy = (iy-jy)**2
        zz = (iz-jz)**2

        dist = np.sqrt(xx + yy + zz)
        contact = 1.0 / (1.0 + np.exp(beta*(dist-dist0)))

        Q_value.append(contact)

    print('{:>10d}    {:16.8f}'.format(itime, np.mean(Q_value)), file=f_out)

f_out.close()


# =============================================================================
