import sys
import glob
import numpy as np

"""
calculate the number of hydrophobic contacts

A hydrophobic contact refers to a contact between heavy atoms (cutoff 5.4 A)
in side chains of hydrophobic residues (ALA, VAL, LEU, ILE, MET, PHE, CYS).
Only residue pairs with a sequence separation equal to or larger than four
residues are considered.

input:  list of pdb files to be analyzed
output: hydrophobic_contacts.dat
"""

dat_dir = sys.argv[1]
list_pdb = sorted(glob.glob('{}/*.pdb'.format(dat_dir)))
ndata = len(list_pdb)
print('total number of pdb files =', ndata)


cutoff  = 5.4
cutoff2 = cutoff * cutoff

hydrophobic_res = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'CYS']
main_atoms = ['H', 'CA', 'C', 'O', 'N']

f_out = open('hydrophobic_contacts.dat', 'w')

for k, file in enumerate(list_pdb):

    n_atom = 0
    l_res  = []
    pdb = []

    """
    https://www.rbvi.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    'ATOM'
    atom_no   = line[6:11]
    atom_name = line[12:16]
    res_name  = line[17:20]
    chain     = line[21]
    res_no    = line[22:26]
    coord_x   = line[30:38]
    coord_y   = line[38:46]
    coord_z   = line[46:54]
    """

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                atom_name = line[12:16].replace(' ', '')
                res_name = line[17:20]
                res = int(line[22:26])

                n_atom += 1
                l_res.append(res) # for count the number of residue

                # only for heavy atoms
                if atom_name[0] == 'H':
                    continue

                # only for side chain atoms
                if atom_name in main_atoms:
                    continue

                # only for hydrophobic residue
                if res_name not in hydrophobic_res:
                    continue

                pdb.append(line.replace('\n',''))

    if k == 0:
        n_res = len(list(set(l_res)))
        print('number of atoms           =', n_atom)
        print('number of residues        =', n_res)
        # for x in pdb:
        #     print(x)

    # count the number of the hydrophobic contacts in the current pdb
    ipair_residues = np.zeros((n_res+1, n_res+1), dtype=int)

    for ires in range(1, n_res+1):
        ires_atoms = [x for x in pdb if int(x[22:26]) == ires]

        for jres in range(ires + 4, n_res+1): # avoid neighboring pair
            jres_atoms = [x for x in pdb if int(x[22:26]) == jres]

            for i_atom in ires_atoms:
                inum = int(i_atom[6:11])
                ix = float(i_atom[30:38])
                iy = float(i_atom[38:46])
                iz = float(i_atom[46:54])

                for j_atom in jres_atoms:
                    jnum = int(j_atom[6:11])
                    jx = float(j_atom[30:38])
                    jy = float(j_atom[38:46])
                    jz = float(j_atom[46:54])

                    xx = (ix-jx)**2
                    yy = (iy-jy)**2
                    zz = (iz-jz)**2

                    dist2 = xx + yy + zz

                    if dist2 <= cutoff2:
                        ipair_residues[ires,jres] = 1

    print('{:>8d} {:>8d}'.format(k+1, np.sum(ipair_residues)), file=f_out)


f_out.close()

# =============================================================================
