import os
import sys
import glob
import numpy as np

"""
find inter-monomer heavy atom contacts using the pre-defined cutoff value

input:  BINARY_natom.dat (info for number of atoms)
        residue_info.dat (info for atoms constituting each residue in protein)

        protein.pdb (complex structure) : ../../CV30_RBD/traj_A/pdb_from_crd/*.pdb

output: heavy_atom_contacts_atoms.dat
        heavy_atom_contacts_residues.dat

"""

dat_dir = sys.argv[1]
list_pdb = sorted(glob.glob('{}/*.pdb'.format(dat_dir)))
ndata = len(list_pdb)
print('number of data = {}'.format(ndata))

# set cutoff
cutoff  = 4.5
cutoff2 = cutoff * cutoff


# read BINARY_natom
atom_start = []
atom_end = []
with open('BINARY_natom.dat', 'r') as f:
    lines = f.readlines()[1:]  # skip first line
    for line in lines:
        istart = int(line[12:17])
        iend   = int(line[18:23])
        atom_start.append(istart)
        atom_end.append(iend)

natom_monomer_1 = atom_end[0] - atom_start[0] + 1
natom_monomer_2 = atom_end[1] - atom_start[1] + 1
natom_BINARY = natom_monomer_1 + natom_monomer_2


# read residue_info
res_m1 = []
res_m2 = []
end_m1 = []
end_m2 = []

with open('residue_info.dat', 'r') as f:
    lines = f.readlines()[1:]  # skip first line (num. of residues)
    for line in lines:
        ires   = int(line[:5])
        iend   = int(line[12:17])

        if iend <= natom_monomer_1: # monomer_1
            res_m1.append(ires)
            end_m1.append(iend)
        else:                       # monomer_2
            res_m2.append(ires)
            end_m2.append(iend)

if natom_BINARY != end_m2[-1]:
    print('error: inconsistency in natom_BINARY')
    sys.exit()

nres_BINARY = len(res_m1) + len(res_m2)
print()
print('number of residues in complex   = {}'.format(nres_BINARY))
print('number of residues in monomer 1 = {}'.format(len(res_m1)))
print('number of residues in monomer 2 = {}'.format(len(res_m2)))
print()
print('number of atoms in complex      = {}'.format(natom_BINARY))
print('number of atoms in monomer 1    = {}'.format(natom_monomer_1))
print('number of atoms in monomer 2    = {}'.format(natom_monomer_2))
print()


# find atom pairs making inter-monomer heavy-atom contacts

work_dir = 'data'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
os.chdir(work_dir)

for k, file in enumerate(list_pdb):
    file = '../../' + file
    frame_dir = os.path.basename(file).replace('.pdb', '')

    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    os.chdir(frame_dir)

    # collect data only for heavy atoms
    data_m1 = []
    data_m2 = []
    str_atom = ''
    str_res = ''

    ipair_residues = np.zeros((nres_BINARY+1, nres_BINARY+1), dtype=int)

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                # atom_no, atom_name,
                # res_name, res_no, chain,
                # x, y, z
                atom = [int(line[6:11]), line[12:16],
                        line[17:20], int(line[22:26]), line[21],
                        float(line[30:38]), float(line[38:46]), float(line[46:54])]

                atom_name = atom[1].replace(' ', '')
                res_no = atom[3]

                if not atom_name[0] == 'H': # only for heavy atoms
                    if res_no < res_m2[0]:
                        data_m1.append(atom)
                    else:
                        data_m2.append(atom)

    n_heavy_atoms = 0
    n_heavy_residues = 0

    for i_atom in data_m1:
        inum  = i_atom[0]
        iatom = i_atom[1]
        ires  = i_atom[2]
        iresnum = i_atom[3]
        ix = i_atom[5]
        iy = i_atom[6]
        iz = i_atom[7]

        for j_atom in data_m2:
            jnum  = j_atom[0]
            jatom = j_atom[1]
            jres  = j_atom[2]
            jresnum = j_atom[3]
            jx = j_atom[5]
            jy = j_atom[6]
            jz = j_atom[7]

            xx = (ix-jx)**2
            yy = (iy-jy)**2
            zz = (iz-jz)**2

            dist2 = xx + yy + zz
            dist  = np.sqrt(dist2)

            if dist2 <= cutoff2:
                n_heavy_atoms += 1
                # heavy_contact_atoms
                str = '{:>5d} {:<4} {:>3d} {} : {:>5d} {:<4} {:>3d} {} : {:8.3f}'.format(
                                    inum, iatom, iresnum, ires,
                                    jnum, jatom, jresnum, jres, dist)
                str_atom = str_atom + str + '\n'

                # check duplication
                if ipair_residues[iresnum,jresnum] == 0:
                    n_heavy_residues += 1
                    ipair_residues[iresnum,jresnum] = 1
                    # heavy_contact_residues
                    str = '{:>3d} {} : {:>3d} {}'.format(iresnum, ires, jresnum, jres)
                    str_res = str_res + str + '\n'


    f_atom = open('heavy_atom_contacts_atoms.dat', 'w')
    print('{:>12d}'.format(n_heavy_atoms), file=f_atom)
    print(str_atom, file=f_atom)
    f_atom.close()

    f_res = open('heavy_atom_contacts_residues.dat', 'w')
    print('{:>12d}'.format(n_heavy_residues), file=f_res)
    print(str_res, file=f_res)
    f_res.close()

    print('{} : atom {} : res {}'.format(frame_dir, n_heavy_atoms, n_heavy_residues))

    os.chdir('../')

# =============================================================================
