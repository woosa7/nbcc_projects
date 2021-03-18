import numpy as np

"""
a native contact refers to a heavy-atom contact in the native structure,
where a heavy-atom contact refers to a pair of heavy atoms

    1) belonging to residues with a sequence separation equal to or larger than four residues,
    2) atom-atom distance is smaller than 4.5 A.

input:  residue_info.dat (info for atoms constituting each residue in protein)
        native_structure.pdb

output: native_contacts_atoms.dat
        native_contacts_residues.dat
"""

cutoff = 4.5
cutoff2 = cutoff * cutoff


# read residue information
l_res, l_start, l_end = np.loadtxt('residue_info.dat',
                                   skiprows=1, unpack=True, dtype=int)
n_res = len(l_res)
n_atom = l_end[-1]

print('total number of residues        :', n_res)
print('total number of atoms           :', n_atom)

# read pdb data of native structure
# atom, atom_no, atom_name, res_name, res_no,
# coord_x, coord_y, coord_z, dummy, dummy, simbol
pdb = np.loadtxt('native_structure.pdb',
                 skiprows=1,
                 max_rows=n_atom,
                 dtype=[('a','S4'),('b',int),('c','S4'),('d','S4'),('e',int),
                        ('f',float),('g',float),('h',float),
                        ('i',float),('j',float),('k','S4')])

"""
(b'ATOM', 1,  b'N',  b'MET', 1, 4.27, -7.378, -7.921, 1., 0., b'N')
(b'ATOM', 5,  b'CA', b'MET', 1, 2.964, -6.831, -7.451, 1., 0., b'C')
(b'ATOM', 7,  b'CB', b'MET', 1, 1.886, -7.084, -8.51, 1., 0., b'C')
(b'ATOM', 10, b'CG', b'MET', 1, 1.749, -5.845, -9.4, 1., 0., b'C')
(b'ATOM', 13, b'SD', b'MET', 1, 1.8, -6.34, -11.146, 1., 0., b'S')
(b'ATOM', 14, b'CE', b'MET', 1, 0.294, -5.485, -11.673, 1., 0., b'C')
(b'ATOM', 18, b'C',  b'MET', 1, 2.579, -7.502, -6.132, 1., 0., b'C')
(b'ATOM', 19, b'O',  b'MET', 1, 2.66, -8.707, -5.986, 1., 0., b'O')
"""

# functions
def find_atom_res_name(atom_no):
    atom = pdb[atom_no-1]
    atom_name = atom[2].decode()
    res_name = atom[3].decode()
    res_no = atom[4]

    # set pdb-type atomname
    if len(atom_name) < 4:
        atom_name = ' ' + atom_name

    return atom_name, res_name, res_no

def find_res_name(res_no):
    ires_atoms = [x for x in pdb if x[4] == res_no]
    ires = ires_atoms[0]
    res_name = ires[3].decode()
    return res_name


# find atom pairs making native contacts
ncontact_atoms = 0
ncontact_residues = 0

# make arrays
ipair_atoms = np.zeros((n_atom+1, n_atom+1), dtype=int)
dist_atoms = np.zeros((n_atom+1, n_atom+1))
ipair_residues = np.zeros((n_res+1, n_res+1), dtype=int)

for ires in range(1, n_res+1):
    ires_atoms = [x for x in pdb if x[4] == ires]

    # only for heavy atoms
    # decode() : byte to string
    ires_atoms = [x for x in ires_atoms if x[2].decode()[0] != 'H']
    # for x in ires_atoms:
    #     print(x)

    for jres in range(ires + 4, n_res+1): # avoid neighboring pair
        jres_atoms = [x for x in pdb if x[4] == jres]
        jres_atoms = [x for x in jres_atoms if x[2].decode()[0] != 'H']

        for i_atom in ires_atoms:
            inum = i_atom[1]
            ix = i_atom[5]
            iy = i_atom[6]
            iz = i_atom[7]

            for j_atom in jres_atoms:
                jnum = j_atom[1]
                jx = j_atom[5]
                jy = j_atom[6]
                jz = j_atom[7]

                xx = (ix-jx)**2
                yy = (iy-jy)**2
                zz = (iz-jz)**2

                dist2 = xx + yy + zz

                if dist2 <= cutoff2:
                  if ipair_atoms[inum,jnum] == 0:
                      ncontact_atoms += 1
                      ipair_atoms[inum,jnum] = 1
                      dist_atoms[inum,jnum] = np.sqrt(dist2)

                  if ipair_residues[ires,jres] == 0:
                      ncontact_residues += 1
                      ipair_residues[ires,jres] = 1


# print native contacts info
f_atom = open('native_contacts_atoms.dat', 'w')
print('{:>12d}'.format(ncontact_atoms), file=f_atom)
for i in range(1, n_atom):
    for j in range(1, n_atom):
        if ipair_atoms[i,j] == 1:
            iatom, irname, ires = find_atom_res_name(i)
            jatom, jrname, jres = find_atom_res_name(j)
            print('{:>5d} {:<4} {:>3d} {} : {:>5d} {:<4} {:>3d} {} : {:8.3f}'.format(
                                                i, iatom, ires, irname,
                                                j, jatom, jres, jrname, dist_atoms[i,j]),
                                                file=f_atom)
f_atom.close()

f_res = open('native_contacts_residues.dat', 'w')
print('{:>12d}'.format(ncontact_residues), file=f_res)
for ires in range(1, n_res+1):
    for jres in range(ires+1, n_res+1):
        if ipair_residues[ires,jres] == 1:
            irname = find_res_name(ires)
            jrname = find_res_name(jres)
            print('{:>3d} {} : {:>3d} {}'.format(ires, irname, jres, jrname), file=f_res)

f_res.close()


# =============================================================================
