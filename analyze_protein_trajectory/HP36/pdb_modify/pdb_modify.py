import os
import sys

pdb_file = sys.argv[1]
chains = sys.argv[2]

"""
Print only ATOM and HETATM lines. TER must be put between the chains.
Select three chains (A, H, L).
Modify the chain id from L to M.
If the residue number starts from 0, it is modified to start from 1.
"""

select_chains = chains.split(',')
print(select_chains)

dict = {} # for counting
pdb = []

with open(pdb_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith(('ATOM','HETATM','TER')):
            chain = line[21]

            # select chains
            if chain not in chains:
                continue

            # change chain L to M
            if chain == 'L':
                new_line = list(line)
                new_line[21] = 'M'
                line = ''.join(new_line)

            pdb.append(line.replace('\n',''))

        # for count number of atoms and residues
        if line.startswith('ATOM'):
            chain = line[21]
            res_no = line[22:26]

            if chain not in dict:
                l_res = [res_no]
            else:
                l_res = dict[chain]
                l_res.append(res_no)

            dict[chain] = l_res


# print chain, num of residue, num of atoms
l_chains = dict.keys()
for chain in l_chains:
    l_res = dict[chain]
    n_atom = len(l_res)
    n_res = len(list(set(l_res))) # set : left unique item.
    print(chain, n_res, n_atom)


# If the residue number starts from 0,
# it is modified to start from 1
need_change_resno = []

for chain in l_chains:
    first_atom = [x for x in pdb if x[21] == chain and x[:4] == 'ATOM'][0]
    # check res_no
    if int(first_atom[22:26]) == 0:
        need_change_resno.append(chain)

for i in range(len(pdb)):
    atom = pdb[i]
    chain = atom[21]
    res_no = int(atom[22:26])

    if chain in need_change_resno:
        new_res_no = res_no + 1
        new_line = list(atom)
        new_line[22:26] = '{:>4d}'.format(new_res_no)
        line = ''.join(new_line)
        pdb[i] = line


# output to file
pdb_modified = os.path.basename(pdb_file).replace('.pdb','_modified.pdb')
with open(pdb_modified, 'w') as f:
    for atom in pdb:
        print(atom, file=f)

    print('END', file=f)


# =============================================================================
