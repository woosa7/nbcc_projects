#!/home/nbcc/anaconda3/envs/prowave/bin/python
import os

resmap = dict()

with open('model.pdb', 'r') as pdb:
    for line in pdb:
        record = line[0:6].strip()
        if record != 'ATOM':
            continue

        atom_ind = line[6:11].strip()
        chain = line[21:22].strip()
        resnum = line[22:26].strip()

        resmap[atom_ind] = "%s %s" % (chain, resnum)

resultmap = dict()

for i in range(40):
    xmua_atm_file = 'analyses/1/xmua_atm_%d.dat' % i
    if not os.path.exists(xmua_atm_file):
        continue

    with open(xmua_atm_file, 'r') as f:
        data = f.readlines()[2:]

    for d in data:
        atom_ind, sfe, _, _ = d.split()
        residue = resmap[atom_ind]
        if residue not in resultmap:
            resultmap[residue] = 0.0
        resultmap[residue] += float(sfe)

    print((i+1)*5, end='\t')
    for _, value in resultmap.items():
        print('%.4f \t' % value, end='')
    print()
