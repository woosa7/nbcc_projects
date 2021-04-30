import os
import sys
import glob
import subprocess

"""
set heavy atom contact pairs as a function of time and computes their population

input:  BINARY_nres.dat
        list for heavy_atom_contacts_residues.dat

output: contact_ID_XXXX_res_YYY_ZZZ_vs_time.dat
        population_contacts.dat

"""

dat_dir = sys.argv[1]
list_pdb = sorted(glob.glob('{}/*/heavy_atom_contacts_residues.dat'.format(dat_dir)))
n_data = len(list_pdb)
print('number of data = {}'.format(n_data))

# read BINARY_nres
res_start = []
res_end = []
with open('BINARY_nres.dat', 'r') as f:
    lines = f.readlines()[1:]  # skip first line
    for line in lines:
        istart = int(line[12:17])
        iend   = int(line[18:23])
        res_start.append(istart)
        res_end.append(iend)

nres_monomer_1 = res_end[0] - res_start[0] + 1
nres_monomer_2 = res_end[1] - res_start[1] + 1
nres_BINARY = nres_monomer_1 + nres_monomer_2

print()
print('number of residues in complex   = {}'.format(nres_BINARY))
print('number of residues in monomer 1 = {}'.format(nres_monomer_1))
print('number of residues in monomer 2 = {}'.format(nres_monomer_2))
print()


# extract heavy atom contact data
contact_data = []
contact_res = []

for k, file in enumerate(list_pdb):
    # ../data/cv30com_rbd_A_0000250/SB_HB_residues.dat --> 0000250
    time_step = file.split('/')[-2].split('_')[-1]

    with open(file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.replace('\n','')
            if len(line) == 0:
                continue

            line = line + ' ' + time_step
            contact_data.append(line)

            ires = int(line[0:3])
            jres = int(line[10:13])
            contact = (ires, jres)
            if not contact in contact_res:
                contact_res.append(contact)


ID_contact = 0

contact_data = sorted(contact_data)
contact_res = sorted(contact_res)

f_population = open('population_contacts.dat', 'w')

for ires, jres in contact_res:
    ID_contact = ID_contact + 1

    outfile = 'contact_ID_{:04d}_res_{:03d}_{:03d}_vs_time.dat'.format(ID_contact, ires, jres)
    f_out = open(outfile, 'w')

    frames = [x for x in contact_data if int(x[0:3]) == ires and int(x[10:13]) == jres]
    for x in frames:
        time_step = int(x[18:])
        print('{:>10} {:>10d}'.format(time_step, ID_contact), file=f_out)
    f_out.close()

    population = len(frames) / n_data * 100
    print('{:>3d}:  {:>3d}  {:>3d} {:>6.2f}'.format(ID_contact, ires, jres, population), file=f_population)


# move files
save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

subprocess.call('mv contact_ID_* {}/'.format(save_dir), shell=True)

# =============================================================================
