import os
import sys
import glob
import subprocess

"""
set SB and HB pairs as a function of time and computes their population

input:  BINARY_nres.dat
        list for SB_HB_residues.dat

output: SB_ID_XXX_res_YYY_ZZZ_vs_time.dat
        HB_ID_XXX_res_YYY_ZZZ_vs_time.dat
        population_SB_HB.dat

"""

dat_dir = sys.argv[1]
list_pdb = sorted(glob.glob('{}/*/SB_HB_residues.dat'.format(dat_dir)))
n_data = len(list_pdb)
print('number of data = {}'.format(n_data))

# read BINARY_natom
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


# extract SB and HB data
SB_data = []
HB_data = []

SB_res = [] # tuple of contact residues : (i, j)
HB_res = []

for k, file in enumerate(list_pdb):
    # ../data/cv30com_rbd_A_0000250/SB_HB_residues.dat --> 0000250
    time_step = file.split('/')[-2].split('_')[-1]

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')

            if line.endswith('SB'):
                line = line + ' ' + time_step
                SB_data.append(line)

                ires = int(line[0:3])
                jres = int(line[10:13])
                contact = (ires, jres)
                if not contact in SB_res:
                    SB_res.append(contact)

            elif line.endswith('HB'):
                line = line + ' ' + time_step
                HB_data.append(line)

                ires = int(line[0:3])
                jres = int(line[10:13])
                contact = (ires, jres)
                if not contact in HB_res:
                    HB_res.append(contact)

            else:
                pass


ID_SB_HB = 0

SB_res = sorted(SB_res)
HB_res = sorted(HB_res)

SB_data = sorted(SB_data)
HB_data = sorted(HB_data)

f_population = open('population_SB_HB.dat', 'w')

# SB
for ires, jres in SB_res:
    ID_SB_HB = ID_SB_HB + 1

    outfile = 'SB_ID_{:03d}_res_{:03d}_{:03d}_vs_time.dat'.format(ID_SB_HB, ires, jres)

    frames = [x for x in SB_data if int(x[0:3]) == ires and int(x[10:13]) == jres]
    f_out = open(outfile, 'w')
    for x in frames:
        time_step = int(x[22:])
        print('{:>10} {:>10d}'.format(time_step, ID_SB_HB), file=f_out)
    f_out.close()

    population = len(frames) / n_data * 100
    print('{:>3d}:  {:>3d}  {:>3d}  SB  {:>6.2f}'.format(ID_SB_HB, ires, jres, population), file=f_population)

# HB
for ires, jres in HB_res:
    ID_SB_HB = ID_SB_HB + 1

    outfile = 'HB_ID_{:03d}_res_{:03d}_{:03d}_vs_time.dat'.format(ID_SB_HB, ires, jres)

    frames = [x for x in HB_data if int(x[0:3]) == ires and int(x[10:13]) == jres]
    f_out = open(outfile, 'w')
    for x in frames:
        time_step = int(x[22:])
        print('{:>10} {:>10d}'.format(time_step, ID_SB_HB), file=f_out)
    f_out.close()

    population = len(frames) / n_data * 100
    print('{:>3d}:  {:>3d}  {:>3d}  HB  {:>6.2f}'.format(ID_SB_HB, ires, jres, population), file=f_population)

f_population.close()

# move files
save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

subprocess.call('mv SB_* {}/'.format(save_dir), shell=True)
subprocess.call('mv HB_* {}/'.format(save_dir), shell=True)

# =============================================================================
