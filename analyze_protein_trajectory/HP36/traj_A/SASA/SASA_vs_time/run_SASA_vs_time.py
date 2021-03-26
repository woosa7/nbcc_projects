import sys
import glob

"""
 "SASA_vs_time" writes a file in which SASA is saved as a function of time

input  : directory of SASA data files

output : SASA_tot_vs_time.dat
         SASA_res_XXX_vs_time.dat
"""

dat_dir = sys.argv[1]

list_files = sorted(glob.glob('{}/*/protein.rsa'.format(dat_dir)))
ndata = len(list_files)
print('total number of data    =', ndata)


# SASA_tot_vs_time.dat
f_tot = open('SASA_tot_vs_time.dat', 'w')

for k, file in enumerate(list_files):
    # /SASA_data/HP36_300K_A_0000250/protein.rsa --> 250
    itime = int(file.split('/')[-2].split('_')[-1])

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('TOTAL'):
                sasa = line.replace('\n','').replace('TOTAL   ','')
                print('{:>10d} {}'.format(itime, sasa), file=f_tot)

f_tot.close()


# SASA_res_XXX_vs_time.dat
data = []
for k, file in enumerate(list_files):
    # /SASA_data/HP36_300K_A_0000250/protein.rsa --> 250
    itime = file.split('/')[-2].split('_')[-1]

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('RES'):
                line = line.replace('\n','')
                line = line + '    ' + itime
                data.append(line)

    if k == 0:
        n_res = len(data)
        print('number of residues      =', n_res)

for i in range(1, n_res+1):
    # select data by residue number
    res_x = [x for x in data if int(x[8:14]) == i]
    f_out = open('SASA_res_{:03d}_vs_time.dat'.format(i), 'w')

    for line in res_x:
        itime = int(line[108:])
        sasa_res    = float(line[14:23])
        sasa_NPside = float(line[29:36])
        sasa_Pside  = float(line[42:49])
        sasa_side   = float(line[55:62])
        sasa_main   = float(line[68:75])
        sasa_NP     = float(line[81:88])
        sasa_P      = float(line[94:101])
        print('{:>8d}{:16.2f}{:16.2f}{:16.2f}{:16.2f}{:16.2f}{:16.2f}{:16.2f}'.format(itime,
                            sasa_res, sasa_NPside, sasa_Pside, sasa_side,
                            sasa_main, sasa_NP, sasa_P), file=f_out)

    f_out.close()


# =============================================================================
