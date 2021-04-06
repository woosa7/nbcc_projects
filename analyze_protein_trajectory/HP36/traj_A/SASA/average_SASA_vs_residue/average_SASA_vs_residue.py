import sys
import glob
import numpy as np

"""
calculate the average of SASA_vs_residue

input  : directory of SASA data files

output : average_SASA_vs_residue.dat

"""

dat_dir = sys.argv[1]

list_files = sorted(glob.glob('{}/*/protein.rsa'.format(dat_dir)))
ndata = len(list_files)


# get number of residues and SASA of all residues
data = []
sasa_all = []
for k, file in enumerate(list_files):
    sasa_sum = 0.

    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('RES'):
                line = line.replace('\n','')
                sasa_res = float(line[14:23])
                sasa_sum += sasa_res
                data.append(line)

    sasa_all.append(sasa_sum)

    if k == 0:
        n_res = len(data)
        print('number of residues    =', n_res)


print('total number of data  =', ndata)
print('average_SASA_all      =', np.mean(sasa_all))

f_out = open('summary_average_SASA_vs_residue.dat', 'w')
print('total number of data  =', ndata, file=f_out)
print('', file=f_out)
print('average_SASA_all      =', np.mean(sasa_all), file=f_out)
print('', file=f_out)
f_out.close()


# calculate the average of SASA_vs_residue
f_out = open('average_SASA_vs_residue.dat', 'w')

for i in range(1, n_res+1):
    # select data by residue number
    res_x = [x for x in data if int(x[8:14]) == i]

    l_res = []
    for line in res_x:
        sasa_res = float(line[14:23])
        l_res.append(sasa_res)

    ave_SASA_res = np.mean(l_res)
    print('{:>5d} {:16.8f}'.format(i, ave_SASA_res), file=f_out)

f_out.close()



# =============================================================================
