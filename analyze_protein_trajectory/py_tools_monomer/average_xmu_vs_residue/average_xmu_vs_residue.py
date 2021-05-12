import sys
import glob
import numpy as np

"""
calculate the average of xmu_vs_residue

input  : directory of xmu decomposition data files

output : average_xmu_vs_residue.dat

"""

dat_dir = sys.argv[1]

list_files = sorted(glob.glob('{}/*/xmu_res.dat'.format(dat_dir)))
ndata = len(list_files)


# read data and caluclate the statistics
data = []
xmu_all = []
for k, file in enumerate(list_files):
    xmu_sum = 0.

    with open(file, 'r') as f:
        lines = f.readlines()[1:]  # skip first line (num. of residues)
        for line in lines:
            line = line.replace('\n','')
            xmu_res = float(line[6:23])
            xmu_sum += xmu_res
            data.append(line)

    xmu_all.append(xmu_sum)

    if k == 0:
        n_res = len(data)
        print('number of residues    =', n_res)


print('total number of data  =', ndata)
print('average_xmu_all       =', np.mean(xmu_all))

f_out = open('summary_average_xmu_vs_residue.dat', 'w')
print('total number of data  =', ndata, file=f_out)
print('', file=f_out)
print('average_xmu_all       =', np.mean(xmu_all), file=f_out)
print('', file=f_out)
f_out.close()


# calculate the average of xmu_vs_residue
f_out = open('average_xmu_vs_residue.dat', 'w')

for i in range(1, n_res+1):
    # select data by residue number
    res_x = [x for x in data if int(x[:6]) == i]

    l_xmu_res = []
    l_xmu_LJ  = []
    l_xmu_cou = []
    l_xmu_sig = []

    for line in res_x:
        xmu_res = float(line[6:23])
        xmu_res_LJ = float(line[23:40])
        xmu_res_cou = float(line[40:57])

        l_xmu_res.append(xmu_res)
        l_xmu_LJ.append(xmu_res_LJ)
        l_xmu_cou.append(xmu_res_cou)
        l_xmu_sig.append(xmu_res*xmu_res)

    ave_xmu_tot = np.mean(l_xmu_res)
    ave_xmu_LJ = np.mean(l_xmu_LJ)
    ave_xmu_cou = np.mean(l_xmu_cou)

    sig_xmu_tot = np.sqrt(np.mean(l_xmu_sig) - ave_xmu_tot**2)

    print('{:>6d} {:16.8f} {:16.8f} {:16.8f} {:16.8f}'.format(i,
                            ave_xmu_tot, ave_xmu_LJ, ave_xmu_cou, sig_xmu_tot),
                            file=f_out)

f_out.close()

# =============================================================================
