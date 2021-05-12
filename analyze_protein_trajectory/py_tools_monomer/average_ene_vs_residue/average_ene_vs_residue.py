import sys
import glob
import numpy as np

"""
calculate the average of ene_vs_residue

input  : path of ene_res files in thm decomposition directory
         "../../traj_A/3D-RISM/thm_decomposition/thma_data/"

output : average_ene_vs_residue.dat

"""

dat_dir = sys.argv[1]

list_files = sorted(glob.glob('{}/*/ene_res.dat'.format(dat_dir)))
ndata = len(list_files)

# read data and caluclate the statistics
data = []
ene_all = []
for k, file in enumerate(list_files):
    ene_sum = 0.

    with open(file, 'r') as f:
        lines = f.readlines()[1:]  # skip first line (num. of residues)
        for line in lines:
            line = line.replace('\n','')
            ene_res = float(line[6:23])
            ene_sum += ene_res
            data.append(line)

    ene_all.append(ene_sum)

    if k == 0:
        n_res = len(data)
        print('number of residues    =', n_res)


print('total number of data  =', ndata)
print('average_ene_all       =', np.mean(ene_all))

f_out = open('summary_average_ene_vs_residue.dat', 'w')
print('total number of data  =', ndata, file=f_out)
print('', file=f_out)
print('average_ene_all       =', np.mean(ene_all), file=f_out)
print('', file=f_out)
f_out.close()


# calculate the average of ene_vs_residue
f_out = open('average_ene_vs_residue.dat', 'w')

for i in range(1, n_res+1):
    # select data by residue number
    res_x = [x for x in data if int(x[:6]) == i]

    l_ene_res = []
    l_ene_LJ  = []
    l_ene_cou = []
    l_ene_sig = []

    for line in res_x:
        ene_res = float(line[6:23])
        ene_res_LJ = float(line[23:40])
        ene_res_cou = float(line[40:57])

        l_ene_res.append(ene_res)
        l_ene_LJ.append(ene_res_LJ)
        l_ene_cou.append(ene_res_cou)
        l_ene_sig.append(ene_res*ene_res)

    ave_ene_tot = np.mean(l_ene_res)
    ave_ene_LJ = np.mean(l_ene_LJ)
    ave_ene_cou = np.mean(l_ene_cou)

    sig_ene_tot = np.sqrt(np.mean(l_ene_sig) - ave_ene_tot**2)

    print('{:>6d} {:16.8f} {:16.8f} {:16.8f} {:16.8f}'.format(i,
                            ave_ene_tot, ave_ene_LJ, ave_ene_cou, sig_ene_tot),
                            file=f_out)

f_out.close()

# =============================================================================
