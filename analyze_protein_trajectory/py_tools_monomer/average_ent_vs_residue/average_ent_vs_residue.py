import sys
import glob
import numpy as np

"""
calculate the average of ent_vs_residue

input  : path of ent_res files in thm decomposition directory
         "../../traj_A/3D-RISM/thm_decomposition/thma_data/"

output : average_ent_vs_residue.dat

"""

dat_dir = sys.argv[1]

list_files = sorted(glob.glob('{}/*/ent_res.dat'.format(dat_dir)))
ndata = len(list_files)

# read data and caluclate the statistics
data = []
ent_all = []
for k, file in enumerate(list_files):
    ent_sum = 0.

    with open(file, 'r') as f:
        lines = f.readlines()[1:]  # skip first line (num. of residues)
        for line in lines:
            line = line.replace('\n','')
            ent_res = float(line[6:23])
            ent_sum += ent_res
            data.append(line)

    ent_all.append(ent_sum)

    if k == 0:
        n_res = len(data)
        print('number of residues    =', n_res)


print('total number of data  =', ndata)
print('average_ent_all       =', np.mean(ent_all))

f_out = open('summary_average_ent_vs_residue.dat', 'w')
print('total number of data  =', ndata, file=f_out)
print('', file=f_out)
print('average_ent_all       =', np.mean(ent_all), file=f_out)
print('', file=f_out)
f_out.close()


# calculate the average of ent_vs_residue
f_out = open('average_ent_vs_residue.dat', 'w')

for i in range(1, n_res+1):
    # select data by residue number
    res_x = [x for x in data if int(x[:6]) == i]

    l_ent_res = []
    l_ent_LJ  = []
    l_ent_cou = []
    l_ent_sig = []

    for line in res_x:
        ent_res = float(line[6:23])
        ent_res_LJ = float(line[23:40])
        ent_res_cou = float(line[40:57])

        l_ent_res.append(ent_res)
        l_ent_LJ.append(ent_res_LJ)
        l_ent_cou.append(ent_res_cou)
        l_ent_sig.append(ent_res*ent_res)

    ave_ent_tot = np.mean(l_ent_res)
    ave_ent_LJ = np.mean(l_ent_LJ)
    ave_ent_cou = np.mean(l_ent_cou)

    sig_ent_tot = np.sqrt(np.mean(l_ent_sig) - ave_ent_tot**2)

    print('{:>6d} {:16.8f} {:16.8f} {:16.8f} {:16.8f}'.format(i,
                            ave_ent_tot, ave_ent_LJ, ave_ent_cou, sig_ent_tot),
                            file=f_out)

f_out.close()

# =============================================================================
