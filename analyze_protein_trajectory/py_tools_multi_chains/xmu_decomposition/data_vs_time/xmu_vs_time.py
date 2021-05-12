import os
import sys
import glob
import numpy as np

"""
xmu_vs_time writes a file in which xmu is saved as a function of time

input  : list of xmu_tot_from_xmua
         list of xmu_res

output : xmu_tot_from_xmua_vs_time.dat
         xmu_res_XXX_vs_time.dat

"""

dat_dir = sys.argv[1]

subdirs = ['BINARY', 'monomer_1', 'monomer_2']

for subdir in subdirs:
    # make and chage directory
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    os.chdir(subdir)

    # ------------------------------------------------------------
    print('processing ... xmu_tot_from_xmua_vs_time.dat in {}'.format(subdir))

    residue_info = '../residue_info_{}.dat'.format(subdir)
    with open(residue_info, 'r') as f:
        line = f.readlines()[0]  # number of residues
        n_res = int(line)

    # list of xmu_tot_from_xmua.dat
    list_files = sorted(glob.glob('../{}/*/{}/xmu_tot_from_xmua.dat'.format(dat_dir, subdir)))

    file_tot = open('xmu_tot_from_xmua_vs_time.dat', 'w')

    for file in list_files:
        # ../data/cv30com_rbd_A_0001000/BINARY/protein.xmu --> 1000
        time_step = int(file.split('/')[-3].split('_')[-1])

        with open(file, 'r') as f:
            line = f.readlines()[0].replace('\n', '')

        print('{:>10d}\t{}'.format(time_step, line), file=file_tot)

    file_tot.close()


    # ------------------------------------------------------------
    print('processing ... xmu_res_XXX_vs_time.dat in {}'.format(subdir))

    # list of xmu_res.dat
    list_files = sorted(glob.glob('../{}/*/{}/xmu_res.dat'.format(dat_dir, subdir)))

    data = []
    for file in list_files:
        # ../data/cv30com_rbd_A_0001000/BINARY/protein.xmu --> 1000
        time_step = int(file.split('/')[-3].split('_')[-1])

        with open(file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                # res_no, xmu_res, xmu_LJ, xmu_cou, time_step
                item = [int(line[:6]), float(line[7:23]), float(line[24:40]),
                        float(line[41:57]), time_step]
                data.append(item)

    # xmu by residue vs. time
    for i in range(1, n_res+1):
        # select data by residue number
        l_xmu_res = [x[1] for x in data if x[0] == i]
        l_xmu_LJ  = [x[2] for x in data if x[0] == i]
        l_xmu_cou = [x[3] for x in data if x[0] == i]
        l_times   = [x[4] for x in data if x[0] == i]

        f_out = open('xmu_res_{:03d}_vs_time.dat'.format(i), 'w')
        for idx in range(len(l_times)):
            print('{:>10d} {:16.8f} {:16.8f} {:16.8f}'.format(
                    l_times[idx], l_xmu_res[idx], l_xmu_LJ[idx], l_xmu_cou[idx],
                    ), file=f_out)

        f_out.close()

    os.chdir('../')
    print()


# =============================================================================
