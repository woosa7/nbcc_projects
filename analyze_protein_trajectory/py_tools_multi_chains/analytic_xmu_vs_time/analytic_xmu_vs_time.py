import os
import sys
import glob
import numpy as np

"""
extract solvation free energy and pmv from protein.xmu

input:  list for xmu files to be analyzed
        dat_dir="../../CV30_RBD/traj_A/3D-RISM/xmu_only_calc/data/"
        sub-directory : BINARY / monomer_1 / monomer_2

output: xmu_vs_time.dat
        pmv_vs_time.dat

"""

subdirs = ['BINARY', 'monomer_1', 'monomer_2']

dat_dir = sys.argv[1]

for subdir in subdirs:
    # change dir
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    os.chdir(subdir)

    # check the number of data
    list_pdb = sorted(glob.glob('../{}/*/{}/protein.xmu'.format(dat_dir, subdir)))
    ndata = len(list_pdb)
    print('{:>10} : total number of data = {}'.format(subdir, ndata))

    # extract data and save them into files
    file_x = open('xmu_vs_time.dat', 'w')
    file_p = open('pmv_vs_time.dat', 'w')
    file_x_250 = open('xmu_vs_time_every_250ps.dat', 'w')
    file_p_250 = open('pmv_vs_time_every_250ps.dat', 'w')

    for k, file in enumerate(list_pdb):
        # ../../../CV30_RBD/traj_A/3D-RISM/xmu_only_calc/data/cv30com_rbd_A_0001000/BINARY/protein.xmu
        itime = file.split('/')[-3]  # cv30com_rbd_A_0001000
        itime = int(itime.split('_')[-1])

        _, values = np.loadtxt(file, dtype="S4, float", unpack=True)

        xmu = values[0]
        pmv = values[1]

        print('{:>10d}\t{:16.8f}'.format(itime, xmu), file=file_x)
        print('{:>10d}\t{:16.8f}'.format(itime, pmv), file=file_p)

        if itime % 250 == 0:
            print('{:>10d}\t{:16.8f}'.format(itime, xmu), file=file_x_250)
            print('{:>10d}\t{:16.8f}'.format(itime, pmv), file=file_p_250)

    file_x.close()
    file_p.close()
    file_x_250.close()
    file_p_250.close()

    os.chdir('../')


# =============================================================================
