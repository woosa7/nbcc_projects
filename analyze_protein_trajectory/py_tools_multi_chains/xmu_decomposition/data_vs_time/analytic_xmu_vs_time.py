import os
import sys
import glob
import numpy as np

"""
extract solvation free energy and pmv from protein.xmu

input:  list for xmu files (protein.xmu) to be analyzed

output: xmu_vs_time.dat
        pmv_vs_time.dat

"""

dat_dir = sys.argv[1]

subdirs = ['BINARY', 'monomer_1', 'monomer_2']

for subdir in subdirs:
    # make and chage directory
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    os.chdir(subdir)

    print('processing ... xmu_vs_time.dat in {}'.format(subdir))

    # list for protein.xmu
    list_files = sorted(glob.glob('../{}/*/{}/protein.xmu'.format(dat_dir, subdir)))

    file_x = open('xmu_vs_time.dat', 'w')
    file_p = open('pmv_vs_time.dat', 'w')

    for file in list_files:
        # ../data/cv30com_rbd_A_0001000/BINARY/protein.xmu --> 1000
        time_step = int(file.split('/')[-3].split('_')[-1])

        _, values = np.loadtxt(file, dtype="S4, float", unpack=True)

        xmu = values[0]
        pmv = values[1]

        print('{:>10d}\t{:16.8f}'.format(time_step, xmu), file=file_x)
        print('{:>10d}\t{:16.8f}'.format(time_step, pmv), file=file_p)

    file_x.close()
    file_p.close()

    os.chdir('../')

print()


# =============================================================================
