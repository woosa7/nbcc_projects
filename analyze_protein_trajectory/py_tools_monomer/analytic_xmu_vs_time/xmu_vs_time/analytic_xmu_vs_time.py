import sys
import glob
import numpy as np

"""
extract solvation free energy and pmv from protein.xmu

input:  list for xmu files to be analyzed
output: xmu_vs_time.dat
        pmv_vs_time.dat
"""

dat_dir = sys.argv[1]
list_pdb = sorted(glob.glob('{}/*.xmu'.format(dat_dir)))
ndata = len(list_pdb)
print('total number of data =', ndata)


# extract data and save them into files
file_x = open('xmu_vs_time.dat', 'w')
file_p = open('pmv_vs_time.dat', 'w')
file_x_250 = open('xmu_vs_time_every_250ps.dat', 'w')
file_p_250 = open('pmv_vs_time_every_250ps.dat', 'w')

for k, file in enumerate(list_pdb):

    # HP36_300K_A_0000010.xmu --> 10
    itime = int(file.split('_')[-1].replace('.xmu',''))

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
