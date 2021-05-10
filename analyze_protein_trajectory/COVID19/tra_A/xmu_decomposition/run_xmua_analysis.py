import os
import sys
import glob
import numpy as np

"""
analyzes decompositions of xmu

input : xmua_atm.dat             (xmu for each atom)
        residue_info.dat         (residue info to get istart and iend)

output: xmu_tot_from_xmua.dat
        xmu_res.dat
"""

dat_dir = sys.argv[1]
list_dirs = sorted(glob.glob('{}/*'.format(dat_dir)))
print('number of directories = {}'.format(len(list_dirs)))

root_dir = os.getcwd()

subdirs = ['BINARY', 'monomer_1', 'monomer_2']

for subdir in subdirs:
    # --------------------------------------------
    # read residue_info
    l_res = []
    l_start = []
    l_end = []

    residue_info = 'residue_info_{}.dat'.format(subdir)
    with open(residue_info, 'r') as f:
        lines = f.readlines()[1:]  # skip first line (num. of residues)
        for line in lines:
            ires   = int(line[:5])
            istart = int(line[6:11])
            iend   = int(line[12:17])

            l_res.append(ires)
            l_start.append(istart)
            l_end.append(iend)

    n_res = len(l_res)
    print('number of residues in {} = {}'.format(subdir, n_res))

    # --------------------------------------------
    # read xmu for each atom from xmua_atm.dat
    for k, frame_dir in enumerate(list_dirs):
        target_dir = '{}/{}'.format(frame_dir, subdir)
        if not os.path.exists(target_dir):
            print('Not exist target dir: %s' % target_dir)
            sys.exit()

        os.chdir(target_dir)

        with open('xmua_atm.dat', 'r') as f:
            lines = f.readlines()

            # only total xmua values
            total_data = lines[1]
            xmua_tot = float(total_data[7:23])
            xmua_LJ  = float(total_data[24:40])
            xmua_cou = float(total_data[41:57])

            # write data
            with open('xmu_tot_from_xmua.dat', 'w') as f_tot:
                print('{:>16.8f} {:>16.8f} {:>16.8f}'.format(xmua_tot, xmua_LJ, xmua_cou), file=f_tot)

            # xmua by atom
            decomp_data = []
            data = lines[2:]
            for line in data:
                # atom_no, xmu_res, xmu_LJ, xmu_cou
                atom = [int(line[:6]), float(line[7:23]), float(line[24:40]), float(line[41:57])]
                decomp_data.append(atom)

        # --------------------------------------------
        # sum xmu_res by residue
        f_res = open('xmu_res.dat', 'w')
        print('{:>12d}'.format(n_res), file=f_res)
        for idx, ires in enumerate(l_res):
            istart = l_start[idx]   # atom start index by residue
            iend = l_end[idx]       # atom end index by residue

            l_xmua_res = [x[1] for x in decomp_data if x[0] >= istart and x[0] <= iend]
            l_xmua_LJ  = [x[2] for x in decomp_data if x[0] >= istart and x[0] <= iend]
            l_xmua_cou = [x[3] for x in decomp_data if x[0] >= istart and x[0] <= iend]

            print('{:>5d} {:>16.8f} {:>16.8f} {:>16.8f}'.format(
                  ires, np.sum(l_xmua_res), np.sum(l_xmua_LJ), np.sum(l_xmua_cou)),
                  file=f_res)

        f_res.close()

        os.chdir(root_dir)

# =============================================================================
