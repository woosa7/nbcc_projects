import os
import sys
import glob
import numpy as np

"""
analyzes decompositions of xmu

input  : xmua_atm_BINARY.dat
         xmua_atm_monomer_1.dat
         xmua_atm_monomer_2.dat

         residue_info_BINARY.dat
         residue_info_monomer_1.dat
         residue_info_monomer_2.dat

output : delta_xmu_tot_from_xmua.dat
         delta_xmua_atm.dat
         delta_xmu_res.dat
"""

dat_dir = sys.argv[1]
list_dirs = sorted(glob.glob('{}/*'.format(dat_dir)))
print('number of directories = {}'.format(len(list_dirs)))

root_dir = os.getcwd()

for k, frame_dir in enumerate(list_dirs):
    target_dir = '{}/delta'.format(frame_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    os.chdir(target_dir)

    # --------------------------------------------
    # delta_xmu_tot_from_xmua
    binary    = '../BINARY/xmu_tot_from_xmua.dat'
    monomer_1 = '../monomer_1/xmu_tot_from_xmua.dat'
    monomer_2 = '../monomer_2/xmu_tot_from_xmua.dat'

    with open(binary, 'r') as f:
        line = f.readlines()[0]
        xmua_tot_BINARY = float(line[:17])
        xmua_LJ_BINARY  = float(line[18:34])
        xmua_cou_BINARY = float(line[35:51])

    with open(monomer_1, 'r') as f:
        line = f.readlines()[0]
        xmua_tot_monomer_1 = float(line[:17])
        xmua_LJ_monomer_1  = float(line[18:34])
        xmua_cou_monomer_1 = float(line[35:51])

    with open(monomer_2, 'r') as f:
        line = f.readlines()[0]
        xmua_tot_monomer_2 = float(line[:17])
        xmua_LJ_monomer_2  = float(line[18:34])
        xmua_cou_monomer_2 = float(line[35:51])

    delta_xmua_tot = xmua_tot_BINARY - (xmua_tot_monomer_1 + xmua_tot_monomer_2)
    delta_xmua_LJ  = xmua_LJ_BINARY - (xmua_LJ_monomer_1 + xmua_LJ_monomer_2)
    delta_xmua_cou = xmua_cou_BINARY - (xmua_cou_monomer_1 + xmua_cou_monomer_2)

    # write data
    with open('delta_xmu_tot_from_xmua.dat', 'w') as f_tot:
        print('{:>16.8f} {:>16.8f} {:>16.8f}'.format(delta_xmua_tot, delta_xmua_LJ, delta_xmua_cou), file=f_tot)


    # --------------------------------------------
    # delta_xmua_atm
    binary    = '../BINARY/xmua_atm.dat'
    monomer_1 = '../monomer_1/xmua_atm.dat'
    monomer_2 = '../monomer_2/xmua_atm.dat'

    data_bi  = []
    data_mono = []

    with open(binary, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            atom = [int(line[:6]), float(line[7:23]), float(line[24:40]), float(line[41:57])]
            data_bi.append(atom)

    # Combine monomer_1 and monomer_2 in data_mono.
    with open(monomer_1, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            atom = [int(line[:6]), float(line[7:23]), float(line[24:40]), float(line[41:57])]
            data_mono.append(atom)

    with open(monomer_2, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            atom = [int(line[:6]), float(line[7:23]), float(line[24:40]), float(line[41:57])]
            data_mono.append(atom)

    n_atom = len(data_bi)
    if not n_atom == len(data_bi):
        print('error: inconsistency in atom number')
        sys.exit()

    f_atom = open('delta_xmua_atm.dat', 'w')
    print('{:>12d}'.format(n_atom), file=f_atom)
    print('      {:>16.8f} {:>16.8f} {:>16.8f}'.format(delta_xmua_tot, delta_xmua_LJ, delta_xmua_cou), file=f_atom)

    for idx in range(n_atom):
        atom_bi = data_bi[idx]
        atom_mono = data_mono[idx]

        xmua_tot = atom_bi[1] - atom_mono[1]
        xmua_LJ  = atom_bi[2] - atom_mono[2]
        xmua_cou = atom_bi[3] - atom_mono[3]

        print('{:>5d} {:>16.8f} {:>16.8f} {:>16.8f}'.format(idx+1, xmua_tot, xmua_LJ, xmua_cou), file=f_atom)

    f_atom.close()

    # --------------------------------------------
    # delta_xmua_res
    binary    = '../BINARY/xmu_res.dat'
    monomer_1 = '../monomer_1/xmu_res.dat'
    monomer_2 = '../monomer_2/xmu_res.dat'

    data_bi  = []
    data_mono = []

    with open(binary, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            res = [int(line[:6]), float(line[7:23]), float(line[24:40]), float(line[41:57])]
            data_bi.append(res)

    # Combine monomer_1 and monomer_2 in data_mono.
    with open(monomer_1, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            res = [int(line[:6]), float(line[7:23]), float(line[24:40]), float(line[41:57])]
            data_mono.append(res)

    with open(monomer_2, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            res = [int(line[:6]), float(line[7:23]), float(line[24:40]), float(line[41:57])]
            data_mono.append(res)

    n_res = len(data_bi)
    if not n_res == len(data_bi):
        print('error: inconsistency in residue number')
        sys.exit()

    f_res = open('delta_xmu_res.dat', 'w')
    print('{:>12d}'.format(n_res), file=f_res)

    for idx in range(n_res):
        res_bi = data_bi[idx]
        res_mono = data_mono[idx]

        xmu_tot = res_bi[1] - res_mono[1]
        xmu_LJ  = res_bi[2] - res_mono[2]
        xmu_cou = res_bi[3] - res_mono[3]

        print('{:>5d} {:>16.8f} {:>16.8f} {:>16.8f}'.format(idx+1, xmu_tot, xmu_LJ, xmu_cou), file=f_res)

    f_res.close()

    # --------------------------------------------
    os.chdir(root_dir)

# =============================================================================
