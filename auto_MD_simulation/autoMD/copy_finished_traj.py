import os
import glob
import subprocess

data_dir  = '/homes/eta/users/aiteam/delta_f_implicit/1000_proteins'
data_dir2 = '/homes/eta/users/aiteam2/delta_f_implicit/1000_proteins'
dir_list  = ['autoMD_0','autoMD_1','autoMD_2','autoMD_3']

for entry in dir_list:
    if not os.path.exists(entry):
        continue

    os.chdir(entry)

    chain_list = sorted(os.listdir('./'))
    chain_list = [x for x in chain_list if x[0] in ['1','2','3','4','5','6','7']]

    for c_code in chain_list:
        target_dir = '{}/{}'.format(data_dir, c_code)
        if os.path.exists(target_dir):
            continue

        target_dir = '{}/{}'.format(data_dir2, c_code)
        if os.path.exists(target_dir):
            continue

        num_eu_files = len(glob.glob('{}/*/traj_*/8_Eu_plus_xmu/Eu_plus_xmu_vs_time.dat'.format(c_code)))
        num_xmu_files = len(glob.glob('{}/*/traj_*/6_calc_xmu/xmu_data/*.xmu'.format(c_code)))

        if num_eu_files == 5 and num_xmu_files == 4500:
            print(entry, c_code)
            cmd = 'cp -r {} /homes/eta/users/aiteam2/delta_f_implicit/1000_proteins/'.format(c_code)
            subprocess.call(cmd, shell=True)

    os.chdir('../')
