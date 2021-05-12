import os
import sys
import subprocess
import time
import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb
import make_3Dimage
import make_3Dimage_LJ_elec
import make_single_features

# ============================================================================
def make_dir(main_dir, sub_dir):
    target_dir = '{}/{}'.format(main_dir, sub_dir)
    print(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# ============================================================================
# check parameter file
params_file = ''
try:
    params_file = sys.argv[1]
except IndexError as e:
    params_file = 'feature_parameters'

if os.path.exists(params_file):
    for line in open(params_file, 'r'):
        if len(line.strip()) > 0:
            exec(line)
else:
    print('Not exist the md_parameters file')
    sys.exit()

conformation_num = 20
ppdb = PandasPdb()

# set directory
work_dir = os.path.abspath(work_dir)

trans_dir = os.path.join(work_dir, 'trans_data')
if not os.path.exists(trans_dir):
    os.makedirs(trans_dir)
    list_dirs = ['3D', 'charge', 'residue', 'RT', 'RT_frac', 'terminal', 'ydata', 'pdb']
    for folder in list_dirs:
        make_dir(trans_dir, folder)

log_error = '{}/log_error'.format(trans_dir)
with open(log_error, "w") as f:
    print('log_error', file=f)

log_process = '{}/log_process'.format(trans_dir)
with open(log_process, "w") as f:
    print('log_process', file=f)

# ============================================================================
# job loop
group_dirs = sorted(next(os.walk(work_dir))[1])
group_dirs = [k for k in group_dirs if k[0].isdigit()]

count_num = 0
for g_code in group_dirs:
    second_dirs = sorted(next(os.walk('{}/{}'.format(work_dir, g_code)))[1])
    for p_code in second_dirs:

        # if not p_code == '5W8OB': continue

        xmu_file = '{}/{}/{}/xmu/xmu_vs_time.dat'.format(work_dir, g_code, p_code)
        if os.path.exists(xmu_file):
            start_time = time.time()
            # ----------------------------------------------------------------
            # extract sequence
            src_dir = '{}/{}/{}'.format(work_dir, g_code, p_code)
            # print('---------------------')
            # print('make features :', src_dir)

            pdb_file = '{}/pdb/{}_1000.pdb'.format(src_dir, p_code)
            target_file = '{}/pdb/{}_1000.pdb'.format(trans_dir, p_code)
            if not os.path.exists(target_file):
                subprocess.call('cp {} {}'.format(pdb_file, target_file), shell=True)

            ppdb.read_pdb(pdb_file)
            df = ppdb.df['ATOM'][['residue_number','residue_name','chain_id']]
            df = df[df['chain_id'] == 'A']
            df_model = df.groupby(['residue_number', 'residue_name']).count()
            res_sequence = df_model.index.get_level_values(1).tolist()
            res_num = len(res_sequence)

            # ----------------------------------------------------------------
            # 3D image
            # if m_3Dimage_basic:
            #     feature_3D = make_3Dimage.Image3D(src_dir, trans_dir, p_code)
            #     result = feature_3D.make_feature()
            #     print('3D image : {}'.format(result))

            # ----------------------------------------------------------------
            # 3D image - atom charge & LJ parameters
            if m_3Dimage_LJ:
                feature_3D_LJ = make_3Dimage_LJ_elec.Image3D_LJ_elec(src_dir, trans_dir, p_code, log_error)
                result = feature_3D_LJ.make_feature()
                print('3D image : {}'.format(result))

            # ----------------------------------------------------------------
            featurizer = make_single_features.Featurizer(src_dir, trans_dir, log_error)

            # residual Gsolv - ydata
            result = featurizer.residual_Gsolv(p_code)
            print('residue Gsolv : {}'.format(result))

            # ----------------------------------------------------------------
            # total charge
            if m_total_charge:
                result1 = featurizer.total_charge(p_code, conformation_num, res_num, res_sequence)
                print('total charge  : {}'.format(result1))

            # ----------------------------------------------------------------
            # total residue
            if m_total_res:
                result2 = featurizer.total_residue(p_code, conformation_num, res_num)
                # print('total residue : {}'.format(result2))

            # ----------------------------------------------------------------
            # residue type
            if m_res_type:
                result3 = featurizer.residue_type(p_code, conformation_num, res_num, res_sequence)
                print('residue type  : {}'.format(result3))

            # ----------------------------------------------------------------
            # freaction of residue type
            if m_res_fraction:
                result4 = featurizer.residue_fraction(p_code, conformation_num, res_num, res_sequence)
                # print('residue fraction : {}'.format(result4))

            # ----------------------------------------------------------------
            # terminal type
            if m_terminal:
                result5 = featurizer.terminal_type(p_code, conformation_num, res_num)
                # print('terminal type : {}'.format(result5))

            # # ----------------------------------------------------------------
            # # salt bridge
            # if m_saltbridge:
            #     result6 = featurizer.salt_bridge(p_code, res_num, res_sequence)
            #
            #     # OpenMM에서 pdb 생성이 잘못된 경우 --> 해당 파일 모두 삭제
            #     if result6 == 'error_pdb':
            #         with open(log_error, "a") as f:
            #             print(p_code, file=f)
            #
            #         subprocess.call('rm -fr {}'.format(src_dir), shell=True)
            #         for folder in list_dirs:
            #             subprocess.call('rm -fr {}/{}/{}_*'.format(trans_dir, folder, p_code), shell=True)
            #
            #         continue
            #
            #     print('salt bridge   : {}'.format(result6))

            # count_num += 1
            # elapsed_time = time.time() - start_time
            # print('elapsed_time : {:.2f}'.format(elapsed_time))
            # with open(log_process, "a") as f:
            #     print('{},{:4d} --- {} : {:.2f}'.format(p_code, res_num, count_num, elapsed_time), file=f)

# ============================================================================
