import os
import sys
import shutil
import glob
import subprocess
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
# import md_calc_sfe

# ===========================================================================
work_dir = '/home/woohyun/auto_simulation'
data_dir = '/homes/epsilon/users/aiteam/sfe_database/protein_data'

cwd = os.getcwd()
ppdb = PandasPdb()

for pdb_prefix in ['4X','4Y']:
    first_dirs = sorted(next(os.walk('{}/{}'.format(work_dir, pdb_prefix)))[1])
    for f_dir in first_dirs:
        chain_code = f_dir[:5]
        pdb_work_dir = '{}/{}/{}'.format(work_dir, pdb_prefix, f_dir)
        if not os.path.exists(pdb_work_dir):
            continue

        os.chdir(pdb_work_dir)
        print(pdb_work_dir)

        model_pdb = './pdb/{}_1000.pdb'.format(chain_code)
        if not os.path.exists(model_pdb):
            print('pdb not exist')
            continue

        ppdb.read_pdb(model_pdb)
        df = ppdb.df['ATOM'][['residue_number','residue_name', 'chain_id']]
        df = df[df['chain_id'] == 'A']
        df_model = df.groupby(['residue_number', 'residue_name']).count()
        res_sequence = df_model.index.get_level_values(1).tolist()
        total_res = len(res_sequence)

        res_xmu_list = sorted(glob.glob('./xmu/xmu_res_*.dat'))
        if total_res == len(res_xmu_list):
            # copy files to data_dir in storage
            src_pdb_dir = os.path.join(pdb_work_dir, 'pdb')
            src_xmu_dir = os.path.join(pdb_work_dir, 'xmu')

            pdb_data_dir = os.path.join(data_dir, '%s/%s' % (pdb_prefix, chain_code))
            if not os.path.exists(pdb_data_dir):
                os.makedirs(pdb_data_dir)

            subprocess.call('cp -r {} {}/'.format(src_pdb_dir, pdb_data_dir), shell=True)
            subprocess.call('cp -r {} {}/'.format(src_xmu_dir, pdb_data_dir), shell=True)
        else:
            print(chain_code, total_res, len(res_xmu_list))

    os.chdir(cwd)
# ===========================================================================


# ===========================================================================
# batch_sfe
# work_dir = '/home/woohyun/auto_simulation/4W'
# progress_log = '{}/progress_log'.format(work_dir)
#
# first_dirs = sorted(next(os.walk(work_dir))[1])
# for f_dir in first_dirs:
#     pdb_work_dir = '{}/{}'.format(work_dir, f_dir)
#     md_calc = md_calc_sfe.rism3d(pdb_work_dir, progress_log)
#     chain_code = f_dir[:5]
#     # Summary Gsolv values
#     total_res = md_calc.run_xmu_summary(chain_code)
#     print('xmu_summary :', total_res)
# ===========================================================================
