import os
import sys
import glob
import subprocess
import time
import numpy as np
from biopandas.pdb import PandasPdb
import make_3Dimage_LJ_elec_amber
import make_single_features

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

ppdb = PandasPdb()

# ------------------------------------------------------
# set directory
work_dir = os.path.abspath(work_dir)

md_dirs = sorted(next(os.walk(work_dir))[1])
print(md_dirs)

trans_dir = './trans_data'
if not os.path.exists(trans_dir):
    os.makedirs(trans_dir)

log_error = '{}/log_error'.format(trans_dir)
with open(log_error, "w") as f:
    print('log_error', file=f)

# ============================================================================
# select pdb files

# for k, p_code in enumerate(md_dirs):
#     time_list = np.loadtxt('{}/{}/xmu_decomposition_vs_time/xmu_tot_from_xmua_vs_time.dat'.format(work_dir, p_code))[:,0]
#     time_list = [int(x) for x in time_list]
#     print(p_code, len(time_list))
#
#     pdb_dir = '{}/{}/pdb_from_crd'.format(work_dir, p_code)
#     f_list = sorted(glob.glob('{}/*.pdb'.format(pdb_dir)))
#     print('--------- remove pdb files not needed')
#     print('Before :', len(f_list))
#
#     for j, f_name in enumerate(f_list):
#         t = f_name.split('/')[-1].split('_')[-1].replace('.pdb','')
#         if not int(t) in time_list:
#             subprocess.call('rm -fr {}'.format(f_name), shell=True)
#
#     f_list = sorted(glob.glob('{}/*.pdb'.format(pdb_dir)))
#     print('After :', len(f_list))

# ============================================================================
# feature generation
for k, p_code in enumerate(md_dirs):
    start_time = time.time()

    slice = 4  # split by 1000

    # ------------------------------------------------------------------------
    # extract sequence
    src_dir = '{}/{}'.format(work_dir, p_code)
    print('---------------------')
    print('make features :', src_dir)

    pdb_list = sorted(glob.glob('{}/pdb_from_crd/*.pdb'.format(src_dir)))
    pdb_file = pdb_list[0]
    conformation_num = len(pdb_list)
    print('conformations :', conformation_num)
    print('slice         :', slice)

    target_dir = os.path.join(trans_dir, 'pdb')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_file = '{}/{}_1000.pdb'.format(target_dir, p_code)
    if not os.path.exists(target_file):
        subprocess.call('cp {} {}'.format(pdb_file, target_file), shell=True)

    ppdb.read_pdb(pdb_file)
    df = ppdb.df['ATOM'][['residue_number','residue_name']] # not exist chain_id
    df_model = df.groupby(['residue_number', 'residue_name']).count()
    res_sequence = df_model.index.get_level_values(1).tolist()
    res_num = len(res_sequence)

    print('residue num   :', res_num)
    # print(res_sequence)
    print()

    # ------------------------------------------------------------------------
    # Gsolv, Gsolv_LJ, Gsolv_elec
    featurizer = make_single_features.Featurizer(src_dir, trans_dir, log_error)

    target_dir = os.path.join(trans_dir, 'ydata')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # residual Gsolv - ydata
    result = featurizer.residual_Gsolv_slice(p_code, xmu_dir='xmu_decomposition_vs_time', pdb_dir='pdb_from_crd', slice=slice)
    print('residue Gsolv : {}'.format(result))

    # ------------------------------------------------------------------------
    # total_charge
    target_dir = os.path.join(trans_dir, 'charge')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    result = featurizer.total_charge_slice(p_code, conformation_num, res_num, res_sequence, slice=slice)
    print('total charge  : {}'.format(result))

    # ------------------------------------------------------------------------
    # residue type
    target_dir = os.path.join(trans_dir, 'RT')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    result = featurizer.residue_type_slice(p_code, conformation_num, res_num, res_sequence, slice=slice)
    print('residue type  : {}'.format(result))

    # ------------------------------------------------------------------------
    # 3D image
    target_dir = os.path.join(trans_dir, '3D')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    feature_3D_LJ = make_3Dimage_LJ_elec_amber.Image3D_LJ_elec(src_dir, trans_dir, p_code, log_error)
    result = feature_3D_LJ.make_feature(xmu_dir='xmu_decomposition_vs_time', pdb_dir='pdb_from_crd', slice=slice)
    print('3D image : {}'.format(result))

    print()

# ============================================================================
