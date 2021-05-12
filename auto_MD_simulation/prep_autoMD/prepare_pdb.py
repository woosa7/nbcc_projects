import os
import sys
import glob
import shutil
import subprocess
import pandas as pd
import md_preprocess
# import md_prep_openmm
# from simtk import openmm as mm, unit

# ---------------------------------------------------------------------
work_dir = os.path.abspath('../data')

force_field = 'ff14SB'

# df = pd.read_csv('../pdb_selected_1000')
# for index, row in df.iterrows():
#     chain_code = row.p_code

chain_code = '3CIJB'

pdb_code = chain_code[:4]
pdb_chain = chain_code[-1]
print(chain_code, pdb_code, pdb_chain)

# 작업 디렉토리 및 실행파일 설정
pdb_file = '{}/{}.pdb'.format(work_dir, pdb_code)

# ---------------------------------------------------------------------
# pdb preprocess
model_file = '{}/{}_H_deleted.pdb'.format(work_dir, chain_code)

# if not os.path.exists(model_file):
pdb_preprocessor = md_preprocess.pdb_preprocess(work_dir)
result = pdb_preprocessor.run_pre_process(pdb_file, pdb_code, pdb_chain, force_field)
print('preprocess  :', result)
print()
