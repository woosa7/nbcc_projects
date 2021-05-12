import os
import pandas as pd

data_dir = '/homes/eta/users/aiteam/delta_f_implicit/1000_proteins'
data_dir2 = '/homes/eta/users/aiteam2/delta_f_implicit/1000_proteins'

df = pd.read_csv('../job_list')
pdb_list = df[df['done'] == 1]

print(pdb_list.shape)

for i, row in pdb_list.iterrows():
    ex = 0

    target_dir = '{}/{}'.format(data_dir, row.p_code)
    if os.path.exists(target_dir):
        ex = 1

    target_dir = '{}/{}'.format(data_dir2, row.p_code)
    if os.path.exists(target_dir):
        ex = 1

    if ex == 0:
        print(row.p_code, row.node, row.gpu)
