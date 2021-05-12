import sys
import subprocess
import pandas as pd

"""
/homes/eta/users/aiteam/HP36_tutorial/tools/calc_Eu_plus_xmu/

input  : tenergy.dat
         xmu_vs_time.dat

output : Eu_vs_time.dat
         Eu_plus_xmu_vs_time.dat
"""

# load data files
traj_dir = '/homes/eta/users/aiteam/HP36_tutorial/traj_A'

subprocess.call('cp {}/anal_tenergy/tenergy.dat .'.format(traj_dir), shell=True)
subprocess.call('cp {}/3D-RISM/xmu_calc/xmu_vs_time/xmu_vs_time.dat .'.format(traj_dir), shell=True)
subprocess.call('cp {}/3D-RISM/xmu_calc/xmu_vs_time/xmu_vs_time_every_250ps.dat .'.format(traj_dir), shell=True)

time_interval = 10  # ps

df_e = pd.read_csv('tenergy.dat', header=None, skiprows=1, delimiter='\s+')
df_e.columns = ['time','Eu','e_bond','e_angle','e_dihedral','e_14LJ','e_14elec','e_LJ','e_elec']
df_e['time'] = df_e['time'] * time_interval
# print(df_e.head())

df_xmu = pd.read_csv('xmu_vs_time.dat', header=None, delimiter='\s+')
df_xmu.columns = ['time','xmu']
# print(df_xmu.head())

num0 = df_e.shape[0]
num1 = df_xmu.shape[0]


# check the number of data in energy.dat and xmu_vs_time.dat
if num0 != num1:
    print('error: inconsistency in the number of data')
    sys.exit()

print('total number of data : %s' % num0)

for i in range(num0):
    time0 = int(df_e.iloc[i].time)
    time1 = int(df_xmu.iloc[i].time)

    if time0 != time1:
        print('error:  inconsistency in times.',time0,time1)
        sys.exit()


# manipulation
df = pd.merge(df_e, df_xmu, on='time')      # time을 기준으로 묶음.
df = df[['time', 'Eu', 'xmu']]              # 필요한 컬럼만 추출
df['Eu_plus_xmu'] = df['Eu'] + df['xmu']    # 새로운 컬럼 생성
# print(df.head())


df.to_csv('Eu_plus_xmu_vs_time.csv', index=None)

# =============================================================================
