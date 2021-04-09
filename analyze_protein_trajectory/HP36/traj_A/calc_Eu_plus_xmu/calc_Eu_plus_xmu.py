import sys
import pandas as pd

"""
/homes/eta/users/aiteam/HP36_tutorial/tools/calc_Eu_plus_xmu/

input  : tenergy.dat
         xmu_vs_time.dat

output : Eu_vs_time.dat
         Eu_plus_xmu_vs_time.dat
"""

# load data files
df_e = pd.read_csv('tenergy.dat', header=None, skiprows=1, delimiter='\s+')
df_e.columns = ['time','Eu','e_bond','e_angle','e_dihedral','e_14LJ','e_14elec','e_LJ','e_elec']
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

# manipulation
time_interval = 10  # ps

file_Eu = open('Eu_vs_time.dat', 'w')
file_total = open('Eu_plus_xmu_vs_time.dat', 'w')
file_Eu_250 = open('Eu_vs_time_every_250ps.dat', 'w')
file_total_250 = open('Eu_plus_xmu_vs_time_every_250ps.dat', 'w')

for i in range(num0):
    time0 = int(df_e.iloc[i].time * time_interval)
    time1 = int(df_xmu.iloc[i].time)

    Eu = df_e.iloc[i].Eu
    xmu = df_xmu.iloc[i].xmu

    if time0 != time1:
        print('error:  inconsistency in times.',time0,time1)
        sys.exit()


    print('{:>5d}\t{:>15.8f}'.format(time0, Eu), file=file_Eu)
    print('{:>5d}\t{:>15.8f}'.format(time0, Eu+xmu), file=file_total)

    if time0 % 250 == 0:
        print('{:>5d}\t{:>15.8f}'.format(time0, Eu), file=file_Eu_250)
        print('{:>5d}\t{:>15.8f}'.format(time0, Eu+xmu), file=file_total_250)


file_Eu.close()
file_total.close()
file_Eu_250.close()
file_total_250.close()

# =============================================================================
