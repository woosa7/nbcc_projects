import sys
import glob
import pandas as pd
import numpy as np

"""
calculate the average of Eu_vs_residue

input  : directory path of anal_renergy
output : average_Eu_vs_residue.dat
"""

dat_dir = sys.argv[1]
list_renergy = sorted(glob.glob('{}/renergy*.dat'.format(dat_dir)))

ndata = len(list_renergy)

for i, file in enumerate(list_renergy):
    df_i = pd.read_csv(file, header=None, skiprows=2, delimiter='\s+')
    df_i.columns = ['res_no','Eu_res','Eu_res_Bond','Eu_res_14LJ','Eu_res_14cou','Eu_res_LJ','Eu_res_cou']

    if i == 0:
        # Create dataframe to add data for each frame.
        n_res = df_i.shape[0]
        zeros = [0.0] * n_res

        dict = {'Eu_res': zeros,
                'Eu_res_2': zeros,
                'Eu_res_Bond': zeros,
                'Eu_res_LJ': zeros,
                'Eu_res_cou': zeros}

        df_sum = pd.DataFrame.from_dict(dict)

    # From the second data, a value is added to df_sum.
    df_sum['Eu_res']      = df_sum['Eu_res']      + df_i['Eu_res']
    df_sum['Eu_res_2']    = df_sum['Eu_res_2']    + df_i['Eu_res']**2
    df_sum['Eu_res_Bond'] = df_sum['Eu_res_Bond'] + df_i['Eu_res_Bond']
    df_sum['Eu_res_LJ']   = df_sum['Eu_res_LJ']   + df_i['Eu_res_LJ']  + df_i['Eu_res_14LJ']
    df_sum['Eu_res_cou']  = df_sum['Eu_res_cou']  + df_i['Eu_res_cou'] + df_i['Eu_res_14cou']


# calc average
df_sum = df_sum / ndata
df_sum['sig_Eu_res'] = np.sqrt(df_sum['Eu_res_2'] - df_sum['Eu_res']**2)
print(df_sum)

print()

# print out summary
print('total number of data = %s' % ndata)
print('ave_Eu_all      :', np.sum(df_sum['Eu_res']))
print('ave_Eu_all_Bond :', np.sum(df_sum['Eu_res_Bond']))
print('ave_Eu_all_LJ   :', np.sum(df_sum['Eu_res_LJ']))
print('ave_Eu_all_cou  :', np.sum(df_sum['Eu_res_cou']))


# =============================================================================
