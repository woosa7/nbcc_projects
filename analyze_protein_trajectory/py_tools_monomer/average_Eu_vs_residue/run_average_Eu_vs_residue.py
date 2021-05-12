import sys
import glob
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
    rno, Eu_res, Eu_Bond, Eu_14LJ, Eu_14cou, Eu_LJ, Eu_cou = np.loadtxt(file, skiprows=2, unpack=True)
    n_res = len(rno)

    if i == 0:
        zeros = [0.0] * n_res
        ave_Eu_res = zeros
        ave2_Eu_res = zeros
        ave_Eu_res_Bond = zeros
        ave_Eu_res_LJ = zeros
        ave_Eu_res_cou = zeros

    ave_Eu_res      = [a + b for a, b in zip(ave_Eu_res, Eu_res)]
    ave2_Eu_res     = [a + b**2 for a, b in zip(ave2_Eu_res, Eu_res)]
    ave_Eu_res_Bond = [a + b for a, b in zip(ave_Eu_res_Bond, Eu_Bond)]
    ave_Eu_res_LJ   = [a + b + c for a, b, c in zip(ave_Eu_res_LJ, Eu_LJ, Eu_14LJ)]
    ave_Eu_res_cou  = [a + b + c for a, b, c in zip(ave_Eu_res_cou, Eu_cou, Eu_14cou)]


# calc average
ave_Eu_res      = [x / ndata for x in ave_Eu_res]
ave_Eu_res_Bond = [x / ndata for x in ave_Eu_res_Bond]
ave_Eu_res_LJ   = [x / ndata for x in ave_Eu_res_LJ]
ave_Eu_res_cou  = [x / ndata for x in ave_Eu_res_cou]

ave2_Eu_res     = [x / ndata for x in ave2_Eu_res]
sig_Eu_res      = [np.sqrt(a - b**2) for a, b in zip(ave2_Eu_res, ave_Eu_res)]


# output to files
f_res = open('average_Eu_vs_residue.dat', 'w')
for k in range(len(ave_Eu_res)):
    print('{:>5d} {:16.8f} {:16.8f}'.format(k+1, ave_Eu_res[k], sig_Eu_res[k]), file=f_res)
f_res.close()

f_bond = open('average_Eu_Bond_vs_residue.dat', 'w')
for k in range(len(ave_Eu_res_Bond)):
    print('{:>5d} {:16.8f}'.format(k+1, ave_Eu_res_Bond[k]), file=f_bond)
f_bond.close()

f_LJ = open('average_Eu_LJ_vs_residue.dat', 'w')
for k in range(len(ave_Eu_res_LJ)):
    print('{:>5d} {:16.8f}'.format(k+1, ave_Eu_res_LJ[k]), file=f_LJ)
f_LJ.close()

f_cou = open('average_Eu_cou_vs_residue.dat', 'w')
for k in range(len(ave_Eu_res_cou)):
    print('{:>5d} {:16.8f}'.format(k+1, ave_Eu_res_cou[k]), file=f_cou)
f_cou.close()

f_summary = open('summary_average_Eu_vs_residue.dat', 'w')
print('total number of data = %s' % ndata, file=f_summary)
print('ave_Eu_all      :{:16.8f}'.format(np.sum(ave_Eu_res)), file=f_summary)
print('ave_Eu_all_Bond :{:16.8f}'.format(np.sum(ave_Eu_res_Bond)), file=f_summary)
print('ave_Eu_all_LJ   :{:16.8f}'.format(np.sum(ave_Eu_res_LJ)), file=f_summary)
print('ave_Eu_all_cou  :{:16.8f}'.format(np.sum(ave_Eu_res_cou)), file=f_summary)
f_summary.close()

print('total number of data = %s' % ndata)
print('ave_Eu_all      :{:16.8f}'.format(np.sum(ave_Eu_res)))
print('ave_Eu_all_Bond :{:16.8f}'.format(np.sum(ave_Eu_res_Bond)))
print('ave_Eu_all_LJ   :{:16.8f}'.format(np.sum(ave_Eu_res_LJ)))
print('ave_Eu_all_cou  :{:16.8f}'.format(np.sum(ave_Eu_res_cou)))


# =============================================================================
