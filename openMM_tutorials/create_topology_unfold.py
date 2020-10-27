import subprocess
from biopandas.pdb import PandasPdb

# ---------------------------------------------------------------------------
# tLeap for unfolded structure
# ---------------------------------------------------------------------------
pdb_code = '1VIMC'

raw_pdb = '{}_H_deleted.pdb'.format(pdb_code)

ppdb = PandasPdb()
ppdb.read_pdb(raw_pdb)

# extract sequences
df_pro = ppdb.df['ATOM'][['residue_number','residue_name', 'chain_id']]
# df_pro = df_pro[df_pro['chain_id'] == 'A']
df_model = df_pro.groupby(['residue_number', 'residue_name']).count()
res_sequence = df_model.index.get_level_values(1).tolist()
res_num = len(res_sequence)

# ---------------------------------------------------------------------------
# tleap for implicit water simulation
seq_list = '{\n'
for k, res in enumerate(res_sequence):
    r_no = k+1
    if r_no == 1:
        seq_list += 'N{} '.format(res)
    elif r_no == res_num:
        seq_list += 'C{}'.format(res)
    else:
        seq_list += '{} '.format(res)
        if r_no % 20 == 0:   # 20 residue by 1 line
            seq_list += '\n'

seq_list += '}'
print(res_num, seq_list)

leap_script1 = 'leap_1.in'
with open(leap_script1, "w") as f:
    print('source leaprc.protein.ff14SB', file=f)
    print('prot = sequence {}'.format(seq_list), file=f)
    print('set default PBradii mbondi2', file=f)   # to prepare the prmtop file
    print('saveamberparm prot {0}.top {0}.inpcrd'.format(pdb_code), file=f)
    print('savepdb prot {}.pdb'.format(pdb_code), file=f)
    print('quit', file=f)

subprocess.call('tleap -s -f {}'.format(leap_script1), shell=True)

# tleap for pql file generation
# leap_script2 = 'leap_2.in'
# with open(leap_script2, "w") as f:
#     print('source leaprc.protein.ff14SB', file=f)
#     print('source leaprc.water.tip3p', file=f)
#     print('prot = loadpdb {}.pdb'.format(pdb_code), file=f)
#     print('solvateBox prot TIP3PBOX 10.0 iso', file=f)      # solvate Model / Buffer size / iso
#     print('addIons prot Na+ 0', file=f)
#     print('addIons prot Cl- 0', file=f)
#     print('set default PBradii mbondi2', file=f)
#     print('saveamberparm prot {0}_water.top {0}_initial.crd'.format(pdb_code), file=f)
#     print('savepdb prot {}_water.pdb'.format(pdb_code), file=f)
#     print('quit', file=f)
#
# subprocess.call('tleap -s -f {}'.format(leap_script2), shell=True)

# ---------------------------------------------------------------------------
