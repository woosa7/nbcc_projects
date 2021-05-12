import pandas as pd

"""
/homes/eta/users/aiteam/HP36_tutorial/py_tools/set_atom_info/run_set_atom_info.py

"set_atom_info" creats a file on the atom information to be referred to in the analysis

input file  : protein.pql
output file : atom_info.dat

set imain(i) : 1 if i-th atom is a main chain atom, 0 otherwise
    main chain atoms : N, H, CA, HA, C, O
    N-terminal       : H1, H2, H3
    C-terminal       : OXT
    GLY              : HA2, HA3
"""

# read pql file
df = pd.read_csv('protein.pql', header=None, skiprows=1, delimiter='\s+')
df.columns = ['atom','atom_no','atom_name','res_name','res_no','coord_x','coord_y','coord_z']
# print(df.head())
# print(df.tail())


# generate atom info
atom_count = df.shape[0]

df_res = df.groupby(['res_no']).count()
res_count = df_res.shape[0]
print(atom_count, res_count)

main_atoms = ['N','H','CA','HA','C','O','H1','H2','H3','OXT','HA2','HA3']

output_file = 'atom_info.dat'
f = open(output_file, "w")

print('{:>5d}{:>10d}'.format(atom_count, res_count), file=f)

for i, row in df.iterrows():
    is_main = 0
    if row.atom_name in main_atoms:
        is_main = 1

    # set pdb-type atomname
    atom_name = row.atom_name
    if len(atom_name) < 4:
        atom_name = ' ' + atom_name

    print('{:>5d} {:<4}{:>6d}{:>6d}'.format(row.atom_no, atom_name, row.res_no, is_main), file=f)


f.close()

# =============================================================================
