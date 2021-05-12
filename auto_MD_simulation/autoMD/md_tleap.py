import os
import subprocess
from biopandas.pdb import PandasPdb

# ---------------------------------------------------------------------------
# tLeap
# ---------------------------------------------------------------------------
class tLeap:
    def __init__(self, pdb_code, work_dir, prep_pdb_dir):
        self.pdb_code = pdb_code
        self.work_dir = work_dir  # folded_dir or unfolded_dir

        self.tleap_dir = os.path.join(work_dir, '1_tleap')
        try:
            os.makedirs(self.tleap_dir)
        except OSError:
            pass

        # copy H deleted pdb file
        subprocess.call('cp {}/{}_H_deleted.pdb {}/'.format(prep_pdb_dir, pdb_code, self.tleap_dir), shell=True)


    def tleap_folded(self):
        initial_file = '{}/{}_initial.crd'.format(self.tleap_dir, self.pdb_code)
        if os.path.exists(initial_file):
            return initial_file

        cwd = os.getcwd()
        os.chdir(self.tleap_dir)

        raw_pdb = '{}_H_deleted.pdb'.format(self.pdb_code)

        # tleap for implicit water simulation
        leap_script1 = 'leap_1.in'
        with open(leap_script1, "w") as f:
            print('source leaprc.protein.ff14SB', file=f)
            print('prot = loadpdb {}'.format(raw_pdb), file=f)
            print('set default PBradii mbondi2', file=f)   # to prepare the prmtop file
            print('saveamberparm prot {0}.top {0}.inpcrd'.format(self.pdb_code), file=f)
            print('savepdb prot {}.pdb'.format(self.pdb_code), file=f)
            print('quit', file=f)

        subprocess.call('tleap -s -f {}'.format(leap_script1), shell=True)

        # tleap for pql file generation
        leap_script2 = 'leap_2.in'
        with open(leap_script2, "w") as f:
            print('source leaprc.protein.ff14SB', file=f)
            print('source leaprc.water.tip3p', file=f)
            print('prot = loadpdb {}'.format(raw_pdb), file=f)
            print('solvateBox prot TIP3PBOX 10.0 iso', file=f)      # solvate Model / Buffer size / iso
            print('addIons prot Na+ 0', file=f)
            print('addIons prot Cl- 0', file=f)
            print('set default PBradii mbondi2', file=f)
            print('saveamberparm prot {0}_water.top {0}_initial.crd'.format(self.pdb_code), file=f)
            print('savepdb prot {}_water.pdb'.format(self.pdb_code), file=f)
            print('quit', file=f)

        subprocess.call('tleap -s -f {}'.format(leap_script2), shell=True)

        os.chdir(cwd)

        return initial_file


    def tleap_unfolded(self):
        initial_file = '{}/{}.inpcrd'.format(self.tleap_dir, self.pdb_code)
        if os.path.exists(initial_file):
            return initial_file

        cwd = os.getcwd()
        os.chdir(self.tleap_dir)

        raw_pdb = '{}_H_deleted.pdb'.format(self.pdb_code)

        ppdb = PandasPdb()
        ppdb.read_pdb(raw_pdb)

        # extract sequences
        df_pro = ppdb.df['ATOM'][['residue_number','residue_name', 'chain_id']]
        df_pro = df_pro[df_pro['chain_id'] == 'A']
        df_model = df_pro.groupby(['residue_number', 'residue_name']).count()
        res_sequence = df_model.index.get_level_values(1).tolist()
        res_num = len(res_sequence)

        # ---------------------------------------------------------------------------
        # tleap for implicit water simulation
        seq_list = '{'
        for k, res in enumerate(res_sequence):
            if k == 0:
                seq_list += 'N{} '.format(res)
            elif k == (res_num - 1):
                seq_list += 'C{}'.format(res)
            else:
                seq_list += '{} '.format(res)

            # protect error of tleap
            if k%30 == 0:
                seq_list += '\n'

        seq_list += '}'
        print(res_num, seq_list)

        leap_script1 = 'leap_1.in'
        with open(leap_script1, "w") as f:
            print('source leaprc.protein.ff14SB', file=f)
            print('prot = sequence {}'.format(seq_list), file=f)
            print('set default PBradii mbondi2', file=f)   # to prepare the prmtop file
            print('saveamberparm prot {0}.top {0}.inpcrd'.format(self.pdb_code), file=f)
            print('savepdb prot {}.pdb'.format(self.pdb_code), file=f)
            print('quit', file=f)

        subprocess.call('tleap -s -f {}'.format(leap_script1), shell=True)

        # tleap for pql file generation --> same as folded state pql file --> copy

        os.chdir(cwd)

        return initial_file

# ---------------------------------------------------------------------------
