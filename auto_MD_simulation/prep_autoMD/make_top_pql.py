import os
import ssl
import urllib
import numpy as np
import pandas as pd
import subprocess
import md_preprocess
import time
import sys

df = pd.read_csv('/homes/epsilon/users/aiteam/sfe_database/protein_data/prowise_list_all')
print(df.shape)

root_dir = '../md_simulation'
log_file = 'log_process'
with open(log_file, 'w') as f:
    print('log_process', file=f)

force_field = 'amber14/protein.ff14SB.xml'
solvent_model = 'TIP3PBOX'
buffer_size = 10

start_time = time.time()

count_done = 0

group_dirs = sorted(next(os.walk(root_dir))[1])
group_dirs = [k for k in group_dirs if k[0].isdigit()]

for g_code in group_dirs:
    second_dirs = sorted(next(os.walk('{}/{}'.format(root_dir, g_code)))[1])
    for p_code in second_dirs:
        pdb_id = p_code[:4]
        chain_id = p_code[-1]

        df_sel = df.query("pcode == '{}'".format(p_code))
        r_size = int(df_sel['rsize'])
        print(g_code, p_code, pdb_id, chain_id, r_size)

        pdb_dir = '{}/{}/{}'.format(root_dir, g_code, p_code)

        # -------------------------------------
        # download from rcsb
        # pdb_file = '{}/{}.pdb'.format(pdb_dir, pdb_id)
        # if not os.path.exists(pdb_file):
        #     print()
        #     print('---------------- download : ', pdb_id)
        #     ssl._create_default_https_context = ssl._create_unverified_context
        #     url = "https://files.rcsb.org/download/{}.pdb".format(pdb_id)
        #     try:
        #         pfile = urllib.request.urlopen(url)
        #         with open(pdb_file, 'wb') as output:
        #             output.write(pfile.read())
        #     except urllib.error.HTTPError:
        #         print('{} --- NotFound : HTTPError'.format(pdb_id))

        # -------------------------------------
        # pdb preprocess
        model_pdb = '{}/{}_model.pdb'.format(pdb_dir, p_code)
        print('model_pdb  :', model_pdb)

        # if not os.path.exists(model_pdb):
        #     pdb_preprocessor = md_preprocess.pdb_preprocess(pdb_dir)
        #     result = pdb_preprocessor.run_pre_process(pdb_file, pdb_id, chain_id, force_field)
        #     print('preprocess  :', result)

        # -------------------------------------
        # t-leap : make topology & initial.crd
        top_file = '{}/{}.top'.format(pdb_dir, p_code)
        crd_file = '{}/{}_initial.crd'.format(pdb_dir, p_code)
        pdb_file = '{}/{}_initial.pdb'.format(pdb_dir, p_code)

        if not os.path.exists(top_file):
            print('---------------- tleap : ', p_code)
            tleap_pdb = '{}/{}_H_deleted.pdb'.format(pdb_dir, p_code)
            subprocess.call('pdb4amber -i {} -o {} -y'.format(model_pdb, tleap_pdb), shell=True)

            leapin_file = '{}/leap.in'.format(pdb_dir)
            with open(leapin_file, "w") as f:
                print('source /opt/apps/amber18/dat/leap/cmd/leaprc.protein.ff14SB', file=f)
                print('source /opt/apps/amber18/dat/leap/cmd/leaprc.water.tip3p', file=f)

                print('prot = loadpdb {}'.format(tleap_pdb), file=f)

                print('solvateBox prot {} {} iso'.format(solvent_model, buffer_size), file=f)
                print('addIons prot Na+ 0', file=f)
                print('addIons prot Cl- 0', file=f)
                print('saveAmberParm prot {} {}'.format(top_file, crd_file), file=f)
                print('quit', file=f)

            subprocess.call('tleap -f {}'.format(leapin_file), shell=True)
            subprocess.call('ambpdb -p {} < {} > {}'.format(top_file, crd_file, pdb_file), shell=True)

            os.remove('{}/{}_H_deleted_nonprot.pdb'.format(pdb_dir, p_code))
            os.remove('{}/{}_H_deleted_renum.txt'.format(pdb_dir, p_code))
            os.remove('{}/{}_H_deleted_sslink'.format(pdb_dir, p_code))

        # -------------------------------------
        # make protein.pql
        pql_file = '{}/{}.pql'.format(pdb_dir, p_code)
        if not os.path.exists(pql_file):
            print('---------------- make pql : ', p_code)

            anal_in = '{}/pql_anal.in'.format(pdb_dir)
            with open(anal_in, "w") as f:
                # set topology file
                print('parm {}'.format(top_file), file=f)
                # crd file
                print('trajin {}'.format(crd_file), file=f)
                # centering (do not strip water for pqlout)
                print('center :1-{} mass origin'.format(r_size), file=f)
                print('image origin center familiar', file=f)
                # pql file (put noframeout at the end when only parameter file is necessary)
                print('pqlout :1-{} out {} noframeout'.format(r_size, pql_file), file=f)

            subprocess.call('cpptraj-15 -i {}'.format(anal_in), shell=True)

        count_done += 1
        with open(log_file, 'a') as f:
            print('{:4d} {}'.format(count_done, p_code), file=f)

        # if count_done == 1: break

# -------------------------------------
elapsed_time = time.time() - start_time
print('===== elapsed_time : {:.2f} min'.format(elapsed_time/60))

# =============================================================================
