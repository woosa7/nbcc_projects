import os
import sys
import subprocess
import shutil
from pdblib import pre_process_api as api
from pdblib.models import PDB, RESPROT, MUTMAP

class pdb_preprocess:
    def __init__(self, workdir):
        self.work_dir = workdir

    def run_pre_process(self, input_pdb, pdb_code, selected_chain, force_field):
        print('')
        print('---------------- run_pre_process')

        pdb_file_path = os.path.abspath(input_pdb)
        model_pdb_path = '{}/{}{}_H_deleted.pdb'.format(self.work_dir, pdb_code, selected_chain)

        print(pdb_file_path)
        print(model_pdb_path)

        if pdb_file_path != model_pdb_path:
            shutil.copy2(pdb_file_path, model_pdb_path)

        p = PDB(model_pdb_path, force_field)

        # ------------------------------------------------------------------
        # select model - 첫번째 모델 선택
        models = api.get_models(model_pdb_path)
        api.select_model(model_pdb_path, model_pdb_path, 0)

        # ------------------------------------------------------------------
        # select chain
        chains = api.get_chains(model_pdb_path)
        selected_chains = []
        if selected_chain:
            selected_chains = [selected_chain]
        else:
            selected_chains = list(chains)    # 선택된 chain이 없으면 모든 chain 사용

        print('selected_chains :', selected_chains)
        api.select_chains(model_pdb_path, model_pdb_path, selected_chains)

        # ------------------------------------------------------------------
        # water & hydrogens - remove all
        water_residues = api.get_water_residues(model_pdb_path)
        api.strip_water(model_pdb_path, model_pdb_path)

        # remove all hydrogens if has any
        api.remove_hydrogens(model_pdb_path, model_pdb_path)

        # # ------------------------------------------------------------------
        # # insertion code for residue - 자동화는 default로 처리
        # # 1H4W
        # # ('A', 184, 'GLY', '')
        # # ('A', 184, 'PHE', 'A')
        # icode_residues = api.get_icode_residues(model_pdb_path)
        # if icode_residues:
        #     selections = []
        #     for key, residues in icode_residues.items():
        #         residues = sorted(residues.keys(), key=lambda x: x[3])
        #         default = '%s %s' % (residues[0][2], residues[0][3])
        #         # for res in residues:
        #         #     print(res)
        #         selections.append((key[0], key[1], default))
        #
        #     if selections:
        #         print('insertion code :', len(selections))
        #         api.select_icode_residues(model_pdb_path, model_pdb_path, selections)
        #
        # # ------------------------------------------------------------------
        # # alternative location for atom - 자동화는 default로 처리
        # # 1H4W
        # # ATOM     81  N  ALEU A  27      62.201  31.206  55.988  0.50 13.53           N
        # # ATOM     89  N  BLEU A  27      62.198  31.212  55.987  0.50 12.96           N
        #
        # altloc_atoms = api.get_altloc_atoms(model_pdb_path)
        # if altloc_atoms:
        #     selections = []
        #     for key, atoms in altloc_atoms.items():
        #         atoms = sorted(atoms, key=lambda x: x[4])
        #         default = 'A'
        #         # for atom in atoms:
        #         #     print(atom)
        #         selections.append((key[0], key[1], key[2], key[3], default))
        #
        #     if selections:
        #         print('alternative atoms :', len(selections))
        #         api.select_altloc_atoms(model_pdb_path, model_pdb_path, selections)
        #
        # # ------------------------------------------------------------------
        # # hetero residues - default : Delete All (-1)
        # hetero_residues = api.get_hetero_residues(model_pdb_path)
        #
        # if hetero_residues:
        #     print('hetero residues :', len(hetero_residues))
        #     api.select_hetero_residue(model_pdb_path, model_pdb_path, '-1')
        #
        # # ------------------------------------------------------------------
        # # add missing atoms
        # # api.add_missing_atoms(model_pdb_path, model_pdb_path)
        #
        # # ------------------------------------------------------------------
        # # mutation of non-standard residues
        # nsresidues = api.get_non_standard_residues(model_pdb_path)
        #
        # mutations = []
        # for res in nsresidues:
        #     chain, resnum, resname = res
        #     try:
        #         default = MUTMAP[resname]
        #     except KeyError:
        #         default = 'GLY'
        #
        #     # print(chain, resnum, resname)
        #     mutations.append((chain, resnum, default))
        #
        # if mutations:
        #     print('non-standard residues :', len(nsresidues))
        #     api.mutate_non_standard_residues(model_pdb_path, model_pdb_path, mutations)
        #
        # # ------------------------------------------------------------------
        # # define disulfide bonds
        # # p.connects = []
        # disulfide_bond_candidates = api.get_disulfide_bond_candidates(model_pdb_path)
        #
        # cands = []
        # for cand in disulfide_bond_candidates:
        #     cand1, cand2, distance = cand
        #     if distance < 2.5:
        #         # print(cand1, cand2, distance)
        #         cands.append(cand)
        #
        # if cands:
        #     print('disulfide bonds :', len(disulfide_bond_candidates))
        #     api.build_disulfide_bonds(model_pdb_path, model_pdb_path, cands)
        #
        # # ------------------------------------------------------------------
        # # solvent ions
        # solvent_ions = api.get_solvent_ions(model_pdb_path)
        #
        # selections = []
        # for solvent_ion in solvent_ions:
        #     print(solvent_ion)
        #     selections.append(solvent_ion)
        #
        # if selections:
        #     print('solvent ions :', len(solvent_ions))
        #     api.delete_solvent_ions(model_pdb_path, model_pdb_path, selections)
        #
        # # ------------------------------------------------------------------
        # # protonation states : HIS, HIE, HID, HIP - Default: HIE
        # unknown_protonation_states = api.get_unknown_protonation_states(model_pdb_path)
        #
        # mutations = []
        # for res in unknown_protonation_states:
        #     # print(res)
        #     mutations.append((res[0], res[1], 'HIE'))
        #
        # if mutations:
        #     print('Histidine protonation :', len(mutations))
        #     api.mutate_protonation_states(model_pdb_path, model_pdb_path, mutations)


        # ------------------------------------------------------------------
        # Add missing hydrogens and finalize (PDBFixer)
        api.done(model_pdb_path, model_pdb_path)

        return model_pdb_path
