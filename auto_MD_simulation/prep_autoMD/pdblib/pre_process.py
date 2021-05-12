"""
PDB Preprocess v2.0

"""
import os
import sys
import argparse
import shutil
from utils.pdblib import pre_process_api as api
from utils.pdblib.models import PDB, RESPROT, MUTMAP

# try:
#     from utils.pdblib import pre_process_api as api
#     from utils.pdblib.models import PDB, RESPROT, MUTMAP
# except ImportError:
#     sys.path.append('/home/nbcc/www/prowave/')
#     from utils.pdblib import pre_process_api as api
#     from utils.pdblib.models import PDB, RESPROT, MUTMAP


PARSER = argparse.ArgumentParser()
PARSER.add_argument('pdb_file')
PARSER.add_argument('--simple', action='store_true')  # Only default for strip water, icode, altloc
PARSER.add_argument('--auto', action='store_true')  # All default


def main(**kwargs):
    """
    PDB Preprocess
    :param kwargs:
    :return:
    """
    auto = kwargs.get('auto')
    simple = kwargs.get('simple')
    selected_chain = kwargs.get('chain')

    pdb_file_path = os.path.abspath(kwargs['pdb_file'])
    model_pdb_path = os.path.join(os.path.dirname(pdb_file_path), 'model.pdb')

    if pdb_file_path != model_pdb_path:
        shutil.copy2(pdb_file_path, model_pdb_path)

    p = PDB(model_pdb_path)

    # select model
    models = api.get_models(model_pdb_path)
    if len(models) > 1 and not auto:
        sel = input('Select a model [%d - %d]: ' % (1, len(models)))

        if not sel.isdigit() or int(sel) not in range(1, len(models)):
            print('Please enter only one number between %d - %d' % (1, len(models)))
            sys.exit(-1)
        api.select_model(model_pdb_path, model_pdb_path, int(sel) - 1)
    else:
        api.select_model(model_pdb_path, model_pdb_path, 0)

    # select chain
    chains = api.get_chains(model_pdb_path)
    print(list(chains))
    if selected_chain:
        selected_chains = [selected_chain]
    elif auto:
        selected_chains = list(chains)[:]
    else:
        sel = input('Select chains (separate by space, Default: All): ') or ' '.join(chains)
        selected_chains = sel.split()

    if not set(selected_chains) <= set(chains):
        print('You have entered wrong chain index.')
        sys.exit(-1)

    api.select_chains(model_pdb_path, model_pdb_path, selected_chains)

    water_residues = api.get_water_residues(model_pdb_path)
    if water_residues:
        if not auto and not simple:
            for res in water_residues:
                print(res)

            answer = (input('Do you like strip water? (Y/n): ') or 'y').lower()
            if answer not in ('y', 'n'):
                print('Please enter Y or N')
                sys.exit(-1)
        else:
            answer = 'y'

        if answer == 'y':
            api.strip_water(model_pdb_path, model_pdb_path)

    # remove all hydrogens if has any
    api.remove_hydrogens(model_pdb_path, model_pdb_path)

    icode_residues = api.get_icode_residues(model_pdb_path)
    if icode_residues:
        selections = []
        for key, residues in icode_residues.items():
            residues = sorted(residues.keys(), key=lambda x: x[3])
            default = '%s %s' % (residues[0][2], residues[0][3])

            if not auto and not simple:
                for res in residues:
                    print(res)

                sel = input('Please select insertion code for residue %s %d (Default: %s): ' % (
                    key[0], key[1], default)) or default

                try:
                    resname, icode = sel.split()
                except ValueError:
                    resname = sel
                    icode = ''

                if resname.strip() not in [res[2] for res in residues] or icode not in [res[3] for res in residues]:
                    print('Please enter valid insertion code')
                    sys.exit(-1)
            else:
                sel = default

            selections.append((key[0], key[1], sel))

        if selections:
            api.select_icode_residues(model_pdb_path, model_pdb_path, selections)

    altloc_atoms = api.get_altloc_atoms(model_pdb_path)
    if altloc_atoms:
        selections = []
        for key, atoms in altloc_atoms.items():
            atoms = sorted(atoms, key=lambda x: x[4])

            default = 'A'
            if not auto and not simple:
                for atom in atoms:
                    print(atom)

                sel = input('Please select alternative location for atom %s %d %s %s (Default: %s): ' % (
                    key[0], key[1], key[2], key[3], default
                )) or default
                if sel != 'A' and sel not in [atom[4] for atom in atoms]:
                    print('Please enter valid altlocation code')
                    sys.exit(-1)
            else:
                sel = default

            selections.append((key[0], key[1], key[2], key[3], sel))

        if selections:
            api.select_altloc_atoms(model_pdb_path, model_pdb_path, selections)

    hetero_residues = api.get_hetero_residues(model_pdb_path)
    if hetero_residues:
        if not auto:
            print('%-5d Delete All' % -1)
            resnums = ['-1']
            for chain, residues in hetero_residues.items():
                for res in residues:
                    resnums.append('%s%d' % (chain, res[0]))
                    print('%1s%-4d %s' % (chain, res[0], res[1]))
            sel = input('Please enter residue number to keep or -1 for delete all ligands (Default: -1): ') or '-1'
            if sel not in resnums:
                print('Please enter valid residue number or enter -1')
                sys.exit(-1)
        else:
            sel = '-1'

        api.select_hetero_residue(model_pdb_path, model_pdb_path, sel)

    # add missing atoms
    print('Add missing atoms (PDBFixer)')
    api.add_missing_atoms(model_pdb_path, model_pdb_path)

    nsresidues = api.get_non_standard_residues(model_pdb_path)
    if nsresidues and not auto:
        print('Non standard residues:')

    mutations = []
    for res in nsresidues:
        chain, resnum, resname = res
        try:
            default = MUTMAP[resname]
        except KeyError:
            default = 'GLY'

        if not auto:
            print(chain, resnum, resname)
            sel = input(
                'Please enter mutation target (standard version) of non standard residue (Default: %s): ' % default
            ) or default

            if sel not in RESPROT:
                print('Please enter standard residue name.')
                sys.exit(-1)
        else:
            sel = default

        mutations.append((chain, resnum, sel))

    api.mutate_non_standard_residues(model_pdb_path, model_pdb_path, mutations)

    # clear disulfide bonds described in PDB
    p.connects = []

    disulfide_bond_candidates = api.get_disulfide_bond_candidates(model_pdb_path)
    if disulfide_bond_candidates and not auto:
        print('Disulfide bond candidates')

    cands = []
    for cand in disulfide_bond_candidates:
        if not auto:
            print(cand)
            answer = (input('build disulfide bond? (Y/n): ') or 'y').lower()

            if answer not in ('y', 'n'):
                print('Please enter Y or N')
                sys.exit(-1)
        else:
            answer = 'y'

        if answer == 'y':
            cands.append(cand)

    api.build_disulfide_bonds(model_pdb_path, model_pdb_path, cands)

    solvent_ions = api.get_solvent_ions(model_pdb_path)
    if solvent_ions and not auto:
        print('Solvent ions:')

    selections = []
    for solvent_ion in solvent_ions:
        if not auto:
            print(solvent_ion)
            answer = (input('Do you want to keep this solvent? (y/N): ') or 'n').lower()

            if answer not in ('y', 'n'):
                print('Please enter Y or N')
                sys.exit(-1)
        else:
            answer = 'n'

        if answer == 'n':
            selections.append(solvent_ion)

    api.delete_solvent_ions(model_pdb_path, model_pdb_path, selections)

    unknown_protonation_states = api.get_unknown_protonation_states(model_pdb_path)
    if unknown_protonation_states and not auto:
        print('Unknown protonation states:')

    mutations = []
    for res in unknown_protonation_states:
        if not auto:
            print(res)
            sel = input(
                'Please enter proper protonation state of HIS residue (HIE/HID/HIP/HIS(automatic), Default: HIE): '
            ) or 'HIE'

            if sel not in ('HIS', 'HIE', 'HID', 'HIP'):
                print('Please enter protonation state (HIE/HID/HIP/HIS)')
                sys.exit(-1)
        else:
            sel = 'HIE'

        mutations.append((res[0], res[1], sel))

    api.mutate_protonation_states(model_pdb_path, model_pdb_path, mutations)

    print('Add missing hydrogens and finalize (PDBFixer)')
    api.done(model_pdb_path, model_pdb_path)


if __name__ == '__main__':
    args = PARSER.parse_args()
    print('------------ pre_process.py')
    print(args)

    main(pdb_file=args.pdb_file, auto=args.auto, simple=args.simple)


# -----------------------------------------------------------------------------
