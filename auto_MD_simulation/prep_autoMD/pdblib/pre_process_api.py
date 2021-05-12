#!/usr/bin/envs python
import os
# import sys
import json
import shutil
import subprocess
# sys.path.append(os.getcwd())
from pdblib.models import PDB


def save_protocol(target, key, value):
    try:
        with open(target, 'r') as f:
            protocol = json.load(f)
    except FileNotFoundError:
        protocol = dict()

    protocol[key] = value
    with open(target, 'w') as f:
        json.dump(protocol, f, indent=2)


def get_models(src):
    p = PDB(src)
    return p.models


def select_model(src, dest, model):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    p.select_model(int(model))
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'model', model)
    p.save()


def get_chains(src):
    p = PDB(src)
    p.reload()
    return p.chains


def select_chains(src, dest, chains):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    p.select_chains(chains)
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'chains', chains)


def remove_hydrogens(src, dest):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    p.reload()
    p.remove_hydrogens()


def get_water_residues(src):
    p = PDB(src)
    return p.water_residues


def strip_water(src, dest):
    if src != dest:
        shutil.copy2(src, dest)

    p = PDB(dest)
    p.strip_water()
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'strip_water', True)


def get_icode_residues(src):
    p = PDB(src)
    return p.ambiguous_residues


def select_icode_residues(src, dest, selections):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    for chain, resnum, resname in selections:
        p.select_icode_residue(chain, resnum, resname)
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'icode_residues', selections)


def get_altloc_atoms(src):
    p = PDB(src)
    return p.ambiguous_atoms


def select_altloc_atoms(src, dest, selections):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    for chain, resnum, resname, atomname, altloc in selections:
        p.select_altloc_atom(chain, resnum, resname, atomname, altloc)
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'altloc_atoms', selections)


def get_hetero_residues(src):
    p = PDB(src)
    return p.find_hetero_residues()


def select_hetero_residue(src, dest, selection):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    p.select_hetero_residues(selection[0], int(selection[1:]))
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'hetero_residue_selected', selection)
    p.extract_ligand()


def add_missing_atoms(src, dest):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    p.add_missing_atoms()


def get_non_standard_residues(src):
    p = PDB(src)
    return p.find_non_standard_residues()


def mutate_non_standard_residues(src, dest, mutations):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    for chain, resnum, new_resname in mutations:
        p.mutate_residue(chain, resnum, new_resname)
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'non_standard_residue_mutations', mutations)


def get_disulfide_bond_candidates(src):
    p = PDB(src)
    return p.find_disulfide_bond_candidates()


def build_disulfide_bonds(src, dest, cands):
    from ast import literal_eval
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    # clear disulfide bonds described in PDB
    p.connects = []
    for cand in cands:
        if isinstance(cand, str):
            cand = literal_eval(cand)
        p.build_disulfide_bond(cand)
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'disulfide_bond_builds', cands)


def get_solvent_ions(src):
    p = PDB(src)
    return p.solvents


def delete_solvent_ions(src, dest, selections):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    for chain, resname in selections:
        p.delete_solvent(chain, resname)
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'removed_solvents', selections)


def get_unknown_protonation_states(src):
    p = PDB(src)
    return p.his_residues


def mutate_protonation_states(src, dest, mutations):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    p.select_model(0)
    for chain, resnum, new_resname in mutations:
        p.mutate_residue(chain, resnum, new_resname)
    save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'unknown_protonation_state_mutations',
                  mutations)


def done(src, dest, ligand_net_charge='0'):
    if src != dest:
        shutil.copy2(src, dest)
    p = PDB(dest)
    p.done()
    # p.process_ligand(ligand_net_charge)
    # ligand 관련 코드는 모두 무시.
    # save_protocol(os.path.join(os.path.dirname(dest), 'protocol.json'), 'num_residues', len(p.residues))


def run(src, func, args):
    """
    invoke PDB preprocess within python

    :param src: source file path
    :param func: preprocess function
    :param args: preprocess arguments
    :return: run result
    """
    fetch_func = {
        # if second value of tuple True, convert result to list with list comprehension
        'models': (get_models, False),
        'chains': (get_chains, True),
        'hetero_residues': (get_hetero_residues, False),
        'icode_residues': (get_icode_residues, False),
        'altloc_atoms': (get_altloc_atoms, False),
        'non_standard_residues': (get_non_standard_residues, False),
        'disulfide_bond_candidates': (get_disulfide_bond_candidates, False),
        'solvent_ions': (get_solvent_ions, False),
        'unknown_protonation_states': (get_unknown_protonation_states, False),
    }

    alt_func = {
        # if second value of tuple True, pass args with json.parse, if None, not passes any arguments
        'select_model': (select_model, False),
        'select_chains': (select_chains, True),
        'strip_water': (strip_water, None),
        'remove_hydrogens': (remove_hydrogens, None),
        'select_hetero_residue': (select_hetero_residue, False),
        'select_icode_residues': (select_icode_residues, True),
        'select_altloc_atoms': (select_altloc_atoms, True),
        'add_missing_atoms': (add_missing_atoms, None),
        'mutate_non_standard_residues': (mutate_non_standard_residues, True),
        'build_disuflide_bonds': (build_disulfide_bonds, True),
        'delete_solvent_ions': (delete_solvent_ions, True),
        'mutate_protonation_states': (mutate_protonation_states, True),
        'done': (done, None),
    }

    if func in fetch_func:
        f, is_list = fetch_func[func]
        if is_list:
            return [x for x in f(src)]
        else:
            return f(src)

    if func in alt_func:
        f, is_json = alt_func[func]
        if is_json:
            f(src, src, json.loads(args))
        elif is_json is None:
            f(src, src)
        else:
            f(src, src, args)

    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('func')
    parser.add_argument('--args')
    parser.add_argument('--batch', action='store_true')
    _args = parser.parse_args()
    run_result = run(_args.src, _args.func, _args.args)
    if run_result:
        print(run_result)
