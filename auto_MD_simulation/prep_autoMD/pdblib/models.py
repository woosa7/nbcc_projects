"""
DATA Model class for PDB file


"""
import os
import random
import shutil
import numpy as np
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
from simtk.openmm import app

# PDB Index for ATOM
RECORD = 0
SERIAL = 1
ATOMNAME = 2
ALTLOC = 3
RESNAME = 4
CHAIN = 5
RESNUM = 6
ICODE = 7
X = 8
Y = 9
Z = 10
OCC = 11
TFACTOR = 12
SEGID = 13
ELSYMBOL = 14
CHARGE = 15

# constants for handle residues
RESPROT = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL', 'HID', 'HIE', 'HIN', 'HIP', 'CYX', 'ASH', 'GLH',
           'LYH', 'ACE', 'NME', 'GL4', 'AS4')

RESNA = ('C', 'G', 'U', 'A', 'DC', 'DG', 'DT', 'DA')

RESSOLV = ('WAT', 'HOH', 'AG', 'AL', 'Ag', 'BA', 'BR', 'Be', 'CA', 'CD', 'CE',
           'CL', 'CO', 'CR', 'CS', 'CU', 'CU1', 'Ce', 'Cl-', 'Cr', 'Dy', 'EU',
           'EU3', 'Er', 'F', 'FE', 'FE2', 'GD3', 'HE+', 'HG', 'HZ+', 'Hf',
           'IN', 'IOD', 'K', 'K+', 'LA', 'LI', 'LU', 'MG', 'MN', 'NA', 'NH4',
           'NI', 'Na+', 'Nd', 'PB', 'PD', 'PR', 'PT', 'Pu', 'RB', 'Ra', 'SM',
           'SR', 'Sm', 'Sn', 'TB', 'TL', 'Th', 'Tl', 'Tm', 'U4+', 'V2+', 'Y',
           'YB2', 'ZN', 'Zr')

AMBER_SUPPORT_RES = set(RESPROT + RESNA + RESSOLV)

AA_ATOMS = {
    'ALA': ('N', 'H', 'CA', 'HA', 'CB', 'HB1', 'HB2', 'HB3', 'C', 'O'),
    'ARG': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'NE', 'HE', 'CZ',
            'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'C', 'O'),
    'ASH': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'OD2', 'HD2', 'C', 'O'),
    'ASN': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'ND2', 'HD21', 'HD22', 'C', 'O'),
    'ASP': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'OD2', 'C', 'O'),
    'CYM': ('N', 'H', 'CA', 'HA', 'CB', 'HB3', 'HB2', 'SG', 'C', 'O'),
    'CYS': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'HG', 'C', 'O'),
    'CYX': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'C', 'O'),
    'GLH': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2', 'HE2', 'C', 'O'),
    'GLN': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'NE2', 'HE21', 'HE22', 'C', 'O'),
    'GLU': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2', 'C', 'O'),
    'GLY': ('N', 'H', 'CA', 'HA2', 'HA3', 'C', 'O'),
    'HID': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2', 'CD2', 'HD2', 'C', 'O'),
    'HIE': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'ND1', 'CE1', 'HE1', 'NE2', 'HE2', 'CD2', 'HD2', 'C', 'O'),
    'HIP': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2', 'HE2', 'CD2', 'HD2',
            'C', 'O'),
    'HYP': ('N', 'CD', 'HD22', 'HD23', 'CG', 'HG', 'OD1', 'HD1', 'CB', 'HB2', 'HB3', 'CA', 'HA', 'C', 'O'),
    'ILE': ('N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG2', 'HG21', 'HG22', 'HG23', 'CG1', 'HG12', 'HG13', 'CD1', 'HD11',
            'HD12', 'HD13', 'C', 'O'),
    'LEU': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22',
            'HD23', 'C', 'O'),
    'LYN': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ',
            'HZ2', 'HZ3', 'C', 'O'),
    'LYS': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ',
            'HZ1', 'HZ2', 'HZ3', 'C', 'O'),
    'MET': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'SD', 'CE', 'HE1', 'HE2', 'HE3', 'C', 'O'),
    'PHE': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'HZ', 'CE2', 'HE2', 'CD2',
            'HD2', 'C', 'O'),
    'PRO': ('N', 'CD', 'HD2', 'HD3', 'CG', 'HG2', 'HG3', 'CB', 'HB2', 'HB3', 'CA', 'HA', 'C', 'O'),
    'SER': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'OG', 'HG', 'C', 'O'),
    'THR': ('N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG2', 'HG21', 'HG22', 'HG23', 'OG1', 'HG1', 'C', 'O'),
    'TRP': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'NE1', 'HE1', 'CE2', 'CZ2', 'HZ2', 'CH2',
            'HH2', 'CZ3', 'HZ3', 'CE3', 'HE3', 'CD2', 'C', 'O'),
    'TYR': ('N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'OH', 'HH', 'CE2',
            'HE2', 'CD2', 'HD2', 'C', 'O'),
    'VAL': ('N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'C', 'O'),
}

MUTMAP = {
    'SEP': 'SER',
    'TPO': 'THR',
    'PTR': 'TYR',
    'PDS': 'ASP',
    'PHL': 'ASP',
    'MLY': 'LYS',
    'CSP': 'CYS',
    'MSE': 'MET',
    'GMA': 'GLU',
    'OCS': 'CYS',
    'CSX': 'CYS',
}


class PDB:
    """
    DATA Model class for PDB file

    """
    def __init__(self, filepath, force_field='amber14/protein.ff14SB.xml'):
        self.filepath = filepath
        self.connects = []
        self.selected_atoms = []
        self.seqres = dict()  # SEQRES described in PDB file
        self.force_field = force_field

    @staticmethod
    def _record(s):
        """
        get record field of line

        :param s:
        :return:
        """
        return s[:6].strip()

    @staticmethod
    def _atom(s):
        """
        Parsing template for ATOM and HETATM record

        :param s:
        :return:
        """
        return [
            s[:6].strip(),  # RECORD
            int(s[6:11]),  # SERIAL
            s[12:16].strip(),  # ATOMNAME
            s[16:17].strip(),  # ALTLOC
            s[17:20].strip(),  # RESNAME
            s[21:22].strip(),  # CHAIN
            int(s[22:26]),  # RESNUM
            s[26:27].strip(),  # ICODE
            float(s[30:38]),  # X
            float(s[38:46]),  # Y
            float(s[46:54]),  # Z
            float(s[54:60]) if s[54:60].strip() else 0.0,  # OCC
            float(s[60:66]) if s[54:60].strip() else 0.0,  # TFACTOR
            s[72:76].strip(),  # SEGID
            s[76:78].strip(),  # ELSYMBOL
            s[78:80].strip(),  # CHARGE
        ]

    @staticmethod
    def _ter(s):
        """
        Parsing template for TER record

        :param s:
        :return:
        """
        return [
            s[:6].strip(),  # RECORD
            int(s[6:11]),  # SERIAL
            '',
            '',
            s[17:20].strip(),  # RESNAME
            s[21:22].strip(),  # CHAIN
            int(s[22:26]),  # RESNUM
            s[26:27].strip(),  # ICODE
        ]

    def _isatom(self, atom_or_line):
        """
        check line of file is atom record of not

        :param atom_or_line:
        :return:
        """
        if isinstance(atom_or_line, list):
            return atom_or_line[RECORD] in ('ATOM', 'HETATM')
        else:
            return self._record(atom_or_line) in ('ATOM', 'HETATM')

    def save(self, output_file=None):
        """
        save pdb structure to output file

        :param output_file:
        :return:
        """
        if not output_file:
            output_file = self.filepath

        with open(output_file, 'w') as f:
            for atom in self.selected_atoms:
                if self._isatom(atom):
                    f.write('%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%3.1s\n' % tuple(atom[:]))
                elif atom[RECORD] == 'TER':
                    f.write('%-6s%5d %s%s     %3s %1s%4d%1s\n' % tuple(atom[:]))

            for connect in self.connects:
                f.write('%-6s%5s%5s%5s%5s%5s\n' % tuple(connect[:6]))

            f.write('END\n')

    def reload(self):
        self.select_model(0, dry=True)

    def delete_atom_at(self, index):
        """
        delete atom at position

        :param index:
        :return:
        """
        try:
            self.selected_atoms[index] = [None]
        except IndexError:
            pass

    @property
    def models(self):
        with open(self.filepath, 'r') as f:
            pdb_content = f.readlines()

        _models = []
        _current_model = 0
        for line in pdb_content:
            if self._record(line) == 'ENDMDL':
                _models.append(_current_model)
                _current_model += 1

        if not _models:
            _models.append(_current_model)

        return _models

    @property
    def chains(self):
        self.reload()

        _chains = set()
        for _atom in self.selected_atoms:
            _chains.add(_atom[CHAIN])

        return sorted(_chains)

    @property
    def residues(self):
        self.reload()

        _residues = list()
        for _atom in self.selected_atoms:
            if not self._isatom(_atom):
                continue

            if _atom[RESNAME] in ('WAT', 'HOH'):
                continue

            res = (_atom[CHAIN], _atom[RESNUM], _atom[RESNAME])
            if res not in _residues:
                _residues.append(res)
        return _residues

    @property
    def ambiguous_residues(self):
        self.reload()

        _ambiguous_residues = dict()
        for i, _atom in enumerate(self.selected_atoms):
            if not self._isatom(_atom):
                continue

            # insertion code residues
            key = (_atom[CHAIN], _atom[RESNUM])
            value = (_atom[CHAIN], _atom[RESNUM], _atom[RESNAME], _atom[ICODE])

            if key not in _ambiguous_residues:
                _ambiguous_residues[key] = dict()

            if value not in _ambiguous_residues[key]:
                _ambiguous_residues[key][value] = [i, i]
            else:
                _ambiguous_residues[key][value][1] = i

        ret = dict()
        for key, residues in _ambiguous_residues.items():
            if len(list(residues.keys())) > 1:
                ret[key] = residues
        return ret

    @property
    def ambiguous_atoms(self):
        self.reload()

        _ambiguous_atoms = dict()
        for i, _atom in enumerate(self.selected_atoms):
            if not self._isatom(_atom):
                continue

            if not _atom[ALTLOC]:
                continue

            key = (_atom[CHAIN], _atom[RESNUM], _atom[RESNAME], _atom[ATOMNAME])
            if key not in _ambiguous_atoms:
                _ambiguous_atoms[key] = set()
            row = (_atom[CHAIN], _atom[RESNUM], _atom[RESNAME], _atom[ATOMNAME], _atom[ALTLOC], i)
            _ambiguous_atoms[key].add(row)
        # return _ambiguous_atoms

        ret = dict()
        for key, atoms in _ambiguous_atoms.items():
            # if len(list(atoms)) > 1:
            ret[key] = sorted(atoms, key=lambda x: x[4])
        return ret

    @property
    def water_atoms_indexes(self):
        self.reload()

        _water_atom_indexes = []
        for i, _atom in enumerate(self.selected_atoms):
            if not self._isatom(_atom):
                continue

            if _atom[RESNAME] in ('WAT', 'HOH'):
                _water_atom_indexes.append(i)
        return _water_atom_indexes

    @property
    def water_residues(self):
        self.reload()

        _residues = []
        for i in self.water_atoms_indexes:
            _atom = self.selected_atoms[i]
            res = (_atom[CHAIN], _atom[RESNUM], _atom[RESNAME])
            if res not in _residues:
                _residues.append(res)
        return _residues

    @property
    def ligands(self):
        self.reload()

        _ligands = dict()
        for chain in self.chains:
            _check_ter = False
            for i, _atom in enumerate(self.selected_atoms):
                if _atom[CHAIN] != chain:
                    continue

                if _atom[RECORD] == 'TER':
                    _check_ter = True
                    continue

                if not _check_ter and _atom[RECORD] != 'HETATM':
                    continue

                if _atom[RESNAME] in ('WAT', 'HOH'):
                    continue

                key = (_atom[CHAIN], _atom[RESNUM], _atom[RESNAME])
                if key not in _ligands:
                    _ligands[key] = [i, i]
                else:
                    _ligands[key][1] = i
        return _ligands

    @property
    def cys_atom_indexes(self):
        self.reload()

        _cys_atom_indexes = []
        for i, _atom in enumerate(self.selected_atoms):
            if _atom[RESNAME] in ('CYS', 'CYX', 'CYM') and _atom[ATOMNAME] == 'SG':
                _cys_atom_indexes.append(i)

        return _cys_atom_indexes

    @property
    def solvents(self):
        self.reload()

        _solvents = set()
        for i, _atom in enumerate(self.selected_atoms):
            if _atom[RESNAME] in RESSOLV:
                _solvents.add((_atom[CHAIN], _atom[RESNAME]))
        return list(_solvents)

    @property
    def his_residues(self):
        self.reload()

        _his_residues = set()
        for i, _atom in enumerate(self.selected_atoms):
            if _atom[RESNAME] in ('HIS', 'HID', 'HIE', 'HIP'):
                _his_residues.add((_atom[CHAIN], _atom[RESNUM], _atom[RESNAME]))
        return _his_residues

    def select_model(self, model, dry=False):
        with open(self.filepath, 'r') as f:
            pdb_content = f.readlines()

        self.connects = []
        self.selected_atoms = []
        _current_model = 0
        for line in pdb_content:
            if self._record(line) == 'ENDMDL':
                _current_model += 1
                continue

            if _current_model > model:
                break

            if _current_model != model:
                continue

            if self._record(line) == 'TER':
                self.selected_atoms.append(self._ter(line))

            if self._record(line) == 'CONECT':
                self.connects.append(
                    [self._record(line), line[6:11].strip(), line[11:16].strip(), line[16:21].strip(),
                     line[21:26].strip(), line[26:31].strip()] + [None] * 10
                )

            if self._isatom(line):
                self.selected_atoms.append(self._atom(line))

        if not dry:
            # reload method would not call remove_hydrogens() and save().
            # it only loads all atoms to selected_atoms from file
            self.remove_hydrogens()
            self.connects = []
            self.save()

    def select_chains(self, chains):
        self.reload()

        _selected_atoms = []
        for _atom in self.selected_atoms:
            if _atom[CHAIN] in chains:
                _selected_atoms.append(_atom)
        self.selected_atoms = _selected_atoms
        self.save()

    def add_missing_atoms(self):
        """
        Add missing atoms

        :return:
        """
        def chain_length(topology):
            ret = []
            for chain in topology.chains():
                count = 0
                for _ in chain.residues():
                    count += 1
                ret.append(count)
            return ret

        pf = PDBFixer(filename=self.filepath)
        pf.findMissingResidues()
        pf.findMissingAtoms()

        chain_len = chain_length(pf.topology)

        # remove N or C terminal missing
        del_keys = []
        int_keys = []  # internal missing residue keys
        for mr in pf.missingResidues:
            # N-term or C-term
            if mr[1] == 0 or chain_len[mr[0]] == mr[1]:
                del_keys.append(mr)
            else:
                int_keys.append(mr)

        for dk in del_keys:
            del pf.missingResidues[dk]

        chains = list(self.chains)
        replaced_res = []
        for ik in int_keys:
            for index, res in enumerate(pf.missingResidues[ik]):
                if res in AMBER_SUPPORT_RES:
                    continue

                if res in MUTMAP:
                    replaced_res.append((chains[ik[0]], ik[1] + index + 1, pf.missingResidues[ik][index]))
                    pf.missingResidues[ik][index] = MUTMAP[res]
                else:
                    replaced_res.append((chains[ik[0]], ik[1] + index + 1, pf.missingResidues[ik][index]))
                    pf.missingResidues[ik][index] = 'GLY'

        # to aviod template not found error on non standard missing residue
        # change missing residue names to standard amino acids
        # and add missing atoms then change name again its original
        pf.addMissingAtoms()
        with open(self.filepath, 'w') as f:
            PDBFile.writeFile(pf.topology, pf.positions, f, True)

        self.reload()
        self.change_resname(replaced_res)

        # pdb fixer bug for minus residue number
        for _atom in self.selected_atoms:
            if _atom[RESNUM] > 9990:
                _atom[RESNUM] -= 10000

        self.save(output_file=self.filepath)

    def change_resname(self, residues):
        """
        Change resname

        :param residues:
        :return:
        """
        self.reload()

        res_map = dict()
        for chain, resnum, new_resname in residues:
            res_map[(chain, resnum)] = new_resname

        for _atom in self.selected_atoms:
            if not self._isatom(_atom):
                continue

            key = (_atom[CHAIN], _atom[RESNUM])
            if key in res_map:
                _atom[RESNAME] = res_map[key]

        self.save()

    def remove_hydrogens(self):
        self.reload()

        for ind, atom in enumerate(self.selected_atoms):
            if self._isatom(atom):
                if atom[ELSYMBOL] in ('H', 'D'):
                    self.delete_atom_at(ind)
        self.save()

    def strip_water(self):
        """
        remove all water atoms from structure

        :return:
        """
        self.reload()

        for ind in self.water_atoms_indexes:
            self.delete_atom_at(ind)
        self.save()

    def select_icode_residue(self, chain, resnum, icode):
        self.reload()
        residues = self.ambiguous_residues[(chain, resnum)]

        for res, index_range in residues.items():
            _icode = res[3]
            if _icode == icode:  # selected icode
                continue

            # else delete from PDB
            for i in range(index_range[0], index_range[1] + 1):
                self.delete_atom_at(i)
        self.save()

    def select_altloc_atom(self, chain, resnum, resname, atomname, altloc):
        self.reload()
        atoms = self.ambiguous_atoms[(chain, resnum, resname, atomname)]
        for atom in atoms:
            if atom[4] == altloc:
                continue
            self.delete_atom_at(atom[5])
        self.save()

    def find_hetero_residues(self):
        ret = dict()
        for chain, resnum, resname in self.ligands:
            if chain not in ret:
                ret[chain] = [(resnum, resname)]
            else:
                ret[chain].append((resnum, resname))
        return ret

    def select_hetero_residues(self, chain, resnum):
        self.reload()
        for key, inds in self.ligands.items():
            _chain, _resnum, resname = key
            if _chain == chain and _resnum == resnum:
                continue

            i, j = inds
            for i in range(i, j + 1):
                self.delete_atom_at(i)
        self.save()

    def mutate_residue(self, chain, resnum, new_resname):
        self.reload()
        for i, _atom in enumerate(self.selected_atoms):
            if not self._isatom(_atom):
                continue

            if _atom[CHAIN] != chain:
                continue

            if _atom[RESNUM] != resnum:
                continue

            if new_resname in AA_ATOMS and _atom[ATOMNAME] not in AA_ATOMS[new_resname]:
                self.delete_atom_at(i)
            else:
                self.selected_atoms[i][RESNAME] = new_resname
                self.selected_atoms[i][RECORD] = 'ATOM'
        self.save()

    def delete_solvent(self, chain, resname):
        self.reload()

        if resname not in RESSOLV:
            return  # do nothing

        for i, _atom in enumerate(self.selected_atoms):
            if not self._isatom(_atom) and not _atom[RECORD] == 'TER':
                continue

            if _atom[CHAIN] != chain:
                continue

            if _atom[RESNAME] == resname:
                self.delete_atom_at(i)
        self.save()

    def find_non_standard_residues(self):
        """
        should select_chains first.
        multiple model not supported.

        :return:
        """
        self.reload()

        ret = []
        for _atom in self.selected_atoms:
            if not self._isatom(_atom):
                continue

            if _atom[RESNAME] in AMBER_SUPPORT_RES:
                continue

            # if _atom[ATOMNAME] != 'CA':
            #     continue
            key = (_atom[CHAIN], _atom[RESNUM], _atom[RESNAME])

            if key not in ret and key not in self.ligands:
                ret.append(key)
        return ret

    def find_disulfide_bond_candidates(self, cutoff=3.0):
        """
        Find disulfide bond candidates

        :param cutoff:
        :return:
        """
        from scipy.spatial import distance

        ret = []
        self.reload()
        for chain in self.chains:
            coords = []
            for index in self.cys_atom_indexes:
                _atom = self.selected_atoms[index]
                if _atom[CHAIN] != chain:
                    continue

                coords.append((_atom[X], _atom[Y], _atom[Z]))

            if not coords:
                continue

            coords = np.array(coords)
            dist = distance.pdist(coords, metric='euclidean')
            cands = dict()
            idist = 0
            for i in self.cys_atom_indexes:
                iatm = self.selected_atoms[i]
                if iatm[CHAIN] != chain:
                    continue

                for j in self.cys_atom_indexes:
                    if i >= j:
                        continue

                    jatm = self.selected_atoms[j]
                    if iatm[CHAIN] != jatm[CHAIN]:
                        continue

                    d = dist[idist]
                    idist += 1

                    if d >= cutoff:
                        continue

                    key = (iatm[SERIAL], iatm[CHAIN], iatm[RESNUM], iatm[RESNAME], iatm[ATOMNAME])
                    if key not in cands or cands[key][2] > d:
                        cands[key] = (
                            (iatm[SERIAL], iatm[CHAIN], iatm[RESNUM], iatm[RESNAME], iatm[ATOMNAME]),
                            (jatm[SERIAL], jatm[CHAIN], jatm[RESNUM], jatm[RESNAME], jatm[ATOMNAME]),
                            d,
                        )

            ret += [value for key, value in cands.items()]
        return ret

    def build_disulfide_bond(self, cand):
        iatm, jatm, distance = cand
        self.connects.append(['CONECT', iatm[0], jatm[0], '', '', ''] + [None] * 10)
        self.connects.append(['CONECT', jatm[0], iatm[0], '', '', ''] + [None] * 10)
        self.mutate_residue(iatm[1], iatm[2], 'CYX')
        self.mutate_residue(jatm[1], jatm[2], 'CYX')
        self.save()

    def done(self):
        """
        Finalize pdb pre processing
        :return:
        """
        # delete all ligands
        self.select_hetero_residues('-', 1)

        # backup old disulfide bond info
        old_cands = self.find_disulfide_bond_candidates()

        # load residue variants
        res = []
        ligs = []
        residues = self.residues

        for chain, resnum, resname in residues:
            if resname in ('HIP', 'HID', 'HIE', 'ASH', 'CYX', 'CYS', 'GLH', 'LYN'):
                res.append(resname)
            # check ligand
            elif (chain, resnum, resname) in self.ligands:
                ligs.append(None)
            else:
                res.append(None)

        res = res + ligs

        # add missing atoms for mutated residues
        pf = PDBFixer(filename=self.filepath)
        # here we don't need to find missing residues because we have already add missing residues
        pf.missingResidues = {}
        pf.findMissingAtoms()
        pf.addMissingAtoms()

        # add missing hydrogens
        ff = app.ForceField(self.force_field)
        mod = app.Modeller(pf.topology, pf.positions)
        # random.seed(15)
        # try:
        #     mod.addHydrogens(variants=res, forcefield=ff)
        # except:
        #     print('******* Error in addHydrogens()')
        #     pass

        with open(self.filepath, 'w') as f:
            PDBFile.writeFile(mod.topology, mod.positions, f)

        self.reload()
        new_cands = self.find_disulfide_bond_candidates()
        if new_cands:
            connects = []
            for old, new in zip(old_cands, new_cands):
                if old[0][3] == 'CYX' and old[1][3] == 'CYX':
                    connects.append(['CONECT', new[0][0], new[1][0], new[1][0], '', ''] + [None] * 11)
                    connects.append(['CONECT', new[1][0], new[0][0], new[0][0], '', ''] + [None] * 11)
            self.connects = connects
            self.save()

    def extract_ligand(self):
        # extract ligand residue to _ligand.pdb file
        if self.ligands:
            chain, resnum, resname = list(self.ligands.keys())[0]
            for i, _atom in enumerate(self.selected_atoms):
                if not self._isatom(_atom):
                    self.selected_atoms[i] = [None]
                    continue

                if _atom[CHAIN] != chain:
                    self.selected_atoms[i] = [None]
                    continue

                if _atom[RESNUM] != resnum:
                    self.selected_atoms[i] = [None]
                    continue

                if _atom[RESNAME] != resname:
                    self.selected_atoms[i] = [None]
                    continue

            self.connects = []
            work_dir = os.path.dirname(self.filepath)
            self.save(output_file=os.path.join(work_dir, '_ligand.pdb'))

    def process_ligand(self, net_charge='0'):
        import subprocess
        work_dir = os.path.dirname(self.filepath)
        ligand_h_pdb = os.path.join(work_dir, '_ligand_h.pdb')
        ligand_h_tmp_pdb = os.path.join(work_dir, '_ligand_h_tmp.pdb')
        ligand_mol2 = os.path.join(work_dir, '_ligand.mol2')
        ligand_mol2_frcmod = os.path.join(work_dir, '_ligand.mol2.frcmod')

        if not os.path.exists(ligand_h_pdb):
            return

        # delete all ligands
        # self.select_hetero_residues('-', 1)

        num_atoms = self.selected_atoms[-1][SERIAL]
        # source = os.path.join(os.path.dirname(self.filepath), '_ligand_h.pdb')
        # target = os.path.join(os.path.dirname(self.filepath), '_ligand_h_tmp.pdb')
        source = ligand_h_pdb
        target = ligand_h_tmp_pdb
        with open(source, 'r') as src:
            with open(target, 'w') as f:
                src.seek(0)
                hydrogens = []
                hydrogen_serials = []
                atoms = []
                atom_serials = []
                connections = []
                for line in src:
                    if line[:6] == 'CONECT':
                        connections.append(
                            ['CONECT', line[6:11].strip(), line[11:16].strip(), line[16:21].strip(), '', ''])
                        continue

                    if not self._isatom(line):
                        continue

                    if line[76:78].strip() == 'H':
                        hydrogen_serials.append(line[6:11].strip())
                        hydrogens.append('HETATM%5d ' + line[12:])
                        continue

                    # print(line, file=f, end='')
                    atom_serials.append(line[6:11].strip())
                    atoms.append('HETATM%5d ' + line[12:])

                for i, atom in enumerate(atoms):
                    new_serial = num_atoms + i + 1
                    for j, connection in enumerate(connections):
                        for k, serial in enumerate(connection):
                            if serial == atom_serials[i]:
                                connection[k] = str(new_serial)
                        connections[j] = connection

                    f.write(atom % new_serial)

                for i, hydrogen in enumerate(hydrogens):
                    new_serial = num_atoms + len(atoms) + i + 1
                    for j, connection in enumerate(connections):
                        for k, serial in enumerate(connection):
                            if serial == hydrogen_serials[i]:
                                connection[k] = str(new_serial)
                        connections[j] = connection
                    f.write(hydrogen % new_serial)

        shutil.copy2(target, source)
        os.remove(target)

        import sys
        bin_dir = os.path.dirname(sys.executable)
        antechamber_exe = os.path.join(bin_dir, 'antechamber')
        parmchk_exe = os.path.join(bin_dir, 'parmchk')

        # subprocess.check_call('reduce -BUILD _ligand_h.pdb > _ligand_h.pdb', shell=True)
        cmd1 = '%s -i %s -fi pdb -o %s -fo mol2 -c bcc -s 2 -nc %s' % (antechamber_exe, ligand_h_pdb, ligand_mol2, net_charge)
        out1 = subprocess.check_output(cmd1, shell=True)
        print(out1.decode())

        cmd2 = '%s -i %s -f mol2 -o %s' % (parmchk_exe, ligand_mol2, ligand_mol2_frcmod)
        out2 = subprocess.check_output(cmd2, shell=True)
        print(out2.decode())

        from glob import glob
        for file in glob('ANTECHAMBER*'):
            os.remove(file)

        self.generate_ligand_xml()

        ligand = PDB(os.path.join(work_dir, '_ligand_h.pdb'))
        ligand.reload()
        self.selected_atoms += ligand.selected_atoms
        self.connects += connections
        self.save()

    def generate_ligand_xml(self):
        from openmoltools.utils import create_ffxml_file
        work_dir = os.path.dirname(self.filepath)
        cwd = os.getcwd()
        os.chdir(work_dir)
        create_ffxml_file(['_ligand.mol2'], ['_ligand.mol2.frcmod'], '_ligand.xml')
        os.chdir(cwd)
