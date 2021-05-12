import os
import sys
import subprocess
import shutil
import mdtraj as mdt
from simtk import unit
from simtk.openmm import app, XmlSerializer

class prep_openmm:
    def __init__(self, workdir, error_log):
        self.work_dir = workdir
        self.prep_dir = os.path.join(workdir, 'prep')
        try:
            os.makedirs(self.prep_dir)
        except OSError:
            pass

        self.cation = 'Na+'
        self.anion = 'Cl-'

        self.error_log = error_log


    def generate_openmm_files(self, chain_code, force_field, solvent_model, buffer_size):
        print('')
        print('---------------- run_prep_openmm')

        solvent_ff_map = {
            'TIP3PBOX': 'amber14/tip3p.xml',
            'SPCBOX': 'amber14/spce.xml',
            'TIP4PEWBOX': 'amber14/tip4pew.xml',
        }

        if 'amber99sbildn' in force_field:
            solvent_ff_map = {
                'TIP3PBOX': 'tip3p.xml',
                'SPCBOX': 'spce.xml',
                'TIP4PEWBOX': 'tip4pew.xml',
            }

        wmodel_map = {
            'TIP3PBOX': 'tip3p',
            'SPCBOX': 'spce',
            'TIP4PEWBOX': 'tip4pew',
        }

        cwd = os.getcwd()
        os.chdir(self.prep_dir)
        subprocess.call('cp {}/{}_model.pdb .'.format(self.work_dir, chain_code), shell=True)

        try:
            pdb = app.PDBFile('{}_model.pdb'.format(chain_code))
        except AttributeError as e:
            print(e)
            with open(self.error_log, "a") as f:
                print('{} --- prep_openmm : {}'.format(chain_code, e), file=f)
            return 'Exception'

        mdl = app.Modeller(pdb.topology, pdb.positions)

        print(force_field)
        forcefield = app.ForceField(force_field, solvent_ff_map[solvent_model])

        # ions are mandatory
        try:
            mdl.addSolvent(forcefield, model=wmodel_map[solvent_model], padding=buffer_size/10.0, positiveIon=self.cation, negativeIon=self.anion)
        except ValueError as e:
            print(e)
            with open(self.error_log, "a") as f:
                print('{} --- prep_openmm : {}'.format(chain_code, e), file=f)
            return 'Exception'

        import gzip
        # save the solvated model as a PDB
        with gzip.open('model_solv.pdb.gz', 'wt') as f:
            app.PDBFile.writeFile(mdl.topology, mdl.positions, f)

        # center and image the pdb file
        t = mdt.load('model_solv.pdb.gz')
        t.image_molecules(inplace=True)
        t.center_coordinates(mass_weighted=True)
        with gzip.open('model_solv.pdb.gz', 'wt') as f:
            app.PDBFile.writeFile(mdl.topology, t.xyz[0] * 10.0, f)

        # Unlike AMBER topology, cutoff distance is an input of the "system"
        # Here, we temporary fix the cutoff distance to 10.0 A
        # In the future, the interface should be changed.
        system = forcefield.createSystem(
            mdl.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True
        )

        # save the topology (system) xml file
        with open('model.xml', 'w') as f:
            f.write(XmlSerializer.serialize(system))

        model_xml_path = '{}/model.xml'.format(self.prep_dir)

        os.chdir(cwd)

        return model_xml_path
