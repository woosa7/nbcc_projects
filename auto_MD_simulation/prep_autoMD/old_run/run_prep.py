#!/home/nbcc/anaconda3/envs/prowave/bin/python
import os
import json
import argparse
import subprocess
import shutil
import tempfile
import mdtraj as mdt
from simtk import unit
from simtk.openmm import app, XmlSerializer


PARSER = argparse.ArgumentParser()
PARSER.add_argument('work_dir')
PARSER.add_argument('--solvent_model')
PARSER.add_argument('--buffer_size')
PARSER.add_argument('--cation')
PARSER.add_argument('--anion')
PARSER.add_argument('--batch', action='store_true')
PARSER.add_argument('--dependency')


def write_prep_input(work_dir, solvent_model, buffer_size, cation, anion):
    prep_dir = os.path.join(work_dir, 'prep')

    try:
        os.makedirs(prep_dir)
    except OSError:
        pass

    with open(os.path.join(prep_dir, 'input.json'), 'w') as f:
        json.dump({
            'work_dir': work_dir,
            'solvent_model': solvent_model,
            'buffer_size': int(buffer_size),
            'cation': cation,
            'anion': anion,
        }, f, indent=2)


def generate_openmm_system(work_dir):
    tempdir = tempfile.mkdtemp()

    prep_dir = os.path.join(work_dir, 'prep')
    shutil.copy2('%s/input.json' % prep_dir, os.path.join(tempdir, 'input.json'))
    shutil.copy2(os.path.join(work_dir, 'model.pdb'), os.path.join(tempdir, 'model.pdb'))
    if os.path.exists(os.path.join(work_dir, '_ligand.xml')):
        shutil.copy2(os.path.join(work_dir, '_ligand.xml'), os.path.join(tempdir, '_ligand.xml'))

    cwd = os.getcwd()
    os.chdir(tempdir)

    with open('input.json') as f:
        prep_input = json.load(f)

    solvent_model = prep_input['solvent_model']
    buffer_size = prep_input['buffer_size']
    cation = prep_input['cation']
    anion = prep_input['anion']

    solvent_ff_map = {
        # 'TIP3PBOX': 'amber14/tip3p.xml',
        'TIP3PBOX': 'tip3p.xml',
        'SPCBOX': 'spce.xml',
        'TIP4PEWBOX': 'tip4pew.xml',
    }

    wmodel_map = {
        'TIP3PBOX': 'tip3p',
        'SPCBOX': 'spce',
        'TIP4PEWBOX': 'tip4pew',
    }

    pdb = app.PDBFile('model.pdb')
    mdl = app.Modeller(pdb.topology, pdb.positions)

    if not os.path.exists('_ligand.xml'):
        forcefield = app.ForceField('amber99sbildn.xml', solvent_ff_map[solvent_model])
    else:
        import sys
        env_dir = os.path.dirname(os.path.dirname(sys.executable))

        gaff_path = os.path.join(env_dir, 'lib/python3.6/site-packages/openmoltools/parameters/gaff.xml')
        forcefield = app.ForceField('amber99sbildn.xml', gaff_path, '_ligand.xml', solvent_ff_map[solvent_model])

    # ions are mandatory
    mdl.addSolvent(forcefield, model=wmodel_map[solvent_model], padding=buffer_size / 10.0, positiveIon=cation,
                   negativeIon=anion)

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

    shutil.copy2('model_solv.pdb.gz', os.path.join(work_dir, 'prep/model_solv.pdb.gz'))
    shutil.copy2('model.xml', os.path.join(work_dir, 'prep/model.xml'))
    os.chdir(cwd)
    shutil.rmtree(tempdir)


def system_run(cmd, stdout=None, stderr=None):
    # print(*cmd)

    stdout_f = None
    if stdout:
        stdout_f = open(stdout, 'w')

    stderr_f = None
    if stderr:
        stderr_f = open(stderr, 'w')

    exitcode = subprocess.call(cmd, stdout=stdout_f, stderr=stderr_f)

    if stdout_f:
        stdout_f.close()

    if stderr_f:
        stderr_f.close()

    return exitcode


def submit_batch(work_dir, solvent_model, buffer_size, cation, anion, dependency=None):
    prep_dir = os.path.join(work_dir, 'prep')
    stdout = os.path.join(prep_dir, 'stdout')
    stderr = os.path.join(prep_dir, 'stderr')

    cmd = [
        '/usr/local/bin/sbatch',
        '--nodes', '1',
        '--time', '168:00:00',
        '--job-name', 'PREP',
        '--ntasks', '1',
        # 'exclude', 'node[7-8]',
        '--output', stdout,
        '--error', stderr,
        '--partition', 'prowave',
    ]

    if dependency:
        cmd += ['--dependency', dependency]

    cmd += [
        # os.path.abspath(__file__),
        # sys.executable, '-m', 'utils.run_prep',
        os.path.abspath(__file__),
        work_dir,
        '--solvent_model', solvent_model,
        '--buffer_size', buffer_size,
        '--cation', cation,
        '--anion', anion,
    ]

    try:
        msg = subprocess.check_output(cmd).decode().strip()
        print(msg)
        if msg.startswith('Submitted batch job'):
            batch_jobid = int(msg.split()[3])

            with open(os.path.join(prep_dir, 'status'), 'w') as f:
                f.write('submitted %d' % batch_jobid)

            return batch_jobid
        else:
            return 'Failed'
    except FileNotFoundError:
        return '-1'


if __name__ == '__main__':
    args = PARSER.parse_args()
    _params = {
        'work_dir': args.work_dir,
        'solvent_model': args.solvent_model if args.solvent_model else 'TIP3PBOX',
        'buffer_size': args.buffer_size if args.buffer_size else '10',
        'cation': args.cation if args.cation else 'Na+',
        'anion': args.anion if args.anion else 'Cl-'
    }
    write_prep_input(**_params)
    if args.batch:
        if args.dependency:
            _params['dependency'] = args.dependency
        result = submit_batch(**_params)
        print(result)
    else:
        generate_openmm_system(os.path.abspath(args.work_dir))
