import os
import json
import shutil
import tempfile
import argparse
import subprocess
import mdtraj as mdt
from utils.rsm_util import generate_system, generate_input, post_process


def run_sfe(work_dir, mode, system=None, t=None, i=0, anal_serial='1'):
    """

    :param work_dir:
    :param mode:
    :param system:
    :param t:
    :param i:
    :param anal_serial:
    :return:
    """
    tempdir = tempfile.mkdtemp()

    shutil.copy2(os.path.join(work_dir, 'model.pdb'), os.path.join(tempdir, 'model.pdb'))

    if not system:
        try:
            system, t = generate_system(tempdir)
            # t = mdt.load(os.path.join(tempdir, 'prep/model_initial.pdb'))
        finally:
            if not os.path.exists(os.path.join(work_dir, 'prep')):
                os.makedirs(os.path.join(work_dir, 'prep'))

            shutil.copy2(os.path.join(tempdir, 'prep/model.xml'), os.path.join(work_dir, 'prep/model.xml'))
            shutil.copy2(os.path.join(tempdir, 'prep/model_initial.pdb'),
                         os.path.join(work_dir, 'prep/model_initial.pdb'))

    box_size = 128.0
    grid_size = 128

    params = {
        'mode': mode,
        'box_size': box_size,
        'grid_size': grid_size,
        'forcefiled': 'ff99sbildn',
    }

    with open(os.path.join(work_dir, 'analyses/%s/rism.in' % anal_serial), 'w') as f:
        json.dump(params, f, indent=2)

    rism_input = generate_input(tempdir, mode, box_size, grid_size, system, t, i)
    os.makedirs(os.path.join(tempdir, 'logs'))

    try:

        out_f = open(os.path.join(tempdir, 'logs/%s.out.log' % rism_input), 'w')
        err_f = open(os.path.join(tempdir, 'logs/%s.error.log' % rism_input), 'w')
        cwd = os.getcwd()
        os.chdir(tempdir)
        subprocess.call([
            '/opt/nbcc/bin/rism3d-x',
            os.path.join(tempdir, rism_input),
            '%d' % i], stdout=out_f, stderr=err_f)
        out_f.close()
        err_f.close()
        os.chdir(cwd)

        post_process(tempdir, mode, i)

        if not os.path.exists(os.path.join(work_dir, 'analyses/%s' % anal_serial)):
            os.makedirs(os.path.join(work_dir, 'analyses/%s' % anal_serial))

        shutil.copy2(os.path.join(tempdir, rism_input), os.path.join(work_dir, 'analyses/%s/%s' % (anal_serial, rism_input)))
        shutil.copy2(os.path.join(tempdir, 'frame_%d.xmu' % i),
                     os.path.join(work_dir, 'analyses/%s/frame_%d.xmu' % (anal_serial, i)))

        try:
            shutil.copy2(os.path.join(tempdir, 'frame_%d.thm' % i),
                         os.path.join(work_dir, 'analyses/%s/frame_%d.thm' % (anal_serial, i)))
        except FileNotFoundError:
            pass

        try:
            shutil.copy2(os.path.join(tempdir, 'xmua_atm.dat'),
                         os.path.join(work_dir, 'analyses/%s/xmua_atm_%d.dat' % (anal_serial, i)))
        except FileNotFoundError:
            pass

        try:
            shutil.copy2(os.path.join(tempdir, 'enea_atm.dat'),
                         os.path.join(work_dir, 'analyses/%s/enea_atm_%d.dat' % (anal_serial, i)))
        except FileNotFoundError:
            pass

        try:
            shutil.copy2(os.path.join(tempdir, 'enta_atm.dat'),
                         os.path.join(work_dir, 'analyses/%s/enta_atm_%d.dat' % (anal_serial, i)))
        except FileNotFoundError:
            pass

        if i == 0:
            shutil.copy2(os.path.join(tempdir, 'result.out'), os.path.join(work_dir, 'analyses/%s/result.out' % anal_serial))
        else:
            shutil.copy2(os.path.join(tempdir, 'result.out'), os.path.join(work_dir, 'analyses/%s/result_%d.out' % (anal_serial, i)))
    finally:
        if not os.path.exists(os.path.join(work_dir, 'logs')):
            os.makedirs(os.path.join(work_dir, 'logs'))

        shutil.copy2(os.path.join(tempdir, 'logs/%s.out.log' % rism_input),
                     os.path.join(work_dir, 'logs/%s.out.log' % rism_input))
        shutil.copy2(os.path.join(tempdir, 'logs/%s.error.log' % rism_input),
                     os.path.join(work_dir, 'logs/%s.error.log' % rism_input))
        shutil.rmtree(tempdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir')
    parser.add_argument('mode')

    args = parser.parse_args()
    run_sfe(args.work_dir, args.mode)
