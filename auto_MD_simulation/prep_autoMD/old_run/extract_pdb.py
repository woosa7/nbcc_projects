#!/homes/epsilon/users/junwon/anaconda3/envs/prowave/bin/python
"""
Run 3D-RISM for ProWaVE
"""
import os
import argparse
import mdtraj as mdt


PARSER = argparse.ArgumentParser()
PARSER.add_argument('work_dir')


def run(work_dir):
    # atom selection
    md_dcd = os.path.join(work_dir, 'md/1/md.dcd')
    topology = os.path.join(work_dir, 'prep/model_solv.pdb.gz')

    t = mdt.load(md_dcd, top=topology)
    t.image_molecules(inplace=True)
    t.center_coordinates(mass_weighted=True)

    sel = t.topology.select('protein')

    nframe = t.xyz.shape[0]

    try:
        os.makedirs(os.path.join(work_dir, 'analyses/1/pdb'))
    except OSError:
        pass

    for iframe in [x for x in range(nframe) if x % 10 == (nframe-1) % 10]:
        structure = t[iframe]
        structure.save_pdb(os.path.join(work_dir, 'analyses/1/pdb/frame_%d.pdb' % iframe))


if __name__ == '__main__':
    args = PARSER.parse_args()
    run(os.path.abspath(args.work_dir))
