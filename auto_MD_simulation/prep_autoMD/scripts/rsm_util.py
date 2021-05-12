"""
NBCC 3D-RISM utility
"""
import os
import subprocess
import math
import numpy as np
import mdtraj as mdt
from simtk import unit, openmm as mm
from simtk.openmm import app


def generate_system(work_dir):
    pdb = app.PDBFile(os.path.join(work_dir, 'model.pdb'))
    mdl = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')

    mdl.addSolvent(forcefield, model='tip3p', padding=10.0 / 10.0, positiveIon='Na+',
                   negativeIon='Cl-')

    system = forcefield.createSystem(
        mdl.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True
    )

    if not os.path.exists(os.path.join(work_dir, 'prep')):
        os.makedirs(os.path.join(work_dir, 'prep'))

    # save the solvated model as a PDB
    with open(os.path.join(work_dir, 'prep/model_initial.pdb'), 'w') as f:
        app.PDBFile.writeFile(mdl.topology, mdl.positions, f)

    # save the topology (system) xml file
    with open(os.path.join(work_dir, 'prep/model.xml'), 'w') as f:
        f.write(mm.XmlSerializer.serialize(system))

    # add force (restraints)
    # res_weight = 10.0 * 418.4  # kcal/mol/A^2 to kJ/mol/nm^2
    # force = mm.CustomExternalForce('k*(periodicdistance(x,y,z,x0,y0,z0)^2)')
    # force.addGlobalParameter('k', res_weight)
    # force.addPerParticleParameter('x0')
    # force.addPerParticleParameter('y0')
    # force.addPerParticleParameter('z0')
    # topo = mdl.topology
    # pos = mdl.positions
    #
    # for atom in topo.atoms():
    #     resname = atom.residue.name
    #
    #     # atmname = atom.name
    #     if resname not in ('HOH', 'NA', 'CL'):
    #         ind = atom.index
    #         x0, y0, z0 = pos[ind]
    #         force.addParticle(ind, (x0, y0, z0))
    #
    # system.addForce(force)
    #
    # integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
    # integrator.setConstraintTolerance(0.00001)

    # platform = mm.Platform.findPlatform([])
    # if platform.getName() == 'CUDA':
    #     print('Run simulation with %s' % platform.getName())
    #     properties = {'CudaPrecision': 'mixed'}
    #     simulation = app.Simulation(topo, system, integrator, platform, properties)
    # else:
    #     print('Run simulation with %s' % platform.getName())
    #     simulation = app.Simulation(topo, system, integrator, platform)
    #
    # simulation.context.setPositions(pos)
    # simulation.minimizeEnergy(maxIterations=1000)

    # st = simulation.context.getState(getPositions=True, getEnergy=True)
    # with open('min1.out', 'w') as f:
    #     print('# Epot', file=f)
    #     print('{:.4f}'.format(st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)), file=f)

    # with open(os.path.join(work_dir, 'prep/model_initial.pdb'), 'w') as f:
    #     app.PDBFile.writeFile(topo, st.getPositions(), f)

    t = mdt.load(os.path.join(work_dir, 'prep/model_initial.pdb'))
    return system, t


def generate_input(work_dir, mode, box_size, grid_size, system, t, i=0):
    """
    :param work_dir: work directory
    :param system: openmm system object
    :param t: mdtraj trajectory object
    :param i: frame index number (Optional)
    :return: rism input file stream
    """

    # input header
    header = """ 20130208C
 {0}
 KH
 {1}
  1.0e-6  10000
  {2:5.1f}   {2:5.1f}   {2:5.1f}
  {3}   {3}   {3}
""".format(mode, '/opt/nbcc/common/3D-RISM/tip3p_combined_300K.xsv', box_size, grid_size)

    for idx in range(system.getNumForces()):
        nbforce = system.getForce(idx)
        if isinstance(nbforce, mm.openmm.NonbondedForce):
            break

    coord = t.xyz[0] * 10.0
    avg_coord = np.average(coord.T, axis=1)

    dist = math.sqrt(avg_coord[0]**2 + avg_coord[1]**2 + avg_coord[2]**2)
    if dist > 10.0:
        coord -= avg_coord

    # for c in avg_coord:
    #     if c > 10.0:
    #         coord -= avg_coord
    #         break

    sel = t.topology.select('not (water or name Cl or name Na)')

    # image if traj has multiple frames
    try:
        t.image_molecules(inplace=True)
    except ValueError:
        pass

    # charge / sigma / epsilon
    natom = len(sel)
    joule_per_mole = unit.joule / unit.mole
    charge = np.empty(natom, dtype=np.double)
    sigma = np.empty(natom, dtype=np.double)
    epsilon = np.empty(natom, dtype=np.double)
    for idx in range(natom):
        atmind = int(sel[idx])
        p = nbforce.getParticleParameters(atmind)
        charge[idx] = p[0].value_in_unit(unit.elementary_charge)
        sigma[idx] = p[1].value_in_unit(unit.angstrom)
        epsilon[idx] = p[2].value_in_unit(joule_per_mole)

    rism_input = 'frame_%d.rsm' % i

    with open(os.path.join(work_dir, rism_input), 'w') as f:
        f.write(header)
        f.write('{:5d}\n'.format(natom))
        for i in range(natom):
            atmind = sel[i]
            x, y, z = coord[atmind]
            f.write(' {:16.5f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}{:16.7f}\n'.format(charge[i], sigma[i], epsilon[i], x, y, z))

    return rism_input


def process_xmu(tempdir, result, i=0):
    """

    :param tempdir:
    :param result: result.out file stream
    :param i: frame index number (Optional)
    :return:
    """
    with open(os.path.join(tempdir, 'frame_%d.xmu' % i), 'r') as f:
        for line in f:
            key, value = line.split()
            if key == 'solvation_free_energy':
                print('%s\t%s' % (key, value), file=result)
                break


def process_thm(tempdir, result, i=0):
    """

    :param tempdir:
    :param result: result.out file stream
    :param i: frame index number (Optional)
    :return:
    """
    with open(os.path.join(tempdir, 'frame_%d.thm' % i), 'r') as f:
        for line in f:
            key, value = line.split()
            if key in ('solvation_energy', 'solvation_entropy'):
                print('%s\t%s' % (key, value), file=result)
                continue


def process_xmu_res(tempdir, result, resmap):
    """

    :param tempdir:
    :param result: result.out file stream
    :param resmap: (atom_index, residue name) dictionary
    :return:
    """
    xmua_atm_file = os.path.join(tempdir, 'xmua_atm.dat')
    with open(xmua_atm_file, 'r') as f:
        data = f.readlines()[2:]

    resultmap = dict()
    for d in data:
        atom_ind, sfe, _, _ = d.split()

        try:
            residue = resmap[atom_ind]
        except KeyError:
            continue

        if residue not in resultmap:
            resultmap[residue] = 0.0
        resultmap[residue] += float(sfe)

    for key, value in resultmap.items():
        result.write('%s\t%f\n' % (key, value))


def process_thm_res(tempdir, result, resmap):
    """

    :param tempdir:
    :param result: result.out file stream
    :param resmap: (atom_index, residue name) dictionary
    :return:
    """
    xmua_atm = os.path.join(tempdir, 'xmua_atm.dat')
    enea_atm = os.path.join(tempdir, 'enea_atm.dat')
    enta_atm = os.path.join(tempdir, 'enta_atm.dat')

    xmua_atm_f = open(xmua_atm, 'r')
    enea_atm_f = open(enea_atm, 'r')
    enta_atm_f = open(enta_atm, 'r')

    xmu_data = xmua_atm_f.readlines()[2:]
    ene_data = enea_atm_f.readlines()[2:]
    ent_data = enta_atm_f.readlines()[2:]

    resultmap = dict()
    for xmu_d, ene_d, ent_d in zip(xmu_data, ene_data, ent_data):
        xmu = xmu_d.split()
        ene = ene_d.split()
        ent = ent_d.split()

        try:
            residue = resmap[xmu[0]]
        except KeyError:
            continue

        if residue not in resultmap:
            resultmap[residue] = [0.0, 0.0, 0.0]
        resultmap[residue][0] += float(xmu[1])
        resultmap[residue][1] += float(ene[1])
        resultmap[residue][2] += float(ent[1])

    for key, values in resultmap.items():
        result.write('%s\t%f\t%f\t%f\n' % (key, values[0], values[1], values[2]))

    xmua_atm_f.close()
    enea_atm_f.close()
    enta_atm_f.close()


def post_process(tempdir, mode, i=0):
    res_map = dict()
    with open(os.path.join(tempdir, 'model.pdb'), 'r') as pdb:
        for line in pdb:
            if 'ATOM' in line[0:6]:
                atom_ind, chain, resnum = line[6:11].strip(), line[21:22].strip(), line[22:26].strip()
                res_map[atom_ind] = '%s %s' % (chain, resnum)

    with open(os.path.join(tempdir, 'result.out'), 'w') as result:
        if mode == 'm':
            process_xmu(tempdir, result, i)
        elif mode == 't':
            process_xmu(tempdir, result, i)
            process_thm(tempdir, result, i)
        elif mode == 'a':
            process_xmu(tempdir, result, i)
            process_xmu_res(tempdir, result, res_map)
        elif mode == 'x':
            process_xmu(tempdir, result, i)
            process_thm(tempdir, result, i)
            process_thm_res(tempdir, result, res_map)

    subprocess.check_call(['/bin/cat', os.path.join(tempdir, 'result.out')])
