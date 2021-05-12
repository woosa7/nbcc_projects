import os
import subprocess
from simtk.openmm import app
from simtk import unit, openmm

# ---------------------------------------------------------------------------
# run MD simulation
# ---------------------------------------------------------------------------
class MD:
    def __init__(self, pdb_code, res_size, config, work_dir):
        self.pdb_code = pdb_code
        self.res_size = res_size
        self.config = config

        self.work_dir = work_dir  # folded_dir or unfolded_dir


    def run_simulation(self, traj_dir):
        md_dir = os.path.join(traj_dir, '3_simulation') # traj_01/3_simulation/

        traj_file = '{}/protein.dcd'.format(md_dir)
        if os.path.exists(traj_file):
            return traj_file

        try:
            os.makedirs(md_dir)
        except OSError:
            pass

        cwd = os.getcwd()
        os.chdir(md_dir)

        subprocess.call('cp {}/1_tleap/{}.top {}/protein.top'.format(self.work_dir, self.pdb_code, md_dir), shell=True)
        subprocess.call('cp {}/1_tleap/{}.inpcrd {}/protein.inpcrd'.format(self.work_dir, self.pdb_code, md_dir), shell=True)

        self.openmm_simulation()

        os.chdir(cwd)

        return traj_file


    def openmm_simulation(self):
        # ---------------------------------------------------------------------------
        # Create system
        # GBSA model using OBC2 parameters
        prmtop = app.AmberPrmtopFile('protein.top')
        inpcrd = app.AmberInpcrdFile('protein.inpcrd')

        if self.res_size <= 100:
            system = prmtop.createSystem(implicitSolvent=app.OBC2,
                                         implicitSolventKappa=1.0/unit.nanometer,
                                         constraints=app.HBonds)
        else:
            system = prmtop.createSystem(implicitSolvent=app.OBC2,
                                         implicitSolventKappa=1.0/unit.nanometer,
                                         constraints=app.HBonds,
                                         nonbondedMethod=app.CutoffNonPeriodic,
                                         nonbondedCutoff=1.4*unit.nanometer)

        # Create simulation context
        temp0    = 300.0 # in unit.kelvin
        gamma_ln = 2.0   # in 1/unit.picosecond
        dt       = 0.002 # in unit.picosecond

        integrator = openmm.LangevinIntegrator(temp0*unit.kelvin,
                                               gamma_ln/unit.picoseconds,
                                               dt*unit.picosecond)

        platform = openmm.Platform.getPlatformByName('CUDA')

        GPU_NUM = self.config['gpu_no']
        properties = {'CudaPrecision': 'mixed',
                      'DeviceIndex': '{}'.format(GPU_NUM)}

        simulation = app.Simulation(prmtop.topology, system, integrator, platform, properties)
        simulation.context.setPositions(inpcrd.positions)

        # Minimization
        simulation.minimizeEnergy(maxIterations=1000)

        # Run simulation
        # 1,000 ns = 1 microsecond. (10,000 structures) : md_steps = 500,000,000
        md_time_ns = self.config['md_time_ns']
        md_steps = 500000 * md_time_ns   # 1 ns (10 structures) * time length

        dcd_file = 'protein.dcd'
        dcd_period = 50000     # 100 ps interval (50000 * dt)

        log_file = 'protein.log'
        log_period = 50000     # 100 ps interval (50000 * dt)

        simulation.reporters.append(app.DCDReporter(dcd_file, dcd_period))
        simulation.reporters.append(app.StateDataReporter(log_file, log_period,
                            step=True, potentialEnergy=True, kineticEnergy=True,
                            totalEnergy=True, temperature=True,
                            elapsedTime=True, remainingTime=True, totalSteps=md_steps,
                            separator=' '))

        simulation.step(md_steps)


# ---------------------------------------------------------------------------
