"""
Used for writing CRD files.

"""
from __future__ import absolute_import

import array
import numpy as np
import math
from simtk import unit
from simtk.openmm import Vec3
from simtk.openmm.app.internal.unitcell import computeLengthsAndAngles
import sys

class CRDFile(object):
    """CRDFile provides methods for creating CRD files.

    To use this class, create a CRDFile object, then call writeModel() once for each model in the file."""

    def __init__(self, file, topology, firstStep=0, interval=1, append=False):
        """Create a CRD file and write out the header, or open an existing file to append.

        Parameters
        ----------
        file : file
            A file to write to
        topology : Topology
            The Topology defining the molecular system being written
        dt : time
            The time step used in the trajectory
        firstStep : int=0
            The index of the first step in the trajectory
        interval : int=1
            The frequency (measured in time steps) at which states are written
            to the trajectory
        append : bool=False
            If True, open an existing CRD file to append to.  If False, create a new file.
        """
        self._file = file
        self._topology = topology
        self._firstStep = firstStep
        self._interval = interval
        self._modelCount = 0

        print('topology :', topology)

        if append:
            pass
            # TODO
        else:
            atoms = len(list(self._topology.atoms()))
            header = 'ATOMS {:<14}'.format(atoms)
            # print(len(header), '[{}]'.format(header))
            file.write(header)
            file.write('\n')


    def writeModel(self, positions, periodicBoxVectors=None):
        """Write out a model to the CRD file.

        The periodic box can be specified either by the unit cell dimensions
        (for a rectangular box), or the full set of box vectors (for an
        arbitrary triclinic box).  If neither is specified, the box vectors
        specified in the Topology will be used. Regardless of the value
        specified, no dimensions will be written if the Topology does not
        represent a periodic system.

        Parameters
        ----------
        positions : list
            The list of atomic positions to write
        periodicBoxVectors : tuple of Vec3=None
            The vectors defining the periodic box.
        """

        if len(list(self._topology.atoms())) != len(positions):
            raise ValueError('The number of positions must match the number of atoms')
        if unit.is_quantity(positions):
            positions = positions.value_in_unit(unit.nanometers)
        if any(math.isnan(unit.norm(pos)) for pos in positions):
            raise ValueError('Particle position is NaN')
        if any(math.isinf(unit.norm(pos)) for pos in positions):
            raise ValueError('Particle position is infinite')

        file = self._file

        self._modelCount += 1

        if self._modelCount % 10 == 0:
            print('modelCount :', self._modelCount, '  ---  positions : ', len(positions))

        # ------------------------------------------------------------------
        # Write the data.

        # coordinates
        # 10F8.3 : 10 columns 8 width 3 decimals

        total_vectors = len(positions)
        list_position = []
        for i in range(0, total_vectors):
            x1, y1, z1 = positions[i]
            list_position.extend([x1, y1, z1])

        unit_angst = 10   # nanometers --> angstroms
        list_position = [e * unit_angst for e in list_position]

        col_num = 10
        total_position = len(list_position)
        if total_position % col_num == 0:
            total_line = int(total_position/col_num)
        else:
            total_line = int(total_position/col_num) + 1
        # print(total_vectors, ' atoms * 3 =', total_position, '---', total_line)

        for k in range(0, total_line):
            end_point = col_num + col_num * k
            start_point = end_point - col_num
            if end_point > total_position:
                end_point = total_position
            # print(start_point, end_point)

            data = list_position[start_point:end_point]
            str = ''.join(format(x, "8.3f") for x in data)
            file.write(str)
            file.write('\n')
            # print(str)


        # periodicBoxVectors
        boxVectors = self._topology.getPeriodicBoxVectors()

        if boxVectors is not None:
            if periodicBoxVectors is not None:
                boxVectors = periodicBoxVectors
            elif unitCellDimensions is not None:
                if unit.is_quantity(unitCellDimensions):
                    unitCellDimensions = unitCellDimensions.value_in_unit(unit.nanometers)
                boxVectors = (Vec3(unitCellDimensions[0], 0, 0), Vec3(0, unitCellDimensions[1], 0), Vec3(0, 0, unitCellDimensions[2]))*unit.nanometers

            (a_length, b_length, c_length, alpha, beta, gamma) = computeLengthsAndAngles(boxVectors)
            a_length = a_length * unit_angst
            b_length = b_length * unit_angst
            c_length = c_length * unit_angst

            str = ''.join(format(x, "8.3f") for x in [a_length, b_length, c_length])
            file.write(str)
            file.write('\n')

        # ------------------------------------------------------------------
