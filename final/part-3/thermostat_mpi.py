# !/usr/bin/python
"""
A molecular dynamics solver that simulates the motion of non-interacting particles
in the canonical ensemble using a Langevin thermostat.
Reference: https://github.com/Comp-science-engineering/Tutorials/tree/master/MolecularDynamics
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# Define global physical constants
from scipy.constants import Avogadro, Boltzmann

def wallHitCheck(pos, vels, box):
    """ This function enforces reflective boundary conditions.
    All particles that hit a wall  have their velocity updated
    in the opposite direction.
    @pos: atomic positions (ndarray)
    @vels: atomic velocity (ndarray, updated if collisions detected)
    @box: simulation box size (tuple)
    """
    ndims = len(box)

    for i in range(ndims):
        vels[((pos[:,i] <= box[i][0]) | (pos[:,i] >= box[i][1])),i] *= -1

def integrate(pos, vels, forces, mass,  dt):
    """ A simple forward Euler integrator that moves the system in time 
    @pos: atomic positions (ndarray, updated)
    @vels: atomic velocity (ndarray, updated)
    """
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T

def computeForce(mass, vels, temp, relax, dt):
    """ Computes the Langevin force for all particles
    @mass: particle mass (ndarray)
    @vels: particle velocities (ndarray)
    @temp: temperature (float)
    @relax: thermostat constant (float)
    @dt: simulation timestep (float)
    returns forces (ndarray)
    """
    natoms, ndims = vels.shape

    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T

    force = - (vels * mass[np.newaxis].T) / relax + noise

    return force

def run(**args):
    """ This is the main function that solves Langevin's equations for
    a system of natoms usinga forward Euler scheme, and returns an output
    list that stores the time and the temperture.
    @natoms (int): number of particles
    @temp (float): temperature (in Kelvin)
    @mass (float): particle mass (in Kg)
    @relax (float): relaxation constant (in seconds)
    @dt (float): simulation timestep (s)
    @nsteps (int): total number of steps the solver performs
    @box (tuple): simulation box size (in meters) of size dimensions x 2
    e.g. box = ((-1e-9, 1e-9), (-1e-9, 1e-9)) defines a 2D square
    @ofname (string): filename to write output to
    @freq (int): write output every 'freq' steps
    @[radius]: particle radius (for visualization)
    Returns a list (of size nsteps x 2) containing the time and temperature.
    
    """

    natoms, box, dt, temp = args['natoms'], args['box'], args['dt'], args['temp']
    mass, relax, nsteps   = args['mass'], args['relax'], args['steps']
    ofname, freq, radius = args['ofname'], args['freq'], args['radius']

    dim = len(box)
    pos = np.random.rand(natoms,dim)

    for i in range(dim):
        pos[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * pos[:,i]

    vels = np.random.rand(natoms,dim)
    mass = np.ones(natoms) * mass / Avogadro
    radius = np.ones(natoms) * radius
    step = 0

    output = []

    while step <= nsteps:

        step += 1

        # Compute all forces
        forces = computeForce(mass, vels, temp, relax, dt)

        # Move the system in time
        integrate(pos, vels, forces, mass, dt)

        # Check if any particle has collided with the wall
        wallHitCheck(pos,vels,box)

        # Compute output (temperature)
        ins_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2)) / (Boltzmann * dim * natoms)
        output.append([step * dt, ins_temp])

        if not step%freq:
            #dump.writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)
            writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)
    return np.array(output)


def writeOutput(filename, natoms, timestep, box, **data):
    """ Writes the output (in dump format) """

    axis = ('x', 'y', 'z')

    with open(filename, 'a') as fp:

        fp.write('ITEM: TIMESTEP\n')
        fp.write('{}\n'.format(timestep))

        fp.write('ITEM: NUMBER OF ATOMS\n')
        fp.write('{}\n'.format(natoms))

        fp.write('ITEM: BOX BOUNDS' + ' f' * len(box) + '\n')
        for box_bounds in box:
            fp.write('{} {}\n'.format(*box_bounds))

        for i in range(len(axis) - len(box)):
            fp.write('0 0\n')

        keys = list(data.keys())

        for key in keys:
            isMatrix = len(data[key].shape) > 1

            if isMatrix:
                _, nCols = data[key].shape

                for i in range(nCols):
                    if key == 'pos':
                        data['{}'.format(axis[i])] = data[key][:,i]
                    else:
                        data['{}_{}'.format(key,axis[i])] = data[key][:,i]

                del data[key]

        keys = data.keys()

        fp.write('ITEM: ATOMS' + (' {}' * len(data)).format(*data) + '\n')

        output = []
        for key in keys:
            output = np.hstack((output, data[key]))

        if len(output):
            np.savetxt(fp, output.reshape((natoms, len(data)), order='F'))

# -----------------------------------------------------------------------------
# Your MPI parallelization code should start here. Do not modify the code above.
# -----------------------------------------------------------------------------

# Author: Males-Araujo Yorlan
# Date: June 2025

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                      Libraries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import argparse
from mpi4py import MPI

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                     Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def job_distro(jobs: int = 4,
               n_sims: int = 1000,
               temp_max: int = 500,
               temp_min: int = 200):
    """
    Our function to distribute the temperature jobs
    across multiple MPI ranks.

    Parameters
    ----------
    jobs : int
        Number of jobs.
    n_sims : int
        Total number of simulations.
    temp_max : int
        Maximum temperature
    temp_min : int
        Minimum temperature.

    Returns
    -------
    distributed_temps : ndarray
        Array of temperatures assigned to rank.
    comm : MPI communicator
        MPI communicator object.
    size : int
        Total number of MPI ranks.
    rank : int
        Rank of the current process.
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Range
    temps = np.linspace(temp_min, temp_max, n_sims)

    # To split the temps
    temps_per_rank = len(temps) // jobs
    remainder = len(temps) % jobs

    # Indexes for each rank
    start_index = rank * temps_per_rank + min(rank, remainder)
    end_index = start_index + temps_per_rank + (1 if rank < remainder else 0)

    # Distribute them
    distributed_temps = temps[start_index:end_index]

    # Sync all ranks
    comm.Barrier()

    # Some info
    if rank == 0:
        print("━━" * 30)
        print("  " * 10 + "Distribution info" + "  " * 10)
        print("━━" * 30)
        print(f"> Simulations:        {len(temps)}")
        print(f"> Number of jobs:     {jobs}")
        print(f"> Distro per rank:    {temps_per_rank}")
        if remainder > 0:
            print(f"> Remainder:          {remainder}")
        
    # Sync again
    comm.Barrier()

    # Only if work is assigned
    if len(distributed_temps) > 0:
        print(f"  - Rank {rank} will handle: {distributed_temps.astype(int)}")

    return distributed_temps, comm, size, rank

def parse_args():
    """
    Simpler parser.
    """
    # Initialize
    parser = argparse.ArgumentParser(
        description = "MPI job distribution for temperature simulation."
        )

    # Arguments
    parser.add_argument(
        '-j', '--jobs', 
        type = int, default = 1, 
        help = 'Number of jobs to distribute across MPI ranks.'
    )
    parser.add_argument(
        '-n', '--n_sims',
        type = int, default = 4,
        help = 'Total number of simulations to run.'
    )
    parser.add_argument(
        '--temp_max', 
        type = int, default = 600, 
        help = 'Maximum temperature.'
    )
    parser.add_argument(
        '--temp_min', 
        type = int, default = 100, 
        help = 'Minimum temperature.'
    )
    
    return parser.parse_args()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                    Main execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
        
    # Parse arguments
    args = parse_args()
    n_jobs = args.jobs
    n_sims = args.n_sims
    temp_max = args.temp_max
    temp_min = args.temp_min

    # Get distro
    distributed_temps, comm, size, rank = job_distro(
        jobs     = n_jobs,
        n_sims   = n_sims,
        temp_max = temp_max,
        temp_min = temp_min
    )
    
    # Info
    comm.Barrier()
    if rank == 0:
        print("━━" * 30)
        print("  " * 10 + "Completion times" + "  " * 10)
        print("━━" * 30)

    # Empty list
    all_results = []

    # Atoms
    N_atoms = 20000

    # Start time
    start = time.time()

    # Loop
    for temp in distributed_temps:

        # Set temp
        params = {
            'natoms': N_atoms,
            'temp': temp,
            'mass': 0.001,
            'radius': 120e-12,
            'relax': 1e-13,
            'dt': 1e-15,
            'steps': 10000,
            'freq': 100,
            'box': ((0, 1e-8), (0, 1e-8), (0, 1e-8)),
            'ofname':f'traj-hydrogen-3D-{N_atoms}-{temp:.2f}K.dump'
            }
        
        # Run
        output = run(**params)

        # Save results
        all_results.append((temp, output))

    # End time
    end = time.time()
    elapsed_time = end - start
    
    # Only if work was done
    if elapsed_time > 0.01:
        print(f"> Rank {rank}: {end - start:.2f} s")

    comm.Barrier()
    if rank == 0:
        print("━━" * 30)

     # Gather
    all_gathered = comm.gather(all_results, root = 0)
    execution_times = comm.gather(elapsed_time, root = 0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #                       Save
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if rank == 0:

        # 1. CSV
        if not os.path.exists('mpi_scaling.csv'):
            with open('mpi_scaling.csv', 'w') as f:
                f.write("n_jobs,execution_time\n")
        
        # We use the max time
        total_time = max(execution_times)
        with open('mpi_scaling.csv', 'a') as f:
            f.write(f"{n_jobs},{total_time}\n")

        # 2. Individual plots
        for temp, output in all_results:
            plt.figure(figsize = (8, 5))
            plt.plot(output[:, 0] * 1e12, output[:, 1], lw = 0.5, label = f"{temp:.2f} K")
            plt.xlabel('Time [ps]')
            plt.ylabel('Temperature [K]')
            plt.title(f'Temperature evolution - {temp:.2f} K', fontsize = 14, fontweight = 'bold')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./temperature-{N_atoms}-{temp:.2f}.png")
            plt.close()

        # 3. All together
        plt.figure(figsize = (8, 5))
        for rank_results in all_gathered:
            for temp, output in rank_results:
                plt.plot(output[:, 0] * 1e12, output[:, 1], lw = 0.3)
        
        plt.xlabel('Time [ps]')
        plt.ylabel('Temperature [K]')
        plt.title('Temperature evolution (all)', fontsize = 14, fontweight = 'bold')
        plt.tight_layout()
        plt.savefig(f"./temperature-a{N_atoms}-all.png")
        plt.close()

