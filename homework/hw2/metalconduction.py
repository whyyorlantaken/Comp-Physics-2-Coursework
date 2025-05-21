"""
╔═══════════════════════════════════════════════════╗
║                       AZULA                       ║
╚═══════════════════════════════════════════════════╝
------------- Simulate metal conduction ------------- 

Author: Males-Araujo Yorlan
Date: May 2025

Note: Azula is a character from 
      Avatar: The Last Airbender.
      She is a princess, fire master,
      and can control heat.
"""
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                      Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import sys

import time
import argparse
import numpy as np

from joblib import Parallel, delayed

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                Diffusivities in cm²/s
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

diffusivities_dir = {
    "Copper":    1.11,
    "Iron":      0.23,
    "Aluminum":  0.97,
    "Brass":     0.34,
    "Steel":     0.18,
    "Zinc":      0.63,
    "Lead":      0.22,
    "Titanium":  0.098
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#        Crank-Nicholson method for one metal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SingleSolver:
    """
    Class to solve the heat equation for a rod of some
    metal using the Crank-Nicholson method.
    It's one-dimensional.

    Parameters
    ----------
    metal : str
        The metal to be used. Default is "Copper".
    """
    def __init__(self, metal: str = "Copper"):
        """
        Initialize the solver.
        """
        # Check if the metal is valid
        if metal not in diffusivities_dir:
            raise ValueError(f"Invalid metal: {metal}. " +
                             f"Choose from {list(diffusivities_dir.keys())}.")
        
        # Attributes
        self.metal = metal
        self.diffussivity = diffusivities_dir[metal]

    ##################################################
    #                 Public methods                 #
    ##################################################

    def integrate(self,
                  dt: float = 0.5,
                  dx: float = 0.1,
                  x_min: float = -10,
                  x_max: float = 10,
                  t_max: float = 100,
                  ic_type: str = "Smooth",
                  bc_type: str = "Fixed",
                  noise_factor: float = 0.01) -> None:
        """
        Crank-Nicholson integration method.

        Parameters
        ----------
        dt : float
            Time step.
        dx : float
            Space step.
        x_min : float
            Minimum x value.
        x_max : float
            Maximum x value.
        t_max : float
            Maximum time in seconds.
        ic_type : str
            Type of initial condition. 
            "Smooth" or "Noisy".
        bc_type : str
            Type of boundary condition.
            "Fixed" or "Varying".
        noise_factor : float
            Noise factor for the initial condition.
            Default is 0.01.

        Returns
        -------
        x : np.ndarray
            Space vector.
        t : np.ndarray
            Time vector.
        T : np.ndarray
            Full temperature matrix.
        eq_reached : bool
            True if thermal equilibrium is reached.
        eq_time : float
            Time when thermal equilibrium is reached.
            If not reached, it returns the maximum time.
        """
        # Setup the integrator
        self._integrator_setup(
            dt, 
            dx, 
            x_min, 
            x_max,
            t_max,
            ic_type, 
            bc_type, 
            noise_factor
        )
        
        # Integrate
        for j in range(0, len(self.t) - 1):

            # Copy the initial conditions to the b vector
            b = self.T[1:-1, j].copy()
            
            # Evaluate the right-hand side
            b = np.dot(self.D2, b)
            
            # Append missing values
            b[0]  = b[0]  + self.r_factor * (self.T[0, j+1] + self.T[0, j])
            b[-1] = b[-1] + self.r_factor * (self.T[-1, j+1] + self.T[-1, j])
            
            # Compute solution vector
            sln_b = np.linalg.solve(self.D1, b)
            
            # And add it to the temperature matrix
            self.T[1:-1, j+1] = sln_b

        # Detect thermal equilibrium
        eq_reached, eq_time = self._detect_thermal_eq()

        return self.x, self.t, self.T, eq_reached, eq_time
    
    ##################################################
    #                 Private methods                #
    ##################################################

    def _integrator_setup(self,
                          dt: float = 0.5,
                          dx: float = 0.1,
                          x_min: float = -10,
                          x_max: float = 10,
                          t_max: float = 100,
                          ic_type: str = "Smooth", 
                          bc_type: str = "Fixed",
                          noise_factor: float = 0.01) -> None:
        """
        Setup the integrator-needed variables.

        Parameters
        ----------
        dt : float
            Time step.
        dx : float
            Space step.
        x_min : float
            Minimum x value.
        x_max : float
            Maximum x value.
        t_max : float
            Maximum time in seconds.
        ic_type : str
            Type of initial condition. 
            "Smooth" or "Noisy".
        bc_type : str
            Type of boundary condition.
            "Fixed" or "Varying".
        noise_factor : float
            Noise factor for the initial condition.
            Default is 0.01.
        """
        # Time and space vectors
        self.dt = dt
        self.dx = dx
        self.x  = np.arange(x_min, x_max + dx, dx)
        self.t  = np.arange(0, t_max + dt, dt)

        # R-factor
        self.r_factor = self.diffussivity * dt / dx**2
        
        # Determine conditions based on types
        self._determine_conditions(ic_type, bc_type, noise_factor)

        # Matrix of temperature
        self.T        = np.zeros((len(self.x), len(self.t)))
        self.T[0, :]  = self.bcs[0]  
        self.T[-1, :] = self.bcs[1]
        self.T[:, 0]  = self.ics

        # Matrices for the method
        n = len(self.x)
        self.D1 = self._create_matrix(n, -1)
        self.D2 = self._create_matrix(n, +1)

    def _detect_thermal_eq(self,
                           threshold: float = 0.01,
                           consecutive_steps: int = 2) -> tuple:
        """
        Detect thermal equilibrium in the system.

        Parameters
        ----------
        threshold : float
            Threshold for the difference between consecutive profiles.
            Default is 0.01.
        consecutive_steps : int
            Number of consecutive steps to consider thermal equilibrium.
            Default is 2.

        Returns
        -------
        eq_reached : bool
            True if thermal equilibrium is reached.
        eq_time : float
            Time when thermal equilibrium is reached.
            If not, it returns the maximum time.
        """
        # Average all profiles
        avg_temps = np.mean(self.T, axis = 0)

        # Loop over solutions
        count = 0
        for i in range(1, len(self.t) - 1):

            # Difference between consecutive profiles
            diff = np.abs(avg_temps[i] - avg_temps[i-1])

            # It has to be a persistent small difference
            if diff < threshold:
                count += 1
                if count >= consecutive_steps:
                    return True, self.t[i - (consecutive_steps - 1)]
            else:
                count = 0
        
        return False, self.t[-1]
    
    def _smooth_ic(self) -> np.ndarray:
        """
        Smooth initial condition.

        Returns
        -------
        np.ndarray
            The condition.
        """
        return 175 - 50 * np.cos(np.pi * self.x / 5) - self.x**2
    
    def _noisy_ic(self, noise_factor: float = 0.01) -> np.ndarray:
        """
        Noisy initial condition.

        Parameters
        ----------
        noise_factor : float
            Noise factor for the initial condition.
            Default is 0.01.

        Returns
        -------
        np.ndarray
            The condition.
        """
        # Maximum amplitude
        smooth_profile = self._smooth_ic()
        beta = noise_factor * np.max(smooth_profile)

        # Random noise and apodization function
        f_x = np.random.normal(-1.0, 1.0, len(self.x))
        g_x = np.ones(len(self.x))
        g_x[0] = g_x[-1] = 0.0

        return smooth_profile + beta * f_x * g_x

    def _determine_conditions(self, 
                              ic_type: str, 
                              bc_type: str,
                              noise_factor: float = None) -> None:
        """
        Determine initial and boundary conditions based on types.

        Parameters
        ----------
        ic_type : str
            Type of initial condition. "Smooth" or "Noisy".
        bc_type : str
            Type of boundary condition."Fixed" or "Varying".
        noise_factor : float
            Noise factor for the noisy initial condition.
        """
        # Initial
        if ic_type == "Smooth":
            self.ics = self._smooth_ic()
        elif ic_type == "Noisy":
            self.ics = self._noisy_ic(noise_factor)
        else:
            raise ValueError("Invalid initial condition type." +
                             "Choose 'Smooth' or 'Noisy'.")
        
        # Boundary
        if bc_type == "Fixed":
            self.bcs = [25.0, 25.0]
        elif bc_type == "Varying":
            self.bcs = [25.0 + 0.12 * self.t, 25.0 + 0.27 * self.t]
        else:
            raise ValueError("Invalid boundary condition type." +
                                "Choose 'Fixed' or 'Varying'.")

    def _create_matrix(self, 
                       n: int, 
                       sign: int) -> np.ndarray:
        """
        Create matrices for the Crank-Nicholson method.
        We noted the matrices could be constructed in 
        the same way but with a opposite sign in some places.

        Parameters
        ----------
        n : int
            Size of the matrix.
        sign : int
            Sign to construct the matrix.

        Returns
        -------
        np.ndarray
            The matrix.
        """
        # Main diagonal
        diag = np.diag([2 - sign * 2 * self.r_factor]*(n - 2), 0)
        
        # -1 diagonal
        diag_minus = np.diag([sign * self.r_factor]*(n - 3), -1)
        
        # +1 diagonal
        diag_plus =  np.diag([sign * self.r_factor]*(n - 3), +1)

        # Combine them all
        matrix = diag + diag_minus + diag_plus
        
        return matrix

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#    Multiple metals with optional parallelization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultipleSolver:
    """
    Class to solve the heat equation for multiple metals
    using the Crank-Nicholson method. It can be used to
    run serial or parallel simulations.

    We deliberately did not use inheritance.

    Parameters
    ----------
    metals : list
        List of metals to be used. 
        If None, all 8 metals are used.
    parallel : bool
        If True, the integration is done in parallel.
    n_jobs : int
        Number of jobs for parallel integration.
        Default is 1.
    """
    def __init__(self, 
                 metals: list = None,
                 parallel: bool = False,
                 n_jobs: int = 1):
        """
        Initialize the multiple solver.
        """
        # Metals
        if metals is None:
            self.all_metals = list(diffusivities_dir.keys())
        else:
            self.all_metals = metals

        # The rest
        self.n_jobs = n_jobs
        self.parallel = parallel
        self.integrators = [SingleSolver(metal) for metal in self.all_metals]

    ##################################################
    #                  Public method                 #
    ##################################################

    def integrate(self,
                  dt: float = 0.5,
                  dx: float = 0.1,
                  x_min: float = -10,
                  x_max: float = 10,
                  t_max: float = 100,
                  ic_type: str = "Smooth",
                  bc_type: str = "Fixed",
                  noise_factor: float = 0.01) -> tuple:
        """
        Multiple integration method.

        Parameters
        ----------
        dt : float
            Time step.
        dx : float
            Space step.
        x_min : float
            Minimum x value.
        x_max : float
            Maximum x value.
        t_max : float
            Maximum time in seconds.
        ic_type : str
            Type of initial condition. 
            "Smooth" or "Noisy".
        bc_type : str
            Type of boundary condition.
            "Fixed" or "Varying".
        noise_factor : float
            Noise factor for the initial condition.
            Default is 0.01.

        Returns
        -------
        results : list
            List of results for each metal.
        total_time : float
            Total time for the integration.
        """
        # Info
        print("> Metals to be integrated:")
        for metal in self.all_metals:
            print(f"  - {metal} [{diffusivities_dir[metal]} cm²/s]")

        # Parallel integration
        if self.parallel:

            # Info
            print("━━" * 30)
            print("  "*8 + "PARALLEL INTEGRATION STARTED")
            print("━━" * 30)
            print(f"> Number of jobs: {self.n_jobs}.")
            print()
            print("> Reached thermal equilibrium:")

            # Start time
            start = time.time()

            # Distribute the work
            results = Parallel(n_jobs = self.n_jobs)(
                delayed(integrator.integrate)(
                    dt, 
                    dx, 
                    x_min, 
                    x_max, 
                    t_max, 
                    ic_type, 
                    bc_type,
                    noise_factor
                ) for integrator in self.integrators
            )

            # Thermal equilibrium
            for i, result in enumerate(results):

                # Extract
                _, _, _, eq_reached, eq_time = result
                print(f"  - {self.all_metals[i]}: {eq_reached}, {eq_time:.2f} s")

            # End time
            print()
            total_time = time.time() - start
            print(f"> Total integration time: {total_time:.2f} s.")
            print("━━" * 30)
            print("  "*8 + "PARALLEL INTEGRATION ENDED")
            print("━━" * 30)

        # Sequential integration
        else:
            # Info
            print("━━" * 30)
            print("  "*8 + "SEQUENTIAL INTEGRATION STARTED")
            print("━━" * 30)
            print("> Reached thermal equilibrium:")
            
            # Start time
            start = time.time()

            # Each simulation one by one
            results = [integrator.integrate(
                dt, 
                dx, 
                x_min, 
                x_max, 
                t_max, 
                ic_type, 
                bc_type,
                noise_factor
            ) for integrator in self.integrators]

            # Thermal equilibrium
            for i, result in enumerate(results):

                # Extract and print
                _, _, _, eq_reached, eq_time = result
                print(f"  - {self.all_metals[i]}: {eq_reached}, {eq_time:.2f} s")

            # End time
            print()
            total_time = time.time() - start
            print(f"> Total integration time: {total_time:.2f} s.")
            print("━━" * 30)
            print("  "*8 + "SEQUENTIAL INTEGRATION ENDED")
            print("━━" * 30)

        return results, total_time
    
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#            Parse command line arguments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Done solely for the parallelization part.

def parse_args():
    """
    Parser.

    Returns
    -------
        Parsed arguments.
    """
    # Create it
    parser = argparse.ArgumentParser(description = "Metal conduction simulation.")

    # Integration parameters
    parser.add_argument(
        "-dt",
        type = float,
        default = 0.1,
        help = "Time step"
    )
    parser.add_argument(
        "-dx",
        type = float,
        default = 0.1,
        help = "Space step"
    )
    parser.add_argument(
        "-x_min",
        type = float,
        default = -10,
        help = "Minimum x value."
    )
    parser.add_argument(
        "-x_max",
        type = float,
        default = 10,
        help = "Maximum x value."
    )
    parser.add_argument(
        "-t_max",
        type = float,
        default = 100,
        help = "Maximum time in s for the integration."
    )
    parser.add_argument(
        "-ic",
        type = str,
        default = "Smooth",
        choices = ["Smooth", "Noisy"],
        help = "Type of initial condition."
    )
    parser.add_argument(
        "-bc",
        type = str,
        default = "Fixed",
        choices = ["Fixed", "Varying"],
        help = "Type of boundary condition."
    )

    # Others
    parser.add_argument(
        "-m", "--metals",
        type = str,
        nargs = "+",
        default = None,
        help = "List of metals to be used."
    )
    parser.add_argument(
        "-n", "--n_jobs",
        type = int,
        default = 1,
        help = "Number of jobs for parallel integration."
    )
    parser.add_argument(
        "-p", "--parallel", 
        action = "store_true",
        help = "Use parallel integration."
    )
    parser.add_argument(
        "-l", "--log",
        action = "store_true",
        help = "Save output to a log file."
    )
    return parser.parse_args()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                   Main execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
if __name__ == "__main__":

    # Collect arguments
    args = parse_args()
    dt = args.dt
    dx = args.dx
    x_min = args.x_min
    x_max = args.x_max
    t_max = args.t_max
    ic_type = args.ic
    bc_type = args.bc
    metals = args.metals
    parallel = args.parallel
    n_jobs = args.n_jobs

    # Log file
    if args.log:

        # Folder
        if not os.path.exists("outputfolder"):
            os.makedirs("outputfolder")

        # Save all output
        log_filename = f"outputfolder/azula.{n_jobs}.{'par' if parallel else 'seq'}.log"
        sys.stdout = open(log_filename, "w")

    # Header and info
    print("━━" * 30)
    print("             |                      '||          ")
    print("            |||    ......  ... ...   ||   ....   ")
    print("           |  ||   '  .|'   ||  ||   ||  '' .||  ")
    print("          .''''|.   .|'     ||  ||   ||  .|' ||  ")
    print("         .|.  .||. ||....|  '|..'|. .||. '|..'|'.v1.0")
    print("━━" * 30)
    print("  "*8 + "METAL CONDUCTION SIMULATION")
    print("━━" * 30)
    print(f"> Running on: {os.uname()[1]}")
    print()
    print("> Parameters:")
    print(f"  - dt          {dt:.3f} s")
    print(f"  - dx          {dx:.3f} cm")
    print(f"  - x_min      {x_min:.2f} cm")
    print(f"  - x_max       {x_max:.2f} cm")
    print(f"  - t_max       {t_max:.2f} s")
    print(f"  - ic_type     {ic_type}")
    print(f"  - bc_type     {bc_type}")
    print()

    # Create the solver
    solver = MultipleSolver(
        n_jobs = n_jobs,
        parallel = parallel
    )

    # Integrate
    results = solver.integrate(
        dt = dt,
        dx = dx,
        x_min = x_min,
        x_max = x_max,
        t_max = t_max,
        ic_type = ic_type,
        bc_type = bc_type
    )