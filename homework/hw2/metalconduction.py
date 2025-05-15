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

import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                Diffusivities in cm²/s
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

diffusivities_dir = {
    "Copper":    1.11,
    "Iron":      0.23,
    "Aluminium": 0.97,
    "Brass":     0.34,
    "Steel":     0.18,
    "Zinc":      0.63,
    "Lead":      0.22,
    "Titanium":  0.098
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#               Crank-Nicholson method
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SingleSolver:
    """
    """
    def __init__(self, metal: str = "Copper"):
        """
        Initialize.
        """
        self.metal = metal
        self.diffussivity = diffusivities_dir[metal]

    def integrate(self,
                  dt: float = 0.5,
                  dx: float = 0.1,
                  x_min: float = -10,
                  x_max: float = 10,
                  t_max: float = 100,
                  ic_type: str = "Smooth",
                  bc_type: str = "Fixed") -> None:
        """
        Integrate the temperature distribution.
        """
        # Setup the integrator
        self._integrator_setup(
            dt, 
            dx, 
            x_min, 
            x_max,
            t_max,
            ic_type, 
            bc_type
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

    def _integrator_setup(self,
                          dt: float = 0.5,
                          dx: float = 0.1,
                          x_min: float = -10,
                          x_max: float = 10,
                          t_max: float = 100,
                          ic_type: str = "Smooth", 
                          bc_type: str = "Fixed") -> None:
        """
        Initialize the integrator.
        """
        # Time and space vectors
        self.dt = dt
        self.dx = dx
        self.x = np.arange(x_min, x_max + dx, dx)
        self.t = np.arange(0, t_max + dt, dt)

        # R-factor
        self.r_factor = self.diffussivity * dt / dx**2
        
        # Determine conditions based on types
        self._determine_conditions(ic_type, bc_type)

        # Matrix of temperature
        self.T = np.zeros((len(self.x), len(self.t)))
        self.T[0, :] = self.bcs[0]  
        self.T[-1, :] = self.bcs[1]
        self.T[:, 0] = self.ics

        # Matrices for the method
        n = len(self.x)
        self.D1 = self._create_matrix(n, -1)
        self.D2 = self._create_matrix(n, +1)

    def _detect_thermal_eq(self,
                           threshold: float = 0.05,
                           consecutive_steps: int = 2) -> None:
        """
        Detect the thermal equation.
        """
        # Average all profiles
        avg_temps = np.mean(self.T, axis = 0)

        # Loop over solutions
        count = 0
        for i in range(1, len(self.t) - 1):

            # Difference between consecutive profiles
            diff = np.abs(avg_temps[i] - avg_temps[i-1])

            # Small changes have to be persistent
            if diff < threshold:
                count += 1
                if count >= consecutive_steps:
                    # return print(f"  - {self.metal}: {True}, {self.t[i-(consecutive_steps-1)]:.2f} s")
                    return True, self.t[i-(consecutive_steps - 1)]
            else:
                count = 0
        
        # print(f"> {self.metal}: Thermal eq. not reached in {self.t[-1]:.2f} s :(")
        # return print(f"  - {self.metal}: {False}, {self.t[-1]:.2f} s")
        return False, self.t[-1]

    def _determine_conditions(self, 
                              ic_type: str, 
                              bc_type: str) -> None:
        """
        Determine initial and boundary conditions based on types.
        """
        # Initial
        if ic_type == "Smooth":
            self.ics = 175 - 50 * np.cos(np.pi * self.x / 5) - self.x**2
        elif ic_type == "Noisy":
            # TODO: Implement this part
            pass
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
        Create a tridiagonal matrix for the Crank-Nicholson method.
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
#                  Parallelization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultipleSolver:
    """
    Parallelize the integration.
    """
    def __init__(self, 
                 metals: list = ["Iron", "Lead"], 
                 parallel: bool = False):
        """
        Initialize.
        """
        self.metals = metals
        self.n_jobs = len(metals)
        self.parallel = parallel
        self.integrators = [SingleSolver(metal) for metal in metals]

    def integrate(self,
                  dt: float = 0.5,
                  dx: float = 0.1,
                  x_min: float = -10,
                  x_max: float = 10,
                  t_max: float = 100,
                  ic_type: str = "Smooth",
                  bc_type: str = "Fixed") -> None   :
        """
        Integrate the temperature distribution.
        """
        # Info
        print("> Metals to be integrated:")
        for metal in self.metals:
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

            results = Parallel(n_jobs = self.n_jobs)(
                delayed(integrator.integrate)(
                    dt, 
                    dx, 
                    x_min, 
                    x_max, 
                    t_max, 
                    ic_type, 
                    bc_type
                ) for integrator in self.integrators
            )

            # Thermal equilibrium
            for i, result in enumerate(results):

                # Extract and print
                _, _, _, eq_reached, eq_time = result
                print(f"  - {self.metals[i]}: {eq_reached}, {eq_time:.2f} s")

            # End time
            print()
            print(f"> Total integration time: {(time.time() - start):.2f} s.")
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

            results = [integrator.integrate(
                dt, 
                dx, 
                x_min, 
                x_max, 
                t_max, 
                ic_type, 
                bc_type
            ) for integrator in self.integrators]

            # Thermal equilibrium
            for i, result in enumerate(results):

                # Extract and print
                _, _, _, eq_reached, eq_time = result
                print(f"  - {self.metals[i]}: {eq_reached}, {eq_time:.2f} s")

            # End time
            print()
            print(f"> Total integration time: {(time.time() - start):.2f} s.")
            print("━━" * 30)
            print("  "*8 + "SEQUENTIAL INTEGRATION ENDED")
            print("━━" * 30)

        return results
    

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                      Argparse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    """
    Parse command line arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description = "Metal conduction simulation.")

    # Integration parameters
    parser.add_argument(
        "-dt",
        type = float,
        default = 0.1,
        help = "Time step for the integration."
    )
    parser.add_argument(
        "-dx",
        type = float,
        default = 0.1,
        help = "Space step for the integration."
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
        nargs = "+", 
        default = ["Iron", "Lead"],
        help = "List of metals to simulate."
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
#                      Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
if __name__ == "__main__":

    # Arguments
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

    # Log file
    if args.log:
        log_filename = f"azula.{len(metals)}.{'par' if parallel else 'seq'}.log"
        sys.stdout = open(log_filename, "w")

    # Header
    print("━━" * 30)
    print("             |                      '||          ")
    print("            |||    ......  ... ...   ||   ....   ")
    print("           |  ||   '  .|'   ||  ||   ||  '' .||  ")
    print("          .''''|.   .|'     ||  ||   ||  .|' ||  ")
    print("         .|.  .||. ||....|  '|..'|. .||. '|..'|'.v1.0")
    print("━━" * 30)
    print("  "*8 + "METAL CONDUCTION SIMULATION")
    print("━━" * 30)
    print("> Parameters:")
    print(f"  - dt          {dt:.2f} s")
    print(f"  - dx          {dx:.2f} cm")
    print(f"  - x_min      {x_min:.2f} cm")
    print(f"  - x_max       {x_max:.2f} cm")
    print(f"  - t_max       {t_max:.2f} s")
    print(f"  - ic_type     {ic_type}")
    print(f"  - bc_type     {bc_type}")
    print()

    # Create the solver
    solver = MultipleSolver(
        metals = metals,
        parallel = parallel
    )

    results = solver.integrate(
        dt = 0.1,
        dx = 0.1,
        x_min = -10,
        x_max = 10,
        t_max = 100,
        ic_type = "Smooth",
        bc_type = "Fixed"
    )










