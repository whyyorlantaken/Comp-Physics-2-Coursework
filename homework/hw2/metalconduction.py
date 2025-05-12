"""
╔═══════════════════════════════════════════════════╗
║                     Name                          ║
╚═══════════════════════════════════════════════════╝
------Description-----

Author: Males-Araujo Yorlan
Date: May 2025

[What the name stands for]
"""
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                      Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
# import h5py
# import argparse

import numpy as np
from scipy.integrate import solve_ivp

# import imageio.v2 as imageio
import matplotlib.pyplot as plt
# from IPython.display import Image as IPImage, display

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                     Dictionary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

diffusivities_dir = {
    "Copper":   111.0,
    "Iron":     23.0,
    "Aluminum": 97.0,
    "Brass":    34.0,
    "Steel":    18.0,
    "Zinc":     63.0,
    "Lead":     22.0,
    "Titanium": 9.8
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#               Crank-Nicholson method
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Integrator:
    """
    """
    def __init__(self, metal: str = "Copper"):
        """
        Initialize.
        """
        self.metal = metal
        self.diffussivity = diffusivities_dir[metal]

    def integrate(self,
                  dt: float = 0.01,
                  dx: float = 0.01,
                  x_min: float = -10,
                  x_max: float = 10,
                  t_min: float = 0,
                  t_max: float = 1,
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
            t_min, 
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

        return self.x, self.t, self.T

    def _integrator_setup(self,
                          dt: float = 0.01,
                          dx: float = 0.01,
                          x_min: float = -10,
                          x_max: float = 10,
                          t_min: float = 0,
                          t_max: float = 1,
                          ic_type: str = "Smooth", 
                          bc_type: str = "Fixed") -> None:
        """
        Initialize the integrator.
        """
        # Time and space vectors
        self.dt = dt
        self.dx = dx
        self.x = np.arange(x_min, x_max + dx, dx)
        self.t = np.arange(t_min, t_max + dt, dt)

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


# TEST with working method
if __name__ == "__main__":

    # Create the integrator
    integrator = Integrator(metal="Copper")

    # Integrate
    x, t, T = integrator.integrate(
        dt = 0.05,
        dx = 0.01,
        x_min = -10,
        x_max = 10,
        t_min = 0,
        t_max = 2.0,
        ic_type = "Smooth",
        bc_type = "Fixed"
    )
    
    # Plotting
    R = np.linspace(1, 0, len(t))
    G = 0
    B = np.linspace(0, 1, len(t))

    # FIgure environment
    plt.figure(figsize=(8,3))

    for j in range(len(t)):
        plt.plot(x, T[:, j] , color = [R[j], G, B[j]])

    plt.show()










