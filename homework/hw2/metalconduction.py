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

import numpy as np
import matplotlib.pyplot as plt

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
        self._detect_thermal_eq()

        return self.x, self.t, self.T

    def _integrator_setup(self,
                          dt: float = 0.01,
                          dx: float = 0.01,
                          x_min: float = -10,
                          x_max: float = 10,
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
                    print(f"Thermal equilibrium reached at t = {self.t[i-(consecutive_steps-1)]:.2f} s! :)")
                    return self.t[i-(consecutive_steps-1)]
            else:
                count = 0
        
        print(f"Thermal equilibrium not reached for t = {self.t[-1]:.2f} s :(")
        return self.t[-1]

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

if __name__ == "__main__":

    # Create the integrator
    integrator = Integrator(metal="Steel")

    # Integrate
    x, t, T = integrator.integrate(
        dt = 10,
        dx = 0.01,
        t_max = 2000.0,
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










