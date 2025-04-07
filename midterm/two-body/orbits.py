"""
Celestia package.
"""
import os
import time
import h5py

import numpy as np

import imageio.v2 as imageio
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from scipy.integrate import solve_ivp

# ---------------- Constants ----------------

G = 4 * np.pi**2  # AU^3/M_sun * yr^2
c = 63241.54      # AU/yr

# ----------------- Functions ----------------

def plot_orbit( S_sol, s_radius_x, s_radius_y, save, show, directory, name):
    """
    Plot the orbit.
    """
    # Figure
    plt.figure(figsize=(7, 7))

    # Plots
    if S_sol[0].size == 1:
        plt.scatter(S_sol[0], S_sol[1], label='Start', color='deepskyblue', marker='o', s=20)
    else:
        plt.plot(S_sol[0], S_sol[1], label='Orbit', color='magenta', lw=0.4)
        plt.scatter(S_sol[0][-1], S_sol[1][-1], label='Planet', color='deepskyblue',
                    marker='o', edgecolors='white', s=50, zorder = 10)
        
    
    plt.scatter(0, 0, label='Black hole', marker='o', color='k', s=150, edgecolor='crimson',
                lw = 1.5)
    plt.plot(s_radius_x, s_radius_y, label=r'$r_s$', lw = 0.8, color = "crimson", alpha=0.4)
    
    # Labels and title
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.title('Planet orbit around the black hole')
    plt.legend(loc=(1.05, 0.42))
    
    plt.axis('equal')
    plt.grid(alpha=0.2, ls ='-.', lw = 0.5)

    max_x = np.max(np.abs(S_sol[0]))
    max_y = np.max(np.abs(S_sol[1]))
    maximum = np.max([max_x, max_y])

    plt.xlim(-maximum*1.2, maximum*1.2)
    plt.ylim(-maximum*1.2, maximum*1.2)

    if save:
        plt.savefig(os.path.join(directory, name), dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()

    plt.close()

# ----------------- Classes ----------------

class OrbitBirther:
    """
    Physical constants and unit conversions.
    """
    def __init__(self, 
                 M: float = 5e6,
                 a: float = 1.0,
                 e: float = 0.0,
                 save_start: bool = False):
        """
        Constructor.
        """
        # Parameters
        self.M = M
        self.a = a
        self.e = e

        # Schwarzschild radius
        self.s_radius = self.schwarzschild_radius()

        # Restrictions
        if self.a <= 0 or self.a <= self.s_radius:
            raise ValueError(f"Invalid semi-major axis: {self.a} au. It must be positive and " +
                             f"greater than the Schwarzschild radius ({self.s_radius:.2f} au).")
        
        # Get initial conditions
        self.s0 = self.initial_conditions()

        # Get Schwarzschild components
        angle = np.linspace(0, 2*np.pi, 100)
        self.s_radius_x = self.s_radius * np.cos(angle)
        self.s_radius_y = self.s_radius * np.sin(angle)

        # Save them if needed
        if save_start:
            self.save_start()

    def save_start(self):
        """
        """
        # Create directory if it doesn't exist
        if not os.path.exists("outputfolder"):
            os.makedirs("outputfolder")

        # Save
        plot_orbit(self.s0, self.s_radius_x, self.s_radius_y,
                    True, False, "outputfolder", f"M{self.M:.1e}-a{self.a:.1f}-e{self.e:.3f}.png")

    def initial_conditions(self):
        """
        Calculate initial conditions.
        """
        # Position
        x0 = 0
        y0 = self.a * (1 - self.e)

        # Velocity
        vx0 = -np.sqrt((G * self.M / self.a) * (1 + self.e) / (1 - self.e))
        vy0 = 0

        return np.array([x0, y0, vx0, vy0])

    def schwarzschild_radius(self):
        """
        Calculate the Schwarzschild radius.
        """
        return 2 * G * self.M / c**2
    
class CelestialIntegrator(OrbitBirther):
    """
    Integrator class.
    """
    def __init__(self, 
                 M: float = 5e6,
                 a: float = 1.0,
                 e: float = 0.0,
                 save_start: bool = False):
        """
        Constructor.
        """
        # Parent constructor
        super().__init__(M, a, e, save_start)

    def trapezoidal(self, slope_func):
        """
        Trapezoidal method.
        """
        # Integrate
        for i in range(0, self.steps):
            
            # Compute the slopes
            k1 = np.array(slope_func(self.t_eval[i], self.S[i]))
            k2 = np.array(slope_func(self.t_eval[i] + self.dt, self.S[i] + self.dt*k1))

            # Update the state vector
            self.S[i + 1] = self.S[i] + self.dt*(k1 + k2)/2

        return self.t_eval, self.S
    
    def RK3(self, slope_func):
        """
        Runge-Kutta 3rd order method.
        """
        # Integrate
        for i in range(0, self.steps):

            # Compute the slopes
            k1 = np.array(slope_func(self.t_eval[i], self.S[i]))
            k2 = np.array(slope_func(self.t_eval[i] + self.dt/2, self.S[i] + self.dt*k1/2))
            k3 = np.array(slope_func(self.t_eval[i] + self.dt, self.S[i] - self.dt*k1 + 2*self.dt*k2))

            # Update the state vector
            self.S[i + 1] = self.S[i] + self.dt*(k1 + 4*k2 + k3)/6

        return self.t_eval, self.S

    def relativistic_slope(self, t, S):
        """
        Relativistic ODE system.
        """
        # Unpack
        x, y, vx, vy = S

        # Compute quantities
        r  = np.sqrt(x**2 + y**2)
        L  = np.abs(x*vy - y*vx)
        c1 = -G * self.M/r**3
        c2 = 1 + 3 * L**2 / (c * r)**2

        # Define equations
        dxdt  = vx
        dydt  = vy
        dvxdt = c1 * c2 * x
        dvydt = c1 * c2 * y

        return [dxdt, dydt, dvxdt, dvydt]

    def classical_slope(self, t, S):
        """
        Classical ODE system.
        """
        # Unpack
        x, y, vx, vy = S

        # Compute quantities
        r   = np.sqrt(x**2 + y**2)
        cte = -G * self.M/r**3

        # Define equations
        dxdt  = vx
        dydt  = vy
        dvxdt = cte * x
        dvydt = cte * y

        return [dxdt, dydt, dvxdt, dvydt]

    def integrate(self, 
                  N: float = 2.0,
                  steps: int = 10000,
                  method: str = 'RK3',
                  relativistic: bool = False,
                  save: bool = False):
        """
        Integrate the system.
        """
        # Initialize the integrator
        self.integrator_initializations(N, steps)

        # Slope function
        slope_func = self.relativistic_slope if relativistic else self.classical_slope
        
        # Integrate
        if method == 'RK3':
            t_sol, S_sol = self.RK3(slope_func)
            S_sol = S_sol.T
            
        elif method == 'TPZ':
            t_sol, S_sol = self.trapezoidal(slope_func)
            S_sol = S_sol.T
            
        elif method == 'SPY':
            t_span = (self.t_eval[0], self.t_eval[-1])
            sol = solve_ivp(
                slope_func, 
                self.t_span, 
                self.s0, 
                method="RK45",
                t_eval=self.t_eval, 
                rtol=1e-6, 
                atol=1e-6
            )
            t_sol, S_sol = sol.t, sol.y

        # Exclude data inside the Schwarzschild radius
        t_sol, S_sol = self.schwarzschild_restriction(t_sol, S_sol)

        # Save results
        if save: 
            self.save_solutions(t_sol, S_sol, method, relativistic, N)

        return t_sol, S_sol

    def save_solutions(self, 
                       t_sol: np.ndarray,
                       S_sol: np.ndarray, 
                       method: str, 
                       relativistic: bool,
                       N: float):
        """
        Save simulation results to HDF5 format with core metadata.
        """
        # Create directory if it doesn't exist
        if not os.path.exists("outputfolder"):
            os.makedirs("outputfolder")
        
        # Descriptive filename
        ode_type = "relat" if relativistic else "class"
        filename = f"M{self.M:.1e}-a{self.a:.1f}-e{self.e:.3f}-{ode_type}-{method}.h5"
        filepath = os.path.join("outputfolder", filename)
        
        # Save
        with h5py.File(filepath, 'w') as f:

            # Trajectory data
            f.create_dataset("time", data = t_sol)
            f.create_dataset("x", data = S_sol[0])
            f.create_dataset("y", data = S_sol[1])
            f.create_dataset("vx", data = S_sol[2])
            f.create_dataset("vy", data = S_sol[3])
            
            # Other data as attributes
            f.attrs['units'] = 'solar mass, AU, yr'

            f.attrs['M'] = self.M
            f.attrs['a'] = self.a
            f.attrs['e'] = self.e
            f.attrs['N'] = N
            f.attrs['method'] = method

            f.attrs['schwarzschild_radius'] = self.s_radius

        print(f"Results saved to {filepath}.")

    def schwarzschild_restriction(self, 
                                  t_sol: np.ndarray,
                                  S_sol: np.ndarray):

        # Get the distances
        r_sol = np.sqrt(S_sol[0]**2 + S_sol[1]**2)

        # Inside the Schwarzschild radius
        inside = r_sol < self.s_radius

        # Get the indexes
        coeff = np.where(inside)[0]

        # If the orbit is inside, remove the data
        if coeff.size > 0:
            S_sol = S_sol[:, :coeff[0]]
            t_sol = t_sol[:coeff[0]]

        return t_sol, S_sol

    def integrator_initializations(self,
                                   N: float = 2.0,
                                   steps: int = 1000):
        """
        Initialize the integrator.
        """
        # Time stuff
        self.steps  = steps
        self.t_span = (0, self.max_time(N))
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], self.steps + 1)
        self.dt     = (self.t_span[1] - self.t_span[0]) / self.steps

        # Initial conditions
        self.S    = np.zeros((self.steps + 1, 4))
        self.S[0] = self.s0
    
    def max_time(self, N: float = 2.0):
        """
        Time for integration.
        """
        # Kepler's third law
        period = 2 * np.pi * np.sqrt(self.a**3 / (G * self.M))

        return N * period
    
class ChronoPainter:
    """
    Plots the evolution over time.
    """
    def __init__(self, orbital_history: str):
        """
        Constructor.
        """
        # Load the data
        self.orbital_history = orbital_history

        # Check if the file exists
        if not os.path.exists(self.orbital_history):
            raise FileNotFoundError(f"The file {self.orbital_history} does not exist.")
        
        # Read the data
        self.summon_history()

        print(f"Data loaded from {self.orbital_history}.")

    def summon_history(self):
        """
        Read the data.
        """
        # Load and extract
        with h5py.File(self.orbital_history, 'r') as f:

            # Data
            self.time = f['time'][:]
            self.x    = f['x'][:]
            self.y    = f['y'][:]
            self.vx   = f['vx'][:]
            self.vy   = f['vy'][:]

            # Metadata
            self.M = f.attrs['M']
            self.a = f.attrs['a']
            self.e = f.attrs['e']
            self.N = f.attrs['N']
            self.method = f.attrs['method']
            self.s_radius = f.attrs['schwarzschild_radius']
            self.units = f.attrs['units']

            # Schwarzschild radius components
            angle = np.linspace(0, 2*np.pi, 100)
            self.s_radius_x = self.s_radius * np.cos(angle)
            self.s_radius_y = self.s_radius * np.sin(angle)

        # Reconstruct the solution
        self.S = np.zeros((4, len(self.time)))
        self.S[0], self.S[1], self.S[2], self.S[3] = self.x, self.y, self.vx, self.vy

    def sketch(self, frames):
        """
        Generate images.
        """
        # Create directory
        if not os.path.exists("outputfolder/images"):
            os.makedirs("outputfolder/images")

        # Determine the frame indices
        frame_indices = np.linspace(0, len(self.time)-1, frames, dtype=int)

        # Print
        print("----------------------------------------------")
        print(f"Saving {frames} frames...")
        print("----------------------------------------------")

        # Save images
        for i, index in enumerate(frame_indices):

            # Get the partial solution
            S_partial = self.S[:, :index + 1]

            # Plot
            plot_orbit(
                S_partial,
                self.s_radius_x,
                self.s_radius_y,
                True, False,
                "outputfolder/images",
                f"orbit_{i:03d}.png"
            )

            # Print progress each 10%
            if i % (frames // 10) == 0:
                print(f"Progress: {i / frames:.0%} ({i} images)")
            
        # Print
        print("----------------------------------------------")
        print(f"All {frames} images saved to outputfolder/images.")
            
    def paint(self, frames):
        """
        Generate GIF.
        """
        # Create directory
        if not os.path.exists("outputfolder/images"):
            os.makedirs("outputfolder/images")

        # Print
        print("----------------------------------------------")
        print("            STARTING GIF CREATION")

        # Save images
        self.sketch(frames)

        # Print info
        print("----------------------------------------------")
        print(f"Saving GIF...")
        print("----------------------------------------------")

        # Save gif
        images = []
        for i in range(frames):
            filename = os.path.join("outputfolder/images", f"orbit_{i:03d}.png")
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join("outputfolder", "orbit-test-2.gif"), images, duration=1.0)

        # Print
        print(f"GIF saved to outputfolder/orbit-test-2.gif.")
        print("----------------------------------------------")

        print("            END OF GIF CREATION")
        print("----------------------------------------------")

        # Delete images
        self.burn_sketches()

    def burn_sketches(self):
        """
        Delete images.
        """
        # Remove the images
        for filename in os.listdir("outputfolder/images"):
            if filename.endswith(".png"):
                os.remove(os.path.join("outputfolder/images", filename))

        # Remove the directory
        os.rmdir("outputfolder/images")

        # Print
        print("All images have been deleted.")
        print("----------------------------------------------")


# ----------------- Main ----------------

if __name__ == "__main__":

    #integrator = CelestialIntegrator(M=7.83e6, a=1.0, e=0.0167)
    #t_sol, S_sol = integrator.integrate(N=2, steps=2500, method='SPY', relativistic=True, save=True)

    # Painter instance
    painter = ChronoPainter("outputfolder/M7.8e+06-a1.0-e0.017-relat-SPY.h5")
    painter.paint(frames=10)
