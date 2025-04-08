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
    
class CelestialSeer(OrbitBirther):
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
    def __init__(self, 
                 orbits: tuple = None, 
                 labels: tuple = None,
                 colors: tuple = None):
        """
        Constructor.
        """
        # It has to be a tuple
        if not isinstance(orbits, tuple):
            raise TypeError("The orbits parameter must be a tuple.")
        
        # with strings pointing to the files
        for orbit in orbits:
            if not isinstance(orbit, str):
                raise TypeError("Each element in the orbits tuple must be a string.")
            if not os.path.exists(orbit):
                raise FileNotFoundError(f"The file {orbit} does not exist.")
            
        # Make them attributes
        self.orbits = orbits
        self.labels = labels
        self.colors = colors

        # Load all
        self.designate()

    def designate(self):
        """
        Load all the orbits data.
        """
        # Load all the orbits data
        for i, orbit in enumerate(self.orbits):

            # Read the data
            t, S = self.look_back(orbit)

            # Assign different names
            setattr(self, f't_{i:02d}', t)
            setattr(self, f'S_{i:02d}', S)

    def look_back(self, orbital_history: str):
        """
        Read just the necessary data from the HDF5 file.
        """
        # Load and extract
        with h5py.File(orbital_history, 'r') as f:

            # Data
            time = f['time'][:]
            x    = f['x'][:]
            y    = f['y'][:]
            vx   = f['vx'][:]
            vy   = f['vy'][:]

            # Metadata
            s_radius = f.attrs['schwarzschild_radius']

            # Schwarzschild radius components (shouldn't change per object)
            angle = np.linspace(0, 2*np.pi, 100)
            self.s_radius_x = s_radius * np.cos(angle)
            self.s_radius_y = s_radius * np.sin(angle)

        # Reconstruct the solution
        S = np.zeros((4, len(time)))
        S[0], S[1], S[2], S[3] = x, y, vx, vy

        return time, S

    def sketch(self, frames):
        """
        Generate images.
        """
        # Create directory
        if not os.path.exists("outputfolder/images"):
            os.makedirs("outputfolder/images")

        # Determine the frame indices
        frame_indices = np.linspace(0, len(self.t_00) - 1, frames, dtype=int)

        # Get the max limit
        xlim = self.max_lim()

        # Print
        print("----------------------------------------------")
        print(f"Saving {frames} frames...")
        print("----------------------------------------------")

        # Save images
        for i, index in enumerate(frame_indices):

            # Frame
            plt.figure(figsize=(7, 7))

            # Get the partial solutions
            for j in range(len(self.orbits)):
                S = getattr(self, f'S_{j:02d}')[:, :index+1]
                plt.plot(S[0], S[1], lw = 0.4, label = self.labels[j], color = self.colors[j])
                plt.scatter(S[0][-1], S[1][-1], color='deepskyblue', marker='o',
                            edgecolors='w', s=50, zorder = 10)

            plt.scatter(0, 0, label='Black hole', color='k', s=150, edgecolor='crimson', lw=1.5)
            plt.plot(self.s_radius_x, self.s_radius_y, label=r'$r_s$', lw=0.8, color="crimson", alpha=0.4)
            
            # Set labels and save
            if len(self.orbits) == 1:
                plt.title('Planet orbit around the black hole')
            else:
                plt.title('Planet orbits around the black hole')

            plt.xlabel('x [AU]')
            plt.ylabel('y [AU]')
            plt.axis('equal')
            plt.grid(alpha=0.2, ls='-.', lw=0.5)
            plt.legend(loc=(1.05, 0.42))
            plt.xlim(-xlim*1.2, xlim*1.2)
            plt.ylim(-xlim*1.2, xlim*1.2)
        
            # Save the frame
            plt.savefig(os.path.join("outputfolder/images", f"orbit_{i:03d}.png"), dpi=150, bbox_inches='tight')
            plt.close()

            # Print progress each 10%
            if i % (frames // 10) == 0:
                print(f"Progress: {i / frames:.0%} ({i} images)")
            
        # Print
        print("----------------------------------------------")
        print(f"All {frames} images saved to outputfolder/images.")

    def max_lim(self):
        """
        Determine the max limit.
        """
        # Empty list
        list_max = []

        # Loop through the orbits
        for i in range(len(self.orbits)):

            # Get the solution
            S = getattr(self, f'S_{i:02d}')

            # and its maximum value
            max_x = np.max(np.abs(S[0]))
            max_y = np.max(np.abs(S[1]))
            maximum = np.max([max_x, max_y])

            # Append to the list
            list_max.append(maximum)

        return np.max(np.array(list_max))
            
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
        imageio.mimsave(os.path.join("outputfolder", "orbit-test-4.gif"), images, duration=1.0)

        # Print
        print(f"GIF saved to outputfolder/orbit-test-4.gif.")
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

    integrator = CelestialSeer(M=7.83e6, a=1.0, e=0.0167)
    t_sol, S_sol = integrator.integrate(N=4, steps=2000, method='SPY', relativistic=True, save=True)
    t_sol2, S_sol2 = integrator.integrate(N=4, steps=2000, method='SPY', relativistic=False, save=True)

    # Painter instance
    file1 = "outputfolder/M7.8e+06-a1.0-e0.017-relat-SPY.h5"
    file2 = "outputfolder/M7.8e+06-a1.0-e0.017-class-SPY.h5"

    tupla = (file1, file2)
    colors = ("khaki", "magenta")
    painter = ChronoPainter(orbits=tupla, labels=("Relativistic", "Classical"),
                            colors=colors)
    painter.paint(frames=50)

    # painter = ChronoPainter("outputfolder/M7.8e+06-a1.0-e0.017-relat-SPY.h5")
    # painter.paint(frames=10)
