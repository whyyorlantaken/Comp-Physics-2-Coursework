"""
╔═══════════════════════════════════════════════════╗
║                     ATRAMENTA                     ║
╚═══════════════════════════════════════════════════╝
------Simulate planet orbits around a black hole-----

Author: Males-Araujo Yorlan
Date: April 2025

Note: The name stands for "black ink" in Latin.
"""
# Libraries
import os
import h5py
import argparse

import numpy as np
from scipy.integrate import solve_ivp

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display

# ---------------- Constants ----------------

G = 4 * np.pi**2  # AU^3/M_sun * yr^2
c = 63241.54      # AU/yr

# ----------------- Classes -----------------

class OrbitBirther:
    """
    Class to set up the initial conditions for the orbit
    of a planet around a black hole.

    Parameters
    ----------
    M : float
        Mass of the black hole in solar masses.
    a : float
        Semi-major axis in AU.
    e : float
        Eccentricity of the orbit.
    save_start : bool
        If True, save the initial conditions plot.
    """
    def __init__(self, 
                 M: float = 5e6,
                 a: float = 1.0,
                 e: float = 0.0,
                 save_start: bool = False):
        """
        Initialize the class and orbit parameters.
        """
        # Parameters
        self.M = M
        self.a = a
        self.e = e

        # Schwarzschild radius
        self.s_radius = self.schwarzschild_radius()

        # ----------------------- Validations -----------------------

        # Mass check
        if not isinstance(self.M, (int, float)) or self.M <= 0:
            raise ValueError(f"Invalid mass: {self.M} M_sun." +
                             "It must be a positive number.")

        # Semi-major axis check
        if not isinstance(self.a, (int, float)) or self.a <= 0 or self.a <= self.s_radius:
            raise ValueError(f"Invalid semi-major axis: {self.a} au." +
                              "It must be a positive number greater" +
                              "than the Schwarzschild radius.")
        
        # Eccentricity check
        if not isinstance(self.e, (int, float)) or self.e < 0 or self.e >= 1:
            raise ValueError(f"Invalid eccentricity: {self.e}." + 
                             "It must be a number in the range [0, 1).")
        
        # Last check
        if not isinstance(save_start, bool):
            raise TypeError("save_start must be a boolean.")
        
        # ----------------------- Other stuff -----------------------

        # Get initial conditions
        self.s0 = self.initial_conditions()

        # Get Schwarzschild components
        angle = np.linspace(0, 2*np.pi, 100)
        self.s_radius_x = self.s_radius * np.cos(angle)
        self.s_radius_y = self.s_radius * np.sin(angle)

        # Save start if requested
        if save_start:
            self._save_start()

    def initial_conditions(self) -> np.ndarray:
        """
        Calculate the initial conditions for the orbit. 
        It places the planet at the periapsis.

        Returns
        -------
        np.ndarray
            Initial conditions for the orbit, [x, y, vx, vy].
        """
        # Position
        x0 = 0
        y0 = self.a * (1 - self.e)

        # Velocity
        vx0 = -np.sqrt((G * self.M / self.a) * (1 + self.e) / (1 - self.e))
        vy0 = 0

        return np.array([x0, y0, vx0, vy0])

    def schwarzschild_radius(self) -> float:
        """
        It obtains the magnitude of the Schwarzschild radius.

        Returns
        -------
        float
            Schwarzschild radius in AU.
        """
        return 2 * G * self.M / c**2
    
    def _save_start(self) -> None:
        """
        To save the initial conditions plot.
        """
        # Create directory if not present
        if not os.path.exists("outputfolder"):
            os.makedirs("outputfolder")

        # Save
        plot_orbit(
            self.s0, 
            self.s_radius_x, self.s_radius_y, 
            save = True, 
            show = False, 
            directory = "outputfolder",
            name = f"M{self.M:.1e}-a{self.a:.1f}-e{self.e:.3f}.png")
    
class CelestialSeer(OrbitBirther):
    """
    Class to integrate the equations of motion for a planet
    orbiting a black hole using different methods, slopes and
    other options. It inherits from the OrbitBirther class.

    Parameters
    ----------
    M : float
        Mass of the black hole in solar masses.
    a : float
        Semi-major axis in AU.
    e : float
        Eccentricity of the orbit.
    save_start : bool
        If True, save the initial conditions plot.
    """
    def __init__(self, 
                 M: float = 5e6,
                 a: float = 1.0,
                 e: float = 0.0,
                 save_start: bool = False):
        """
        Initialize the class and orbit parameters.
        """
        # Parent constructor
        super().__init__(M, a, e, save_start)

    #######################################################
    #                   Public methods                    #
    #######################################################

    def integrate(self, 
                  N: float = 2.0,
                  steps: int = 5000,
                  method: str = 'RK3',
                  relativistic: bool = None,
                  save_hdf5: bool = False,
                  gif_name: str = None,
                  frames: int = 50) -> tuple:
        """
        To integrate the equations of motion of a planet
        around a black hole.

        Parameters
        ----------
        N : float
            Number of periods to simulate.
        steps : int
            Number of steps for integration.
        method : str
            Integration method to use from 'RK3', 'TPZ', or 'SPY'.
        relativistic : bool
            If True, use relativistic equations of motion.
        save_hdf5 : bool
            If True, save the simulation results.
        gif_name : str
            Name of the GIF file to save.
        frames : int
            Number of frames for the GIF.

        Returns
        -------
        tuple
            (t_sol, S_sol, filename) - time, state vector, and filename.
        """
        # Check valid method
        if method not in ['RK3', 'TPZ', 'SPY']:
            raise ValueError(f"Invalid method: {method}. Choose from 'RK3', 'TPZ', or 'SPY'.")
        
        # State vector with initial conditions and time stuff
        self._integrator_initializations(N, steps)

        # Determine slope function
        slope_func = self._relativistic_slope if relativistic else self._classical_slope

        # For naming stuff later
        self.relativistic = relativistic
        self.method = method
        ode_type = "relat" if relativistic else "class"
        self.filename = f"M{self.M:.1e}-a{self.a:.1f}-e{self.e:.3f}-{ode_type}-{method}.h5"
        
        # Integration
        if method == 'TPZ':
            t_sol, S_sol = self._trapezoidal(slope_func)
            S_sol = S_sol.T

        elif method == 'RK3':
            t_sol, S_sol = self._rk3(slope_func)
            S_sol = S_sol.T
            
        elif method == 'SPY':
            sol = solve_ivp(
                slope_func, 
                self.t_span, 
                self.s0, 
                method = "RK45",
                t_eval = self.t_eval, 
                rtol = 1e-13, 
                atol = 1e-13
            )
            t_sol, S_sol = sol.t, sol.y

        # Exclude data inside the Schwarzschild radius
        t_sol, S_sol = self._schwarzschild_restriction(t_sol, S_sol)

        # Save results if requested
        if save_hdf5:
            self._save_solutions(t_sol, S_sol, method, N)

            # and show gif also if requested
            if gif_name is not None:
                self._gif(gif_name, frames)

        return t_sol, S_sol, "outputfolder/" + self.filename

    def measure_convergence(self,
                    N: float = 2.0,
                    method: str = 'RK3',
                    relativistic: bool = None,
                    num_steps: tuple = (1000, 5000, 10000)) -> None:
                    
        """
        It computes the mean Euclidean error of the given integration
        method with respect to a higher-order solution across different
        time steps. Multiple integrations are done.

        Parameters
        ----------
        N : float
            Number of periods to simulate.
        method : str
            Integration method to use from 'RK3', 'TPZ', or 'SPY'.
        relativistic : bool
            If True, we use the relativistic slope.
        num_steps : tuple
            Number of steps to use for the integration.
            This changes the step size.

        Returns
        -------
        tuple
            (step_size, errors, coeff) - step sizes, errors and power law coefficient.
        """
        # Empty lists for solutions, error and step sizes
        methd_sol = []
        scipy_sol = []
        errors    = []
        step_size = []

        # Loop over the number of steps
        for steps in num_steps:

            # Integrate with the specified method
            _, S_meth, _ = self.integrate(
                N = N,
                steps = steps,
                method = method,
                relativistic = relativistic
            )

            # and with scipy
            _, S_scpy, _ = self.integrate(
                N = N,
                steps = steps,
                method = 'SPY',
                relativistic = relativistic
            )

            # Append all (self.dt is updated each time `integrate()` is called)
            methd_sol.append(S_meth)
            scipy_sol.append(S_scpy)
            step_size.append(self.dt)

        # Compute the mean Euclidean error for each
        for i in range(len(num_steps)):

            # Get the solutions 
            S_methd = methd_sol[i]
            S_scipy = scipy_sol[i]

            # Compute the error
            norm = self._mean_euclidean_error(S_methd, S_scipy)

            # Append
            errors.append(norm)

        # Estimate the power law coefficient
        coeff = self._pl_coeff(step_size, errors)

        return step_size, errors, coeff

    #######################################################
    #                  Private methods                    #
    #######################################################

    def _integrator_initializations(self,
                                    N: float = 2.0,
                                    steps: int = 1000) -> None:
        """
        Initialize variables for the integrator.

        Parameters
        ----------
        N : float
            Number of periods to simulate.
        steps : int
            Number of steps for integration.
        """
        # Time stuff
        self.steps  = steps
        self.t_span = (0, self._max_time(N))
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], self.steps + 1)
        self.dt     = (self.t_span[1] - self.t_span[0]) / self.steps

        # Initial conditions
        self.S    = np.zeros((self.steps + 1, 4))
        self.S[0] = self.s0   

    def _classical_slope(self,
                         t: float, 
                         S: np.ndarray) -> list:
        """
        Classical slope function for
        orbital dynamics.

        Parameters
        ----------
        t : float
            Time.
        S : np.ndarray
            State vector [x, y, vx, vy].

        Returns
        -------
        list
            List of derivatives [dxdt, dydt, dvxdt, dvydt].
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
    
    def _relativistic_slope(self, 
                            t: float, 
                            S: np.ndarray) -> list:
        """
        Post-Newtonian approximation slope function for 
        orbital dynamics. It includes the correction term
        for relativistic precession of the orbit.

        Parameters
        ----------
        t : float
            Time.
        S : np.ndarray
            State vector [x, y, vx, vy].

        Returns
        -------
        list
            List of derivatives [dxdt, dydt, dvxdt, dvydt].
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

    def _trapezoidal(self, slope_func: callable) -> tuple:
        """
        Trapezoidal method for numerical integration.

        Parameters
        ----------
        slope_func : callable
            Slope function to be integrated.

        Returns
        -------
        tuple
            (time, state vector).
        """
        # Integrate
        for i in range(0, self.steps):
            
            # Compute the slopes
            k1 = np.array(slope_func(self.t_eval[i], self.S[i]))
            k2 = np.array(slope_func(self.t_eval[i] + self.dt, self.S[i] + self.dt*k1))

            # Update the state vector
            self.S[i + 1] = self.S[i] + self.dt*(k1 + k2)/2

        return self.t_eval, self.S
    
    def _rk3(self, slope_func: callable) -> tuple:
        """
        Third-order Runge-Kutta method for numerical integration.

        Parameters
        ----------
        slope_func : callable
            Slope function to be integrated.

        Returns
        -------
        tuple
            (time, state vector).
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

    def _save_solutions(self, 
                        t_sol: np.ndarray,
                        S_sol: np.ndarray, 
                        method: str,
                        N: float) -> None:
        """
        It saves the simulation results to an HDF5 file.

        Parameters
        ----------
        t_sol : np.ndarray
            Time array.
        S_sol : np.ndarray
            State vector array [x, y, vx, vy].
        method : str
            Integration method used.
        N : float
            Number of periods simulated.
        """
        # Create directory if it doesn't exist
        if not os.path.exists("outputfolder"):
            os.makedirs("outputfolder")
        
        # Assign filepath
        filepath = os.path.join("outputfolder", self.filename)
        
        # Save the data
        with h5py.File(filepath, 'w') as f:

            # Time and state vector
            f.create_dataset("time", data = t_sol)
            f.create_dataset("x", data = S_sol[0])
            f.create_dataset("y", data = S_sol[1])
            f.create_dataset("vx", data = S_sol[2])
            f.create_dataset("vy", data = S_sol[3])
            
            # Other data
            f.attrs['units'] = 'solar mass, AU, yr'

            f.attrs['M'] = self.M
            f.attrs['a'] = self.a
            f.attrs['e'] = self.e
            f.attrs['N'] = N
            f.attrs['method'] = method

            f.attrs['schwarzschild_radius'] = self.s_radius

        print(f"Saved to {filepath}.") 
    
    def _max_time(self, N: float = 2.0) -> float:
        """
        Calculate the maximum time for the simulation
        based on Kepler's third law.

        Parameters
        ----------
        N : float
            Number of periods to simulate.

        Returns
        -------
        float
            Maximum time in years.
        """
        # Kepler's third law
        period = 2 * np.pi * np.sqrt(self.a**3 / (G * self.M))

        return N * period
    
    def _schwarzschild_restriction(self, 
                                   t_sol: np.ndarray,
                                   S_sol: np.ndarray) -> tuple:
        """
        To remove the data inside the Schwarzschild radius.

        Parameters
        ----------
        t_sol : np.ndarray
            Time array.
        S_sol : np.ndarray
            State vector array [x, y, vx, vy].

        Returns
        -------
        tuple
            (t_sol, S_sol) - outside Schwarzschild only.
        """
        # Get all the distances
        r_sol = np.sqrt(S_sol[0]**2 + S_sol[1]**2)

        # Determine which are inside the radius
        inside = r_sol < self.s_radius

        # Get the indexes of such points
        coeff = np.where(inside)[0]

        # If there are points inside, remove them
        if coeff.size > 0:
            S_sol = S_sol[:, :coeff[0]]
            t_sol = t_sol[:coeff[0]]

        return t_sol, S_sol
    
    def _gif(self, 
             gif_name: str, 
             frames: int = 100) -> None:
        """
        Create a GIF from the results.

        Parameters
        ----------
        gif_name : str
            Name of the GIF file.
        frames : int
            Number of frames.
        """
        # Initialize the painter class
        painter = ChronoPainter(
                    orbits = ("outputfolder/" + self.filename,),
                    labels = (f"Rel: {self.relativistic}, method: {self.method}",), 
                    colors = ('deepskyblue',)
                )
        
        # Create the GIF
        painter.paint(
            gif_name = gif_name,
            frames   = frames,
            duration = 1.0,
            dpi      = 120
        )

    @staticmethod
    def _mean_euclidean_error(s1: np.ndarray,
                             s2: np.ndarray) -> float:
        """
        Compute the mean euclidean error between two solutios.

        Parameters
        ----------
        s1 : np.ndarray
            First solution.
        s2 : np.ndarray
            Second solution.

        Returns
        -------
        float
            Mean euclidean error.
        """
        return np.sqrt(np.mean((s1 - s2)**2))
    
    @staticmethod
    def _pl_coeff(step_size: list, 
                         errors: list) -> float:
        """
        Estimate the power law coefficient of the
        step size and the error.

        Parameters
        ----------
        step_size : list
            List of step sizes
        errors : list
            List of errors

        Returns
        -------
        float
            Power law coefficient.
        """
        return np.polyfit(np.log(step_size), np.log(errors), 1)[0]
    
class ChronoPainter:
    """
    Class to create a GIF of the orbits of planets
    around a black hole. It loads the data from HDF5
    files and generates images for each frame.
    It can accept multiple orbits.

    Parameters
    ----------
    orbits : tuple
        Tuple of strings with the HDF5 file paths.
    labels : tuple
        Labels for each orbit.
    colors : tuple
        Colors for each orbit.
    """
    def __init__(self, 
                 orbits: tuple = None, 
                 labels: tuple = None,
                 colors: tuple = None):
        """
        Initialize the class and load the orbits data.
        """
        # Tuple of orbits
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

        # Assign different names to all
        self._nominate()

    #######################################################
    #                   Public methods                    #
    #######################################################

    def paint(self, 
              gif_name: str = None,
              frames: int = 100,
              duration: float = 1.0, 
              dpi: int = 120,
              show: bool = False) -> None:
        """
        It creates a GIF from the orbits data.

        Parameters
        ----------
        gif_name : str
            Name of the GIF file.
        frames : int
            Number of frames.
        duration : float
            Duration of the GIF.
        dpi : int
            DPI for the frames.
        show : bool
            If True, show the GIF in a notebook.
        """
        # Create directory for the frames
        if not os.path.exists("outputfolder/sketches"):
            os.makedirs("outputfolder/sketches")

        # Print
        print("----------------------------------------------------------")
        print("                  STARTING GIF CREATION")

        # Save frames
        self._sketch(frames, dpi)

        # Print info
        print("----------------------------------------------------------")
        print(f"Assembling the GIF...")
        print("----------------------------------------------------------")

        # Empty list
        images = []

        # Loop through the images
        for i in range(frames):
            filename = os.path.join("outputfolder/sketches", f"orbit_{i:03d}.png")
            images.append(imageio.imread(filename))

        # Save the GIF with loop
        imageio.mimsave(os.path.join("outputfolder", gif_name),
                        images, 
                        duration = duration,
                        loop = 0)

        # Print
        print(f"GIF saved to outputfolder/{gif_name}.")
        print("----------------------------------------------------------")

        # Delete images
        self._burn_sketches()

        print("                   END OF GIF CREATION")
        print("----------------------------------------------------------")

        # Show the gif if desired
        if show:
            self.show_evolution(gif_name)

    @staticmethod
    def show_evolution(gif_name: str) -> None:
        """
        It's used to show the GIF in a notebook.

        Parameters
        ----------
        gif_name : str
            Name of the GIF file.
        """
        # Show it
        img = IPImage(filename = os.path.join("outputfolder", gif_name))
        display(img)

    #######################################################
    #                   Private methods                   #
    #######################################################

    def _nominate(self) -> None:
        """
        To give different names to the data loaded from
        the HDF5 files.
        """
        # Loop through the orbits
        for i, orbit in enumerate(self.orbits):

            # Read the data
            t, S = self._read(orbit)

            # Assign different names
            setattr(self, f't_{i:02d}', t)
            setattr(self, f'S_{i:02d}', S)

    def _read(self, orbital_history: str) -> tuple:
        """
        To read the HDF5 file and extract the data.

        Parameters
        ----------
        orbital_history : str
            Path to the HDF5 file.

        Returns
        -------
        tuple
            (time, S) - time array and state vector array.
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

            # Schwarzschild components (shouldn't change per object)
            angle = np.linspace(0, 2*np.pi, 100)
            self.s_radius_x = s_radius * np.cos(angle)
            self.s_radius_y = s_radius * np.sin(angle)

            # Same with solar mass
            self.M = f.attrs['M']

        # Reconstruct the solution
        S = np.zeros((4, len(time)))
        S[0], S[1], S[2], S[3] = x, y, vx, vy

        return time, S

    def _sketch(self, 
                frames: int = 100,
                dpi: int = 120) -> None:
        """
        It creates the frames for the GIF.

        Parameters
        ----------
        frames : int
            Number of equally spaced frames to create.
        dpi : int
            DPI for the images.
        """
        # Create directory if it does not exist
        if not os.path.exists("outputfolder/sketches"):
            os.makedirs("outputfolder/sketches")

        # Equally spaced frames
        frame_indices = np.linspace(0, len(self.t_00) - 1, frames, dtype = int)

        # Max limit for the plot
        lim = self._max_lim()

        # Print info
        print("----------------------------------------------------------")
        print(f"Saving {frames} frames...")
        print("----------------------------------------------------------")

        # Start the loop
        for f_idx, t_idx in enumerate(frame_indices):

            # Draw and save
            self._save_sketch(
                frame_idx = f_idx,
                time_idx  = t_idx, 
                lim = lim,
                dpi = dpi
                )

            # Print progress
            if f_idx % (frames // 10) == 0:
                print(f"Progress: {f_idx / frames:.0%} ({f_idx} images)")
            
        # End
        print("----------------------------------------------------------")
        print(f"All {frames} frames saved to outputfolder/sketches.")
        
    def _save_sketch(self, 
                     time_idx: int, 
                     frame_idx: int,
                     lim: float,
                     dpi: int = 120) -> None:
        """
        It plots and saves the orbit at a given time.

        Parameters
        ----------
        time_idx : int
            Index of the time step to plot.
        frame_idx : int
            Index of the frame to save.
        lim : float
            Limit for the plot.
        dpi : int
            DPI for the image.
        """
        # Dark background
        with plt.style.context('dark_background'):

            # Figure
            plt.figure(figsize=(7, 7))

            # Loop over the orbits
            for j in range(len(self.orbits)):

                # Get the orbit solution
                S = getattr(self, f'S_{j:02d}')[:, :time_idx + 1]

                # Plot the position
                plt.plot(S[0], S[1], lw = 0.4, 
                        label = self.labels[j] if self.labels is not None else None, 
                        color = self.colors[j] if self.colors is not None else None)
                plt.scatter(S[0][-1], S[1][-1], 
                            color = 'deepskyblue', marker ='o', 
                            edgecolors = 'w', s = 50, zorder = 10)

            # Black hole and Schwarzschild radius
            plt.scatter(0, 0, label='Black hole',
                        color = 'k', s = 150, 
                        edgecolor = 'crimson', lw = 1.5)
            plt.plot(self.s_radius_x, self.s_radius_y,
                     label= r'$r_s$', lw = 0.8,
                     color="crimson", alpha=0.4)
            
            # Title and labels
            if len(self.orbits) == 1:
                plt.title(f'Planet orbit around the black hole ({self.M:.1e}' + r"$\ M_\odot$)")
            else:
                plt.title(f'Planet orbits around the black hole ({self.M:.1e}' + r"$\ M_\odot$)")

            plt.xlabel('x [AU]')
            plt.ylabel('y [AU]')
            plt.axis('equal')
            plt.grid(alpha = 0.2, ls = '-.', lw = 0.5)
            plt.legend(loc = (1.05, 0.42))
            plt.xlim(-lim * 1.2, lim * 1.2)
            plt.ylim(-lim * 1.2, lim * 1.2)
        
            # Save it
            plt.savefig(os.path.join(
                "outputfolder/sketches",
                f"orbit_{frame_idx:03d}.png"),
                dpi = dpi, 
                bbox_inches = 'tight'
                )
            plt.close() 

    def _burn_sketches(self) -> None:
        """
        It deletes the images used to create the GIF
        since they're no longer needed.
        """
        # Remove all pngs in the folder
        for filename in os.listdir("outputfolder/sketches"):
            if filename.endswith(".png"):
                os.remove(os.path.join("outputfolder/sketches", filename))

        # Also the directory
        os.rmdir("outputfolder/sketches")

        # Print
        print("All frames have been deleted.")
        print("----------------------------------------------------------")

    def _max_lim(self) -> float:
        """
        Calculate the maximum limit for the plot based on
        the distance from the center of the black hole
        of all the orbits.
        We do this to make sure that 1) all orbit points
        are visible and 2) to fix the plot limits.

        Returns
        -------
        float
            Maximum distance of all orbits.
        """
        # Empty list
        list_max = []

        # Loop through the orbits
        for i in range(len(self.orbits)):

            # Get solution
            S = getattr(self, f'S_{i:02d}')

            # and its maximum value
            max_x = np.max(np.abs(S[0]))
            max_y = np.max(np.abs(S[1]))
            maximum = np.max([max_x, max_y])

            # Append to the list
            list_max.append(maximum)

        return np.max(np.array(list_max))

# ---------------- Functions ----------------

# Comment: I was initially planning to use this function
# to save both the start and the images for the gif, but
# things got complicated in the animation part, so I did
# it separately in the end. 

def plot_orbit(S_sol: np.ndarray,
               s_radius_x: np.ndarray, 
               s_radius_y: np.ndarray,
               save: bool = False,
               show: bool = False,
               directory: str = "outputfolder",
               name: str = None) -> None:
    """
    To plot the orbit of a planet around a black hole.

    Parameters
    ----------
    S_sol : np.ndarray
        State vector of the planet [x, y, vx, vy].
    s_radius_x, s_radius_y : np.ndarray
        Schwarzschild radius components.
    save : bool
        If True, save the plot.
    show : bool
        If True, show the plot.
    directory : str
        Directory to save the plot.
    name : str
        Name of the file to save.
    """
    # Dark background
    with plt.style.context('dark_background'):

        # Figure
        plt.figure(figsize=(7, 7))

        # Plots
        if S_sol[0].size == 1:
            plt.scatter(S_sol[0], S_sol[1],
                        label = 'Start', color = 'deepskyblue', 
                        marker = 'o', s = 20)
        else:
            plt.plot(S_sol[0], S_sol[1], 
                     label = 'Orbit', color = 'magenta', lw = 0.4)
            plt.scatter(S_sol[0][-1], S_sol[1][-1],
                        label = 'Planet', color = 'deepskyblue',
                        marker = 'o', edgecolors = 'white', 
                        s = 50, zorder = 10)
            
        # Black hole
        plt.scatter(0, 0, label = 'Black hole', 
                    marker = 'o', color = 'k', 
                    s = 150, edgecolor = 'crimson', lw = 1.5)
        plt.plot(s_radius_x, s_radius_y, 
                 label=r'$r_s$', lw = 0.8, 
                 color = "crimson", alpha = 0.4)
        
        # Labels and title
        plt.xlabel('x [AU]')
        plt.ylabel('y [AU]')
        plt.title('Planet orbit around the black hole')
        plt.legend(loc = (1.05, 0.42))
        plt.axis('equal')
        plt.grid(alpha = 0.2, ls ='-.', lw = 0.5)

        # Set limits
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

def parse_args():
    """
    Set up the command line arguments for both the
    integrator and the animation classes.

    Returns
    -------
    parser.parse_args()
        Parsed command line arguments.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description = 'Orbit around a black hole simulation.')

    # and subparsers
    subparsers = parser.add_subparsers(dest = 'command', help = 'Command to run')
    
    # =========================== CelestialSeer ===========================
    seer_parser = subparsers.add_parser(
        'CelestialSeer', help = 'Run orbit simulation')

    # Arguments
    seer_parser.add_argument(
        '--M', type = float, default = 5e6,
        help = 'Mass of the black hole in solar masses.'
        )
    seer_parser.add_argument(
        '--a', type = float, default = 1.0, 
        help = 'Semi-major axis in AU.'
        )
    seer_parser.add_argument(
        '--e', type = float, default = 0.0167,
        help = 'Eccentricity of the orbit.'
        )
    seer_parser.add_argument(
        '--N', type = float, default = 2.0,
        help = 'Number of periods to simulate.'
        )
    seer_parser.add_argument(
        '--steps', type = int, default = 2000,
        help = 'Number of steps for integration.'
        )
    seer_parser.add_argument(
        '--method', type = str, choices = ['RK3', 'TPZ', 'SPY'], default = 'RK3',
        help = 'Integration method.'
        )
    seer_parser.add_argument(
        '--relativistic', action = 'store_true',
        help = 'Use relativistic equations of motion.'
        )
    seer_parser.add_argument(
        '--save_start', action = 'store_true',
        help = 'Save plot of initial conditions.'
        )
    seer_parser.add_argument(
        '--save_hdf5', action = 'store_true',
        help = 'Save simulation results to HDF5 file.'
        )
    seer_parser.add_argument(
        '--gif_name', type = str, default = None,
        help = 'Name of the GIF file.'
        )
    seer_parser.add_argument(
        '--frames', type = int, default = 100,
        help = 'Number of frames for the GIF.'
        )
    seer_parser.add_argument(
        '--show_gif', action = 'store_true',
        help = 'Show the GIF in a notebook.'
        )

    # =========================== ChronoPainter ===========================
    painter_parser = subparsers.add_parser(
        'ChronoPainter', help = 'Create gif from orbit data')

    # Arguments
    painter_parser.add_argument(
        '--orbits', nargs = '+', required = True,
        help = 'HDF5 files with orbit data.'
        )
    painter_parser.add_argument(
        '--labels', nargs = '+',
        help = 'Labels for each orbit.'
        )
    painter_parser.add_argument(
        '--colors', nargs = '+',
        help = 'Colors for each orbit.'
        )
    painter_parser.add_argument(
        '--gif_name', type = str, default = 'orbit.gif',
        help = 'Name of the GIF file.'
        )
    painter_parser.add_argument(
        '--frames', type = int, default = 10,
        help = 'Number of frames for the GIF.'
        )
    painter_parser.add_argument(
        '--duration', type = float, default = 0.1,
        help = 'Duration between frames in GIF.'
        )
    painter_parser.add_argument(
        '--dpi', type = int, default = 100,
        help = 'DPI for images.'
        )
    
    return parser.parse_args()


# ----------------- Main ----------------

if __name__ == "__main__":

    # Initialize the parser
    args = parse_args()

    # Package name
    print("")
    print("=========================================================")
    print("    ▗▄▖    ■   ▄▄▄ ▗▞▀▜▌▄▄▄▄  ▗▞▀▚▖▄▄▄▄    ■   ▗▞▀▜▌")
    print("   ▐▌ ▐▌▗▄▟▙▄▖█    ▝▚▄▟▌█ █ █ ▐▛▀▀▘█   █ ▗▄▟▙▄▖▝▚▄▟▌")
    print("   ▐▛▀▜▌  ▐▌  █         █   █ ▝▚▄▄▖█   █   ▐▌       ")
    print("   ▐▌ ▐▌  ▐▌                               ▐▌       ")
    print("          ▐▌                               ▐▌       ")

    # Integrator class
    if args.command == 'CelestialSeer':

        # Info
        print("=========================================================")
        print("")
        print("Initializing CelestialSeer (integrator class):")
        print(f"    Black hole mass:        {args.M:.1e} [M_sun]")
        print(f"    Semi-major axis (a):    {args.a:.1f} [AU]")
        print(f"    Eccentricity (e):       {args.e:.3f}")
        print("")

        # Create the integrator
        integrator = CelestialSeer(
            M = args.M, 
            a = args.a, 
            e = args.e,
            save_start = args.save_start
        )

        # Integrate
        print(f"    Integrating equations of motion:")
        print(f"        Periods (N):        {args.N}")
        print(f"        Method:             {args.method}")
        print(f"        Steps:              {args.steps}")
        print(f"        Relativistic:       {args.relativistic}")
        print(f"        Save HDF5:          {args.save_hdf5}")
        print("")

        # Call the integrator
        t_sol, S_sol, _ = integrator.integrate(
            N = args.N,
            steps = args.steps,
            method = args.method,
            relativistic = args.relativistic,
            save_hdf5 = args.save_hdf5,
            gif_name = args.gif_name,
            frames = args.frames
        )
        print("")
        print(f"DONE. Final time on planet: {t_sol[-1]:.1e} Earth years.")
        print("")   
        print("==========================================================")

    # Animation class
    elif args.command == 'ChronoPainter':
        
        # Info
        print("=========================================================")
        print("")
        print("Initializing ChronoPainter (animation class):")
        print(f"    Orbits: ")
        for orbit in args.orbits:
            print(f"        {orbit}")
        print(f"    Labels: {', '.join(args.labels) if args.labels else 'None'}")
        print(f"    Colors: {', '.join(args.colors) if args.colors else 'None'}")
        print("")
        
        # Create the painter
        painter = ChronoPainter(
            orbits = tuple(args.orbits),
            labels = tuple(args.labels) if args.labels is not None else None,
            colors = tuple(args.colors) if args.colors is not None else None
        )

        # Animation
        print(f"    Creating animation:")
        print(f"        GIF name:   {args.gif_name}")
        print(f"        Duration:   {args.duration}")
        print(f"        Frames:     {args.frames}")
        print(f"        DPI:        {args.dpi}")
        print("")
        
        # Call the painter
        painter.paint(
            gif_name = args.gif_name,
            frames   = args.frames,
            duration = args.duration,
            dpi      = args.dpi
        )
        
        print("")
        print("DONE. Animation created successfully.")
        print("")
        print("==========================================================")
