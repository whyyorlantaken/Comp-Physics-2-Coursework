"""
My module.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------- Constants ----------------

G = 4 * np.pi**2  # AU^3/M_sun * yr^2
c = 63241.54      # au/yr

# ----------------- Classes ----------------

class OrbitalSystem:
    """
    Physical constants and unit conversions.
    """
    def __init__(self, 
                 M: float = 5e6,
                 N: float = 2.0,
                 a: float = 1.0,
                 e: float = 0.0,
                 method: str = 'TRAPEZOIDAL',
                 relativistic: bool = False):
        """
        Constructor.
        """
        # Parameters
        self.M = M
        self.N = N
        self.a = a
        self.e = e
        self.method = method
        self.relativistic = relativistic

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
    
class Integrator(OrbitalSystem):
    """
    Integrator class.
    """
    def __init__(self, 
                 M: float = 5e6,
                 N: float = 2.0,
                 a: float = 1.0,
                 e: float = 0.0,
                 method: str = 'TRAPEZOIDAL',
                 relativistic: bool = False,
                 steps: int = 1000):
        """
        Constructor.
        """
        # Parent constructor
        super().__init__(M, N, a, e, method, relativistic)

        # Time stuff
        self.steps  = steps
        self.t_span = (0, self.max_time())
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], self.steps)
        self.dt     = (self.t_span[1] - self.t_span[0]) / self.steps

        # Initial conditions
        self.s0   = self.initial_conditions()
        self.S    = np.zeros((self.steps + 1, 4))
        self.S[0] = self.s0

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
            k3 = np.array(slope_func(self.t_eval[i] + self.dt, self.S[i] + self.dt*k2))

            # Update the state vector
            self.S[i + 1] = self.S[i] + (self.dt/6)*(k1 + 4*k2 + k3)

        return self.t_eval, self.S

    def relativistic_slope(self, t, S):
        """
        Relativistic ODE system.
        """
        # Unpack
        x, y, vx, vy = S

        # Compute quantities
        r   = np.sqrt(x**2 + y**2)
        L   = np.abs(x*vy - y*vx)
        cte = -G * self.M/r**3

        # Define equations
        dxdt  = vx
        dydt  = vy
        dvxdt = cte*(1+3*L**2/(c*r)**2)*x
        dvydt = cte*(1+3*L**2/(c*r)**2)*y

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

    def integrate(self):
        """
        Integrate the system.
        """
        # Slope function
        slope_func = (self.relativistic_slope if self.relativistic 
                      else self.classical_slope)
        
        # Integrate
        if self.method == 'RK3':
            t_sol, S_sol = self.RK3(slope_func)
            
        elif self.method == 'TRAPEZOIDAL':
            t_sol, S_sol = self.trapezoidal(slope_func)
            
        elif self.method == 'SCIPY':
            t_span = (self.t_eval[0], self.t_eval[-1])
            sol = solve_ivp(
                slope_func, 
                self.t_span, 
                self.s0, 
                method="RK45",
                t_eval=self.t_eval, 
                rtol=1e-8, 
                atol=1e-8
            )
            t_sol, S_sol = sol.t, sol.y

        return t_sol, S_sol
    
    def max_time(self):
        """
        Time for integration.
        """
        # Kepler's third law
        t = 2 * np.pi * np.sqrt(self.a**3 / (G * self.M))

        return self.N * t 


# ----------------- Main ----------------

if __name__ == "__main__":

    integrator = Integrator(M=6.5e6, N=10.0, a=1.0, e=0.0, method='SCIPY', relativistic=True,
                            steps=1000)
    t, y = integrator.integrate()

    s_radius = integrator.schwarzschild_radius()

    array = np.linspace(0, 2*np.pi, 100)
    s_radius_x = s_radius * np.cos(array)
    s_radius_y = s_radius * np.sin(array)

    plt.style.use('dark_background')

    plt.figure(figsize=(8, 8))
    plt.plot(y[0], y[1], label='Orbit', lw = 0.5, color='white')
    plt.plot(s_radius_x, s_radius_y, label='Schwarzschild radius', lw = 0.5, color = "yellow")
    plt.scatter(0, 0, label='Black hole', marker='o', color='white')
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.title('Orbits test')
    plt.axis('equal')
    #plt.savefig("orbit.png", dpi=150, bbox_inches='tight')
    #plt.legend()
    plt.grid(alpha=0.1)
    plt.show()
    plt.close()

    

