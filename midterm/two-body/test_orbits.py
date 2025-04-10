import pytest
import numpy as np

from orbits import OrbitBirther, CelestialSeer

class TestModule(object):
    """
    Tests for the OrbitBirther class which handles initial conditions.
    """
    @classmethod
    def setup_class(cls):
        """
        Common test parameters.
        """
        cls.M = 5e6
        cls.a = 1.0
        cls.e = 0.0

    @classmethod
    def teardown_class(cls):
        """
        Clean up after all tests.
        """
        del cls.M
        del cls.a
        del cls.e

    def teardow_method(self):
        """
        Clean up after each test method.
        """
        del self.orbit
        del self.seer

    def setup_method(self):
        """
        Setup for each test method.
        """
        self.orbit = OrbitBirther(
            M = self.M, 
            a = self.a, 
            e = self.e
            )
        self.seer  = CelestialSeer(
            M = self.M, 
            a = self.a, 
            e = self.e
            )

    def test_valid_inputs(self):
        """
        Test that valid inputs are accepted correctly
        """
        # Attribute assignment
        assert self.orbit.M == self.M
        assert self.orbit.a == self.a
        assert self.orbit.e == self.e

        # Mass should be positive
        assert self.orbit.M > 0

        # Semi-major axis should not be greater than the Schwarzschild radius
        assert self.orbit.a > self.orbit.schwarzschild_radius()

        # Initial conditions
        s0 = self.orbit.initial_conditions()
        assert s0[0] == 0
        assert s0[1] == 1.0
        assert s0[2] <  0
        assert s0[3] == 0

    def test_invalid_inputs(self):
        """
        Test invalid inputs
        """
        # Negative mass
        with pytest.raises(ValueError):
            OrbitBirther(M = -1.0, a = self.a, e = self.e)

        # Negative semimajor axis
        with pytest.raises(ValueError):
            OrbitBirther(M = self.M, a = -1.0, e = self.e)

        # Negative eccentricity
        with pytest.raises(ValueError):
            OrbitBirther(M = self.M, a = self.a, e = -0.5)

        # Eccentricity greater than 1
        with pytest.raises(ValueError):
            OrbitBirther(M = self.M, a = self.a, e = 1.5)

    def test_different_outcomes(self):
        """
        """
        # ---------------- METHODS ----------------
        
        # Trapezoidal method
        _, S_tpz, _ = self.seer.integrate(
            N = 2.0, 
            steps = 100,
            method = "TPZ",
            relativistic = True
        )

        # Runge-Kutta 3 method
        _, S_rk3, _ = self.seer.integrate(
            N = 2.0, 
            steps = 100,
            method = "RK3",
            relativistic = True
        )

        # Scipy's RK45 method
        _, S_spy, _ = self.seer.integrate(
            N = 2.0,  
            steps = 100,
            method = "SPY",
            relativistic = True
        )

        # Final positions after integration should be different
        assert not np.allclose(S_tpz[0][-1], S_rk3[0][-1]) \
            or not np.allclose(S_tpz[1][-1], S_rk3[1][-1])
        
        assert not np.allclose(S_tpz[0][-1], S_spy[0][-1]) \
            or not np.allclose(S_tpz[1][-1], S_spy[1][-1])
        
        assert not np.allclose(S_rk3[0][-1], S_spy[0][-1]) \
            or not np.allclose(S_rk3[1][-1], S_spy[1][-1]) 
        
        # ---------------- SLOPES ----------------
        
        # Classical
        _, S_class, _ = self.seer.integrate(
            N = 2.0, 
            steps = 100,
            method = "SPY",
            relativistic = False
        )

        # Similarly
        assert not np.allclose(S_class[0][-1], S_spy[0][-1]) \
            or not np.allclose(S_class[1][-1], S_spy[1][-1])