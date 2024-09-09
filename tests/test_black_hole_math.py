import unittest
import sympy as sy
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.math.black_hole_math import SymbolicExpressions, NumericalFunctions, GeometricFunctions, PhysicalFunctions, Utilities, Optimization, ImpactParameter, Simulation

class TestSymbolicExpressions(unittest.TestCase):

    def setUp(self):
        # Define symbols that might be used in the expressions
        self.M, self.P, self.Q, self.N, self.alpha, self.theta_0, self.r, self.b, self.mdot = sy.symbols('M P Q N alpha theta_0 r b mdot')

    def test_expr_q(self):
        expected = sy.sqrt((self.P - 2*self.M) * (self.P + 6*self.M))
        result = SymbolicExpressions.expr_q()
        self.assertEqual(result, expected)

    def test_expr_b(self):
        expected = sy.sqrt((self.P**3) / (self.P - 2*self.M))
        result = SymbolicExpressions.expr_b()
        self.assertEqual(result, expected)

    def test_expr_k(self):
        expected = sy.sqrt((self.Q - self.P + 6*self.M) / (2*self.Q))
        result = SymbolicExpressions.expr_k()
        self.assertEqual(result, expected)

    def test_expr_zeta_inf(self):
        expected = sy.asin(sy.sqrt((self.Q - self.P + 2*self.M) / (self.Q - self.P + 6*self.M)))
        result = SymbolicExpressions.expr_zeta_inf()
        self.assertEqual(result, expected)

    def test_expr_gamma(self):
        expected = sy.acos(sy.cos(self.alpha) / sy.sqrt(sy.cos(self.alpha)**2 + sy.tan(self.theta_0)**-2))
        result = SymbolicExpressions.expr_gamma()
        self.assertEqual(result, expected)

    @patch('src.math.black_hole_math.SymbolicExpressions.expr_zeta_inf')
    @patch('src.math.black_hole_math.SymbolicExpressions.expr_k')
    def test_expr_u(self, mock_k, mock_zeta_inf):
        # Mock the return values of expr_zeta_inf and expr_k
        mock_zeta_inf.return_value = sy.Symbol('zeta_inf')
        mock_k.return_value = sy.Symbol('k')

        result = SymbolicExpressions.expr_u()
        
        # Check if the result is a Piecewise function
        self.assertIsInstance(result, sy.Piecewise)
        
        # Check if it has two pieces
        self.assertEqual(len(result.args), 2)

    def test_expr_one_plus_z(self):
        expected = (1 + sy.sqrt(self.M / self.r**3) * self.b * sy.sin(self.theta_0) * sy.sin(self.alpha)) / sy.sqrt(1 - 3*self.M / self.r)
        result = SymbolicExpressions.expr_one_plus_z()
        self.assertEqual(result, expected)

    def test_expr_fs(self):
        rstar = self.r / self.M
        expected = (
            ((3 * self.M * self.mdot) / (8 * sy.pi))
            * (1 / ((rstar - 3) * rstar ** (5 / 2)))
            * (
                sy.sqrt(rstar)
                - sy.sqrt(6)
                + (sy.sqrt(3) / 3)
                * sy.ln(
                    ((sy.sqrt(rstar) + sy.sqrt(3)) * (sy.sqrt(6) - sy.sqrt(3)))
                    / ((sy.sqrt(rstar) - sy.sqrt(3)) * (sy.sqrt(6) + sy.sqrt(3)))
                )
            )
        )
        result = SymbolicExpressions.expr_fs()
        self.assertEqual(result, expected)

class TestNumericalFunctions(unittest.TestCase):
    def setUp(self):
        self.periastron = 10.0
        self.bh_mass = 1.0
        self.tol = 1e-6

    def test_calc_q(self):
        expected = np.sqrt((self.periastron - 2*self.bh_mass) * (self.periastron + 6*self.bh_mass))
        result = NumericalFunctions.calc_q(self.periastron, self.bh_mass)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_calc_b_from_periastron(self):
        expected = np.sqrt((self.periastron**3) / (self.periastron - 2*self.bh_mass))
        result = NumericalFunctions.calc_b_from_periastron(self.periastron, self.bh_mass)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_k(self):
        q = NumericalFunctions.calc_q(self.periastron, self.bh_mass)
        expected = np.sqrt((q - self.periastron + 6 * self.bh_mass) / (2 * q))
        result = NumericalFunctions.k(self.periastron, self.bh_mass)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_k2(self):
        q = NumericalFunctions.calc_q(self.periastron, self.bh_mass)
        expected = (q - self.periastron + 6 * self.bh_mass) / (2 * q)
        result = NumericalFunctions.k2(self.periastron, self.bh_mass)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_zeta_inf(self):
        q = NumericalFunctions.calc_q(self.periastron, self.bh_mass)
        arg = (q - self.periastron + 2 * self.bh_mass) / (q - self.periastron + 6 * self.bh_mass)
        expected = np.arcsin(np.sqrt(arg))
        result = NumericalFunctions.zeta_inf(self.periastron, self.bh_mass)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_zeta_r(self):
        r = 20.0
        q = NumericalFunctions.calc_q(self.periastron, self.bh_mass)
        a = (q - self.periastron + 2 * self.bh_mass + (4 * self.bh_mass * self.periastron) / r) / (q - self.periastron + (6 * self.bh_mass))
        expected = np.arcsin(np.sqrt(a))
        result = NumericalFunctions.zeta_r(self.periastron, r, self.bh_mass)
        self.assertAlmostEqual(result, expected, delta=self.tol)

class TestGeometricFunctions(unittest.TestCase):
    def setUp(self):
        self.tol = 1e-6

    def test_cos_gamma(self):
        a = np.pi/4
        incl = np.pi/3
        expected = np.cos(a) / np.sqrt(np.cos(a) ** 2 + 1 / (np.tan(incl) ** 2))
        result = GeometricFunctions.cos_gamma(a, incl)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_cos_gamma_zero_incl(self):
        a = np.pi/4
        incl = 0
        result = GeometricFunctions.cos_gamma(a, incl)
        self.assertAlmostEqual(result, 0, delta=self.tol)

    def test_cos_alpha(self):
        phi = np.pi/4
        incl = np.pi/3
        expected = np.cos(phi) * np.cos(incl) / np.sqrt((1 - np.sin(incl) ** 2 * np.cos(phi) ** 2))
        result = GeometricFunctions.cos_alpha(phi, incl)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_alpha(self):
        phi = np.pi/4
        incl = np.pi/3
        expected = np.arccos(GeometricFunctions.cos_alpha(phi, incl))
        result = GeometricFunctions.alpha(phi, incl)
        self.assertAlmostEqual(result, expected, delta=self.tol)

    def test_ellipse(self):
        r = 10.0
        a = np.pi/4
        incl = np.pi/3
        g = np.arccos(GeometricFunctions.cos_gamma(a, incl))
        expected = r * np.sin(g)
        result = GeometricFunctions.ellipse(r, a, incl)
        self.assertAlmostEqual(result, expected, delta=self.tol)

class TestPhysicalFunctions(unittest.TestCase):
    def setUp(self):
        self.periastron = 10.0
        self.ir_radius = 5.0
        self.ir_angle = np.pi/4
        self.bh_mass = 1.0
        self.incl = np.pi/3
        self.tol = 1e-6

    @patch('src.math.black_hole_math.NumericalFunctions.zeta_inf')
    @patch('src.math.black_hole_math.NumericalFunctions.calc_q')
    @patch('src.math.black_hole_math.NumericalFunctions.k2')
    @patch('src.math.black_hole_math.GeometricFunctions.cos_gamma')
    def test_eq13(self, mock_cos_gamma, mock_k2, mock_calc_q, mock_zeta_inf):
        mock_zeta_inf.return_value = 0.5
        mock_calc_q.return_value = 8.0
        mock_k2.return_value = 0.25
        mock_cos_gamma.return_value = np.cos(np.pi/4)

        result = PhysicalFunctions.eq13(self.periastron, self.ir_radius, self.ir_angle, self.bh_mass, self.incl)
        self.assertIsInstance(result, float)

    def test_redshift_factor(self):
        radius = 20.0
        angle = np.pi/4
        b_ = 5.0
        result = PhysicalFunctions.redshift_factor(radius, angle, self.incl, self.bh_mass, b_)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Redshift factor should be positive

    def test_flux_intrinsic(self):
        r_val = 20.0
        acc = 0.1
        result = PhysicalFunctions.flux_intrinsic(r_val, acc, self.bh_mass)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Flux should be positive

    def test_flux_observed(self):
        r_val = 20.0
        acc = 0.1
        redshift_factor = 1.5
        result = PhysicalFunctions.flux_observed(r_val, acc, self.bh_mass, redshift_factor)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Observed flux should be positive

    @patch('src.math.black_hole_math.NumericalFunctions.calc_q')
    @patch('src.math.black_hole_math.NumericalFunctions.k2')
    @patch('src.math.black_hole_math.NumericalFunctions.zeta_inf')
    def test_phi_inf(self, mock_zeta_inf, mock_k2, mock_calc_q):
        mock_calc_q.return_value = 8.0
        mock_k2.return_value = 0.25
        mock_zeta_inf.return_value = 0.5

        result = PhysicalFunctions.phi_inf(self.periastron, self.bh_mass)
        self.assertIsInstance(result, float)

    def test_mu(self):
        result = PhysicalFunctions.mu(self.periastron, self.bh_mass)
        self.assertIsInstance(result, float)

class TestUtilities(unittest.TestCase):
    def setUp(self):
        self.periastrons = [2.0, 3.0, 4.0, 5.0]
        self.bh_mass = 1.0
        self.tol = 1e-3

    def test_filter_periastrons(self):
        result = Utilities.filter_periastrons(self.periastrons, self.bh_mass, self.tol)
        self.assertNotIn(2.0, result)  # 2.0 should be filtered out
        self.assertIn(3.0, result)
        self.assertIn(4.0, result)
        self.assertIn(5.0, result)

    def test_lambdify(self):
        x, y = sy.symbols('x y')
        expr = x**2 + y**2
        func = Utilities.lambdify((x, y), expr)
        result = func(2, 3)
        self.assertAlmostEqual(result, 13, delta=self.tol)

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.func = lambda periastron, **kwargs: periastron**2 - 4
        self.args = {}
        self.x = [1, 3]
        self.y = [-3, 5]
        self.ind = 0

    def test_midpoint_method(self):
        new_x, new_y, new_ind = Optimization.midpoint_method(self.func, self.args, self.x, self.y, self.ind)
        self.assertEqual(len(new_x), 3)
        self.assertEqual(len(new_y), 3)
        self.assertAlmostEqual(new_x[1], 2)
        self.assertAlmostEqual(new_y[1], 0)
        self.assertEqual(new_ind, 0)

    def test_improve_solutions_midpoint(self):
        result = Optimization.improve_solutions_midpoint(self.func, self.args, self.x, self.y, self.ind, 5)
        self.assertAlmostEqual(result, 2, places=3)

    @patch('src.math.black_hole_math.PhysicalFunctions.eq13')
    def test_calc_periastron(self, mock_eq13):
        mock_eq13.return_value = 1
        result = Optimization.calc_periastron(5, np.pi/4, np.pi/3, 1)
        self.assertIsNotNone(result)

class TestImpactParameter(unittest.TestCase):
    @patch('src.math.black_hole_math.Optimization.calc_periastron')
    @patch('src.math.black_hole_math.NumericalFunctions.calc_b_from_periastron')
    def test_calc_impact_parameter(self, mock_calc_b, mock_calc_periastron):
        mock_calc_periastron.return_value = 5
        mock_calc_b.return_value = 10
        result = ImpactParameter.calc_impact_parameter(np.pi/4, 10, np.pi/3, 0, 1)
        self.assertEqual(result, 10)

    @patch('src.math.black_hole_math.Optimization.calc_periastron')
    @patch('src.math.black_hole_math.GeometricFunctions.ellipse')
    def test_calc_impact_parameter_ellipse(self, mock_ellipse, mock_calc_periastron):
        mock_calc_periastron.return_value = 1
        mock_ellipse.return_value = 5
        result = ImpactParameter.calc_impact_parameter(np.pi/4, 10, np.pi/3, 0, 1)
        self.assertEqual(result, 5)

class TestSimulation(unittest.TestCase):
    def test_reorient_alpha(self):
        alpha = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        n = 1
        result = Simulation.reorient_alpha(alpha, n)
        expected = np.array([np.pi, 3*np.pi/2, 0, np.pi/2])
        np.testing.assert_array_almost_equal(result, expected)

    @patch('src.math.black_hole_math.ImpactParameter.calc_impact_parameter')
    @patch('src.math.black_hole_math.PhysicalFunctions.redshift_factor')
    @patch('src.math.black_hole_math.PhysicalFunctions.flux_observed')
    def test_simulate_flux(self, mock_flux_observed, mock_redshift_factor, mock_calc_impact_parameter):
        alpha = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        r = 10
        theta_0 = np.pi/4
        n = 0
        m = 1
        acc = 0.1
        
        mock_calc_impact_parameter.return_value = np.array([5, 5, 5, 5])
        mock_redshift_factor.return_value = np.array([1.1, 1.1, 1.1, 1.1])
        mock_flux_observed.return_value = np.array([100, 100, 100, 100])
        
        result = Simulation.simulate_flux(alpha, r, theta_0, n, m, acc)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].shape, (4,))
        self.assertEqual(result[1].shape, (4,))
        self.assertEqual(result[2].shape, (4,))
        self.assertEqual(result[3].shape, (4,))

    @patch('src.math.black_hole_math.Simulation.simulate_flux')
    def test_generate_image_data(self, mock_simulate_flux):
        alpha = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        r_vals = [10, 20]
        theta_0 = np.pi/4
        n_vals = [0, 1]
        m = 1
        acc = 0.1
        root_kwargs = {}
        
        mock_simulate_flux.return_value = (alpha, np.array([5, 5, 5, 5]), np.array([1.1, 1.1, 1.1, 1.1]), np.array([100, 100, 100, 100]))
        
        result = Simulation.generate_image_data(alpha, r_vals, theta_0, n_vals, m, acc, root_kwargs)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 16)  # 4 alpha values * 2 r values * 2 n values

if __name__ == '__main__':
    unittest.main()