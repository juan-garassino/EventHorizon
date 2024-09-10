import unittest
import sympy as sy
import numpy as np
import pandas as pd
import scipy.special as scp
from unittest.mock import patch
from src.math.symbolic_expressions import SymbolicExpressions
from src.math.numerical_functions import NumericalFunctions
from src.math.geometric_functions import GeometricFunctions
from src.math.physical_functions import PhysicalFunctions
from src.math.impact_parameter import ImpactParameter
from src.math.utilities import Utilities
from src.math.optimization import Optimization
from src.math.simulation import Simulation

class TestSymbolicExpressions(unittest.TestCase):

    def setUp(self):
        # Define symbols that might be used in the expressions
        self.M, self.P, self.Q, self.N, self.calculate_alpha, self.theta_0, self.radius, self.b, self.mdot = sy.symbols('M P Q N calculate_alpha theta_0 radius b mdot')

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
        expected = sy.acos(sy.cos(self.calculate_alpha) / sy.sqrt(sy.cos(self.calculate_alpha)**2 + sy.tan(self.theta_0)**-2))
        result = SymbolicExpressions.expr_gamma()
        self.assertEqual(result, expected)

    @patch('src.math.symbolic_expressions.SymbolicExpressions.expr_zeta_inf')
    @patch('src.math.symbolic_expressions.SymbolicExpressions.expr_k')
    def test_expr_u(self, mock_k, mock_zeta_inf):
        # Mock the return values of expr_zeta_inf and expr_k
        mock_zeta_inf.return_value = sy.Symbol('calculate_zeta_infinity')
        mock_k.return_value = sy.Symbol('calculate_elliptic_integral_modulus')

        result = SymbolicExpressions.expr_u()
        
        # Check if the result is argument Piecewise function
        self.assertIsInstance(result, sy.Piecewise)
        
        # Check if it has two pieces
        self.assertEqual(len(result.arguments), 2)

    def test_expr_one_plus_z(self):
        expected = (1 + sy.sqrt(self.M / self.radius**3) * self.b * sy.sin(self.theta_0) * sy.sin(self.calculate_alpha)) / sy.sqrt(1 - 3*self.M / self.radius)
        result = SymbolicExpressions.expr_one_plus_z()
        self.assertEqual(result, expected)

    def test_expr_fs(self):
        rstar = self.radius / self.M
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
        self.black_hole_mass = 1.0
        self.tolerance = 1e-6

    def test_calc_q(self):
        expected = np.sqrt((self.periastron - 2*self.black_hole_mass) * (self.periastron + 6*self.black_hole_mass))
        result = NumericalFunctions.calculate_q_parameter(self.periastron, self.black_hole_mass)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_calc_b_from_periastron(self):
        expected = np.sqrt((self.periastron**3) / (self.periastron - 2*self.black_hole_mass))
        result = NumericalFunctions.calculate_impact_parameter(self.periastron, self.black_hole_mass)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_k(self):
        q_parameter = NumericalFunctions.calculate_q_parameter(self.periastron, self.black_hole_mass)
        expected = np.sqrt((q_parameter - self.periastron + 6 * self.black_hole_mass) / (2 * q_parameter))
        result = NumericalFunctions.calculate_elliptic_integral_modulus(self.periastron, self.black_hole_mass)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_k2(self):
        q_parameter = NumericalFunctions.calculate_q_parameter(self.periastron, self.black_hole_mass)
        expected = (q_parameter - self.periastron + 6 * self.black_hole_mass) / (2 * q_parameter)
        result = NumericalFunctions.calculate_squared_elliptic_integral_modulus(self.periastron, self.black_hole_mass)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_zeta_inf(self):
        q_parameter = NumericalFunctions.calculate_q_parameter(self.periastron, self.black_hole_mass)
        arg = (q_parameter - self.periastron + 2 * self.black_hole_mass) / (q_parameter - self.periastron + 6 * self.black_hole_mass)
        expected = np.arcsin(np.sqrt(arg))
        result = NumericalFunctions.calculate_zeta_infinity(self.periastron, self.black_hole_mass)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_zeta_r(self):
        radius = 20.0
        q_parameter = NumericalFunctions.calculate_q_parameter(self.periastron, self.black_hole_mass)
        argument = (q_parameter - self.periastron + 2 * self.black_hole_mass + (4 * self.black_hole_mass * self.periastron) / radius) / (q_parameter - self.periastron + (6 * self.black_hole_mass))
        expected = np.arcsin(np.sqrt(argument))
        result = NumericalFunctions.calculate_zeta_radius(self.periastron, radius, self.black_hole_mass)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

class TestGeometricFunctions(unittest.TestCase):
    def setUp(self):
        self.tolerance = 1e-6

    def test_cos_gamma(self):
        argument = np.pi/4
        inclination = np.pi/3
        expected = np.cos(argument) / np.sqrt(np.cos(argument) ** 2 + 1 / (np.tan(inclination) ** 2))
        result = GeometricFunctions.calculate_cos_gamma(argument, inclination)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_cos_gamma_zero_incl(self):
        argument = np.pi/4
        inclination = 0
        result = GeometricFunctions.calculate_cos_gamma(argument, inclination)
        self.assertAlmostEqual(result, 0, delta=self.tolerance)

    def test_cos_alpha(self):
        phi_black_hole_frame = np.pi/4
        inclination = np.pi/3
        expected = np.cos(phi_black_hole_frame) * np.cos(inclination) / np.sqrt((1 - np.sin(inclination) ** 2 * np.cos(phi_black_hole_frame) ** 2))
        result = GeometricFunctions.calculate_cos_alpha(phi_black_hole_frame, inclination)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_alpha(self):
        phi_black_hole_frame = np.pi/4
        inclination = np.pi/3
        expected = np.arccos(GeometricFunctions.calculate_cos_alpha(phi_black_hole_frame, inclination))
        result = GeometricFunctions.calculate_alpha(phi_black_hole_frame, inclination)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

    def test_ellipse(self):
        radius = 10.0
        argument = np.pi/4
        inclination = np.pi/3
        gamma = np.arccos(GeometricFunctions.calculate_cos_gamma(argument, inclination))
        expected = radius * np.sin(gamma)
        result = GeometricFunctions.calculate_ellipse_radius(radius, argument, inclination)
        self.assertAlmostEqual(result, expected, delta=self.tolerance)

class TestPhysicalFunctions(unittest.TestCase):
    def setUp(self):
        self.periastron = 10.0
        self.emission_radius = 5.0
        self.emission_angle = np.pi/4
        self.black_hole_mass = 1.0
        self.inclination = np.pi/3
        self.tolerance = 1e-6

    @patch('src.math.numerical_functions.NumericalFunctions.calculate_zeta_infinity')
    @patch('src.math.numerical_functions.NumericalFunctions.calculate_q_parameter')
    @patch('src.math.numerical_functions.NumericalFunctions.calculate_squared_elliptic_integral_modulus')
    @patch('src.math.geometric_functions.GeometricFunctions.calculate_cos_gamma')
    def test_eq13(self, mock_cos_gamma, mock_k2, mock_calc_q, mock_zeta_inf):
        mock_zeta_inf.return_value = 0.5
        mock_calc_q.return_value = 8.0
        mock_k2.return_value = 0.25
        mock_cos_gamma.return_value = np.cos(np.pi/4)

        result = PhysicalFunctions.calculate_luminet_equation_13(self.periastron, self.emission_radius, self.emission_angle, self.black_hole_mass, self.inclination)
        self.assertIsInstance(result, float)

    def test_redshift_factor(self):
        radius = 20.0
        angle = np.pi/4
        impact_parameter = 5.0
        result = PhysicalFunctions.calculate_redshift_factor(radius, angle, self.inclination, self.black_hole_mass, impact_parameter)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Redshift factor should be positive

    def test_flux_intrinsic(self):
        r_val = 20.0
        accretion_rate = 0.1
        result = PhysicalFunctions.calculate_intrinsic_flux(r_val, accretion_rate, self.black_hole_mass)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Flux should be positive

    def test_flux_observed(self):
        r_val = 20.0
        accretion_rate = 0.1
        calculate_redshift_factor = 1.5
        result = PhysicalFunctions.calculate_observed_flux(r_val, accretion_rate, self.black_hole_mass, calculate_redshift_factor)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Observed flux should be positive

    @patch('src.math.numerical_functions.NumericalFunctions.calculate_q_parameter')
    @patch('src.math.numerical_functions.NumericalFunctions.calculate_squared_elliptic_integral_modulus')
    @patch('src.math.numerical_functions.NumericalFunctions.calculate_zeta_infinity')
    def test_phi_inf(self, mock_zeta_inf, mock_k2, mock_calc_q):
        mock_calc_q.return_value = 8.0
        mock_k2.return_value = 0.25
        mock_zeta_inf.return_value = 0.5

        result = PhysicalFunctions.calculate_phi_infinity(self.periastron, self.black_hole_mass)
        self.assertIsInstance(result, float)

    def test_mu(self):
        result = PhysicalFunctions.calculate_mu(self.periastron, self.black_hole_mass)
        self.assertIsInstance(result, float)

class TestUtilities(unittest.TestCase):
    def setUp(self):
        self.periastrons = [2.0, 3.0, 4.0, 5.0]
        self.black_hole_mass = 1.0
        self.tolerance = 1e-3

    def test_filter_periastrons(self):
        result = Utilities.filter_periastrons(self.periastrons, self.black_hole_mass, self.tolerance)
        self.assertNotIn(2.0, result)  # 2.0 should be filtered out
        self.assertIn(3.0, result)
        self.assertIn(4.0, result)
        self.assertIn(5.0, result)

    def test_lambdify(self):
        x_values, y_values = sy.symbols('x_values y_values')
        expr = x_values**2 + y_values**2
        function = Utilities.lambdify((x_values, y_values), expr)
        result = function(2, 3)
        self.assertAlmostEqual(result, 13, delta=self.tolerance)

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.function = lambda periastron, **kwargs: periastron**2 - 4
        self.arguments = {}
        self.x_values = [2, 3]
        self.y_values = [-3, 5]
        self.index = 0

    def test_midpoint_method(self):
        new_x, new_y, new_ind = Optimization.apply_midpoint_method(self.function, self.arguments, self.x_values, self.y_values, self.index)
        self.assertEqual(len(new_x), 3)
        self.assertEqual(len(new_y), 3)
        self.assertAlmostEqual(new_x[1], 2)
        self.assertAlmostEqual(new_y[1], 0)
        self.assertEqual(new_ind, 0)

    def test_improve_solutions_midpoint(self):
        result = Optimization.refine_solution_with_midpoint_method(self.function, self.arguments, self.x_values, self.y_values, self.index, 5)
        self.assertAlmostEqual(result, 2, delta=1e-2)  # Use delta or adjust as needed

    @patch('src.math.physical_functions.PhysicalFunctions.calculate_luminet_equation_13')
    def test_calc_periastron(self, mock_eq13):
        mock_eq13.return_value = 1  # Mock return value should be argument valid number
        result = Optimization.calculate_periastron(5, np.pi/4, np.pi/3, 1)
        self.assertIsNotNone(result)  # Check that result is not None
        self.assertIsInstance(result, float)  # Ensure the result is of type float or expected type
        # Add more assertions to validate result correctness
        self.assertGreater(result, 0)  # Example additional check if appropriate

class TestImpactParameter(unittest.TestCase):
    @patch('src.math.optimization.Optimization.calculate_periastron')
    @patch('src.math.numerical_functions.NumericalFunctions.calculate_impact_parameter')
    def test_calc_impact_parameter(self, mock_calc_b, mock_calc_periastron):
        mock_calc_periastron.return_value = 5
        mock_calc_b.return_value = 10
        result = ImpactParameter.calc_impact_parameter(np.pi/4, 10, np.pi/3, 0, 1)
        self.assertEqual(result, 10)

    @patch('src.math.optimization.Optimization.calculate_periastron')
    @patch('src.math.geometric_functions.GeometricFunctions.calculate_ellipse_radius')
    def test_calc_impact_parameter_ellipse(self, mock_ellipse, mock_calc_periastron):
        mock_calc_periastron.return_value = 1
        mock_ellipse.return_value = 5
        result = ImpactParameter.calc_impact_parameter(np.pi/4, 10, np.pi/3, 0, 1)
        self.assertEqual(result, 5)

class TestSimulation(unittest.TestCase):
    def test_reorient_alpha(self):
        calculate_alpha = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        image_order = 1
        result = Simulation.reorient_alpha(calculate_alpha, image_order)
        expected = np.array([np.pi, 3*np.pi/2, 0, np.pi/2])
        np.testing.assert_array_almost_equal(result, expected)

    @patch('src.math.impact_parameter.ImpactParameter.calc_impact_parameter')
    @patch('src.math.physical_functions.PhysicalFunctions.calculate_redshift_factor')
    @patch('src.math.physical_functions.PhysicalFunctions.calculate_observed_flux')
    def test_simulate_flux(self, mock_flux_observed, mock_redshift_factor, mock_calc_impact_parameter):
        calculate_alpha = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        radius = 10
        theta_0 = np.pi/4
        image_order = 0
        m = 1
        accretion_rate = 0.1
        
        mock_calc_impact_parameter.return_value = np.array([5, 5, 5, 5])
        mock_redshift_factor.return_value = np.array([1.1, 1.1, 1.1, 1.1])
        mock_flux_observed.return_value = np.array([100, 100, 100, 100])
        
        result = Simulation.simulate_flux(calculate_alpha, radius, theta_0, image_order, m, accretion_rate)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].shape, (4,))
        self.assertEqual(result[1].shape, (4,))
        self.assertEqual(result[2].shape, (4,))
        self.assertEqual(result[3].shape, (4,))

    @patch('src.math.simulation.Simulation.simulate_flux')
    def test_generate_image_data(self, mock_simulate_flux):
        calculate_alpha = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        r_vals = [10, 20]
        theta_0 = np.pi/4
        n_vals = [0, 1]
        m = 1
        accretion_rate = 0.1
        root_kwargs = {}
        
        mock_simulate_flux.return_value = (calculate_alpha, np.array([5, 5, 5, 5]), np.array([1.1, 1.1, 1.1, 1.1]), np.array([100, 100, 100, 100]))
        
        result = Simulation.generate_image_data(calculate_alpha, r_vals, theta_0, n_vals, m, accretion_rate, root_kwargs)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 16)  # 4 calculate_alpha values * 2 radius values * 2 image_order values

if __name__ == '__main__':
    unittest.main()