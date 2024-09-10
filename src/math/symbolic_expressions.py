import sympy as sy
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and symbols
M, P, Q, radius, theta_0, calculate_alpha, impact_parameters, gamma, N, z_op, F_s, mdot = sy.symbols("M P Q radius theta_0 calculate_alpha impact_parameters gamma N z_op F_s mdot")

class SymbolicExpressions:
    @staticmethod
    def _log_verbose(message: str, verbose: bool):
        if verbose:
            logger.info(message)

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_q(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for Q."""
        logger.info("🔢 Generating expression for Q")
        SymbolicExpressions._log_verbose("🔍 Detailed Q expression calculation in progress", verbose)
        result = sy.sqrt((P - 2*M) * (P + 6*M))
        SymbolicExpressions._log_verbose("✅ Q expression generation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_b(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for impact_parameters, the radial coordinate in the observer's frame."""
        logger.info("📏 Calculating impact parameter expression")
        SymbolicExpressions._log_verbose("🧮 Detailed impact parameter calculation ongoing", verbose)
        result = sy.sqrt((P**3) / (P - 2*M))
        SymbolicExpressions._log_verbose("✅ Impact parameter expression calculation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_k(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for calculate_elliptic_integral_modulus; calculate_elliptic_integral_modulus**2 is used as argument modulus in the elliptic integrals."""
        logger.info("🔄 Computing elliptic integral modulus")
        SymbolicExpressions._log_verbose("📊 Detailed elliptic integral modulus calculation in process", verbose)
        result = sy.sqrt((Q - P + 6*M) / (2*Q))
        SymbolicExpressions._log_verbose("✅ Elliptic integral modulus computation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_zeta_inf(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for calculate_zeta_infinity."""
        logger.info("🔬 Evaluating zeta infinity expression")
        SymbolicExpressions._log_verbose("🔎 Detailed zeta infinity calculation underway", verbose)
        result = sy.asin(sy.sqrt((Q - P + 2*M) / (Q - P + 6*M)))
        SymbolicExpressions._log_verbose("✅ Zeta infinity expression evaluation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_gamma(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for gamma, an angle that relates calculate_alpha and theta_0."""
        logger.info("📐 Deriving gamma angle expression")
        SymbolicExpressions._log_verbose("🧭 Detailed gamma angle calculation in progress", verbose)
        result = sy.acos(sy.cos(calculate_alpha) / sy.sqrt(sy.cos(calculate_alpha)**2 + sy.tan(theta_0)**-2))
        SymbolicExpressions._log_verbose("✅ Gamma angle expression derivation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_u(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for the argument to sine_jacobi in equation 13 of Luminet (1977)."""
        logger.info("🧮 Constructing u expression for sine_jacobi")
        SymbolicExpressions._log_verbose("📝 Detailed u expression calculation ongoing", verbose)
        calculate_zeta_infinity, calculate_elliptic_integral_modulus = SymbolicExpressions.expr_zeta_inf(verbose), SymbolicExpressions.expr_k(verbose)
        result = sy.Piecewise(
            (gamma / (2 * sy.sqrt(P / Q)) + sy.elliptic_f(calculate_zeta_infinity, calculate_elliptic_integral_modulus**2), sy.Eq(N, 0)),
            (
                (gamma - 2 * N * sy.pi) / (2 * sy.sqrt(P / Q))
                - sy.elliptic_f(calculate_zeta_infinity, calculate_elliptic_integral_modulus**2)
                + 2 * sy.elliptic_k(calculate_elliptic_integral_modulus**2),
                True,
            ),
        )
        SymbolicExpressions._log_verbose("✅ U expression construction complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_r_inv(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for 1/radius."""
        logger.info("📉 Formulating inverse radius expression")
        SymbolicExpressions._log_verbose("🔬 Detailed inverse radius calculation in process", verbose)
        Q, calculate_elliptic_integral_modulus = SymbolicExpressions.expr_q(verbose), SymbolicExpressions.expr_k(verbose)
        sine_jacobi = sy.Function("sine_jacobi")
        u = SymbolicExpressions.expr_u(verbose)
        result = (1 / (4*M*P)) * (-(Q - P + 2*M) + (Q - P + 6*M) * sine_jacobi(u, calculate_elliptic_integral_modulus**2)**2)
        SymbolicExpressions._log_verbose("✅ Inverse radius expression formulation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_one_plus_z(verbose: bool = False) -> sy.Expr:
        """Generate an expression for the redshift 1+z."""
        logger.info("🌈 Calculating redshift expression")
        SymbolicExpressions._log_verbose("🔭 Detailed redshift calculation underway", verbose)
        result = (1 + sy.sqrt(M / radius**3) * impact_parameters * sy.sin(theta_0) * sy.sin(calculate_alpha)) / sy.sqrt(1 - 3*M / radius)
        SymbolicExpressions._log_verbose("✅ Redshift expression calculation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_fs(verbose: bool = False) -> sy.Expr:
        """Generate an expression for the flux of an accreting disk."""
        logger.info("💫 Deriving accretion disk flux expression")
        SymbolicExpressions._log_verbose("🌠 Detailed accretion disk flux calculation in progress", verbose)
        rstar = radius / M
        result = (
            ((3 * M * mdot) / (8 * sy.pi))
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
        SymbolicExpressions._log_verbose("✅ Accretion disk flux expression derivation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_f0(verbose: bool = False) -> sy.Expr:
        """Generate an expression for the observed bolometric flux."""
        logger.info("🔆 Computing observed bolometric flux")
        SymbolicExpressions._log_verbose("☀️ Detailed observed bolometric flux calculation ongoing", verbose)
        result = F_s / z_op**4
        SymbolicExpressions._log_verbose("✅ Observed bolometric flux computation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_f0_normalized(verbose: bool = False) -> sy.Expr:
        """Generate an expression for the normalized observed bolometric flux."""
        logger.info("📊 Normalizing observed bolometric flux")
        SymbolicExpressions._log_verbose("📈 Detailed normalized bolometric flux calculation in process", verbose)
        result = SymbolicExpressions.expr_f0(verbose) / ((8 * sy.pi) / (3 * M * mdot))
        SymbolicExpressions._log_verbose("✅ Normalized observed bolometric flux calculation complete", verbose)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_ellipse(verbose: bool = False) -> sy.Expr:
        """Generate argument sympy expression for an calculate_ellipse_radius."""
        logger.info("🔴 Formulating ellipse radius expression")
        SymbolicExpressions._log_verbose("⭕ Detailed ellipse radius calculation underway", verbose)
        result = radius / sy.sqrt(1 + (sy.tan(theta_0)**2) * (sy.cos(calculate_alpha)**2))
        SymbolicExpressions._log_verbose("✅ Ellipse radius expression formulation complete", verbose)
        return result