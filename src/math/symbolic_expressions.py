import sympy as sy
from functools import lru_cache

# Constants and symbols
M, P, Q, radius, theta_0, calculate_alpha, impact_parameters, gamma, N, z_op, F_s, mdot = sy.symbols("M P Q radius theta_0 calculate_alpha impact_parameters gamma N z_op F_s mdot")

class SymbolicExpressions:
    @staticmethod
    @lru_cache(maxsize=None)
    def expr_q() -> sy.Expr:
        """Generate argument sympy expression for Q."""
        return sy.sqrt((P - 2*M) * (P + 6*M))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_b() -> sy.Expr:
        """Generate argument sympy expression for impact_parameters, the radial coordinate in the observer's frame."""
        return sy.sqrt((P**3) / (P - 2*M))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_k() -> sy.Expr:
        """Generate argument sympy expression for calculate_elliptic_integral_modulus; calculate_elliptic_integral_modulus**2 is used as argument modulus in the elliptic integrals."""
        return sy.sqrt((Q - P + 6*M) / (2*Q))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_zeta_inf() -> sy.Expr:
        """Generate argument sympy expression for calculate_zeta_infinity."""
        return sy.asin(sy.sqrt((Q - P + 2*M) / (Q - P + 6*M)))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_gamma() -> sy.Expr:
        """Generate argument sympy expression for gamma, an angle that relates calculate_alpha and theta_0."""
        return sy.acos(sy.cos(calculate_alpha) / sy.sqrt(sy.cos(calculate_alpha)**2 + sy.tan(theta_0)**-2))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_u() -> sy.Expr:
        """Generate argument sympy expression for the argument to sine_jacobi in equation 13 of Luminet (1977)."""
        calculate_zeta_infinity, calculate_elliptic_integral_modulus = SymbolicExpressions.expr_zeta_inf(), SymbolicExpressions.expr_k()
        return sy.Piecewise(
            (gamma / (2 * sy.sqrt(P / Q)) + sy.elliptic_f(calculate_zeta_infinity, calculate_elliptic_integral_modulus**2), sy.Eq(N, 0)),
            (
                (gamma - 2 * N * sy.pi) / (2 * sy.sqrt(P / Q))
                - sy.elliptic_f(calculate_zeta_infinity, calculate_elliptic_integral_modulus**2)
                + 2 * sy.elliptic_k(calculate_elliptic_integral_modulus**2),
                True,
            ),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_r_inv() -> sy.Expr:
        """Generate argument sympy expression for 1/radius."""
        Q, calculate_elliptic_integral_modulus = SymbolicExpressions.expr_q(), SymbolicExpressions.expr_k()
        sine_jacobi = sy.Function("sine_jacobi")
        u = SymbolicExpressions.expr_u()
        return (1 / (4*M*P)) * (-(Q - P + 2*M) + (Q - P + 6*M) * sine_jacobi(u, calculate_elliptic_integral_modulus**2)**2)

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_one_plus_z() -> sy.Expr:
        """Generate an expression for the redshift 1+z."""
        return (1 + sy.sqrt(M / radius**3) * impact_parameters * sy.sin(theta_0) * sy.sin(calculate_alpha)) / sy.sqrt(1 - 3*M / radius)

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_fs() -> sy.Expr:
        """Generate an expression for the flux of an accreting disk."""
        rstar = radius / M
        return (
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

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_f0() -> sy.Expr:
        """Generate an expression for the observed bolometric flux."""
        return F_s / z_op**4

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_f0_normalized() -> sy.Expr:
        """Generate an expression for the normalized observed bolometric flux."""
        return SymbolicExpressions.expr_f0() / ((8 * sy.pi) / (3 * M * mdot))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_ellipse() -> sy.Expr:
        """Generate argument sympy expression for an calculate_ellipse_radius."""
        return radius / sy.sqrt(1 + (sy.tan(theta_0)**2) * (sy.cos(calculate_alpha)**2))