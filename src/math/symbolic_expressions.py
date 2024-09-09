import sympy as sy
from functools import lru_cache

# Constants and symbols
M, P, Q, r, theta_0, alpha, b, gamma, N, z_op, F_s, mdot = sy.symbols("M P Q r theta_0 alpha b gamma N z_op F_s mdot")

class SymbolicExpressions:
    @staticmethod
    @lru_cache(maxsize=None)
    def expr_q() -> sy.Expr:
        """Generate a sympy expression for Q."""
        return sy.sqrt((P - 2*M) * (P + 6*M))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_b() -> sy.Expr:
        """Generate a sympy expression for b, the radial coordinate in the observer's frame."""
        return sy.sqrt((P**3) / (P - 2*M))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_k() -> sy.Expr:
        """Generate a sympy expression for k; k**2 is used as a modulus in the elliptic integrals."""
        return sy.sqrt((Q - P + 6*M) / (2*Q))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_zeta_inf() -> sy.Expr:
        """Generate a sympy expression for zeta_inf."""
        return sy.asin(sy.sqrt((Q - P + 2*M) / (Q - P + 6*M)))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_gamma() -> sy.Expr:
        """Generate a sympy expression for gamma, an angle that relates alpha and theta_0."""
        return sy.acos(sy.cos(alpha) / sy.sqrt(sy.cos(alpha)**2 + sy.tan(theta_0)**-2))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_u() -> sy.Expr:
        """Generate a sympy expression for the argument to sn in equation 13 of Luminet (1977)."""
        zeta_inf, k = SymbolicExpressions.expr_zeta_inf(), SymbolicExpressions.expr_k()
        return sy.Piecewise(
            (gamma / (2 * sy.sqrt(P / Q)) + sy.elliptic_f(zeta_inf, k**2), sy.Eq(N, 0)),
            (
                (gamma - 2 * N * sy.pi) / (2 * sy.sqrt(P / Q))
                - sy.elliptic_f(zeta_inf, k**2)
                + 2 * sy.elliptic_k(k**2),
                True,
            ),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_r_inv() -> sy.Expr:
        """Generate a sympy expression for 1/r."""
        Q, k = SymbolicExpressions.expr_q(), SymbolicExpressions.expr_k()
        sn = sy.Function("sn")
        u = SymbolicExpressions.expr_u()
        return (1 / (4*M*P)) * (-(Q - P + 2*M) + (Q - P + 6*M) * sn(u, k**2)**2)

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_one_plus_z() -> sy.Expr:
        """Generate an expression for the redshift 1+z."""
        return (1 + sy.sqrt(M / r**3) * b * sy.sin(theta_0) * sy.sin(alpha)) / sy.sqrt(1 - 3*M / r)

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_fs() -> sy.Expr:
        """Generate an expression for the flux of an accreting disk."""
        rstar = r / M
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
        """Generate a sympy expression for an ellipse."""
        return r / sy.sqrt(1 + (sy.tan(theta_0)**2) * (sy.cos(alpha)**2))