from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc
# import mpmath

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


def calc_q(periastron: float, bh_mass: float, tol=1e-3) -> float:
    """
    Convert Periastron distance P to the variable Q (easier to work with)
    """
    # limits give no substantial speed improvement
    # if periastron - 2. * bh_mass < tol:
    #     # limit for small values
    #     return .5 * (periastron - 2. * bh_mass) * (periastron + 6. * bh_mass)
    # if 1/periastron < tol:
    #     # limit for large values
    #     return periastron
    # if periastron <= 2*bh_mass:
    #     raise ValueError("Non-physical periastron found (P <= 2M, aka the photon sphere)."
    #                      "If you want to calculate non-physical values, you should implement the mpmath library")
    q = np.sqrt((periastron - 2. * bh_mass) * (periastron + 6. * bh_mass))
    # Q is complex if P < 2M = r_s
    return q


def calc_b_from_periastron(periastron: float, bh_mass: float, tol: float = 1e-5) -> float:
    """
    Get impact parameter b from Periastron distance P
    """
    # limits give no substantial speed improvement
    # if abs(periastron) < tol:  # could physically never happen
    #     print("tolerance exceeded for calc_b_from_P(P_={}, M={}, tol={}".format(periastron, bh_mass, tol))
    #     return np.sqrt(3 * periastron ** 2)
    # WARNING: the paper most definitely has a typo here. The fracture on the right hand side equals b², not b.
    # Just fill in u_2 in equation 3, and you'll see. Only this way do the limits P -> 3M and P >> M hold true,
    # as well as the value for b_c
    return np.sqrt(periastron ** 3 / (periastron - 2. * bh_mass))  # the impact parameter


def k(periastron: float, bh_mass: float) -> float:
    """
    Calculate modulus of elliptic integral
    """
    q = calc_q(periastron, bh_mass)
    # adding limits does not substantially improve speed, nor stability
    # if q < 10e-3:  # numerical stability
    #     return np.sqrt(.5)
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return np.sqrt((q - periastron + 6 * bh_mass) / (2 * q))  # the modulus of the elliptic integral


def k2(periastron: float, bh_mass: float, tol: float = 1e-6):
    """Calculate the squared modulus of elliptic integral"""
    q = calc_q(periastron, bh_mass)
    # adding limits does not substantially improve speed
    # if 1 / periastron <= tol:
    #     # limit of P -> inf, Q -> P
    #     return 0.
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return (q - periastron + 6 * bh_mass) / (2 * q)  # the modulus of the ellipitic integral


def zeta_inf(periastron: float, bh_mass: float, tol: float = 1e-6) -> float:
    """
    Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)
    """
    q = calc_q(periastron, bh_mass)  # Q variable, only call to function once
    arg = (q - periastron + 2 * bh_mass) / (q - periastron + 6 * bh_mass)
    z_inf = np.arcsin(np.sqrt(arg))
    return z_inf


def zeta_r(periastron: float, r: float, bh_mass: float) -> float:
    """
    Calculate the elliptic integral argument Zeta_r for a given value of P and r
    """
    q = calc_q(periastron, bh_mass)
    a = (q - periastron + 2 * bh_mass + (4 * bh_mass * periastron) / r) / (q - periastron + (6 * bh_mass))
    s = np.arcsin(np.sqrt(a))
    return s


def cos_gamma(_a: float, incl: float, tol=10e-5) -> float:
    """
    Calculate the cos of the angle gamma
    """
    if abs(incl) < tol:
        return 0
    return np.cos(_a) / np.sqrt(np.cos(_a) ** 2 + 1 / (np.tan(incl) ** 2))  # real


def cos_alpha(phi: float, incl: float) -> float:
    """Returns cos(angle) alpha in observer frame given angles phi (black hole frame) and
    inclination (black hole frame)"""
    return np.cos(phi) * np.cos(incl) / np.sqrt((1 - np.sin(incl) ** 2 * np.cos(phi) ** 2))


def alpha(phi: float, incl: float):
    """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)"""
    return np.arccos(cos_alpha(phi, incl))


def filter_periastrons(periastron: [], bh_mass: float, tol: float = 10e-3) -> []:
    """
    Removes instances where P == 2*M
    returns indices where this was the case
    """
    return [e for e in periastron if abs(e - 2. * bh_mass) > tol]


def eq13(periastron: float, ir_radius: float, ir_angle: float, bh_mass: float, incl: float, n: int = 0,
         tol=10e-6) -> float:
    """
    Relation between radius (where photon was emitted in accretion disk), a and P.
    P can be converted to b, yielding the polar coordinates (b, a) on the photographic plate

    This function get called almost everytime when you need to calculate some black hole property
    """
    z_inf = zeta_inf(periastron, bh_mass)
    q = calc_q(periastron, bh_mass)
    m_ = k2(periastron, bh_mass)  # modulus of the elliptic integrals. mpmath takes m = k² as argument.
    ell_inf = ellipkinc(z_inf, m_)  # Elliptic integral F(zeta_inf, k)
    g = np.arccos(cos_gamma(ir_angle, incl))

    # Calculate the argument of sn (mod is m = k², same as the original elliptic integral)
    # WARNING: paper has an error here: \sqrt(P / Q) should be in denominator, not numerator
    # There's no way that \gamma and \sqrt(P/Q) can end up on the same side of the division
    if n:  # higher order image
        ell_k = ellipk(m_)  # calculate complete elliptic integral of mod m = k²
        ellips_arg = (g - 2. * n * np.pi) / (2. * np.sqrt(periastron / q)) - ell_inf + 2. * ell_k
    else:  # direct image
        ellips_arg = g / (2. * np.sqrt(periastron / q)) + ell_inf

    # sn is an Jacobi elliptic function: elliptic sine. ellipfun() takes 'sn'
    # as argument to specify "elliptic sine" and modulus m=k²
    sn, cn, dn, ph = ellipj(ellips_arg, m_)
    sn2 = sn * sn
    term1 = -(q - periastron + 2. * bh_mass) / (4. * bh_mass * periastron)
    term2 = ((q - periastron + 6. * bh_mass) / (4. * bh_mass * periastron)) * sn2

    return 1. - ir_radius * (term1 + term2)  # solve this for zero


def midpoint_method(func, args: Dict, __x, __y, __ind):
    new_x = __x
    new_y = __y

    x_ = [new_x[__ind], new_x[__ind + 1]]  # interval of P values
    inbetween_x = np.mean(x_)  # new periastron value, closer to solution yielding 0 for ea13
    new_x.insert(__ind + 1, inbetween_x)  # insert middle P value to calculate

    y_ = [new_y[__ind], new_y[__ind + 1]]  # results of eq13 given the P values
    # calculate the P value inbetween
    inbetween_solution = func(periastron=inbetween_x, **args)
    new_y.insert(__ind + 1, inbetween_solution)
    y_.insert(1, inbetween_solution)
    ind_of_sign_change_ = np.where(np.diff(np.sign(y_)))[0]
    new_ind = __ind + ind_of_sign_change_[0]

    return new_x, new_y, new_ind  # return x and y refined in relevant regions, as well as new index of sign change


def improve_solutions_midpoint(func, args, x, y, index_of_sign_change, iterations) -> float:
    """
    To increase precision.
    Recalculate each solution in :arg:`solutions` using the provided :arg:`func`.
    Achieves an improved solution be re-evaluating the provided :arg:`func` at a new
    :arg:`x`, inbetween two pre-existing values for :arg:`x` where the sign of :arg:`y` changes.
    Does this :arg:`iterations` times
    """
    index_of_sign_change_ = index_of_sign_change
    new_x = x
    new_y = y
    new_ind = index_of_sign_change_  # location in X and Y where eq13(P=X[ind]) equals Y=0
    for iteration in range(iterations):
        new_x, new_y, new_ind = midpoint_method(func=func, args=args, __x=new_x, __y=new_y, __ind=new_ind)
    updated_periastron = new_x[new_ind]
    return updated_periastron


def calc_periastron(_r, incl, _alpha, bh_mass, midpoint_iterations=100, plot_inbetween=False,
                    n=0, min_periastron=1., initial_guesses=20) -> float:
    """
        Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value
        This periastron can be converted to an impact parameter b, yielding the observer frame coordinates (b, alpha).
        Does this by generating range of periastron values, evaluating eq13 on this range and using a midpoint method
        to iteratively improve which periastron value solves equation 13.
        The considered initial periastron range must not be lower than min_periastron (i.e. the photon sphere),
        otherwise non-physical solutions will be found. These are interesting in their own right (the equation yields
        complex solutions within radii smaller than the photon sphere!), but are for now outside the scope of this project.
        Must be large enough to include solution, hence the dependency on the radius (the bigger the radius of the
        accretion disk where you want to find a solution, the bigger the periastron solution is, generally)

        Args:
            _r (float): radius on the accretion disk (BH frame)
            incl (float): inclination of the black hole
            _alpha: angle along the accretion disk (BH frame and observer frame)
            bh_mass (float): mass of the black hole
            midpoint_iterations (int): amount of midpoint iterations to do when searching a periastron value solving eq13
            plot_inbetween (bool): plot
        """

    # angle = (_alpha + n*np.pi) % (2 * np.pi)  # Assert the angle lies in [0, 2 pi]

    def get_plot(X, Y, solution, radius=_r):
        fig = plt.figure()
        plt.title("Eq13(P)\nr={}, a={}".format(radius, round(_alpha, 5)))
        plt.xlabel('P')
        plt.ylabel('Eq13(P)')
        plt.axhline(0, color='black')
        plt.plot(X, Y)
        plt.scatter(solution, 0, color='red')
        return plt

    # TODO: an x_range between [min - 2.*R] seems to suffice for isoradials < 30M, but this is guesstimated
    periastron_range = list(np.linspace(min_periastron, 2. * _r, initial_guesses))
    y_ = [eq13(P_value, _r, _alpha, bh_mass, incl, n) for P_value in periastron_range]  # values of eq13
    ind = np.where(np.diff(np.sign(y_)))[0]  # only one solution should exist
    periastron_solution = periastron_range[ind[0]] if len(ind) else None  # initial guesses for P

    if (periastron_solution is not None) and (not np.isnan(
            periastron_solution)):  # elliptic integral found a periastron solving equation 13
        args_eq13 = {"ir_radius": _r, "ir_angle": _alpha, "bh_mass": bh_mass, "incl": incl, "n": n}
        periastron_solution = \
            improve_solutions_midpoint(func=eq13, args=args_eq13,
                                       x=periastron_range, y=y_, index_of_sign_change=ind[0],
                                       iterations=midpoint_iterations)  # get better P values
    if plot_inbetween:
        get_plot(periastron_range, y_, periastron_solution).show()
    return periastron_solution


def calc_impact_parameter(_r, incl, _alpha, bh_mass, midpoint_iterations=100, plot_inbetween=False,
                          n=0, min_periastron=1., initial_guesses=20, use_ellipse=True) -> float:
    """
    Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value
    This periastron is then converted to an impact parameter b, yielding the observer frame coordinates (b, alpha).
    Does this by generating range of periastron values, evaluating eq13 on this range and using a midpoint method
    to iteratively improve which periastron value solves equation 13.
    The considered initial periastron range must not be lower than min_periastron (i.e. the photon sphere),
    otherwise non-physical solutions will be found. These are interesting in their own right (the equation yields
    complex solutions within radii smaller than the photon sphere!), but are for now outside the scope of this project.
    Must be large enough to include solution, hence the dependency on the radius (the bigger the radius of the
    accretion disk where you want to find a solution, the bigger the periastron solution is, generally)

    Args:
        _r (float): radius on the accretion disk (BH frame)
        incl (float): inclination of the black hole
        _alpha: angle along the accretion disk (BH frame and observer frame)
        bh_mass (float): mass of the black hole
        midpoint_iterations (int): amount of midpoint iterations to do when searching a periastron value solving eq13
        plot_inbetween (bool): plot
    """

    # angle = (_alpha + n*np.pi) % (2 * np.pi)  # Assert the angle lies in [0, 2 pi]

    periastron_solution = calc_periastron(_r, incl, _alpha, bh_mass, midpoint_iterations, plot_inbetween, n,
                                          min_periastron, initial_guesses)
    if periastron_solution is None or periastron_solution <= 2.*bh_mass:
        # No periastron was found, or a periastron was found, but it's non-physical
        # Assume this is because the image of the photon trajectory might have a periastron,
        # but it does not actually move towards this, but away from the black hole
        # these are generally photons at the front of the accretion disk: use the ellipse function
        # (the difference between the two goes to 0 as alpha approaches 0 or 2pi)
        return ellipse(_r, _alpha, incl)
    elif periastron_solution > 2.*bh_mass:
        b = calc_b_from_periastron(periastron_solution, bh_mass)
        return b
    else:
        # Should never happen
        # why was no P found?
        # fig = plt.figure()
        # plt.plot(x_, y_)
        # plt.show()
        raise ValueError(f"No solution was found for the periastron at (r, a) = ({_r}, {_alpha}) and incl={incl}")


def phi_inf(periastron, M):
    q = calc_q(periastron, M)
    ksq = (q - periastron + 6. * M) / (2. * q)
    z_inf = zeta_inf(periastron, M)
    phi = 2. * (np.sqrt(periastron / q)) * (ellipk(ksq) - ellipkinc(z_inf, ksq))
    return phi


def mu(periastron, bh_mass):
    return float(2 * phi_inf(periastron, bh_mass) - np.pi)


def ellipse(r, a, incl) -> float:
    """Equation of an ellipse, reusing the definition of cos_gamma.
    This equation can be used for calculations in the Newtonian limit (large P = b, small a)
    or to visualize the equatorial plane."""
    g = np.arccos(cos_gamma(a, incl))
    b_ = r * np.sin(g)
    return b_


def flux_intrinsic(r, acc, bh_mass):
    r_ = r / bh_mass
    log_arg = ((np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
              ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
    f = (3. * bh_mass * acc / (8 * np.pi)) * (1 / ((r_ - 3) * r ** 2.5)) * \
        (np.sqrt(r_) - np.sqrt(6) + 3 ** -.5 * np.log10(log_arg))
    return f


def flux_observed(r, acc, bh_mass, redshift_factor):
    flux_intr = flux_intrinsic(r, acc, bh_mass)
    return flux_intr / redshift_factor ** 4


def redshift_factor(radius, angle, incl, bh_mass, b_):
    """
    Calculate the gravitational redshift factor (1 + z), ignoring cosmological redshift.
    """
    # WARNING: the paper is absolutely incomprehensible here. Equation 18 for the redshift completely
    # leaves out important factors. It should be:
    # 1 + z = (1 - Ω*b*cos(η)) * (-g_tt -2Ω*g_tϕ - Ω²*g_ϕϕ)^(-1/2)
    # The expressions for the metric components, Ω and the final result of Equation 19 are correct though
    # TODO perhaps implement other metrics? e.g. Kerr, where g_tϕ != 0
    # gff = (radius * np.sin(incl) * np.sin(angle)) ** 2
    # gtt = - (1 - (2. * M) / radius)
    z_factor = (1. + np.sqrt(bh_mass / (radius ** 3)) * b_ * np.sin(incl) * np.sin(angle)) * \
               (1 - 3. * bh_mass / radius) ** -.5
    return z_factor

"""A collection of expressions for equations in Luminet's paper used for generating images."""
import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.special as sp
import sympy as sy

import util


def expr_q() -> sy.Symbol:
    """Generate a sympy expression for Q.

    Returns
    -------
    sy.Symbol
        Symbolic expression for Q
    """
    p, m = sy.symbols("P, M")
    return sy.sqrt((p - 2 * m) * (p + 6 * m))


def expr_b() -> sy.Symbol:
    """Generate a sympy expression for b, the radial coordinate in the observer's frame.

    See equation 5 in Luminet (1977) for reference.

    Note: the original paper has an algebra error here; dimensional analysis indicates that the
    square root of P**3/(P-2M) must be taken.

    Returns
    -------
    sy.Symbol
        Symbolic expression for b
    """
    p, m = sy.symbols("P, M")
    return sy.sqrt((p**3) / (p - 2 * m))


def expr_r_inv() -> sy.Symbol:
    """Generate a sympy expression for 1/r.

    See equation 13 from Luminet (1977) for reference.

    Note: this equation has an algebra error; see the docstring for expr_u.

    Returns
    -------
    sy.Symbol
        Symbolic expression for 1/r
    """
    p, m, q, u, k = sy.symbols("P, M, Q, u, k")
    sn = sy.Function("sn")

    return (1 / (4 * m * p)) * (-(q - p + 2 * m) + (q - p + 6 * m) * sn(u, k**2) ** 2)


def expr_u() -> sy.Symbol:
    """Generate a sympy expression for the argument to sn in equation 13 of Luminet (1977).

    See equation 13 from Luminet (1977) for reference.

    Note: the original paper has an algebra error here; sqrt(P/Q) should _divide_ gamma, not
    multiply it.

    Returns
    -------
    sy.Symbol
        Symbolic expression for the argument of sn
    """
    gamma, z_inf, k, p, q, n = sy.symbols("gamma, zeta_inf, k, P, Q, N")
    return sy.Piecewise(
        (gamma / (2 * sy.sqrt(p / q)) + sy.elliptic_f(z_inf, k**2), sy.Eq(n, 0)),
        (
            (gamma - 2 * n * sy.pi) / (2 * sy.sqrt(p / q))
            - sy.elliptic_f(z_inf, k**2)
            + 2 * sy.elliptic_k(k**2),
            True,
        ),
    )


def expr_gamma() -> sy.Symbol:
    """Generate a sympy expression for gamma, an angle that relates alpha and theta_0.

    See equation 10 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Symbolic expression for gamma
    """
    alpha, theta_0 = sy.symbols("alpha, theta_0")
    return sy.acos(sy.cos(alpha) / sy.sqrt(sy.cos(alpha) ** 2 + sy.tan(theta_0) ** -2))


def expr_k() -> sy.Symbol:
    """Generate a sympy expression for k; k**2 is used as a modulus in the elliptic integrals.

    See equation 12 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Symbolic expression for k
    """
    p, m, q = sy.symbols("P, M, Q")
    return sy.sqrt((q - p + 6 * m) / (2 * q))


def expr_zeta_inf() -> sy.Symbol:
    """Generate a sympy expression for zeta_inf.

    See equation 12 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Symbolic expression of zeta_inf
    """
    p, m, q = sy.symbols("P, M, Q")
    return sy.asin(sy.sqrt((q - p + 2 * m) / (q - p + 6 * m)))


def expr_ellipse() -> sy.Symbol:
    """Generate a sympy expression for an ellipse.

    In the newtonian limit, isoradials form these images.

    Returns
    -------
    sy.Symbol
        Symbolic expression for an ellipse viewed at an inclination of theta_0
    """
    r, alpha, theta_0 = sy.symbols("r, alpha, theta_0")
    return r / sy.sqrt(1 + (sy.tan(theta_0) ** 2) * (sy.cos(alpha) ** 2))


def impact_parameter(
    alpha: npt.NDArray[np.float64],
    r_value: float,
    theta_0: float,
    n: int,
    m: float,
    objective_func: Optional[Callable] = None,
    **root_kwargs
) -> npt.NDArray[np.float64]:
    """Calculate the impact parameter for each value of alpha.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Polar angle in the observer's frame of reference
    r_value : float
        Isoradial distance for which the impact parameter is to be calculated
    theta_0 : float
        Inclination of the observer with respect to the accretion disk plane normal
    n : int
        Order of the calculation; n=0 corresponds to the direct image, n>0 are ghost images
    m : float
        Mass of the black hole
    objective_func : Optional[Callable]
        Objective function whose roots are the periastron distances for a given b(alpha)
    root_kwargs
        Additional arguments are passed to fast_root

    Returns
    -------
    npt.NDArray[np.float64]
        Impact parameter b for each value of alpha. If no root of the objective function for the
        periastron is found for a particular value of alpha, the value of an ellipse is used at that
        point.
    """
    if objective_func is None:
        objective_func = lambda_objective()

    ellipse = lambdify(["r", "alpha", "theta_0"], expr_ellipse())
    b = lambdify(["P", "M"], expr_b())

    p_arr = util.fast_root(
        objective_func,
        np.linspace(2.1, 50, 1000),
        alpha,
        [theta_0, r_value, n, m],
        **root_kwargs
    )
    return np.where(np.isnan(p_arr), ellipse(r_value, alpha, theta_0), b(p_arr, m))


def reorient_alpha(alpha: Union[float, npt.NDArray[float]], n: int) -> float:
    """Reorient the polar angle on the observation coordinate system.

    From Luminet's paper:

        "...the observer will detect generally two images, a direct (or primary) image at polar
        coordinates (b^(d), alpha) and a ghost (or secundary) image at (b^(g), alpha + pi)."

    This function adds pi to the polar angle for ghost images, and returns the original angle for
    direct images.

    Parameters
    ----------
    alpha : float
        Polar angle alpha in the observer's "sensor" coordinate system.
    n : int
        Order of the image which is being calculated. n=0 corresponds to the direct image, while n>0
        corresponds to ghost images.

    Returns
    -------
    float
        Reoriented polar angle
    """
    return np.where(np.asarray(n) > 0, (alpha + np.pi) % (2 * np.pi), alpha)


def lambdify(*args, **kwargs) -> Callable:
    """Lambdify a sympy expression from Luminet's paper.

    Luminet makes use of the sn function, which is one of as Jacobi's elliptic functions. Sympy
    doesn't (yet) support this function, so lambdifying it requires specifying the correct scipy
    routine.

    Arguments are passed diretly to sympy.lambdify; if "modules" is specified, the user must specify
    which function to call for 'sn'.

    Parameters
    ----------
    *args
        Arguments are passed to sympy.lambdify
    **kwargs
        Additional kwargs passed to sympy.lambdify

    Returns
    -------
    Callable
        Lambdified expression
    """
    kwargs["modules"] = kwargs.get(
        "modules",
        [
            "numpy",
            {
                "sn": lambda u, m: sp.ellipj(u, m)[0],
                "elliptic_f": lambda phi, m: sp.ellipkinc(phi, m),
                "elliptic_k": lambda m: sp.ellipk(m),
            },
        ],
    )
    return sy.lambdify(*args, **kwargs)


def objective() -> sy.Symbol:
    """Generate a sympy expression for the objective function.

    The objective function has roots which are periastron distances for isoradials.

    Returns
    -------
    sy.Symbol
        Symbolic expression for the objective function
    """
    r = sy.symbols("r")
    return 1 - r * expr_r_inv()


def lambda_objective() -> Callable[[float, float, float, float, int, float], float]:
    """Generate a lambdified objective function.

    Returns
    -------
    Callable[(float, float, float, float, int, float), float]
        Objective function whose roots yield periastron distances for isoradials. The function
        signature is

            s(P, alpha, theta_0, r, n, m)
    """
    s = (
        objective()
        .subs({"u": expr_u()})
        .subs({"zeta_inf": expr_zeta_inf()})
        .subs({"gamma": expr_gamma()})
        .subs({"k": expr_k()})
        .subs({"Q": expr_q()})
    )
    return lambdify(("P", "alpha", "theta_0", "r", "N", "M"), s)


def expr_fs() -> sy.Symbol:
    """Generate an expression for the flux of an accreting disk.

    See equation 15 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Sympy expression for Fs, the radiation flux of an accreting disk
    """
    m, rstar, mdot = sy.symbols(r"M, r^*, \dot{m}")

    return (
        ((3 * m * mdot) / (8 * sy.pi))
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


def expr_r_star() -> sy.Symbol:
    """Generate an expression for r^*, the radial coordinate normalized by the black hole mass.

    Returns
    -------
    sy.Symbol
        Sympy expression for the radial coordinate normalized by the black hole mass.
    """
    m, r = sy.symbols("M, r")
    return r / m


def expr_one_plus_z() -> sy.Symbol:
    """Generate an expression for the redshift 1+z.

    See equation 19 in Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol()
        Sympy expression for the redshift of the accretion disk
    """
    m, r, theta_0, alpha, b = sy.symbols("M, r, theta_0, alpha, b")
    return (1 + sy.sqrt(m / r**3) * b * sy.sin(theta_0) * sy.sin(alpha)) / sy.sqrt(
        1 - 3 * m / r
    )


def expr_f0() -> sy.Symbol:
    """Generate an expression for the observed bolometric flux.

    Returns
    -------
    sy.Symbol
        Sympy expression for the raw bolometric flux.
    """
    fs, opz = sy.symbols("F_s, z_op")
    return fs / opz**4


def expr_f0_normalized() -> sy.Symbol:
    """Generate an expression for the normalized observed bolometric flux.

    Units are in (8*pi)/(3*M*Mdot).

    Returns
    -------
    sy.Symbol
        Sympy expression for the normalized bolometric flux.
    """
    m, mdot = sy.symbols(r"M, \dot{m}")
    return expr_f0() / ((8 * sy.pi) / (3 * m * mdot))


def lambda_normalized_bolometric_flux() -> Callable[[float, float, float], float]:
    """Generate the normalized bolometric flux function.

    See `generate_image` for an example of how to use this.

    Returns
    -------
    Callable[(float, float, float), float]
        The returned function takes (1+z, r, M) as arguments and outputs the normalized bolometric
        flux of the black hole.
    """
    return sy.lambdify(
        ("z_op", "r", "M"),
        (
            expr_f0()
            .subs({"F_s": expr_fs()})
            .subs({"M": 1, r"\dot{m}": 1})
            .subs({"r^*": expr_r_star()})
        )
        / (3 / (8 * sy.pi)),
    )


def simulate_flux(
    alpha: npt.NDArray[np.float64],
    r: float,
    theta_0: float,
    n: int,
    m: float,
    objective_func: Optional[Callable] = None,
    root_kwargs: Optional[Dict[Any, Any]] = None,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Simulate the bolometric flux for an accretion disk near a black hole.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Polar angle in the observer's frame of reference
    r : float
        Isoradial distance for which the impact parameter is to be calculated
    theta_0 : float
        Inclination of the observer with respect to the accretion disk plane normal
    n : int
        Order of the calculation; n=0 corresponds to the direct image, n>0 are ghost images
    m : float
        Mass of the black hole
    root_kwargs : Dict[Any, Any]
        Additional arguments are passed to fast_root
    objective_func : Optional[Callable]
        Objective function whose roots are the periastron distances for a given b(alpha)

    Returns
    -------
    Tuple[npt.NDArray[np.float64], ...]
        reoriented alpha, b, 1+z, and observed bolometric flux
    """
    flux = lambda_normalized_bolometric_flux()
    one_plus_z = sy.lambdify(["alpha", "b", "theta_0", "M", "r"], expr_one_plus_z())
    root_kwargs = root_kwargs if root_kwargs else {}

    b = impact_parameter(alpha, r, theta_0, n, m, objective_func, **root_kwargs)
    opz = one_plus_z(alpha, b, theta_0, m, r)

    return reorient_alpha(alpha, n), b, opz, flux(opz, r, m)


def generate_image_data(
    alpha: npt.NDArray[np.float64],
    r_vals: Iterable[float],
    theta_0: float,
    n_vals: Iterable[int],
    m: float,
    root_kwargs,
) -> pd.DataFrame:
    """Generate the data needed to produce an image of a black hole.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Alpha values at which the bolometric flux is to be simulated; angular coordinate in the
        observer's frame of reference
    r_vals : Iterable[float]
        Orbital radius of a section of the accretion disk from the center of the black hole
    theta_0 : float
        Inclination of the observer, in radians, with respect to the the normal of the accretion
        disk
    n_vals : Iterable[int]
        Order of the calculation; n=0 corresponds to the direct image, n>0 are ghost images
    m : float
        Mass of the black hole
    root_kwargs : Dict
        All other kwargs are passed to the `impact_parameter` function

    Returns
    -------
    pd.DataFrame
        Simulated data; columns are alpha, b, 1+z, r, n, flux, x, y
    """
    a_arrs = []
    b_arrs = []
    opz_arrs = []
    r_arrs = []
    n_arrs = []
    flux_arrs = []

    opz = sy.lambdify(["alpha", "b", "theta_0", "M", "r"], expr_one_plus_z())

    for n in n_vals:
        with mp.Pool(mp.cpu_count()) as pool:
            args = [(alpha, r, theta_0, n, m, None, root_kwargs) for r in r_vals]
            for r, (alpha_reoriented, b, opz, flux) in zip(
                r_vals, pool.starmap(simulate_flux, args)
            ):
                a_arrs.append(alpha_reoriented)
                b_arrs.append(b)
                opz_arrs.append(opz)
                r_arrs.append(np.full(b.size, r))
                n_arrs.append(np.full(b.size, n))
                flux_arrs.append(flux)

    df = pd.DataFrame(
        {
            "alpha": np.concatenate(a_arrs),
            "b": np.concatenate(b_arrs),
            "opz": np.concatenate(opz_arrs),
            "r": np.concatenate(r_arrs),
            "n": np.concatenate(n_arrs),
            "flux": np.concatenate(flux_arrs),
        }
    )

    df["x"] = df["b"] * np.cos(df["alpha"])
    df["y"] = df["b"] * np.sin(df["alpha"])
    return df


if __name__ == '__main__':
    M = 1
    solver_params = {'initial_guesses': 10,
                     'midpoint_iterations': 10,
                     'plot_inbetween': False,
                     'minP': 3.01 * M}
    # writeFramesEq13(5, solver_params=solver_params)
