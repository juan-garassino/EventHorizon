import os.path
import pandas as pd
import matplotlib.cm as cm
import matplotlib.collections as mcoll
import matplotlib.image as img
from collections import OrderedDict
from bh_math import *
from bh_utils import *
from tqdm import tqdm
import configparser

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors

class Isoradial:
    def __init__(self, radius, incl, bh_mass, order=0, params=None, plot_params=None, angular_properties=None):
        self.M = bh_mass  # mass of the black hole containing this isoradial
        self.t = incl  # inclination of observer's plane
        self.radius = radius
        self.order = order
        self.params = params if params is not None else {}
        self.angular_properties = params["isoradial_angular_parameters"] if params \
            else angular_properties if angular_properties \
            else {}
        self.solver_params = params["isoradial_solver_parameters"] if params else self.__read_default_solver_params()
        self.find_redshift_params = {
            'force_redshift_solution': False,  # force finding a redshift solution on the isoradial
            'max_force_iter': 5  # only make this amount of iterations when forcing finding a solution
        }
        self.plot_params = plot_params if plot_params else \
            {'save_plot': True,
             'plot_ellipse': False,
             'redshift': False,
             'linestyle': '-',
             'key': "",
             'face_color': 'black',
             'line_color': 'white',
             'text_color': 'white',
             'alpha': 1.,
             'show_grid': False,
             'orig_background': False,
             'legend': False,
             'title': "Isoradials for R = {}".format(radius)}  # default values
        self.radii_b = []
        self.angles = []
        self.cartesian_co = self.X, self.Y = [], []
        self.redshift_factors = []

        self.calculate()

    def __read_default_solver_params(self):
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('parameters.ini')
        return {key: eval(val) for key, val in config["solver_parameters"].items()}

    def calculate_coordinates(self, _tqdm=False):
        """Calculates the angles (alpha) and radii (b) of the photons emitted at radius self.radius as they would appear
        on the observer's photographic plate. Also saves the corresponding values for the impact parameters (P).

        Args:

        Returns:
            tuple: Tuple containing the angles (alpha) and radii (b) for the image on the observer's photographic plate
        """

        start_angle = self.angular_properties['start_angle']
        end_angle = self.angular_properties['end_angle']
        angular_precision = self.angular_properties['angular_precision']

        angles = []
        impact_parameters = []
        t = np.linspace(start_angle, end_angle, angular_precision)
        if _tqdm:
            t = tqdm(t, desc='Calculating isoradial R = {}'.format(self.radius), position=2, leave=False)
        for alpha_ in t:
            b_ = calc_impact_parameter(self.radius, self.t, alpha_, self.M, n=self.order, **self.solver_params)
            if b_ is not None:
                angles.append(alpha_)
                impact_parameters.append(b_)
        if self.order > 0:  # TODO: fix dirty manual flip for ghost images
            angles = [a_ + np.pi for a_ in angles]

        # flip image if necessary
        if self.t > np.pi / 2:
            angles = [(a_ + np.pi) % (2 * np.pi) for a_ in angles]
        if self.angular_properties['mirror']:  # by default True. Halves computation time for calculating full isoradial
            # add second half of image (left half if 0° is set at South)
            angles += [(2 * np.pi - a_) % (2 * np.pi) for a_ in angles[::-1]]
            impact_parameters += impact_parameters[::-1]
        self.angles = angles
        self.radii_b = impact_parameters
        self.X, self.Y = polar_to_cartesian_lists(self.radii_b, self.angles, rotation=-np.pi / 2)
        self.cartesian_co = self.X, self.Y
        return angles, impact_parameters

    def calc_redshift_factors(self):
        """Calculates the redshift factor (1 + z) over the line of the isoradial"""
        redshift_factors = [redshift_factor(radius=self.radius, angle=angle, incl=self.t, bh_mass=self.M, b_=b_)
                            for b_, angle in zip(self.radii_b, self.angles)]
        self.redshift_factors = redshift_factors
        return redshift_factors

    def calculate(self):
        self.calculate_coordinates()
        self.calc_redshift_factors()

    def find_angle(self, z) -> [int]:
        """Returns angle at which the isoradial redshift equals some value z
        Args:
            z: The redshift value z. Do not confuse with redshift factor 1 + z"""
        indices = np.where(np.diff(np.sign([redshift - z - 1 for redshift in self.redshift_factors])))[0]
        return [self.angles[i] for i in indices if len(indices)]

    def get_b_from_angle(self, angle: float):
        # TODO: this method only works if angles augment from index 0 to end
        # if image is flipped, then the mod operator makes it so they jump back to 0 about halfway
        # yielding a fake intersection
        d = [abs(a_ % (2 * np.pi) - angle % (2 * np.pi)) for a_ in self.angles]
        mn = min(d)
        res = [i for i, val in enumerate(d) if val == mn]
        return self.radii_b[res[0]] if len(res) else None

    def plot(self, _ax=None, plot_params=None, show=False, colornorm=(0, 1)):
        def make_segments(x, y):
            """
            Create list of line segments from x and y coordinates, in the correct format
            for LineCollection: an array of the form numlines x (points per line) x 2 (x
            and y) array
            """

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            return segments

        def colorline(
                __ax, __x, __y, z=None, cmap=plt.get_cmap('RdBu_r'), norm=plt.Normalize(*colornorm),
                linewidth=3):
            """
            http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
            http://matplotlib.org/examples/pylab_examples/multicolored_line.html
            Plot a colored line with coordinates x and y
            Optionally specify colors in the array z
            Optionally specify a colormap, a norm function and a line width
            """

            # Default colors equally spaced on [0,1]:
            if z is None:
                z = np.linspace(0.0, 1.0, len(__x))

            # Special case if a single number:
            if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
                z = np.array([z])

            z = np.asarray(z)

            segments = make_segments(__x, __y)
            lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                                      linewidth=linewidth, alpha=self.plot_params['alpha'])
            lc.set_array(z)
            __ax.add_collection(lc)
            # mx = max(segments[:][:, 1].flatten())
            # _ax.set_ylim((0, mx))
            return __ax

        if not _ax:
            ir_fig = plt.figure(figsize=(5, 5))
            ir_ax = ir_fig.add_subplot()
        else:
            ir_ax = _ax

        if not plot_params:
            plot_params = self.plot_params

        # Plot isoradial
        if self.plot_params['redshift']:
            ir_ax = colorline(ir_ax, self.X, self.Y, z=[e - 1 for e in self.redshift_factors],
                              cmap=cm.get_cmap('RdBu_r'))  # red-blue colormap reversed to match redshift
        else:
            ir_ax.plot(self.X, self.Y, color=plot_params['line_color'],
                       alpha=plot_params['alpha'], linestyle=self.plot_params['linestyle'])
        if self.plot_params['legend']:
            plt.legend(prop={'size': 16})
        if len(self.X) and len(self.Y):
            mx = np.max([np.max(self.X), np.max(self.Y)])
            mx *= 1.1
            ir_ax.set_xlim([-mx, mx])
            ir_ax.set_ylim([-mx, mx])
        if show:
            # ax.autoscale_view(scalex=False)
            # ax.set_ylim([0, ax.get_ylim()[1] * 1.1])
            plt.show()
        return plt, ir_ax

    def calc_between(self, ind):
        """
        Calculates the impact parameter and redshift factor at the
        isoradial angle between place ind and ind + 1

        Args:
            ind: the index denoting the location at which the middle point should be calculated. The impact parameter,
            redshift factor, b (observer plane) and alpha (observer/BH coordinate system) will be calculated on the
            isoradial between location ind and ind + 1

        Returns:
            None: Nothing. Updates the isoradial.
        """
        mid_angle = .5 * (self.angles[ind] + self.angles[ind + 1])
        b_ = calc_impact_parameter(self.radius, self.t, mid_angle, self.M, **self.solver_params)
        z_ = redshift_factor(self.radius, mid_angle, self.t, self.M, b_)
        self.radii_b.insert(ind + 1, b_)
        self.angles.insert(ind + 1, mid_angle)
        self.redshift_factors.insert(ind + 1, z_)

    def force_intersection(self, redshift):
        # TODO: improve this method, currently does not seem to work
        """
        If you know a redshift should exist on the isoradial, use this function to calculate the isoradial until
        it finds it. Useful for when the redshift you're looking for equals (or is close to) the maximum
        redshift along some isoradial line.

        Only works if the redshift can be found within the isoradial begin and end angle.
        """

        if len(self.angles) == 2:
            self.calc_between(0)
        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        cross = np.where(np.diff(np.sign(diff)))[0]
        if len(cross):
            return diff  # intersection is found

        it = 0
        while len(cross) == 0 and it < self.find_redshift_params['max_force_iter']:
            # calc derivatives
            delta = [e - b for b, e in zip(self.redshift_factors[:-1], self.redshift_factors[1:])]
            # where does the redshift go back up/down before it reaches the redshift we want to find
            initial_guess_indices = np.where(np.diff(np.sign(delta)))[0]
            new_ind = initial_guess_indices[0]  # initialize the initial guess.
            self.calc_between(new_ind)  # insert more accurate solution
            diff = [redshift + 1 - z_ for z_ in self.redshift_factors]  # calc new interval
            cross = np.where(np.diff(np.sign(diff)))[0]
            it += 1
            # plt.plot(self.angles, [redshift + 1 - z_ for z_ in self.redshift_factors])
            # plt.axvline(0)
            # plt.show()
        return diff

    def calc_redshift_location_on_ir(self, redshift, cartesian=False):
        """
        Calculates which location on the isoradial has some redshift value (not redshift factor)
        Doest this by means of a midpoint method, with midpoint_steps steps (defined in parameters.ini).
        The (b, alpha, z) coordinates of the isoradial are calculated closer and closer to the desired z.
        It does not matter all that much how high the isoradial resolution is, since midpoint_steps is
        much more important to find an accurate location.
        """

        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        # if self.find_redshift_params['force_redshift_solution']:
        #     pass  # TODO, force_intersection does not always seem to work
        #     diff = self.force_intersection(redshift)
        initial_guess_indices = np.where(np.diff(np.sign(diff)))[0]

        angle_solutions = []
        b_solutions = []
        if len(initial_guess_indices):
            for s in range(len(initial_guess_indices)):  # generally, two solutions exists on a single isoradial
                new_ind = initial_guess_indices[s]  # initialize the initial guess.
                for _ in range(self.solver_params["midpoint_iterations"]):
                    self.calc_between(new_ind)  # insert more accurate solution
                    diff_ = [redshift + 1 - z_ for z_ in
                             self.redshift_factors[new_ind:new_ind + 3]]  # calc new interval
                    start = np.where(np.diff(np.sign(diff_)))[0]  # returns index where the sign changes
                    new_ind += start[0]  # index of new redshift solution in refined isoradial
                # append average values of final interval
                angle_solutions.append(.5 * (self.angles[new_ind] + self.angles[new_ind + 1]))
                b_solutions.append(.5 * (self.radii_b[new_ind] + self.radii_b[new_ind + 1]))
                # update the initial guess indices, as the indexing has changed due to inserted solutions
                initial_guess_indices = [e + self.solver_params["midpoint_iterations"] for e in initial_guess_indices]
            if cartesian:
                return polar_to_cartesian_lists(b_solutions, angle_solutions)
        return angle_solutions, b_solutions

    def plot_redshift(self, fig=None, ax=None, show=True):
        """
        Plots the redshift values along the isoradial line in function of the angle<
        """
        fig_ = fig if fig else plt.figure()
        ax_ = ax if ax else fig_.add_subplot()
        ax_.plot(self.angles, [z - 1 for z in self.redshift_factors])
        plt.title("Redshift values for isoradial\nR={} | M = {}".format(20, M))
        ax_.set_xlim([0, 2 * np.pi])
        if show:
            plt.show()
