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
from isoradials import Isoradial
from isoredshift import Isoredshift

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


class BlackHole:
    def __init__(self, mass=1., inclination=80, acc=10e-8):
        """Initialise black hole with mass and accretion rate
        Set viewer inclination above equatorial plane
        """
        self.t = inclination * np.pi / 180
        self.M = mass
        self.acc = acc  # accretion rate
        self.critical_b = 3 * np.sqrt(3) * self.M
        self.settings = {}  # All settings: see below
        self.plot_params = {}
        self.ir_parameters = {}
        self.angular_properties = {}
        self.iz_solver_params = {}
        self.solver_params = {}
        self.__read_parameters()

        self.disk_outer_edge = 50. * self.M
        self.disk_inner_edge = 6. * self.M
        self.disk_apparent_outer_edge = self.calc_apparent_outer_disk_edge()  # outer edge after curving through spacetime
        self.disk_apparent_inner_edge = self.calc_apparent_inner_disk_edge()  # inner edge after curving through spacetime

        self.isoradials = {}
        self.isoredshifts = {}

    def __read_parameters(self):
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('parameters.ini')
        for i, section in enumerate(config.sections()):
            self.settings[section] = {key: eval(val) for key, val in config[section].items()}
        self.plot_params = self.settings["plot_params"]
        self.ir_parameters["isoradial_angular_parameters"] = self.angular_properties = self.settings[
            "isoradial_angular_parameters"]
        self.ir_parameters["isoradial_solver_parameters"] = self.solver_params = self.settings["solver_parameters"]
        self.iz_solver_params = self.settings["isoredshift_solver_parameters"]

    def apparent_inner_edge(self, cartesian=True, scale=.99):
        """
        The apparent inner edge of the black hole (not the apparent inner disk edge). THis takes eventual ghost images
        into account
        """
        b = []
        a = []
        for a_ in np.linspace(0, 2 * np.pi, self.angular_properties["angular_precision"]):
            a.append(a_)
            if np.pi / 2 < a_ < 3 * np.pi / 2:
                b.append(self.critical_b * scale)
            else:
                b_ = min(self.critical_b, self.disk_apparent_inner_edge.get_b_from_angle(a_))
                b.append(b_ * scale)
        if not cartesian:
            return b, a
        else:
            return polar_to_cartesian_lists(b, a, rotation=-np.pi / 2)

    def plot_photon_sphere(self, _ax, c='red'):
        # plot black hole itself
        x, y = self.apparent_inner_edge(cartesian=True)
        _ax.plot(x, y, color=c, zorder=0)
        # plot critical value of b
        x_, y_ = polar_to_cartesian_lists([5.2] * 2 * self.angular_properties["angular_precision"],
                                          np.linspace(-np.pi, np.pi, 2 * self.angular_properties["angular_precision"]))
        _ax.fill(x_, y_, facecolor="none", edgecolor='white', zorder=0, hatch='\\\\\\\\', alpha=.5, linewidth=.5)
        # plot black hole itself
        x_, y_ = polar_to_cartesian_lists([2 * self.M] * 2 * self.angular_properties["angular_precision"],
                                          np.linspace(-np.pi, np.pi, 2 * self.angular_properties["angular_precision"]))
        _ax.fill(x_, y_, facecolor='none', zorder=0, edgecolor='white', hatch='////')
        return _ax

    def calc_apparent_inner_disk_edge(self):
        ir = Isoradial(radius=self.disk_inner_edge, incl=self.t, order=0,
                       params=self.ir_parameters, bh_mass=self.M)
        ir.radii_b = [.99 * b for b in ir.radii_b]  # scale slightly down?  # TODO
        ir.X, ir.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)
        return ir

    def calc_apparent_outer_disk_edge(self):
        ir = Isoradial(radius=self.disk_outer_edge, incl=self.t, order=0,
                       params=self.ir_parameters, bh_mass=self.M)
        ir.X, ir.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)
        return ir

    def get_apparent_outer_edge_radius(self, angle, rotation=0):
        return self.disk_apparent_outer_edge.get_b_from_angle(angle + rotation)

    def get_apparent_inner_edge_radius(self, angle, rotation=0):
        return self.disk_apparent_inner_edge.get_b_from_angle(angle + rotation)

    def plot_apparent_inner_edge(self, _ax, linestyle='--'):
        # plot black hole (photon sphere)
        # TODO why don't I use the Isoradial class for this?
        x = []
        y = []
        impact_parameters = []
        angles = np.linspace(0, 2 * np.pi, 2 * self.angular_properties["angular_precision"])
        for a in angles:
            b = self.get_apparent_inner_edge_radius(a)
            impact_parameters.append(b)
            rot = -np.pi / 2 if self.t < np.pi / 2 else np.pi / 2
            x_, y_ = polar_to_cartesian_lists([b], [a], rotation=rot)
            x.append(x_)
            y.append(y_)

        _ax.plot(x, y, zorder=0, linestyle=linestyle, linewidth=2. * self.plot_params["linewidth"])
        return _ax

    def get_figure(self):
        _fig = plt.figure(figsize=(10, 10))
        _ax = _fig.add_subplot(111)
        plt.axis('off')  # command for hiding the axis.
        _fig.patch.set_facecolor(self.plot_params['face_color'])
        _ax.set_facecolor(self.plot_params['face_color'])
        if self.plot_params['show_grid']:
            _ax.grid(color='grey')
            _ax.tick_params(which='both', labelcolor=self.plot_params['text_color'],
                            labelsize=15)
        else:
            _ax.grid()
        _ax.set_ylim(self.plot_params["ax_lim"])
        _ax.set_xlim(self.plot_params["ax_lim"])
        return _fig, _ax

    def calc_isoredshifts(self, redshifts=None):
        if redshifts is None:
            redshifts = [-.15, 0., .1, .20, .5]

        def get_dirty_isoradials(__bh):
            # an array of quick and dirty isoradials for the initial guesses of redshifts
            isoradials = []  # for initial guesses
            for radius in np.linspace(__bh.disk_inner_edge, __bh.disk_outer_edge,
                                      __bh.iz_solver_params["initial_radial_precision"]):
                isoradial = Isoradial(radius, __bh.t, __bh.M,
                                      params=__bh.ir_parameters)
                isoradials.append(isoradial)
            return isoradials

        dirty_isoradials = get_dirty_isoradials(self)
        t = tqdm(redshifts, desc="Calculating redshift", position=0)
        for redshift in t:
            t.set_description("Calculating redshift {}".format(redshift))
            dirty_ir_copy = dirty_isoradials.copy()
            # spawn an isoredshift instance and calc coordinates based on dirty isoradials
            iz = Isoredshift(inclination=self.t, redshift=redshift, bh_mass=self.M,
                             solver_parameters=self.iz_solver_params, from_isoradials=dirty_ir_copy)
            # iteratively improve coordinates and closing tip of isoredshift
            iz.improve()
            self.isoredshifts[redshift] = iz
        return self.isoredshifts

    def add_isoradial(self, isoradial, radius, order):
        """
        Add isoradial to dict of isoradials. Each key is a radius corresponding to
        some set of isoradials. Each value is again a dict, with as keys the order
        of the isoradial (usually just 0 for direct and 1 for ghost image)
        """
        if radius in self.isoradials.keys():
            self.isoradials[radius][order] = isoradial
        else:
            self.isoradials[radius] = {order: isoradial}

    def calc_isoradials(self, direct_r: [], ghost_r: []):
        progress_bar = tqdm(range(len(direct_r) + len(ghost_r)), position=0, leave=False)
        # calc ghost images
        progress_bar.set_description("Ghost images")
        self.plot_params['alpha'] = .5
        for radius in sorted(ghost_r):
            progress_bar.update(1)
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=1,
                                  params=self.ir_parameters, plot_params=self.plot_params)
            self.add_isoradial(isoradial, radius, 1)

        # calc direct images
        progress_bar.set_description("Direct images")
        self.plot_params['alpha'] = 1.
        for radius in sorted(direct_r):
            progress_bar.update(1)
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=0,
                                  params=self.ir_parameters, plot_params=self.plot_params)
            self.add_isoradial(isoradial, radius, 0)

    def plot_isoradials(self, direct_r: [], ghost_r: [], show=False):
        """Given an array of radii for the direct image and/or ghost image, plots the corresponding
        isoradials.
        Calculates the isoradials according to self.root_params
        Plots the isoradials according to self.plot_params"""

        def plot_ellipse(__r, __ax, incl):
            ax_ = __ax
            a = np.linspace(-np.pi, np.pi, 2 * self.angular_properties['angular_precision'])
            ell = [ellipse(__r, a_, incl) for a_ in a]
            x, y = polar_to_cartesian_lists(ell, a)
            ax_.plot(x, y, color='red', zorder=-1)
            return ax_

        self.calc_isoradials(direct_r, ghost_r)
        _fig, _ax = self.get_figure()
        color_range = (-1, 1)

        # plot background
        if self.plot_params['orig_background']:
            image = img.imread('bh_background.png')
            scale = (940 / 30 * 2. * M)  # 940 px by 940 px, and 2M ~ 30px
            _ax.imshow(image, extent=(-scale / 2, scale / 2, -scale / 2, scale / 2))
        else:
            _ax.set_facecolor('black')

        # plot ghost images
        self.plot_params['alpha'] = .5
        for radius in sorted(ghost_r):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = self.isoradials[radius][1]
            plt_, _ax = isoradial.plot(_ax, self.plot_params, colornorm=color_range)

        # plot direct images
        self.plot_params['alpha'] = 1.
        for radius in sorted(direct_r):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = self.isoradials[radius][0]
            plt_, _ax = isoradial.plot(_ax, self.plot_params, colornorm=color_range)

        if self.plot_params['plot_ellipse']:  # plot ellipse
            for radius in direct_r:
                _ax = plot_ellipse(radius, _ax, self.t)
        if self.plot_params['plot_core']:
            _ax = self.plot_apparent_inner_edge(_ax, 'red')

        plt.title(f"Isoradials for M={self.M}", color=self.plot_params['text_color'])
        if show:
            plt.show()
        if self.plot_params['save_plot']:
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            _fig.savefig(name, dpi=300, facecolor=self.plot_params['face_color'])
        return _fig, _ax

    def write_frames(self, func, direct_r=None, ghost_r=None, step_size=5):
        """
        Given some function that produces  fig and ax, this method sets increasing values for the inclination,
        plots said function and write it out as a frame.
        """
        if ghost_r is None:
            ghost_r = [6, 10, 20, 30]
        if direct_r is None:
            direct_r = [6, 10, 20, 30]
        steps = np.linspace(0, 180, 1 + (0 - 180) // step_size)
        for a in tqdm(steps, position=0, desc='Writing frames'):
            self.t = a
            bh.plot_params['title'] = 'inclination = {:03}°'.format(int(a))
            fig_, ax_ = func(direct_r, ghost_r, ax_lim=self.plot_params["ax_lim"])
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            fig_.savefig('movie/' + name, dpi=300, facecolor=self.plot_params['face_color'])
            plt.close()  # to not destroy your RAM

    def plot_isoredshifts(self, redshifts=None, plot_core=False):
        if redshifts is None:
            redshifts = [-.2, -.15, 0., .15, .25, .5, .75, 1.]
        _fig, _ax = self.get_figure()  # make new figure

        bh.calc_isoredshifts(redshifts=redshifts).values()

        for redshift, irz in self.isoredshifts.items():
            r_w_s, r_wo_s = irz.split_co_on_solutions()
            if len(r_w_s.keys()):
                split_index = irz.split_co_on_jump()
                if split_index is not None:
                    plt.plot(irz.y[:split_index], [-e for e in irz.x][:split_index],
                             linewidth=self.plot_params["linewidth"])
                    plt.plot(irz.y[split_index + 1:], [-e for e in irz.x][split_index + 1:],
                             linewidth=self.plot_params["linewidth"])
                else:
                    plt.plot(irz.y, [-e for e in irz.x],
                             linewidth=self.plot_params["linewidth"])  # todo: why do i need to flip x

        if plot_core:
            _ax = self.plot_apparent_inner_edge(_ax, linestyle='-')
        plt.suptitle("Isoredshift lines for M={}".format(self.M))
        plt.show()
        return _fig, _ax

    def sample_points(self, n_points=1000, f=None, f2=None):
        """
        # TODO: sample separately for direct and ghost image?
        Samples points on the accretion disk. This sampling is not done uniformly, but a bias is added towards the
        center of the accretion disk, as the observed flux is exponentially bigger here and this needs the most
        precision.
        Both the direct and ghost image for each point is calculated. It's coordinates (polar and cartesian),
        redshift and
        :param min_radius:
        :param max_radius:
        :param n_points: Amount of points to sample. 10k takes about 6 minutes and gives ok precision mostly
        :param f:
        :param f2:
        :return:
        """
        if f is None:
            f = f"Points/points_incl={int(self.t * 180 / np.pi)}.csv"
        if f2 is None:
            f2 = f"Points/points_secondary_incl={int(self.t * 180 / np.pi)}.csv"
        df = pd.read_csv(f, index_col=0) if os.path.exists('./{}'.format(f)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'angle', 'z_factor', 'flux_o'])
        df2 = pd.read_csv(f2, index_col=0) if os.path.exists('./{}'.format(f2)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'angle', 'z_factor', 'flux_o'])

        min_radius_ = self.disk_inner_edge
        max_radius_ = self.disk_outer_edge
        t = tqdm(range(n_points), desc="Sampling points for direct and ghost image")
        for _ in t:
            t.update(1)
            # r = minR_ + maxR_ * np.sqrt(np.random.random())  # uniformly sampling a circle's surface
            theta = np.random.random() * 2 * np.pi
            r = min_radius_ + max_radius_ * np.random.random()  # bias towards center (where the interesting stuff is)
            b_ = calc_impact_parameter(r, incl=self.t, _alpha=theta, bh_mass=self.M, **self.solver_params)
            b_2 = calc_impact_parameter(r, incl=self.t, _alpha=theta, bh_mass=self.M, **self.solver_params, n=1)
            if b_ is not None:
                x, y = polar_to_cartesian_lists([b_], [theta], rotation=-np.pi / 2)
                redshift_factor_ = redshift_factor(r, theta, self.t, self.M, b_)
                f_o = flux_observed(r, self.acc, self.M, redshift_factor_)
                df = pd.concat([df,
                                pd.DataFrame.from_dict({'X': x, 'Y': y, 'impact_parameter': b_,
                                                        'angle': theta,
                                                        'z_factor': redshift_factor_, 'flux_o': f_o})])
            if b_2 is not None:
                x, y = polar_to_cartesian_lists([b_2], [theta], rotation=-np.pi / 2)
                redshift_factor_2 = redshift_factor(r, theta, self.t, self.M, b_2)
                F_o2 = flux_observed(r, self.acc, self.M, redshift_factor_2)
                df2 = pd.concat([df2,
                                 pd.DataFrame.from_dict({'X': x, 'Y': y, 'impact_parameter': b_2,
                                                         'angle': theta,
                                                         'z_factor': redshift_factor_2, 'flux_o': F_o2})
                                 ])
        df.to_csv(f)
        df2.to_csv(f2)

    def plot_points(self, power_scale=.9, levels=100):
        """
        Plot the points written out by samplePoints()
        :param levels: amount of levels in matplotlib contour plot
        :param power_scale: powers_cale to apply to flux. No power_scale = 1. Anything lower than 1 will make the
        dim points pop out more.
        :return:
        """

        def plot_direct_image(_ax, points, _levels, _min_flux, _max_flux, _power_scale):
            # direct image
            points.sort_values(by="angle", inplace=True)
            points_ = points.iloc[[b_ <= self.get_apparent_outer_edge_radius(a_) for b_, a_ in
                                   zip(points["impact_parameter"], points["angle"])]]
            fluxes = [(abs(fl + _min_flux) / (_max_flux + _min_flux)) ** _power_scale for fl in points_['flux_o']]
            _ax.tricontourf(points_['X'], points_['Y'], fluxes, cmap='Greys_r',
                            levels=_levels, norm=plt.Normalize(0, 1), nchunk=2
                            )
            br = self.calc_apparent_inner_disk_edge()
            _ax.fill_between(br.X, br.Y, color='black', zorder=1)  # to fill Delauney triangulation artefacts with black
            return _ax

        def plot_ghost_image(_ax, points, _levels, _min_flux, _max_flux, _power_scale):
            # ghost image
            points_inner = points.iloc[[b_ < self.get_apparent_inner_edge_radius(a_ + np.pi) for b_, a_ in
                                        zip(points["impact_parameter"], points["angle"])]]
            points_outer = points.iloc[[b_ > self.get_apparent_outer_edge_radius(a_ + np.pi) for b_, a_ in
                                        zip(points["impact_parameter"], points["angle"])]]
            # _ax.plot(self.disk_apparent_inner_edge.X, self.disk_apparent_inner_edge.Y)
            for i, points_ in enumerate([points_inner, points_outer]):
                points_.sort_values(by=['flux_o'], ascending=False)
                fluxes = [(abs(fl + _min_flux) / (_max_flux + _min_flux)) ** _power_scale for fl in
                          points_['flux_o']]
                _ax.tricontourf(points_['X'], [-e for e in points_['Y']], fluxes, cmap='Greys_r',
                                norm=plt.Normalize(0, 1), levels=_levels, nchunk=2, zorder=1 - i
                                )
            x, y = self.apparent_inner_edge(cartesian=True)
            _ax.fill_between(x, y, color='black', zorder=1)  # to fill Delauney triangulation artefacts with black
            x, y = self.calc_apparent_outer_disk_edge().cartesian_co
            _ax.fill_between(x, y, color='black', zorder=0)  # to fill Delauney triangulation artefacts with black
            return _ax

        _fig, _ax = self.get_figure()
        if self.plot_params["plot_disk_edges"]:
            _ax.plot(self.disk_apparent_outer_edge.X, self.disk_apparent_outer_edge.Y, zorder=4)
            _ax.plot(self.disk_apparent_inner_edge.X, self.disk_apparent_inner_edge.Y, zorder=4)

        points1 = pd.read_csv(f"Points/points_incl={round(self.t * 180 / np.pi)}.csv")
        points2 = pd.read_csv(f"Points/points_secondary_incl={round(self.t * 180 / np.pi)}.csv")
        max_flux = max(max(points1['flux_o']), max(points2['flux_o']))
        min_flux = 0

        _ax = plot_direct_image(_ax, points1, levels, min_flux, max_flux, power_scale)
        _ax = plot_ghost_image(_ax, points2, levels, min_flux, max_flux, power_scale)

        _ax.set_xlim((-40, 40))
        _ax.set_ylim((-40, 40))

        plt.savefig('SampledPoints_incl={}.png'.format(self.t), dpi=300, facecolor='black')
        plt.show()
        return _fig, _ax

    def plot_isoredshifts_from_points(self, levels=None, extension="png"):
        # TODO add ghost image

        if levels is None:
            levels = [-.2, -.15, -.1, -0.05, 0., .05, .1, .15, .2, .25, .5, .75]

        _fig, _ax = self.get_figure()
        points = pd.read_csv(f"points_incl={int(round(self.t * 180 / np.pi))}.csv")
        br = self.calc_apparent_inner_disk_edge()
        color_map = plt.get_cmap('RdBu_r')

        # points1 = addBlackRing(self, points1)
        levels_ = [-.2, -.15, -.1, -0.05, 0., .05, .1, .15, .2, .25, .5, .75]
        _ax.tricontour(points['X'], points['Y'] if self.t <= np.pi / 2 else [-e for e in points['Y']],
                       # TODO why do I have to flip it myself
                       [e for e in points['z_factor']], cmap=color_map,
                       norm=plt.Normalize(0, 2),
                       levels=[e + 1 for e in levels_],
                       nchunk=2,
                       linewidths=2)
        _ax.fill_between(br.X, br.Y, color='black', zorder=2)
        plt.show()
        _fig.savefig(f"Plots/Isoredshifts_incl={str(int(180 * self.t / np.pi)).zfill(3)}.{extension}",
                     facecolor='black', dpi=300)
        return _fig, _ax

if __name__ == '__main__':
    M = 2.  # Black hole mass
    incl = 66  # Inclination angle in degrees
    bh = BlackHole(inclination=incl, mass=M)

    # Setting black hole parameters
    bh.disk_outer_edge = 150 * M
    bh.iz_solver_params['radial_precision'] = 50
    bh.angular_properties["angular_precision"] = 300

    # ---------------------- Plot Isoredshifts ------------------------
    # Plot isoredshifts with core
    fig, ax = bh.plot_isoredshifts(
        redshifts=[-.2, -.15, -.1, -.05, 0., .05, .1, .15, .25, .5, .75],
        plot_core=True
    )
    # Save isoredshifts plot as SVG
    fig.savefig(f"isoredshifts_{incl}_2.svg", facecolor="#F0EAD6")

    # ---------------------- Sample Points ----------------------------
    # Sampling 1000 points to scatter plot later
    bh.sample_points(n_points=1000, f=None, f2=None)

    # Plotting the sampled points
    bh.plot_points()

    # ------------------- Plot Isoredshifts from Points ----------------
    # Optionally plot isoredshifts based on the sampled points
    bh.plot_isoredshifts_from_points(show=True)

    # ----------------------- Plot Edge -------------------------------
    # Plot the disk's outer edge (if implemented in BlackHole)
    bh.plot_edge(show=True)

    # ------------------- Plot Redshift Isoradials ---------------------
    # Calculate and plot redshift isoradials (constant radius and redshift curves)
    bh.calc_isoradials([10, 20], [])
    bh.plot_isoradials([10, 20], [10, 20], show=True)
