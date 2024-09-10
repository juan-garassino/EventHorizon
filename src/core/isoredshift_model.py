# isoredshift_model.py

from .base_model import BaseModel
from .isoradial_model import Isoradial
from ..utils.coordinates import polar_to_cartesian_lists
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..utils.coordinates import polar_to_cartesian_single, get_angle_around

class Isoredshift(BaseModel):
    def __init__(self, config: dict, inclination: float, redshift: float, mass: float, 
                 isoradials: Optional[List[Isoradial]] = None):
        super().__init__(config)
        self.inclination = inclination
        self.redshift = redshift
        self.mass = mass
        self.radii_w_coordinates_dict: Dict[float, List[List[float]]] = {}
        self.coordinates_with_radii_dict: Dict[Tuple[float, float], float] = {}
        self.ir_radii_w_co: List[float] = []
        self.angles: List[float] = []
        self.radii: List[float] = []
        self.x_values: List[float] = []
        self.y_values: List[float] = []
        self.max_radius: float = 0

        if isoradials is not None:
            self.calc_from_isoradials(isoradials)
        self._update()

    def calc_from_isoradials(self, isoradials: List[Isoradial], cartesian: bool = False):
        solutions = {}
        for ir in isoradials:
            argument, radius = ir.calc_redshift_location_on_ir(self.redshift, cartesian=cartesian)
            solutions[ir.radius] = [argument, radius]
        self.radii_w_coordinates_dict = solutions
        self._update()

    def improve(self):
        r_w_s, r_wo_s = self.split_co_on_solutions()
        if len(r_w_s) > 0:
            self.recalc_isoradials_wo_redshift_solutions(plot_inbetween=False)
            self.improve_tip(iteration_count=self.config["isoredshift_solver_parameters"]["retry_tip"])
            for image_order in range(self.config["isoredshift_solver_parameters"]["times_inbetween"]):
                self.improve_between_all_solutions_once()
                self.order_coordinates(plot_title="calculating inbetween",
                                       plot_inbetween=self.config["isoredshift_solver_parameters"]["plot_inbetween"])

    def split_co_on_jump(self, threshold: float = 2):
        def dist(x_values, y_values):
            x1, x2 = x_values
            y1, y2 = y_values
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        self.order_coordinates()
        self._update()
        x_values, y_values = polar_to_cartesian_lists(self.radii, self.angles)
        distances = [dist((x1, x2), (y1, y2)) for x1, x2, y1, y2 in zip(x_values[:-1], x_values[1:], y_values[:-1], y_values[1:])]
        mx2, mx = sorted(distances)[-2:]
        if mx > threshold * mx2:
            split_ind = np.where(distances == mx)[0][0]
            if not abs(np.diff(np.sign(self.x_values[split_ind:split_ind + 2]))) > 0:
                split_ind = None
        else:
            split_ind = None
        return split_ind

    def _update(self):
        self.ir_radii_w_co = [key for key, val in self.radii_w_coordinates_dict.items() if len(val[0]) > 0]
        self.angles, self.radii = self._extract_co_from_solutions_dict()
        self.x_values, self.y_values = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)
        self.order_coordinates()

    def _extract_co_from_solutions_dict(self) -> Tuple[List[float], List[float]]:
        argument = []
        radius = []
        for key, val in self.radii_w_coordinates_dict.items():
            if len(val[0]) > 0:
                angles, radii = val
                argument.extend(angles)
                radius.extend(radii)
        return argument, radius

    def order_coordinates(self, plot_title: str = "", plot_inbetween: bool = False):
        co = list(zip(self.angles, self.radii))
        x_values, y_values = polar_to_cartesian_lists(self.radii, self.angles)
        cx, cy = np.mean(x_values), np.mean(y_values)
        order_around = [0.3 * cx, 0.8 * cy]

        sorted_co = sorted(co, key=lambda polar_point: get_angle_around(order_around, 
                                                                        polar_to_cartesian_single(*polar_point)))

        self.angles, self.radii = [e[0] for e in sorted_co], [e[1] for e in sorted_co]
        self.x_values, self.y_values = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)

    def split_co_on_solutions(self) -> Tuple[Dict[float, List[List[float]]], Dict[float, List[List[float]]]]:
        keys_w_s = [key for key, val in self.radii_w_coordinates_dict.items() if len(val[0]) > 0]
        keys_wo_s = [key for key, val in self.radii_w_coordinates_dict.items() if len(val[0]) == 0]
        dict_w_s = {key: self.radii_w_coordinates_dict[key] for key in keys_w_s}
        dict_wo_s = {key: self.radii_w_coordinates_dict[key] for key in keys_wo_s}
        return dict_w_s, dict_wo_s

    def recalc_isoradials_wo_redshift_solutions(self, plot_inbetween: bool = False):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s) > 0 and len(r_w_so) > 0:
            argument, radius = self.recalc_redshift_on_closest_isoradial_wo_z()
            self.order_coordinates(plot_title="improving tip angular")
            r_w_so, r_wo_s = self.split_co_on_solutions()
            while len(argument) > 0 and len(r_wo_s) > 0:
                argument, radius = self.recalc_redshift_on_closest_isoradial_wo_z()
                r_w_s, r_wo_s = self.split_co_on_solutions()
                self.order_coordinates(plot_inbetween=plot_inbetween, plot_title="improving tip angular")

    def recalc_redshift_on_closest_isoradial_wo_z(self) -> Tuple[List[float], List[float]]:
        r_w_s, r_wo_s = self.split_co_on_solutions()
        angle_interval, _ = self.radii_w_coordinates_dict[max(r_w_s.keys())]
        assert len(angle_interval) > 1, f"1 or less angles found for corresponding isoradial R={max(r_w_s)}"
        closest_r_wo_s = min(r_wo_s.keys())
        begin_angle, end_angle = angle_interval
        if end_angle - begin_angle > np.pi:
            begin_angle, end_angle = end_angle, begin_angle
        argument, b = self.calc_redshift_on_ir_between_angles(closest_r_wo_s, begin_angle, end_angle,
                                                       angular_precision=
                                                       self.config['isoredshift_solver_parameters']['retry_angular_precision'],
                                                       mirror=False)
        
        print(f"Redshift solutions: argument={argument}, b={b}, closest_r_wo_s={closest_r_wo_s}")  # Debugging output
        if len(argument) > 0:
            self._add_solutions(argument, b, closest_r_wo_s)
        return argument, b

    def _add_solutions(self, angles: List[float], impact_parameters: List[float], radius_ir: float):
        for angle, impact_parameter in zip(angles, impact_parameters):
            print(f"Adding solution: angle={angle}, impact_parameter={impact_parameter}, radius_ir={radius_ir}")  # Debugging output
            if radius_ir in self.radii_w_coordinates_dict:
                if len(self.radii_w_coordinates_dict[radius_ir][0]) > 0:
                    self.radii_w_coordinates_dict[radius_ir][0].append(angle)
                    self.radii_w_coordinates_dict[radius_ir][1].append(impact_parameter)
                else:
                    self.radii_w_coordinates_dict[radius_ir] = [[angle], [impact_parameter]]
            else:
                self.radii_w_coordinates_dict[radius_ir] = [[angle], [impact_parameter]]
            self.coordinates_with_radii_dict[(angle, impact_parameter)] = radius_ir
        self._update()

    def improve_between_all_solutions_once(self):
        self.order_coordinates()
        co = list(zip(self.angles, self.radii))
        
        # Debugging output
        print("Coordinates with Radii Dict:", self.coordinates_with_radii_dict)
        
        for b, e in zip(co[:-1], co[1:]):
            if b not in self.coordinates_with_radii_dict or e not in self.coordinates_with_radii_dict:
                print(f"Missing keys: {b}, {e}")
                continue  # Skip this iteration if keys are missing
            r_inbetw = 0.5 * (self.coordinates_with_radii_dict[b] + self.coordinates_with_radii_dict[e])
            begin_angle, end_angle = b[0], e[0]
            if end_angle - begin_angle > np.pi:
                begin_angle, end_angle = end_angle, begin_angle
            argument, radius = self.calc_redshift_on_ir_between_angles(r_inbetw, begin_angle - 0.1, end_angle + 0.1,
                                                           plot_inbetween=False,
                                                           title=f'between p{b} and p{e}',
                                                           force_solution=True)
            if len(argument) > 0:
                self._add_solutions(argument, radius, r_inbetw)

    def improve_tip(self, iteration_count: int = 6):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s) > 0:
            for it in range(iteration_count):
                self.calc_ir_before_closest_ir_wo_z()
                self.order_coordinates(plot_title=f"Improving tip iteration {it}",
                                       plot_inbetween=self.config["isoredshift_solver_parameters"]["plot_inbetween"])

    def calc_ir_before_closest_ir_wo_z(self, angular_margin: float = 0.3):
        r_w_s, r_wo_s = self.split_co_on_solutions()
        angle_interval, _ = self.radii_w_coordinates_dict[max(r_w_s.keys())]
        if len(r_wo_s) > 0 and len(r_w_s) > 0:
            first_r_wo_s = min(r_wo_s.keys())
            last_r_w_s = max(r_w_s.keys())
            inbetween_r = 0.5 * (first_r_wo_s + last_r_w_s)
            begin_angle, end_angle = angle_interval
            if end_angle - begin_angle > np.pi:
                begin_angle, end_angle = end_angle, begin_angle
            argument, radius = self.calc_redshift_on_ir_between_angles(inbetween_r, begin_angle - angular_margin,
                                                           end_angle + angular_margin,
                                                           angular_precision=
                                                           self.config['isoredshift_solver_parameters']['retry_angular_precision'],
                                                           mirror=False)
            if len(argument) > 0:
                self._add_solutions(argument, radius, inbetween_r)
            else:
                self.radii_w_coordinates_dict[inbetween_r] = [[], []]

    def calc_redshift_on_ir_between_angles(self, radius: float, begin_angle: float = 0, end_angle: float = np.pi,
                                           angular_precision: int = 3, mirror: bool = False,
                                           plot_inbetween: bool = False, title: str = '',
                                           force_solution: bool = False) -> Tuple[List[float], List[float]]:
        # Create Isoradial instance with correct parameter names
        ir = Isoradial(config=self.config, radius=radius, inclination=begin_angle, mass=self.mass, order=0)

        ir.find_redshift_params['force_redshift_solution'] = force_solution
        argument, radius = ir.calc_redshift_location_on_ir(self.redshift, cartesian=False)
        return argument, radius