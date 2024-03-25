from dataclasses import dataclass
from typing import Optional
from math import pi
from pathlib import Path
import yaml

@dataclass
class BallConstants:

    gravity: float = 9.81               # m/s^2
    radius: float = 20 * 10**-3         # m
    mass: float = 0.0027                # kg
    coefficient_of_drag: float = 0.47   # unitless
    coefficient_of_lift: float = 1.23   # unitless
    air_density: float = 1.205          # kg/m^3

    # These values are calculated from the above constants
    _cross_sectional_area: Optional[float] = None
    _diameter: Optional[float] = None
    _air_drag_factor: Optional[float] = None      
    _magnus_factor: Optional[float] = None

    def __post_init__(self):
        self._diameter = 2 * self.radius
        self._cross_sectional_area = pi * self.radius**2
        self._air_drag_factor = -0.5 * self.air_density * self.coefficient_of_drag * self._cross_sectional_area / self.mass
        self._magnus_factor = self.coefficient_of_lift * self._diameter**3 * self.air_density / (2 * pi * self.mass)

    @property
    def diameter(self):
        return self._diameter
    
    @property
    def air_drag_factor(self):
        return self._air_drag_factor
    
    @property
    def magnus_factor(self):
        return self._magnus_factor

    @staticmethod
    def load_from_yaml(yaml_file: Path) -> 'BallConstants':
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            return BallConstants(**data["ball"])
    