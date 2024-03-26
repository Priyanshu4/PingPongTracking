from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class TableConstants:
    width: float = 1.53         # m
    length: float = 2.74        # m
    height: float = 0.76        # m
    net_height: float = 0.1525  # m
    net_width: float = 1.835    # m

    @staticmethod
    def load_from_yaml(yaml_file: Path) -> 'TableConstants':
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            return TableConstants(**data["ball"])

    
