from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"


def enforce_entry_point():
    """ Enforces the entry point of the Python modules in this project.
        All Python modules with a main function that import other modules must run from the project root.
        This function ensures that the entry point is the project root.
        If not, it gives a helpful error message.
    """
    if Path.cwd() != PROJECT_ROOT:
        raise Exception(
            f"Please run this script from the project root using python -m syntax."
        )


