import os

__version__ = "1.0.0"
__root_dir__ = os.path.dirname(__file__)

from MCRT import modules, data_processor, visualize, cifs, assets, compared_models, models
from MCRT.run import run
__all__ = [
    "data_processor",
    "modules",
    "run",
    "visualize",
    "cifs",
    "assets",
    "compared_models",
    "models",
    __version__,
]
