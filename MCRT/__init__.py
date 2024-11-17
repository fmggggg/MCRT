import os

__version__ = "1.0.0"
__root_dir__ = os.path.dirname(__file__)

from MCRT import modules, data_processor, visualize, assets,  models
from MCRT.run import run
__all__ = [
    "data_processor",
    "modules",
    "run",
    "visualize",
    "assets",
    "models",
    __version__,
]
