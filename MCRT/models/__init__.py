import os
from MCRT import __root_dir__

DEFAULT_PRETRAIN_MODEL_PATH = os.path.join(
    __root_dir__, "models/"
)

DEFAULT_MCRT_PATH = os.path.join(__root_dir__, "models/MCRT.ckpt")

DEFAULT_FINETUNED_MODEL_PATH = os.path.join(__root_dir__, "models/finetuned/")
