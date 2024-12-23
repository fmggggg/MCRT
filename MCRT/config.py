import os
from sacred import Experiment
from MCRT.modules.utils import _set_load_path, _loss_names
from MCRT import __root_dir__

ex = Experiment("pretrained_MCRT", save_git_info=False)

@ex.config
def config():
    # model
    exp_name = "pretraining_MCRT"
    seed = 0
    loss_names = _loss_names({
        "map": 0,
        "apc": 0,  
        "sgp": 0,
        "sep": 0,
        "cdp": 0,
        "adp": 0,
        "aap": 0,
        "ucp": 0,})
    visualize=True
    # train_val_test_split=[0.8, 0.1, 0.1]
    # dataset_seed=123

    max_graph_len = 1023  # number of maximum atoms in P1 unit cell
    read_from_pickle = True # if read structure and angle list from pickle, if False, read cif and generate it in real time

    # graph setting, for cgcnn
    if_conv = False #for cgcnn, the data load method for cgcnn is different from other GNNs.
    max_num_nbr = 12
    atom_fea_len = 64
    nbr_fea_len = 41
    n_conv = 3
    mask_probability = 0.15 # proportion of masked atoms

    # for alignn
    if_alignn = True
    alignn_num_conv = 4
    alignn_hidden_dim = 256
    alignn_rbf_distance_dim = 40 # RDF expansion dimension for edge distance
    alignn_rbf_triplet_dim = 40 # RDF expansion dimension for triplet angle
    alignn_batch_norm = True
    alignn_dropout = 0.0
    alignn_residual = False

    # chemical positional embedding 
    angle_nbr = 8 # number of nbr atoms when generating the angle matrix, int between 2-8
    pos_emb = "relative" #  "absolute" or "relative" or "both"
    # pretext tasks
    num_ap = 200 # number of atom pairs from each crystal, can't be odd
    n_dist = 100 # number of atom pairs for distance prediction
    n_angle = 100 # number of atom pairs for angle prediction

    # sep_weights = [4.43535968,68.57053843,76.72758503,99.97791198,50.52052353,9.96930181,2.19511,78.12710143,89.54104273,90.02648945,91.84479127,66.14665125,87.25459766,81.96008386,100.35775213,94.87247941,99.90788809,92.78795367,100.04803458,95.63538094,35.27829775,2.21395585] # generated by weights_log = 1 / np.log(epsilon=1.01 + frequencies)
    #sep_weights = [1,1,1,1,1,1,1]
    sep_weights = [9.58304127,3.39176836,8.91170359,5.50722613,1.93331989,9.76274322,1.97087159]# generated by weights_log = 1 / np.log(epsilon=1.1 + frequencies)
    # transformer setting
    num_blocks = 12
    hid_dim = 768
    num_heads = 12
    mlp_ratio = 4
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.1
    attn_drop = 0.0
    drop_path_rate = 0.1

    # for persistence images
    if_image = True #if use image modality
    img_size=(50,50)
    patch_size=(5,5)
    in_chans=1

    # for energy grid
    if_grid = False # if use energy grid, for moftransformer only
    grid_img_size = 30
    grid_patch_size = 5  # length of patch
    grid_in_chans = 1  # channels of grid image

    # downstream
    downstream = ""
    n_classes = 0

    # Optimizer Setting
    optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 1e-4
    weight_decay = 1e-2
    decay_power = (
        1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    )
    max_epochs = 100
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.05  # int or float ( max_steps * warmup_steps)
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # PL Trainer Setting
    resume_from = None
    val_check_interval = 1.0
    test_only = False
    test_to_csv = True
    cls_to_csv = False

    # below params varies with the environment
    root_dataset = os.path.join(__root_dir__, "cifs","test")
    log_dir = "logs/"
    batch_size = 512  # desired batch size; for gradient accumulation
    per_gpu_batchsize = 32  # you should define this manually with per_gpu_batch_size
    accelerator = "auto"
    devices = "auto"
    num_nodes = 1
    strategy = "ddp"
    load_path = _set_load_path(None) # path must be 'MCRT' (for finetune), None (for pretrain), or path to *.ckpt (for using finetuned model)

    num_workers = 6  # the number of cpu's core
    precision = '32-true' 
    """
    precision (Union[Literal[64, 32, 16], Literal['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true'], 
    Literal['64', '32', '16', 'bf16'], None]) , Double precision (64, '64' or '64-true'), 
    full precision (32, '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed'). 
    """

    # normalization target
    mean = None
    std = None

