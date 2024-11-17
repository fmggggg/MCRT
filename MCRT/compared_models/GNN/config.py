# pylint: disable-all
from sacred import Experiment

ex = Experiment("crystal-gnn")

alignn_config = {}


@ex.config
def config():
    exp_name = "alignn"
    seed = 123
    test_only = False

    # prepare_data
    source = "folder"
    database_name = ""
    target = "Triptycene_energy"
    data_dir = r"D:\Projects\MyProjects\MCRT\MCRT\compared_models\GNN\cifs\Triptycene_ALL_CIF"
    classification_threshold = None
    split_seed = 123
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    keep_data_order = False

    # dataset
    compute_line_graph = True
    neighbor_strategy = "k-nearest"
    cutoff = 8.0
    max_neighbors = 12
    use_canonize = True
    # dataloader
    batch_size = 32
    per_gpu_batchsize = 2 # step batch size
    num_workers = 1  # This should be 0 to use dataloader with dgl graph
    pin_memory = False
    use_ddp = False

    # model
    model_name = "alignn"  # "schnet", "cgcnn", "alignn"
    num_conv = 4
    hidden_dim = 256
    rbf_distance_dim = 40  # RDF expansion dimension for edge distance
    rbf_triplet_dim = 40  # RDF expansion dimension for triplet angle
    batch_norm = True
    residual = False
    dropout = 0.0
    num_classes = 1  # if higher than 1, classification mode is activated

    # normalizer (only when num_classes == 1)
    mean = None  # when mean is None, it will be calculated from train data
    std = None  # when std is None, it will be calculated from train data


    # optimizer
    optimizer = "adamw"  # "adma", "sgd", "adamw"
    lr = 1e-3 # learning rate
    weight_decay = 1e-5
    scheduler = "constant"  # "constant", "cosine", "reduce_on_plateau", "constant_with_warmup"

    # training
    devices = 1  # number of GPUs to use
    accelerator = "gpu"  # "cpu", "gpu"
    max_epochs = 50
    deterministic = True  # set True for reproducibility
    log_dir = "./logs"
    load_path = None  # to load pretrained model
    resume_from = None


###########
# default #
###########
@ex.named_config
def schnet():
    exp_name = "schnet"
    model_name = "schnet"


@ex.named_config
def cgcnn():
    exp_name = "cgcnn"
    model_name = "cgcnn"


@ex.named_config
def alignn():
    exp_name = "alignn"
    model_name = "alignn"


############
# matbench #
############
@ex.named_config
def matbench_schnet():
    exp_name = "schnet"
    model_name = "schnet"
    log_dir = "./GNN/logs/matbench"


@ex.named_config
def matbench_cgcnn():
    exp_name = "cgcnn"
    model_name = "cgcnn"
    log_dir = "./GNN/logs/matbench"


@ex.named_config
def matbench_alignn():
    exp_name = "alignn"
    model_name = "alignn"
    log_dir = "./GNN/logs/matbench"
