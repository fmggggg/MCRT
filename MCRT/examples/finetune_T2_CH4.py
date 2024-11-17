import MCRT
import os

__root_dir__ = os.path.dirname(__file__)
root_dataset = "D:/Projects/MyProjects/MCRT/MCRT/cifs/Additional_Computational_Data/T2_Predicted_Structures"
log_dir = './logs/finetune/T2_methane'
downstream = "T2_Methane_capacity"

# kwargs (optional)
loss_names = {
        "classification": 0,
        "regression": 1,}
max_epochs = 50
batch_size = 32  # desired batch size; for gradient accumulation
per_gpu_batchsize = 2
num_workers = 12
mean = 52.249 # T2_Methane_capacity
std = 38.011
pos_emb = "relative"
seed = 0
alignn_num_conv = 4
alignn_hidden_dim = 256
if_conv = False
if_alignn = True
test_only = False
test_to_csv = True
learning_rate = 1e-4
if_image = True
if_grid = False
load_path  = "D:/Projects/MyProjects/MCRT/logs/server/pretrain_multi_modal/seed1/version_4/epoch=39-step=44160.ckpt" # multi_modal


if __name__ == '__main__':
    MCRT.run(root_dataset, downstream,log_dir=log_dir,\
             max_epochs=max_epochs,\
             loss_names=loss_names,\
             batch_size=batch_size,\
             per_gpu_batchsize=per_gpu_batchsize,\
             num_workers = num_workers,\
             load_path =load_path ,\
             if_conv = if_conv, \
             if_alignn = if_alignn, \
             alignn_num_conv = alignn_num_conv, \
             alignn_hidden_dim = alignn_hidden_dim, \
             pos_emb = pos_emb, \
             seed = seed, \
             if_image = if_image, \
             if_grid = if_grid, \
             test_only = test_only,\
             test_to_csv = test_to_csv,\
             learning_rate = learning_rate, \
          #    resume_from=resume_from,\
             mean=mean, std=std )