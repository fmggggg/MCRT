import MCRT
import os

__root_dir__ = os.path.dirname(__file__)
root_dataset = os.path.join(__root_dir__,"cifs","1000CSP")
log_dir = './logs/finetune/1000CSP_delta'
downstream = "1000CSP_delta"

# kwargs (optional)
loss_names = {
        "classification": 0,
        "regression": 1,}
max_epochs = 50
batch_size = 32  # desired batch size; for gradient accumulation
per_gpu_batchsize = 8
num_workers = 12
mean = 6.886 # 1000CSP
std = 6.801
test_only = True
test_to_csv = True
learning_rate = 1e-4
load_path  = "path/to/MCRT.ckpt" 
if __name__ == '__main__':
    MCRT.run(root_dataset, downstream,log_dir=log_dir,\
             max_epochs=max_epochs,\
             loss_names=loss_names,\
             batch_size=batch_size,\
             per_gpu_batchsize=per_gpu_batchsize,\
             num_workers = num_workers,\
             load_path =load_path ,\
             test_only = test_only,\
             test_to_csv = test_to_csv,\
             learning_rate = learning_rate, \
          #    resume_from=resume_from,\
             mean=mean, std=std )