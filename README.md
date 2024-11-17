# MCRT
Molecular Crystal Representation from Transformer

## Install
#### Option 1: Directly install
1. Create a Conda environment with Python 3.8:
```python
conda create -y -n MCRT python=3.8
conda activate MCRT
```
2. Install PyTorch=2.1.1 [(from here)](https://pytorch.org/get-started/previous-versions/) and DGL [(from here)](https://www.dgl.ai/pages/start.html) based on your CUDA version and OS.
3. Install other packages:
```python
cd /path/to/MCRT
pip install -r requirements.txt
```
#### Option 2: Via Apptainer (easier and faster)
It would be easier to use Apptainer [(install from here)](https://apptainer.org/docs/user/main/quick_start.html) because you don't have to deal with any unexpected errors when you install the environment. You only have to install apptainer, and we provided the pre-defined images [here]().

## Prepare dataset
#### Prepare persistence images
The persistence images are generated using adapted [moleculetda](https://github.com/a1k12/moleculetda). We also provide 2 options to install it.
##### Option 1: Directly install
Create a Conda environment with Python 3.11:
```python
conda create -y -n persistent python=3.11
conda activate persistent
pip install moleculetda tqdm numpy
```
Usage:
```python
conda activate persistent
cd /path/to/moleculetda/src
python -m moleculetda.cif_to_image --cif_path ../cifs/your_cif_folder --paral 16
```
You can parallal the generation by setting --paral.
##### Option 2: Via Apptainer (easier and faster)
Download the pre-defined image [here]().
Usage:
```python
cd /path/to/moleculetda
apptainer exec /path/to/moleculetda_container.sif python3 /path/to/cif_to_image.py --cif_path /path/to/cif_path
```
#### Prepare pickles (optional)
Pickles include the pre-calculated positional embedding matrix and pre-training labels. It's an optional procedure for finetuning now because we have implemented the generation of graphs in real time for finetuneing. But for pretraining, the label for tasks are time-consuming to generate, it should be generated like this:
```python
conda activate MCRT
python /path/to/cif_to_dataset.py --cif_path /path/to/cif_path --paral 16 --type pretrain 
```
You can generate pickle for fintuning too, which may be a little bit faster than generating them in real time. But it depends on your CPU and GPU, since the generation is on CPU, if your CPU is fast or GPU is slow, there would be no difference since the bottleneck is the model training on GPU. If you want to generate pickles for finetuning:
```python
conda activate MCRT
python /path/to/cif_to_dataset.py --cif_path /path/to/cif_path --paral 16 --type finetune 
```
#### dataset split
The dataset split is defined by a json file named dataset_split.json:
```json
{
  "train": ["SUYYIV","UYUGED"],
  "val": ["GASVUR","IHOZAH"],
  "test": ["LUMSER","DUGXUY"],
}
```
One can generate it by yourself or by using split_dataset.py which we provided.
```python
python /path/to/split_dataset.py --cif /path/to/cif_path --split 0.8 0.1 0.1
```
#### dataset structure
When you finished the generation above, you should make sure the dataset structure is like this: 
your_dataset/
├── cifs/containing cif files
├── imgs/containing persistence images
├── pickles/(optional for finetuning) containing pickles
├── dataset_split.json
└── downstream.csv
## To fineture
```python
import MCRT
import os

__root_dir__ = os.path.dirname(__file__)
root_dataset = os.path.join(__root_dir__,"cifs","your_dataset")
log_dir = './logs/your_dataset_logs'
downstream = "downstream" # name of downstream.csv

loss_names = {"classification": 0,"regression": 1,} # for regression
max_epochs = 50 # training epochs
batch_size = 32  # desired batch size; for gradient accumulation
per_gpu_batchsize = 8 # batch size per step
num_workers = 12 # num of CPU workers
mean = 0 # mean value of your dataset
std = 1 # standard deviation of your dataset

test_to_csv = True # if True, save test set results
load_path  = "/path/to/pretrained.ckpt" 

if __name__ == '__main__':
    MCRT.run(root_dataset, downstream,log_dir=log_dir,\
             max_epochs=max_epochs,\
             loss_names=loss_names,\
             batch_size=batch_size,\
             per_gpu_batchsize=per_gpu_batchsize,\
             num_workers = num_workers,\
             load_path =load_path ,\
             test_to_csv = test_to_csv,\
             mean=mean, std=std )
```
Usage:
make a python file named finetune.py under the root dir of MCRT, and run it as module:
```python
conda activate MCRT
cd /path/to/MCRT
python -m MCRT.finetune
```
## To test finetuned model
```python
import MCRT
import os

__root_dir__ = os.path.dirname(__file__)
root_dataset = os.path.join(__root_dir__,"cifs","your_dataset")
log_dir = './logs/your_dataset_logs'
downstream = "downstream" # name of downstream.csv

loss_names = {"classification": 0,"regression": 1,} # for regression
max_epochs = 50 # training epochs
batch_size = 32  # desired batch size; for gradient accumulation
per_gpu_batchsize = 8 # batch size per step
num_workers = 12 # num of CPU workers
mean = 0
std = 1

test_only=True # test the model
test_to_csv = True # if True, save test set results
load_path  = "/path/to/finetuned.ckpt" 

if __name__ == '__main__':
    MCRT.run(root_dataset, downstream,log_dir=log_dir,\
             max_epochs=max_epochs,\
             loss_names=loss_names,\
             batch_size=batch_size,\
             per_gpu_batchsize=per_gpu_batchsize,\
             num_workers = num_workers,\
             load_path =load_path ,\
             test_only =test_only ,\
             test_to_csv = test_to_csv,\
             mean=mean, std=std )
```
Usage:
make a python file named test_model.py under the root dir of MCRT, and run it as module:
```python
conda activate MCRT
cd /path/to/MCRT
python -m MCRT.test_model
```