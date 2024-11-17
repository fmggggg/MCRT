This code is adapted from [crystal-gnn](https://github.com/hspark1212/crystal-gnn) to train on customized dataset.

To train CGCNN and ALIGNN:
1. prepare pickles：
        conda activate MCRT
        cd MCRT\compared_models
        python -m GNN.datasets.cal_graph --cif_path \path\to\cifs

2. run training:
        set parameters in config.py
        python -m GNN.run