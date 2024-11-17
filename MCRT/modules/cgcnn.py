import random

import torch
import torch.nn as nn
from MCRT.data_processor.dataset import Dataset

def main():
    dataset_test=Dataset(r'D:\projects\MCRT\MCRT\cifs\test','',["cdp","sgp"])
    batch=[dataset_test[i] for i in range(32)]
    collated_batch=dataset_test.collate(batch,1023,8)
    # print(collated_batch["padded_distance_matrices"].shape)
    GraphEmbeddingsmodel=GraphEmbeddings(atom_fea_len=64, nbr_fea_len=41, max_graph_len=1023, hid_dim=512, n_conv=3,mask_probability=0.15)
    new_atom_fea, atom_label,atm_label, mask=GraphEmbeddingsmodel(atom_num=collated_batch["atom_num"], nbr_idx=collated_batch["nbr_fea_idx"], nbr_fea=collated_batch["nbr_fea"], \
                                crystal_atom_idx=collated_batch["crystal_atom_idx"], atm_list=collated_batch["atm_list"])
    print(atom_label,torch.sum(atom_label != -1).item() /torch.sum(mask != 0).item() ,mask)

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Args:
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns:

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """

        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # [N, M, atom_fea_len]

        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                # [N, atom_fea_len] -> [N, M, atom_fea_len] -> v_i
                atom_nbr_fea,  # [N, M, atom_fea_len] -> v_j
                nbr_fea,
            ],  # [N, M, nbr_fea_len] -> u(i,j)_k
            dim=2,
        )
        # [N, M, atom_fea_len*2+nrb_fea_len]

        total_gated_fea = self.fc_full(total_nbr_fea)  # [N, M, atom_fea_len*2]
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(
            N, M, self.atom_fea_len * 2
        )  # [N, M, atom_fea_len*2]
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # [N, M, atom_fea_len]
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # [N, atom_fea_len]
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)  # [N, atom_fea_len]
        return out


class GraphEmbeddings(nn.Module):
    """
    1.mask atoms (depends on if_mask_atom) # moved to dataset
    2.infer
    3.reconstruct batch

    modified from (https://github.com/txie-93/cgcnn),(https://github.com/hspark1212/MOFTransformer)
    """

    def __init__(
        self, atom_fea_len, nbr_fea_len, max_graph_len, hid_dim, n_conv=3,if_conv=True
    ):
        super().__init__()
        self.if_conv = if_conv
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_graph_len = max_graph_len 
        self.hid_dim = hid_dim 
        if if_conv:
            self.embedding = nn.Embedding(119, atom_fea_len)  # 119 -> max(atomic number)
            self.convs = nn.ModuleList(
                [
                    ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                    for _ in range(n_conv)
                ]
            )
            self.fc = nn.Linear(atom_fea_len, hid_dim)
        else:
            self.hid_dim_embedding = nn.Embedding(119, hid_dim)


    def forward(
        self, atom_num, nbr_idx, nbr_fea, crystal_atom_idx
    ):
        """
        Args:
            atom_num (tensor): [N', 1]
            nbr_idx (tensor): [N', M]
            nbr_fea (tensor): [N', M, nbr_fea_len]
            crystal_atom_idx (list): [B]
        Returns:
            new_atom_fea (tensor): [B, max_graph_len, hid_dim]
            mask (tensor): [B, max_graph_len]
        """
        if self.if_conv:
            assert self.nbr_fea_len == nbr_fea.shape[-1]
            atom_fea = self.embedding(atom_num)  # [N', atom_fea_len]
            for conv in self.convs:
                    atom_fea = conv(atom_fea, nbr_fea, nbr_idx)  # [N', atom_fea_len]
            atom_fea = self.fc(atom_fea)  # [N', hid_dim]
        else:
            atom_fea = self.hid_dim_embedding(atom_num)  # [N', hid_dim]

        new_atom_fea, mask = self.reconstruct_batch(atom_fea, crystal_atom_idx)
        # [B, max_graph_len, hid_dim],  [B, max_graph_len]
        return new_atom_fea, mask

    def apply_mask_to_atoms(self, atom_fea, atom_num, mask_embedding_layer, mask_probability):
        """
        Applies masking logic to atom features.

        Args:
        atom_fea (torch.Tensor): Tensor of atom features, size [num_atoms, feature_dim].
        atom_num (torch.Tensor): Tensor of atom types, labels for each atom.
        mask_embedding_layer (nn.Embedding): Embedding layer to generate features for [MASK] token.
        mask_probability (float): Probability of applying mask, default is 15%.

        Returns:
        masked_atom_fea (torch.Tensor): Tensor of atom features after applying mask.
        labels (torch.Tensor): Labels tensor for loss computation.
        """

        # # Initialize labels with a special value -1, which indicates to ignore
        # labels = torch.full_like(atom_num, -1).to(atom_fea)

        # # Generate mask
        # mask = torch.rand(atom_fea.size(0)) < mask_probability

        # # Apply mask
        # masked_atom_fea = atom_fea.clone()
        # for i in range(masked_atom_fea.size(0)):
        #     if mask[i]:
        #         labels[i] = atom_num[i]  # Set the correct label for masked atom
        #         decision = torch.rand(1).item()
        #         if decision < 0.8:  # 80% chance to replace with [MASK]
        #             masked_atom_fea[i] = mask_embedding_layer(torch.tensor([0], dtype=torch.long).to(atom_fea.device))
        #         elif decision < 0.9:  # 10% chance for random replacement
        #             random_atom_idx = torch.randint(0, atom_fea.size(0), (1,)).item()
        #             masked_atom_fea[i] = atom_fea[random_atom_idx]
        #         # Remaining 10% chance to keep unchanged, label set to -1 (ignore)

        labels = torch.full_like(atom_num, -1).to(atom_fea.device)
        # Generate mask
        mask = torch.rand(atom_fea.size(0), device=atom_fea.device) < mask_probability
        masked_atom_fea = atom_fea.clone()
        # Masked indices
        masked_indices = mask.nonzero(as_tuple=True)[0]
        masked_mask=torch.rand(masked_indices.size(0), device=atom_fea.device)
        # 80% chance to replace with [MASK]
        mask_80_percent = masked_mask < 0.8
        indices_80_percent = masked_indices[mask_80_percent]
        masked_atom_fea[indices_80_percent] = mask_embedding_layer(torch.zeros_like(indices_80_percent))
        # 10% chance for random replacement
        mask_10_percent = (masked_mask >= 0.8) & (masked_mask < 0.9)
        indices_10_percent = masked_indices[mask_10_percent]
        random_atom_indices = torch.randint(0, atom_fea.size(0), indices_10_percent.size(), device=atom_fea.device)
        masked_atom_fea[indices_10_percent] = atom_fea[random_atom_indices]
        # Update labels for masked atoms
        labels[masked_indices] = atom_num[masked_indices]
        return masked_atom_fea, labels # only masked atoms will have labels

    def reconstruct_batch(self, atom_fea, crystal_atom_idx):
        batch_size = len(crystal_atom_idx)
        new_atom_fea = torch.zeros(batch_size, self.max_graph_len, self.hid_dim).to(atom_fea)
        mask_label = torch.zeros(batch_size, self.max_graph_len).to(atom_fea) # this mask is for label valid position
        for i, idx in enumerate(crystal_atom_idx):
                length = len(idx)
                new_atom_fea[i, :length] = atom_fea[idx]
                mask_label[i, :length] = 1  # valid position set to 1

        return new_atom_fea, mask_label # this mask is for label valid position
if __name__ == "__main__":
    main()  