import torch
import torch.nn as nn
import torch.nn.functional as F
import random

#test
from MCRT.data_processor.dataset import Dataset
from MCRT.modules.cgcnn import GraphEmbeddings
from MCRT.modules.McrtTransformer import MCRT_Transformer
def main():
    dataset_test=Dataset(r'D:\projects\MCRT\MCRT\cifs\test','train',1023,4)
    batch=[dataset_test[i] for i in range(32)]
    collated_batch=dataset_test.collate(batch)
    hid_dim=512
    GraphEmbeddingsmodel=GraphEmbeddings(atom_fea_len=64, nbr_fea_len=41, max_graph_len=1023, hid_dim=hid_dim, n_conv=3,mask_probability=0.15)
    new_atom_fea, atom_label,atm_label, mask=GraphEmbeddingsmodel(atom_num=collated_batch["atom_num"], nbr_idx=collated_batch["nbr_fea_idx"], nbr_fea=collated_batch["nbr_fea"], \
                                crystal_atom_idx=collated_batch["crystal_atom_idx"], atm_list=collated_batch["atm_list"],if_mask_atom=False)
    print(atom_label[0][:30])
    # McrtTransformer=MCRT_Transformer(
    #     num_blocks=12,
    #     dim=hid_dim,
    #     num_heads=8,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     qk_scale=None,
    #     drop=0.0,
    #     attn_drop=0.0,
    #     drop_path_rate=0.1,
    #     act_layer=nn.GELU,
    #     norm_layer=nn.LayerNorm
    # )
    # new_atom_fea=McrtTransformer(new_atom_fea,mask)

    print(atm_label[0][:20])
    APC=APCHead(hid_dim,100)
    predictions,atom_pairs, ap_labels = APC(new_atom_fea,atm_label)
    # print(predictions[0])
    print(atom_pairs[0][:20])
    print(atm_label[0][atom_pairs[0]][:20])
    print(ap_labels[0][:20])


class Pooler(nn.Module):
    """
    head for [CLS]
    """
    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.index = index

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class MAPHead(nn.Module):
    """
    head for Masked Atom Prediction
    """

    def __init__(self, hid_dim, num_atom_types=118):  # NUMBER of atom types , label need to -1 when compare
        super().__init__()
        self.fc = nn.Linear(hid_dim, num_atom_types)

    def forward(self, x):
        x = self.fc(x)  # [B, max_graph_len, num_atom_types]
        return x # need to consider invalid positions afterwards
    
class APCHead(nn.Module):
    """
    head for Atom pair classification        
    m: number of atom pairs from each crystal (m can't be odd, if odd, the final position of atom_pairs will be filled -1) ,
       total num of pairs for a batch is B*m,
       half of m atom pairs are from same molecule, remaining from different molecules
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, new_atom_fea, atom_pairs,ap_labels):
        # new_atom_fea: [B, max_graph_len, hid_dim]

        # Select atom pairs and their labels from atm_label
        # atom_pairs, ap_labels = self.select_atom_pairs(atm_label, self.m)  # [B, m, 2], [B, m]
        atom_fea_pairs = self.get_atom_pairs_features(new_atom_fea, atom_pairs)  # [B, m, 2*hid_dim]
        # Handle symmetry of atom pairs
        hid_dim=new_atom_fea.size(2)
        atom_fea_pairs_reversed = torch.cat((atom_fea_pairs[:, :, hid_dim:], atom_fea_pairs[:, :, :hid_dim]), dim=-1)
        predictions_original = self.mlp(atom_fea_pairs)  # [B, m, 1]
        predictions_reversed = self.mlp(atom_fea_pairs_reversed)  # [B, m, 1]
        # Sum the predictions from both orders
        predictions = (predictions_original + predictions_reversed) / 2  # [B, m, 1]
        predictions = predictions.squeeze(-1)  # [B, m]
        return predictions,atom_pairs, ap_labels # [B, m],[B, m, 2], [B, m] # need to consider invalid positions (-1) afterwards
    
    def get_atom_pairs_features(self, new_atom_fea, atom_pairs): 
        B, m, _ = atom_pairs.size()
        hid_dim = new_atom_fea.size(-1)
        atom_pairs = atom_pairs.to(new_atom_fea.device)
        valid_pairs_mask = (atom_pairs[:, :, 0] != -1) & (atom_pairs[:, :, 1] != -1)
        # replace -1 with 0 to prevent invalid index in next step
        atom_pairs = torch.where(atom_pairs == -1, 0, atom_pairs)
        atom_fea_pairs = torch.zeros(B, m, 2 * hid_dim, dtype = new_atom_fea.dtype, device=new_atom_fea.device)
        
        atom1_feats = new_atom_fea.gather(1, atom_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, hid_dim))
        atom2_feats = new_atom_fea.gather(1, atom_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, hid_dim))

        new_atom_fea_pairs = torch.cat((atom1_feats, atom2_feats), dim=-1)
        atom_fea_pairs[valid_pairs_mask] = new_atom_fea_pairs[valid_pairs_mask]
        return atom_fea_pairs
    
    # def select_atom_pairs(self, atm_label, m):
    #     """    
    #     m: number of atom pairs from each crystal,so total num of pairs is B*m,
    #     half of m atom pairs are from same molecule, remaining from different molecules,
    #     if possible number of atom pairs < m, number of selected pairs = max_pairs*2, max_pairs = min(m // 2, same_molecule_pairs_count, diff_molecule_pairs_count)
    #     """
    #     B, max_graph_len = atm_label.size()
    #     atom_pairs = torch.full((B, m, 2), -1, dtype=torch.long)  # Use -1 for invalid pairs
    #     ap_labels = torch.full((B, m), -1, dtype=torch.long)  # Use -1 for invalid ap_labels

    #     for i in range(B):
    #         # Collect information about molecules in the crystal
    #         molecule_atoms = {}
    #         for idx, molecule_id in enumerate(atm_label[i]):
    #             if molecule_id != -1:
    #                 molecule_atoms.setdefault(molecule_id.item(), []).append(idx)

    #         # Calculate the number of possible same molecule pairs
    #         same_molecule_pairs_count = sum(len(atoms) * (len(atoms) - 1) // 2 for atoms in molecule_atoms.values())
    #         # Calculate the number of possible diff molecule pairs
    #         diff_molecule_pairs_count = 0
    #         molecule_ids = list(molecule_atoms.keys())
    #         for idx1, mol1 in enumerate(molecule_ids):
    #             for mol2 in molecule_ids[idx1 + 1:]:
    #                 diff_molecule_pairs_count += len(molecule_atoms[mol1]) * len(molecule_atoms[mol2])

    #         max_pairs = min(m // 2, same_molecule_pairs_count, diff_molecule_pairs_count)

    #         selected_pairs = []
    #         selected_pairs_set = set()

    #         # Randomly select same molecule pairs
    #         while len(selected_pairs) < max_pairs:
    #             mol = random.choice(list(molecule_atoms.keys()))
    #             if len(molecule_atoms[mol]) > 1:
    #                 atom1, atom2 = random.sample(molecule_atoms[mol], 2)
    #                 pair = tuple(sorted((atom1, atom2)))  # Ensure the pair is sorted
    #                 if pair not in selected_pairs_set:
    #                     selected_pairs.append(pair)
    #                     ap_labels[i, len(selected_pairs) - 1] = 1  # Same molecule
    #                     selected_pairs_set.add(pair)

    #         # Select different molecule pairs
    #         while len(selected_pairs) < 2 * max_pairs:
    #             mol1, mol2 = random.sample(list(molecule_atoms.keys()), 2)
    #             atom1, atom2 = random.choice(molecule_atoms[mol1]), random.choice(molecule_atoms[mol2])
    #             pair = tuple(sorted((atom1, atom2)))  # Ensure the pair is sorted
    #             if pair not in selected_pairs_set:
    #                 selected_pairs.append(pair)
    #                 ap_labels[i, len(selected_pairs) - 1] = 0  # Different molecules
    #                 selected_pairs_set.add(pair)

    #         # atom_pairs[i, :len(selected_pairs)] = torch.tensor(selected_pairs)
    #         if len(selected_pairs) > 0:  # Ensure there are selected pairs
    #                 selected_pairs_tensor = torch.tensor(selected_pairs, dtype=torch.long)
    #                 atom_pairs[i, :selected_pairs_tensor.size(0)] = selected_pairs_tensor

    #     return atom_pairs, ap_labels #  [B, m, 2],[B, m]

    # def get_atom_pairs_features(self, new_atom_fea, atom_pairs):
    #     B, m, _ = atom_pairs.size()
    #     hid_dim = new_atom_fea.size(-1)
    #     atom_fea_pairs = torch.zeros(B, m, 2 * hid_dim).to(new_atom_fea)

    #     for i in range(B):
    #         for j in range(m):
    #             if atom_pairs[i, j, 0] != -1 and atom_pairs[i, j, 1] != -1:
    #                 atom1_idx = atom_pairs[i, j, 0]
    #                 atom2_idx = atom_pairs[i, j, 1]
    #                 atom_fea_pairs[i, j] = torch.cat((new_atom_fea[i, atom1_idx], new_atom_fea[i, atom2_idx]))

    #     return atom_fea_pairs

class ADPHead(nn.Module):
    """
    head for Atom Distance Prediction 
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, new_atom_fea, dist_pairs,dist_labels):
        # new_atom_fea: [B, max_graph_len, hid_dim]

        atom_fea_pairs = self.get_atom_pairs_features(new_atom_fea, dist_pairs)  # [B,n_dist, 2*hid_dim]
        # Handle symmetry of atom pairs
        hid_dim=new_atom_fea.size(2)
        atom_fea_pairs_reversed = torch.cat((atom_fea_pairs[:, :, hid_dim:], atom_fea_pairs[:, :, :hid_dim]), dim=-1)
        predictions_original = self.mlp(atom_fea_pairs)  # [B,n_dist, 1]
        predictions_reversed = self.mlp(atom_fea_pairs_reversed)  # [B,n_dist, 1]
        # Sum the predictions from both orders
        predictions = (predictions_original + predictions_reversed) / 2  # [B,n_dist, 1]
        predictions = predictions.squeeze(-1)  # [B,n_dist]
        return predictions,dist_pairs, dist_labels # [B,n_dist],[B,n_dist, 2], [B,n_dist] # need to consider invalid positions (-1.0) afterwards
    
    def get_atom_pairs_features(self, new_atom_fea, atom_pairs): 
        B, m, _ = atom_pairs.size()
        hid_dim = new_atom_fea.size(-1)
        atom_pairs = atom_pairs.to(new_atom_fea.device)
        valid_pairs_mask = (atom_pairs[:, :, 0] != -1) & (atom_pairs[:, :, 1] != -1)
        # replace -1 with 0 to prevent invalid index in next step
        atom_pairs = torch.where(atom_pairs == -1, 0, atom_pairs)
        atom_fea_pairs = torch.zeros(B, m, 2 * hid_dim, dtype = new_atom_fea.dtype, device=new_atom_fea.device)
        
        atom1_feats = new_atom_fea.gather(1, atom_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, hid_dim))
        atom2_feats = new_atom_fea.gather(1, atom_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, hid_dim))

        new_atom_fea_pairs = torch.cat((atom1_feats, atom2_feats), dim=-1)
        atom_fea_pairs[valid_pairs_mask] = new_atom_fea_pairs[valid_pairs_mask]
        return atom_fea_pairs
    
class AAPHead(nn.Module):
    """
    head for Atom Angle Prediction 
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, new_atom_fea, angle_pairs,angle_labels):
        # new_atom_fea: [B, max_graph_len, hid_dim]

        atom_fea_pairs = self.get_atom_pairs_features(new_atom_fea, angle_pairs)  # [B,n_angle, 3*hid_dim]
        # Handle symmetry of atom pairs
        hid_dim=new_atom_fea.size(2)
        atom_fea_pairs_reversed = torch.cat((atom_fea_pairs[:, :, -hid_dim:],atom_fea_pairs[:, :, hid_dim:-hid_dim], atom_fea_pairs[:, :, :hid_dim]), dim=-1)
        predictions_original = self.mlp(atom_fea_pairs)  # [B,n_angle, 1]
        predictions_reversed = self.mlp(atom_fea_pairs_reversed)  # [B,n_angle, 1]
        # Sum the predictions from both orders
        predictions = (predictions_original + predictions_reversed) / 2  # [B,n_angle, 1]
        predictions = predictions.squeeze(-1)  # [B,n_angle]
        return predictions,angle_pairs, angle_labels # [B,n_angle],[B,n_angle, 2], [B,n_angle] # need to consider invalid positions (-100.0) afterwards
    
    def get_atom_pairs_features(self, new_atom_fea, atom_pairs): 
        B, m, _ = atom_pairs.size()
        hid_dim = new_atom_fea.size(-1)
        atom_pairs = atom_pairs.to(new_atom_fea.device)
        valid_pairs_mask = (atom_pairs[:, :, 0] != -1) & (atom_pairs[:, :, 1] != -1) & (atom_pairs[:, :, 2] != -1)
        # replace -1 with 0 to prevent invalid index in next step
        atom_pairs = torch.where(atom_pairs == -1, 0, atom_pairs)
        atom_fea_pairs = torch.zeros(B, m, 3 * hid_dim, dtype = new_atom_fea.dtype, device=new_atom_fea.device)
        
        atom1_feats = new_atom_fea.gather(1, atom_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, hid_dim))
        atom2_feats = new_atom_fea.gather(1, atom_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, hid_dim))
        atom3_feats = new_atom_fea.gather(1, atom_pairs[:, :, 2].unsqueeze(-1).expand(-1, -1, hid_dim))

        new_atom_fea_pairs = torch.cat((atom1_feats, atom2_feats, atom3_feats), dim=-1)
        atom_fea_pairs[valid_pairs_mask] = new_atom_fea_pairs[valid_pairs_mask]
        return atom_fea_pairs
    
class SGPHead(nn.Module):
    """
    head for Space group prediction
    """

    def __init__(self, hid_dim, num_space_groups=230):# NUMBER of space groups , label need to -1 when compare
        super().__init__()
        self.fc = nn.Linear(hid_dim, num_space_groups)  
    def forward(self, x):
        x = self.fc(x)
        return x

class SEPHead(nn.Module):
    """
    head for Symmetry element prediction
    """

    def __init__(self, hid_dim, num_symmetry_elements=7):# NUMBER of Symmetry elements
        super().__init__()
        self.fc = nn.Linear(hid_dim, num_symmetry_elements)  
    def forward(self, x):
        x = self.fc(x)
        return x

class UCPHead(nn.Module):
    """
    head for Unit cell prediction
    """

    def __init__(self, hid_dim, num_cell_para=6):# NUMBER of cell parameters
        super().__init__()
        self.fc = nn.Linear(hid_dim, num_cell_para)  
    def forward(self, x):
        x = self.fc(x)
        return x
    
class CDPHead(nn.Module):
    """
    head for Crystal density prediction 
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hid_dim)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        if x.size(0) > 1:
            x = self.bn(x)
        x = self.fc(x)
        return x
    
class ClassificationHead(nn.Module):
    """
    head for Classification
    """

    def __init__(self, hid_dim, n_classes):
        super().__init__()
        self.average_pooling = AveragePooling()
        if n_classes == 2:
            self.fc = nn.Linear(hid_dim, 1)
            self.binary = True
        else:
            self.fc = nn.Linear(hid_dim, n_classes)
            self.binary = False
            
    def forward(self, x):
        x = self.fc(x)
        return x, self.binary

class RegressionHead(nn.Module):
    """
    head for Regression
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

# class ClassificationHead(nn.Module):
#     """
#     head for Classification
#     """

#     def __init__(self, hid_dim, n_classes):
#         super().__init__()
#         self.average_pooling = AveragePooling()
#         if n_classes == 2:
#             self.fc_global = nn.Linear(hid_dim, 1)
#             self.fc_graph = nn.Linear(hid_dim, 1)
#             self.binary = True
#         else:
#             self.fc_global = nn.Linear(hid_dim, n_classes)
#             self.fc_graph = nn.Linear(hid_dim, n_classes)
#             self.binary = False

#     def forward(self, cls, graph, mask):
#         """
#         cls shape: [B,hid_dim]
#         graph shape: [B,max_graph_len,hid_dim]
#         mask shape: [B,max_graph_len]
#         """
#         graph_pooled = self.average_pooling(graph, mask)# [B,hid_dim]
#         x=self.fc_global(cls)+self.fc_graph(graph_pooled)

#         return x, self.binary
    

# class RegressionHead(nn.Module):
#     """
#     head for Regression
#     """

#     def __init__(self, hid_dim):
#         super().__init__()
#         # self.attention_pooling = AttentionPooling(hid_dim)
#         self.average_pooling = AveragePooling()
#         self.fc_global = nn.Linear(hid_dim, 1)
#         self.fc_graph = nn.Linear(hid_dim, 1)
#     def forward(self, cls, graph, mask):
#         """
#         cls shape: [B,hid_dim]
#         graph shape: [B,max_graph_len,hid_dim]
#         mask shape: [B,max_graph_len]
#         """
#         graph_pooled = self.average_pooling(graph, mask)# [B,hid_dim]
#         x=self.fc_global(cls)+self.fc_graph(graph_pooled)
#         return x
    
# class RegressionHead(nn.Module):
#     """
#     head for Regression
#     """

#     def __init__(self, hid_dim):
#         super().__init__()
#         # self.attention_pooling = AttentionPooling(hid_dim)
#         self.average_pooling = AveragePooling()
#         self.global_read_out = MLPReadout(hid_dim, 1, 1)
#         self.graph_read_out = MLPReadout(hid_dim, 1, 1)
#         # self.read_out = MLPReadout(2*hid_dim, 1)
#     def forward(self, cls, graph, mask):
#         """
#         cls shape: [B,hid_dim]
#         graph shape: [B,max_graph_len,hid_dim]
#         mask shape: [B,max_graph_len]
#         """
#         graph_pooled = self.average_pooling(graph, mask)# [B,hid_dim]
#         x=self.global_read_out(cls)+self.graph_read_out(graph_pooled)

#         # below not perform well
#         # graph_pooled = F.softplus(self.fc(F.softplus(graph_pooled)))# [B,hid_dim]
#         # concat = torch.cat((cls, graph_pooled), dim=-1)# [B,2*hid_dim]
#         # x = self.read_out(concat)
#         return x


# class ClassificationHead(nn.Module):
#     """
#     head for Classification
#     """

#     def __init__(self, hid_dim, n_classes):
#         super().__init__()
#         self.attention_pooling = AttentionPooling(hid_dim)
#         self.fc=nn.Linear(hid_dim, hid_dim)
#         if n_classes == 2:
#             self.read_out = MLPReadout(2*hid_dim, 1)
#             self.binary = True
#         else:
#             self.read_out = MLPReadout(2*hid_dim, n_classes)
#             self.binary = False

#     def forward(self, cls, graph, mask):
#         """
#         cls shape: [B,hid_dim]
#         graph shape: [B,max_graph_len,hid_dim]
#         mask shape: [B,max_graph_len]
#         """
#         graph_pooled = self.attention_pooling(graph, mask)# [B,hid_dim]
#         # graph_pooled = F.softplus(self.fc(F.softplus(graph_pooled)))# [B,hid_dim]
#         concat = torch.cat((cls, graph_pooled), dim=-1)# [B,2*hid_dim]
#         x = self.read_out(concat)
#         return x, self.binary

class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()

    def forward(self, x, mask=None):
        """
        x shape: [B, max_graph_len, hid_dim]
        mask shape: [B, max_graph_len], where valid positions are 1, and invalid positions are 0
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)  
            x_masked = x * mask_expanded.float() 
            sum_pooled = torch.sum(x_masked, dim=1)
            valid_counts = mask.sum(dim=1, keepdim=True)  # [B, 1]
            pooled = sum_pooled / valid_counts.float()  
        else:
            pooled = torch.mean(x, dim=1)
        return pooled  # [B, hid_dim]
    
class AttentionPooling(nn.Module):
    def __init__(self, hid_dim):
        super(AttentionPooling, self).__init__()
        # Linear layer to compute attention weights
        self.attention_fc = nn.Linear(hid_dim, 1)

    def forward(self, x, mask=None):
        """
        x shape: [B,max_graph_len,hid_dim]
        mask shape: [B,max_graph_len], where valid positions are 1, and invalid positions are 0
        """
        attention_scores = self.attention_fc(x).squeeze(-1)  # [B,max_graph_len]
        
        if mask is not None:
            # Set scores at invalid positions to negative infinity
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [B,max_graph_len,1]
        pooled = torch.sum(x * attention_weights, dim=1)  # [B,hid_dim]
        self.last_attention_weights = attention_weights
        
        return pooled # [B,hid_dim]

    def get_last_attention_weights(self):
        """
        Return the attention weights used in the last forward pass
        """
        if self.last_attention_weights is None:
            raise ValueError("Attention weights have not been computed. Perform a forward pass first.")
        return self.last_attention_weights

class MLPReadout(nn.Module):
    """
    Readout function.https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/layers/mlp_readout_layer.py
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        L: int = 2,
    ):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
if __name__ == "__main__":
    main()  