from typing import Tuple, Any, Dict

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

class MLPReadout(nn.Module):
    """
    Readout function.
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
class GatedGCNLayer(nn.Module):
    """ResGatedGCN: Residual Gated Graph ConvNets

    "An Experimental Study of Neural Networks for Variable Graph"
    ICLR (2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        e: torch.Tensor,
    ):
        """
        Args:
            g (dgl.DGLGraph): DGLGraph
            h (torch.Tensor): embedded node features
            e (torch.Tensor): embedded edge features
        Return:
            h (torch.Tensor): updated node features
            e (torch.Tensor): updated edge features
        """
        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["Dh"] = self.D(h)
        g.ndata["Eh"] = self.E(h)
        g.edata["e"] = e
        g.edata["Ce"] = self.C(e)

        g.apply_edges(fn.u_add_v("Dh", "Eh", "DEh"))
        g.edata["e"] = g.edata["DEh"] + g.edata["Ce"]  # updated edge features (e^_ij)
        g.edata["sigma"] = torch.sigmoid(g.edata["e"])  # sigma(e^_ij)
        # numerator
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        # denominator
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["Ah"] + g.ndata["sum_sigma_h"] / (
            g.ndata["sum_sigma"] + 1e-6
        )  # updated node features
        h = g.ndata["h"]
        e = g.edata["e"]

        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)

        h = F.silu(h)
        e = F.silu(e)

        if self.residual:
            h = h_in + h
            e = e_in + e

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e
class ALIGNNLayer(nn.Module):
    """ALIGNN layer.

    "Atomistic Line Graph Neural Network
    for improved materials property predictions"
    npj Comput. Mater. (2021).
    https://www.nature.com/articles/s41524-021-00650-1
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim (int): dimension of input features
            output_dim (int): dimension of output features
            batch_norm (bool, optional): whether to use batch normalization. Defaults to False.
            residual (bool, optional): whether to use residual connection. Defaults to False.
            dropout (float, optional): a ratio of dropout. Defaults to 0.0.
        """
        super().__init__()
        self.node_update = GatedGCNLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            batch_norm=batch_norm,
            residual=residual,
            dropout=dropout,
        )
        self.edge_update = GatedGCNLayer(
            input_dim=output_dim,
            output_dim=input_dim,
            batch_norm=batch_norm,
            residual=residual,
            dropout=dropout,
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        h: torch.Tensor,
        e: torch.Tensor,
        l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            g (dgl.DGLGraph): DGLGraph
            lg (dgl.DGLGraph): line graph of g
            h (torch.Tensor): embedded node features
            e (torch.Tensor): embedded edge features
            l (torch.Tensor): embedded line graph features (edge pair)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            updated node features, updated edge features, updated line graph features
        """
        # node update
        h, m = self.node_update(g, h, e)
        # edge update
        e, l = self.edge_update(lg, m, l)

        return h, e, l
class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    @torch.no_grad()
    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class ALIGNN(nn.Module):
    """ALIGNN model. without pooling and readout

    "Atomistic Line Graph Neural Network
    for improved materials property predictions"
    npj Comput. Mater. (2021).
    https://www.nature.com/articles/s41524-021-00650-1
    """

    def __init__(self, num_conv,hidden_dim,rbf_distance_dim,rbf_triplet_dim,batch_norm,dropout,residual,transformer_hidden_dim,max_graph_len ):
        super().__init__()
        # config
        self.num_conv = num_conv
        self.hidden_dim = hidden_dim
        self.rbf_distance_dim = rbf_distance_dim
        self.rbf_triplet_dim = rbf_triplet_dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual = residual
        self.transformer_hidden_dim = transformer_hidden_dim
        self.max_graph_len = max_graph_len 

        # layers
        self.node_embedding = nn.Embedding(103, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.rbf_distance_dim, self.hidden_dim)
        self.angle_embedding = nn.Linear(self.rbf_triplet_dim, self.hidden_dim)
        self.rbf_expansion_distance = RBFExpansion(
            vmin=0, vmax=8, bins=self.rbf_distance_dim
        )
        self.rbf_expansion_triplet = RBFExpansion(
            vmin=-1, vmax=1, bins=self.rbf_triplet_dim
        )
        self.conv_layers = nn.ModuleList(
            [
                ALIGNNLayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=self.dropout,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.gated_gcn_layers = nn.ModuleList(
            [
                GatedGCNLayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=self.dropout,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.fc = nn.Linear(self.hidden_dim, self.transformer_hidden_dim)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward propagation.

        Args:
            batch (Dict[str, Any]): batch data including graph, line graph (optional)
            and target

        Returns:
            torch.Tensor: predicted target values (logits)
        """
        graph = batch["graph"]
        line_graph = batch["line_graph"]
        crystal_atom_idx = batch["crystal_atom_idx"]

        # node embedding
        node_attrs = graph.ndata["atomic_number"]
        node_feats = self.node_embedding(node_attrs)
        # edge embedding
        edge_attrs = self.rbf_expansion_distance(graph.edata["distance"])
        edge_feats = self.edge_embedding(edge_attrs)
        # angle (edge pair) embedding
        angle_attrs = self.rbf_expansion_triplet(line_graph.edata["angle"])
        angle_feats = self.angle_embedding(angle_attrs)
        # conv layers
        for conv in self.conv_layers:
            node_feats, edge_feats, angle_feats = conv(
                graph, line_graph, node_feats, edge_feats, angle_feats
            )
        # gated gcn layers
        for gated_gcn in self.gated_gcn_layers:
            node_feats, _ = gated_gcn(graph, node_feats, edge_feats)
        # adapt dim    
        node_feats = self.fc(node_feats)  
        new_node_feats, mask = self.reconstruct_batch(node_feats, crystal_atom_idx)  
        return new_node_feats, mask
    
    def reconstruct_batch(self, atom_fea, crystal_atom_idx):
        batch_size = len(crystal_atom_idx)
        new_atom_fea = torch.zeros(batch_size, self.max_graph_len, self.transformer_hidden_dim).to(atom_fea)
        mask_label = torch.zeros(batch_size, self.max_graph_len).to(atom_fea) # this mask is for label valid position
        for i, idx in enumerate(crystal_atom_idx):
                length = len(idx)
                new_atom_fea[i, :length] = atom_fea[idx]
                mask_label[i, :length] = 1  # valid position set to 1

        return new_atom_fea, mask_label # this mask is for label valid position