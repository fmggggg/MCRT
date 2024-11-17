"""
project distance matrix and angle matrix to hid_dim
"""
import torch
import torch.nn as nn
import math
from MCRT.data_processor.dataset import Dataset

def main():
    dataset_test=Dataset(r'D:\Projects\MyProjects\MCRT\MCRT\cifs\test','train',1023,4)
    batch=[dataset_test[i] for i in range(32)]
    collated_batch=dataset_test.collate(batch)
    print(collated_batch["padded_angle_matrices"])
    angle_MLP_Projector=MLP_Projector(
        in_features=6,
        hidden_features=512,
        out_features=512,
        act_layer=nn.GELU,
        drop=0.0,)
    angleoutput=angle_MLP_Projector(collated_batch["padded_angle_matrices"],collated_batch["mask_angle_matrices"])
    print(angleoutput.shape)

    dist_MLP_Projector=MLP_Projector(
        in_features=1023,
        hidden_features=512,
        out_features=512,
        act_layer=nn.GELU,
        drop=0.0,)
    output=dist_MLP_Projector(collated_batch["padded_distance_matrices"],collated_batch["mask_distance_matrices"])
    print(output.shape)
    

class MLP_Projector(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if mask.dim() == 3:
            mask = mask.any(dim=-1, keepdim=True)
        x = x * mask
        return x



class PositionalEncoding3D(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(PositionalEncoding3D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, x, pos):
        pos = pos * 10
        device = x.device
        div = torch.exp(torch.arange(0., self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim)).to(device).type(x.dtype)
        for i in range(3):
            pe = torch.zeros(x.shape, device=device, dtype=x.dtype)
            pe[..., 0::2] = torch.sin(pos[..., i].unsqueeze(-1) * div)
            pe[..., 1::2] = torch.cos(pos[..., i].unsqueeze(-1) * div)
            x += pe
        return self.dropout(x)

class PositionalEncoding3D_new(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(PositionalEncoding3D_new, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        self.pe_linear=nn.Linear(3*embed_dim,embed_dim)

    def forward(self, x, pos):
      
        pos = pos * 10

       
        div = torch.exp(torch.arange(0., self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim)).double().cuda()
        for i in range(3):
            pe = torch.zeros(x.shape).cuda()
            pe[..., 0::2] = torch.sin(pos[..., i].unsqueeze(2) * div)
            pe[..., 1::2] = torch.cos(pos[..., i].unsqueeze(2) * div)
            if i==0:
                pe_all=pe
            else:
                pe_all=torch.cat((pe_all,pe),dim=-1)
        pe_final=self.pe_linear(pe_all)
        x += pe_final
        return self.dropout(x)

if __name__ == "__main__":
    main()  