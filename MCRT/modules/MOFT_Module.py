from typing import Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from MCRT.modules.cgcnn import GraphEmbeddings
from MCRT.modules.alignn import ALIGNN
from MCRT.modules.projector import MLP_Projector,PositionalEncoding3D
from MCRT.modules.McrtTransformer import MCRT_Transformer
from MCRT.modules.MOFT_transformer import MOFT_Transformer
from MCRT.modules import objectives, heads ,utils

import numpy as np
from sklearn.metrics import r2_score


class MOFT_Module(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.vis = config["visualize"]
        self.pos_emb = config["pos_emb"]
        self.if_alignn = config["if_alignn"]
        self.if_conv = config["if_conv"]
        self.if_image = config["if_image"]
        self.if_grid = config["if_grid"]

        if self.if_grid:
            assert self.if_image == False,print("make sure set if_image to False if grid is enabled!")
            print("if_grid = True")
        if self.if_image:
            assert self.if_grid == False,print("make sure set if_grid to False if image is enabled!")
            print("if_image = True")

        print("Using positional embedding:",self.pos_emb)
        # graph embedding
        if not self.if_alignn:
            self.graph_embeddings = GraphEmbeddings(
                atom_fea_len=config["atom_fea_len"],
                nbr_fea_len=config["nbr_fea_len"],
                max_graph_len=config["max_graph_len"],
                hid_dim=config["hid_dim"],
                n_conv=config["n_conv"],
                if_conv=config["if_conv"]
            )
            self.graph_embeddings.apply(objectives.init_weights)

        if self.if_alignn:
            self.GNN_embeddings = ALIGNN( # adjust later to make it flexible
                num_conv=config["alignn_num_conv"],
                hidden_dim=config["alignn_hidden_dim"],
                rbf_distance_dim=config["alignn_rbf_distance_dim"],
                rbf_triplet_dim=config["alignn_rbf_triplet_dim"],
                batch_norm=config["alignn_batch_norm"],
                dropout=config["alignn_dropout"],
                residual=config["alignn_residual"],
                transformer_hidden_dim=config["hid_dim"],
                max_graph_len=config["max_graph_len"],
            )
            self.GNN_embeddings.apply(objectives.init_weights)

        # chemical positional embedding
        if self.pos_emb == "relative":
            self.dist_MLP_Projector=MLP_Projector(
                in_features=config["max_graph_len"],
                hidden_features=config["hid_dim"],
                out_features=config["hid_dim"],
                drop=config["drop_rate"],)
            self.dist_MLP_Projector.apply(objectives.init_weights)
            self.angle_MLP_Projector=MLP_Projector(
                in_features=int(config["angle_nbr"]*(config["angle_nbr"]-1)/2),
                hidden_features=config["hid_dim"],
                out_features=config["hid_dim"],
                drop=config["drop_rate"],)
            self.angle_MLP_Projector.apply(objectives.init_weights)
        elif self.pos_emb == "absolute":
            self.pe_3D = PositionalEncoding3D(embed_dim=config["hid_dim"], dropout=config["drop_rate"])
            self.pe_3D.apply(objectives.init_weights)
        elif self.pos_emb=='both':
            self.dist_MLP_Projector=MLP_Projector(
                in_features=config["max_graph_len"],
                hidden_features=config["hid_dim"],
                out_features=config["hid_dim"],
                drop=config["drop_rate"],)
            self.dist_MLP_Projector.apply(objectives.init_weights)
            self.angle_MLP_Projector=MLP_Projector(
                in_features=int(config["angle_nbr"]*(config["angle_nbr"]-1)/2),
                hidden_features=config["hid_dim"],
                out_features=config["hid_dim"],
                drop=config["drop_rate"],)    
            self.angle_MLP_Projector.apply(objectives.init_weights)
            self.pe_3D = PositionalEncoding3D(embed_dim=config["hid_dim"], dropout=config["drop_rate"])   
            self.pe_3D.apply(objectives.init_weights)     
        else:
            print(f"Please set the pos_emb ('relative' or 'absolute' or 'both'), now it's {self.pos_emb}, the model will run without pos_emb")

        # transformer
        if not self.if_grid:
            print("Transformer using MCRT_Transformer")
            self.transformer = MCRT_Transformer(
                num_blocks=config["num_blocks"],
                dim=config["hid_dim"],
                num_heads=config["num_heads"],
                img_size=config["img_size"], 
                patch_size=config["patch_size"], 
                in_chans=config["in_chans"], 
                mlp_ratio=config["mlp_ratio"],
                qkv_bias=config["qkv_bias"],
                qk_scale=config["qk_scale"],
                drop=config["drop_rate"],
                attn_drop=config["attn_drop"],
                drop_path_rate=config["drop_path_rate"],
            )
        else:
            print("Transformer using MOFT_Transformer")
            self.transformer = MOFT_Transformer(
                num_blocks=config["num_blocks"],
                dim=config["hid_dim"],
                num_heads=config["num_heads"],
                grid_img_size=config["grid_img_size"],
                grid_patch_size=config["grid_patch_size"],
                grid_in_chans=config["grid_in_chans"],
                mlp_ratio=config["mlp_ratio"],
                qkv_bias=config["qkv_bias"],
                qk_scale=config["qk_scale"],
                drop=config["drop_rate"],
                attn_drop=config["attn_drop"],
                drop_path_rate=config["drop_path_rate"],
            )            
            self.volume_embeddings = nn.Linear(1, config["hid_dim"])
            self.volume_embeddings.apply(objectives.init_weights)
            
        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)
        # sep token
        self.sep_embeddings = nn.Linear(1, config["hid_dim"])
        self.sep_embeddings.apply(objectives.init_weights)
        # scale token
        self.birth_1d_embeddings = nn.Linear(1, config["hid_dim"])
        self.birth_1d_embeddings.apply(objectives.init_weights)
        self.pers_1d_embeddings = nn.Linear(1, config["hid_dim"])
        self.pers_1d_embeddings.apply(objectives.init_weights)
        self.birth_2d_embeddings = nn.Linear(1, config["hid_dim"])
        self.birth_2d_embeddings.apply(objectives.init_weights)
        self.pers_2d_embeddings = nn.Linear(1, config["hid_dim"])
        self.pers_2d_embeddings.apply(objectives.init_weights)

        # token type embeddings
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # pooler
        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # ===================== loss =====================
        if config["loss_names"]["map"] > 0:
            self.map_head = heads.MAPHead(config["hid_dim"])
            self.map_head.apply(objectives.init_weights)

        if config["loss_names"]["apc"] > 0:
            self.apc_head = heads.APCHead(config["hid_dim"])
            self.apc_head.apply(objectives.init_weights)

        if config["loss_names"]["adp"] > 0:
            self.adp_head = heads.ADPHead(config["hid_dim"])
            self.adp_head.apply(objectives.init_weights)

        if config["loss_names"]["aap"] > 0:
            self.aap_head = heads.AAPHead(config["hid_dim"])
            self.aap_head.apply(objectives.init_weights)

        if config["loss_names"]["sgp"] > 0:
            self.sgp_head = heads.SGPHead(config["hid_dim"])
            self.sgp_head.apply(objectives.init_weights)

        if config["loss_names"]["sep"] > 0:
            self.sep_head = heads.SEPHead(config["hid_dim"])
            self.sep_head.apply(objectives.init_weights)

        if config["loss_names"]["cdp"] > 0:
            self.cdp_head = heads.CDPHead(config["hid_dim"])
            self.cdp_head.apply(objectives.init_weights)
        

        # ===================== Downstream =====================
        if config["load_path"] != "" and not config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        if self.hparams.config["loss_names"]["regression"] > 0:
            self.regression_head = heads.RegressionHead(config["hid_dim"])
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]

        if self.hparams.config["loss_names"]["classification"] > 0:
            n_classes = config["n_classes"]
            self.classification_head = heads.ClassificationHead(config["hid_dim"], n_classes)
            self.classification_head.apply(objectives.init_weights)

        utils.set_metrics(self)
        self.current_tasks = list()
        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        self.test_logits = []
        self.test_labels = []
        self.test_cifid = []
        self.write_log = True

    def infer(
        self,
        batch,
    ):
        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]  # [N'], N' is all atoms in one batch
        nbr_idx = batch["nbr_fea_idx"]  # [N', M]
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        if self.if_image:
            image = batch["image"]  # [B,chan,h,w]
            max_birth_1d = batch["max_birth_1d"]#list [B] 
            max_persistence_1d = batch["max_persistence_1d"]#list [B]
            max_birth_2d = batch["max_birth_2d"]#list [B]
            max_persistence_2d = batch["max_persistence_2d"]#list [B]
            # for different dim of pers imgs, we should treat them separately
            image_channels = [] 
            for i in range(image.shape[1]):
                single_channel_img = image[:, i, :, :].unsqueeze(1)  # keep [B,1,h,w]
                image_channels.append(single_channel_img)

        if self.if_grid:
            grid = batch["grid"]  # [B, C, H, W, D]
            volume = batch["volume"]  # list [B]

        if self.if_conv:
            # get graph embeds
            (graph_embeds, # [B, max_graph_len, hid_dim]
            graph_masks # [B, max_graph_len]
            ) = self.graph_embeddings(atom_num=atom_num, nbr_idx=nbr_idx, nbr_fea=nbr_fea, \
                    crystal_atom_idx=crystal_atom_idx)
        
        if self.if_alignn:
            # graph_masks = batch["mask"] # [B, max_graph_len]
            (graph_embeds, # [B, max_graph_len, hid_dim]
            graph_masks # [B, max_graph_len]
            ) = self.GNN_embeddings(batch) # [B, max_graph_len, hid_dim]

        # add chemical_position_embeddings
        if self.pos_emb == "relative":
            padded_distance_matrices = batch["padded_distance_matrices"] # [B,max_graph_len,max_graph_len]
            mask_distance_matrices = batch["mask_distance_matrices"] # [B,max_graph_len,max_graph_len]
            padded_angle_matrices = batch["padded_angle_matrices"] # [B,max_graph_len,angle_nbr*(angle_nbr-1)/2]
            mask_angle_matrices = batch["mask_angle_matrices"]# [B,max_graph_len,angle_nbr*(angle_nbr-1)/2]
            dist_embeddings=self.dist_MLP_Projector(padded_distance_matrices,mask_distance_matrices) # [B, max_graph_len, hid_dim]
            angle_embeddings=self.angle_MLP_Projector(padded_angle_matrices,mask_angle_matrices) # [B, max_graph_len, hid_dim]
            graph_embeds = graph_embeds + dist_embeddings + angle_embeddings
        elif self.pos_emb == "absolute":
            padded_abs_pos = batch["padded_abs_pos"] # [B,max_graph_len,3]
            graph_embeds = self.pe_3D(graph_embeds,padded_abs_pos)
        elif self.pos_emb == "both":
            padded_distance_matrices = batch["padded_distance_matrices"] # [B,max_graph_len,max_graph_len]
            mask_distance_matrices = batch["mask_distance_matrices"] # [B,max_graph_len,max_graph_len]
            padded_angle_matrices = batch["padded_angle_matrices"] # [B,max_graph_len,angle_nbr*(angle_nbr-1)/2]
            mask_angle_matrices = batch["mask_angle_matrices"]# [B,max_graph_len,angle_nbr*(angle_nbr-1)/2]
            dist_embeddings=self.dist_MLP_Projector(padded_distance_matrices,mask_distance_matrices) # [B, max_graph_len, hid_dim]
            angle_embeddings=self.angle_MLP_Projector(padded_angle_matrices,mask_angle_matrices) # [B, max_graph_len, hid_dim]
            graph_embeds = graph_embeds + dist_embeddings + angle_embeddings
            padded_abs_pos = batch["padded_abs_pos"] # [B,max_graph_len,3]
            graph_embeds = self.pe_3D(graph_embeds,padded_abs_pos)
        else:
            pass

        # add class embeds to graph_embeds
        cls_tokens = torch.zeros(len(crystal_atom_idx)).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(crystal_atom_idx), 1).to(graph_masks)  # [B, 1]

        graph_embeds = torch.cat(
            [cls_embeds, graph_embeds], dim=1
        )  # [B, 1+max_graph_len, hid_dim]
        graph_masks = torch.cat([cls_mask, graph_masks], dim=1)  # [B, 1+max_graph_len]

        # get image embeds
        if self.if_image:
            # for 1d image
            (
                image_embeds_1d,  # [B, num_patches, hid_dim]
                image_masks_1d,  # [B, num_patches]
            ) = self.transformer.image_embed_1d(
                image_channels[0]
            )        
            # for 2d image
            (
                image_embeds_2d,  # [B, num_patches, hid_dim]
                image_masks_2d,  # [B, num_patches]
            ) = self.transformer.image_embed_2d(
                image_channels[1]
            )  

            # add sep embeds to image_embeds
            sep_tokens = torch.zeros(len(crystal_atom_idx)).to(image_embeds_1d)  # [B]
            sep_embeds = self.sep_embeddings(sep_tokens[:, None, None])  # [B, 1, hid_dim]
            sep_mask = torch.ones(len(crystal_atom_idx), 1).to(image_masks_1d)  # [B, 1]   
            # scale embeds
            max_birth_1d = torch.FloatTensor(max_birth_1d).to(image_embeds_1d)  # [B]
            birth_1d_embeds = self.birth_1d_embeddings(max_birth_1d[:, None, None])  # [B, 1, hid_dim]
            birth_1d_mask = torch.ones(max_birth_1d.shape[0], 1).to(image_masks_1d) # [B, 1]  

            max_persistence_1d = torch.FloatTensor(max_persistence_1d).to(image_embeds_1d)  # [B]
            pers_1d_embeds = self.pers_1d_embeddings(max_persistence_1d[:, None, None])  # [B, 1, hid_dim]
            pers_1d_mask = torch.ones(max_persistence_1d.shape[0], 1).to(image_masks_1d) # [B, 1] 

            max_birth_2d = torch.FloatTensor(max_birth_2d).to(image_embeds_2d)  # [B]
            birth_2d_embeds = self.birth_2d_embeddings(max_birth_2d[:, None, None])  # [B, 1, hid_dim]
            birth_2d_mask = torch.ones(max_birth_2d.shape[0], 1).to(image_masks_2d) # [B, 1] 

            max_persistence_2d = torch.FloatTensor(max_persistence_2d).to(image_embeds_2d)  # [B]
            pers_2d_embeds = self.pers_2d_embeddings(max_persistence_2d[:, None, None])  # [B, 1, hid_dim]
            pers_2d_mask = torch.ones(max_persistence_2d.shape[0], 1).to(image_masks_2d) # [B, 1] 

            image_embeds = torch.cat(
                [sep_embeds, image_embeds_1d,birth_1d_embeds,pers_1d_embeds,image_embeds_2d,birth_2d_embeds,pers_2d_embeds], dim=1
            )  # [B, 1+num_patches, hid_dim]
            image_masks = torch.cat([sep_mask, image_masks_1d,birth_1d_mask,pers_1d_mask,image_masks_2d,birth_2d_mask,pers_2d_mask], dim=1)  # [B, 1+num_patches]


            # add token_type_embeddings
            graph_embeds = graph_embeds + self.token_type_embeddings(
                torch.zeros_like(graph_masks, device=self.device).long()
            )
            image_embeds = image_embeds + self.token_type_embeddings(
                torch.ones_like(image_masks, device=self.device).long()
            )

            # concat graph and image
            co_embeds = torch.cat(
                [graph_embeds, image_embeds], dim=1
            )  # [B, final_max_len, hid_dim]
            co_masks = torch.cat(
                [graph_masks, image_masks], dim=1
            )  # [B, final_max_len]        

            x = co_embeds
            mask = co_masks
        else:
            x = graph_embeds
            mask = graph_masks

        if self.if_grid:
                        # get grid embeds
            (
                grid_embeds,  # [B, max_grid_len+1, hid_dim]
                grid_masks,  # [B, max_grid_len+1]
            ) = self.transformer.visual_embed_3d(
                grid,
            )
            # add volume embeds to grid_embeds
            volume = torch.FloatTensor(volume).to(grid_embeds)  # [B]
            volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
            volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)

            grid_embeds = torch.cat(
                [grid_embeds, volume_embeds], dim=1
            )  # [B, max_grid_len+2, hid_dim]
            grid_masks = torch.cat([grid_masks, volume_mask], dim=1)  # [B, max_grid_len+2]

            # add token_type_embeddings
            graph_embeds = graph_embeds + self.token_type_embeddings(
                torch.zeros_like(graph_masks, device=self.device).long()
            )
            grid_embeds = grid_embeds + self.token_type_embeddings(
                torch.ones_like(grid_masks, device=self.device).long()
            )

            co_embeds = torch.cat(
                [graph_embeds, grid_embeds], dim=1
            )  # [B, final_max_len, hid_dim]
            co_masks = torch.cat(
                [graph_masks, grid_masks], dim=1
            )  # [B, final_max_len, hid_dim]
            x = co_embeds
            mask = co_masks
        else:
            x = graph_embeds
            mask = graph_masks

            
        # transformer process
        attn_weights = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=mask)
            if self.vis:
                attn_weights.append(_attn)
        x = self.transformer.norm(x) # [B, max_graph_len+num_patches+2, hid_dim]


        if self.if_image:
            # split
            graph_feats, image_feats = (
                x[:, : graph_embeds.shape[1]],
                x[:, graph_embeds.shape[1] :],
            )  # [B, 1+max_graph_len, hid_dim], [B, 1+num_patches, hid_dim]

            cls_feats = self.pooler(x)  # [B, hid_dim]
            graph_feats = graph_feats[:, 1:] # [B, max_graph_len, hid_dim]
            graph_masks = graph_masks[:, 1:] # [B, max_graph_len]
            image_feats = image_feats[:, 1:] # [B, num_patches, hid_dim]
            image_masks = image_masks[:, 1:] # [B, num_patches]
            ret = {
                "graph_feats": graph_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "graph_masks": graph_masks, 
                "image_feats": image_feats,
                "image_masks": image_masks,      
                "cif_id": cif_id,
                "attn_weights": attn_weights,
                "image":image_channels,# list 2 [B,1,h,w]
                "max_birth_1d":max_birth_1d,#list [B]
                "max_persistence_1d":max_persistence_1d,
                "max_birth_2d":max_birth_2d,
                "max_persistence_2d":max_persistence_2d,
            }

        elif self.if_grid:
            graph_feats, grid_feats = (
                x[:, : graph_embeds.shape[1]],
                x[:, graph_embeds.shape[1] :],
            )  # [B, max_graph_len, hid_dim], [B, max_grid_len+2, hid_dim]

            cls_feats = self.pooler(x)  # [B, hid_dim]

            ret = {
                "graph_feats": graph_feats,
                "grid_feats": grid_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "graph_masks": graph_masks,
                "grid_masks": grid_masks,
                "cif_id": cif_id,
                "attn_weights": attn_weights,
            }

        else: #graph only
            graph_feats = x[:, 1:] # [B, max_graph_len, hid_dim]
            graph_masks=graph_masks[:, 1:] # [B, max_graph_len]
            cls_feats = self.pooler(x)  # [B, hid_dim]

            ret = {
                "graph_feats": graph_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "graph_masks": graph_masks,       
                "cif_id": cif_id,
                "attn_weights": attn_weights,
            }
        return ret

    def forward(self, batch):
        ret = dict()
        infer=self.infer(batch)
        ret.update(infer)
        if len(self.current_tasks) == 0:
            return ret
        
        if "map" in self.current_tasks:
            ret.update(objectives.compute_map(self, infer, batch))
        if "apc" in self.current_tasks:
            ret.update(objectives.compute_apc(self, infer, batch))
        if "adp" in self.current_tasks:
            ret.update(objectives.compute_adp(self, infer, batch))
        if "aap" in self.current_tasks:
            ret.update(objectives.compute_aap(self, infer, batch))
        if "sgp" in self.current_tasks:
            ret.update(objectives.compute_sgp(self, infer, batch))
        if "sep" in self.current_tasks:
            ret.update(objectives.compute_sep(self, infer, batch))
        if "cdp" in self.current_tasks:
            ret.update(objectives.compute_cdp(self, infer, batch))

        # regression
        if "regression" in self.current_tasks:
            normalizer = utils.Normalizer(self.mean, self.std)
            ret.update(objectives.compute_regression(self, batch, normalizer,infer))
        # classification
        if "classification" in self.current_tasks:
            ret.update(objectives.compute_classification(self, batch,infer))

        return ret

    def on_train_start(self):
        utils.set_task(self)
        self.write_log = True

    def training_step(self, batch, batch_idx):
        output = self(batch)
        # total_loss = sum([v for k, v in output.items() if "loss" in k])
        total_loss = 0
        for k, v in output.items():
            if "loss" in k:
                task_name = k.split("_")[0]
                weight = self.hparams.config["loss_names"].get(task_name, 1)  # default weight = 1
                total_loss += weight * v
                
        return total_loss

    def on_train_epoch_end(self):
        utils.epoch_wrapup(self)

    def on_validation_start(self):
        utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def on_validation_epoch_end(self) -> None:
        utils.epoch_wrapup(self)

    def on_test_start(self,):
        utils.set_task(self)
    
    def test_step(self, batch, batch_idx):
        output = self(batch)
        output = {
            k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()
        }  # update cpu for memory

        if 'regression_logits' in output.keys():
            self.test_logits += output["regression_logits"].tolist()
            self.test_labels += output["regression_labels"].tolist()
        return output

    def on_test_epoch_end(self):
        utils.epoch_wrapup(self)

        # calculate r2 score when regression
        if len(self.test_logits) > 1:
            r2 = r2_score(
                np.array(self.test_labels), np.array(self.test_logits)
            )
            self.log(f"test/r2_score", r2, sync_dist=True)
            self.test_labels.clear()
            self.test_logits.clear()

    def configure_optimizers(self):
        return utils.set_schedule(self)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # return optimizer
    
    def on_predict_start(self):
        self.write_log = False
        utils.set_task(self)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        
        if 'classification_logits' in output:
            if self.hparams.config['n_classes'] == 2:
                output['classification_logits_index'] = torch.round(output['classification_logits']).to(torch.int)
            else:
                softmax = torch.nn.Softmax(dim=1)
                output['classification_logits'] = softmax(output['classification_logits'])
                output['classification_logits_index'] = torch.argmax(output['classification_logits'], dim=1)

        output = {
            k: (v.cpu().tolist() if torch.is_tensor(v) else v)
            for k, v in output.items()
            if ('logits' in k) or ('labels' in k) or 'cif_id' == k
        }

        return output
    
    def on_predict_epoch_end(self, *args):
        self.test_labels.clear()
        self.test_logits.clear()

    def on_predict_end(self, ):
        self.write_log = True

    def lr_scheduler_step(self, scheduler, *args):
        if len(args) == 2:
            optimizer_idx, metric = args
        elif len(args) == 1:
            metric, = args
        else:
            raise ValueError('lr_scheduler_step must have metric and optimizer_idx(optional)')

        if pl.__version__ >= '2.0.0':
            scheduler.step()
        else:
            scheduler.step()


