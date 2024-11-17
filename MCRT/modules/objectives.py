import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def compute_regression(pl_module, batch, normalizer,infer):
    logits = pl_module.regression_head(infer["cls_feats"]).squeeze(-1)  # [B]
    # for complex head
    # logits = pl_module.regression_head(infer["cls_feats"],infer["graph_feats"],infer["graph_masks"]).squeeze(-1)  # [B]

    target = [float(item) for item in batch["target"]]
    labels = torch.FloatTensor(target).to(logits.device)  # [B]
    assert len(labels.shape) == 1

    # normalize encode if config["mean"] and config["std], else pass
    labels = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "regression_loss": loss,
        "regression_logits": normalizer.decode(logits),
        "regression_labels": normalizer.decode(labels),
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_regression_loss")(ret["regression_loss"])
    mae = getattr(pl_module, f"{phase}_regression_mae")(
        mean_absolute_error(ret["regression_logits"], ret["regression_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"regression/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"regression/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_classification(pl_module, batch, infer):

    logits, binary = pl_module.classification_head(
        infer["cls_feats"]
    )  # [B, output_dim]
    # for complex head
    # logits, binary = pl_module.classification_head(
    #     infer["cls_feats"],infer["graph_feats"],infer["graph_masks"]
    # )  # [B, output_dim]
    target = [int(item) for item in batch["target"]]
    labels = torch.LongTensor(target).to(logits.device)  # [B]
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
    else:
        loss = F.cross_entropy(logits, labels)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_classification_loss")(
        ret["classification_loss"]
    )
    acc = getattr(pl_module, f"{phase}_classification_accuracy")(
        ret["classification_logits"], ret["classification_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"classification/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"classification/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_map(pl_module, infer ,batch):
    map_logits = pl_module.map_head(infer["graph_feats"])  # [B, max_graph_len, num_atom_types]
    map_labels = batch["atom_label"].to(dtype=torch.long, device=map_logits.device)  # [B]

    mask = map_labels != -1  # [B, max_image_len]
    map_labels = map_labels-1 #atom number need to -1

    # masking
    map_logits = map_logits[mask]  # [valid_N, num_atom_types]
    map_labels = map_labels[mask].long()  # [valid_N]

    map_loss = F.cross_entropy(map_logits, map_labels)

    ret = {
        "map_loss": map_loss,
        "map_logits": map_logits,
        "map_labels": map_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_map_loss")(ret["map_loss"])
    acc = getattr(pl_module, f"{phase}_map_accuracy")(
        ret["map_logits"], ret["map_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"map/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"map/{phase}/accuracy", acc, sync_dist=True)

    return ret

def compute_apc(pl_module, infer, batch):
    apc_logits,atom_pairs,apc_labels = pl_module.apc_head(infer["graph_feats"],batch["atom_pairs"],batch["ap_labels"],)# [B, m],[B, m, 2], [B, m]
    device = pl_module.device 
    apc_logits = apc_logits.view(-1).float().to(device) 
    apc_labels = apc_labels.view(-1).float().to(device)
    valid_mask = apc_labels != -1
    apc_loss = F.binary_cross_entropy_with_logits(
        input=apc_logits[valid_mask], target=apc_labels[valid_mask]
    )

    ret = {
        "apc_loss": apc_loss,
        "apc_logits": apc_logits,
        "apc_labels": apc_labels,
        # "atom_pairs": atom_pairs
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_apc_loss")(ret["apc_loss"])
    acc = getattr(pl_module, f"{phase}_apc_accuracy")(
        ret["apc_logits"], ret["apc_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"apc/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"apc/{phase}/accuracy", acc, sync_dist=True)

    return ret

def compute_adp(pl_module, infer, batch):
    adp_logits,dist_pairs,dist_labels = pl_module.adp_head(infer["graph_feats"],batch["dist_pairs"],batch["dist_labels"],)# [B,n_dist],[B,n_dist, 2], [B,n_dist]
    device = pl_module.device 
    adp_logits = adp_logits.view(-1).float().to(device) 
    adp_labels = dist_labels.view(-1).float().to(device)
    valid_mask = adp_labels != -1.0

    adp_loss = F.mse_loss(adp_logits[valid_mask], adp_labels[valid_mask])
    ret = {
        "adp_loss": adp_loss,
        "adp_logits": adp_logits,
        "adp_labels": adp_labels,
        # "dist_pairs": dist_pairs
    }
    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_adp_loss")(ret["adp_loss"])
    mae = getattr(pl_module, f"{phase}_adp_mae")(
        mean_absolute_error(ret["adp_logits"], ret["adp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"adp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"adp/{phase}/mae", mae, sync_dist=True)
    return ret

def compute_aap(pl_module, infer, batch):
    aap_logits,angle_pairs,angle_labels = pl_module.aap_head(infer["graph_feats"],batch["angle_pairs"],batch["angle_labels"],)# [B,n_angle],[B,n_angle, 2], [B,n_angle]
    device = pl_module.device 
    aap_logits = aap_logits.view(-1).float().to(device) 
    aap_labels = angle_labels.view(-1).float().to(device)
    valid_mask = aap_labels != -100.0

    aap_loss = F.mse_loss(aap_logits[valid_mask], aap_labels[valid_mask])
    ret = {
        "aap_loss": aap_loss,
        "aap_logits": aap_logits,
        "aap_labels": aap_labels,
        # "angle_pairs": angle_pairs
    }
    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_aap_loss")(ret["aap_loss"])
    mae = getattr(pl_module, f"{phase}_aap_mae")(
        mean_absolute_error(ret["aap_logits"], ret["aap_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"aap/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"aap/{phase}/mae", mae, sync_dist=True)
    return ret

def compute_sgp(pl_module, infer, batch):
    sgp_logits = pl_module.sgp_head(infer["cls_feats"])  # [B, hid_dim]
    sgp_labels = torch.LongTensor(batch["space_group"]).to(sgp_logits.device)  # [B]
    sgp_labels = sgp_labels-1 #space group number need to -1 to match the logit
    sgp_loss = F.cross_entropy(sgp_logits, sgp_labels)  # [B]
    
    ret = {
        "sgp_loss": sgp_loss,
        "sgp_logits": sgp_logits,
        "sgp_labels": sgp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_sgp_loss")(ret["sgp_loss"])
    acc = getattr(pl_module, f"{phase}_sgp_accuracy")(
        ret["sgp_logits"], ret["sgp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"sgp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"sgp/{phase}/accuracy", acc, sync_dist=True)

    return ret

def compute_sep(pl_module, infer, batch):
    sep_logits = pl_module.sep_head(infer["cls_feats"])  # [B, num_symmetry_elements], multi-hot
    sep_labels = torch.Tensor(batch["symm_elem"]).to(sep_logits.device)  # [B, num_symmetry_elements]
    sep_weights=pl_module.hparams.config["sep_weights"]
    sep_weights = torch.tensor(sep_weights, dtype=torch.float).to(sep_logits.device) 
    def weighted_bce_with_logits_loss(input, target, weights):
        input = torch.sigmoid(input)
        loss = -weights * (target * torch.log(input) + (1 - target) * torch.log(1 - input))
        return loss.mean()
    sep_loss = weighted_bce_with_logits_loss(sep_logits, sep_labels,sep_weights)

    ret = {
        "sep_loss": sep_loss,
        "sep_logits": sep_logits,
        "sep_labels": sep_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_sep_loss")(ret["sep_loss"])
    acc = getattr(pl_module, f"{phase}_sep_accuracy")(
        ret["sep_logits"], ret["sep_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"sep/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"sep/{phase}/accuracy", acc, sync_dist=True)

    return ret

def compute_cdp(pl_module, infer, batch):
    cdp_logits = pl_module.cdp_head(infer["cls_feats"]).squeeze(-1)  # [B]
    cdp_labels = torch.FloatTensor(batch["density"]).to(cdp_logits.device)

    assert len(cdp_labels.shape) == 1

    cdp_loss = F.mse_loss(cdp_logits, cdp_labels)

    ret = {
        "cdp_loss": cdp_loss,
        "cdp_logits": cdp_logits,
        "cdp_labels": cdp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_cdp_loss")(ret["cdp_loss"])
    mae = getattr(pl_module, f"{phase}_cdp_mae")(
        mean_absolute_error(ret["cdp_logits"], ret["cdp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"cdp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"cdp/{phase}/mae", mae, sync_dist=True)

    return ret

def compute_ucp(pl_module, infer, batch):
    ucp_logits = pl_module.ucp_head(infer["cls_feats"])  # [B, 6]
    ucp_labels = torch.FloatTensor(batch["ucp_labels"]).to(ucp_logits.device)  # [B, 6]
    assert len(ucp_labels.shape) == 2
    assert ucp_labels.shape[1] == 6

    ucp_loss = F.mse_loss(ucp_logits, ucp_labels)
    # print(ucp_loss)
    ret = {
        "ucp_loss": ucp_loss,
        "ucp_logits": ucp_logits,
        "ucp_labels": ucp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ucp_loss")(ret["ucp_loss"])
    mae = getattr(pl_module, f"{phase}_ucp_mae")(
        mean_absolute_error(ret["ucp_logits"], ret["ucp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"ucp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"ucp/{phase}/mae", mae, sync_dist=True)

    return ret