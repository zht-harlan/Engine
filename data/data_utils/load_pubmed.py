import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def get_raw_text_pubmed(use_text=False, seed=0):
    dataset = Planetoid("./datasets", "PubMed", transform=T.NormalizeFeatures())
    data = dataset[0]

    rng = np.random.default_rng(seed)
    node_id = np.arange(data.num_nodes)
    rng.shuffle(node_id)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[node_id[: int(data.num_nodes * 0.6)]] = True
    val_mask[node_id[int(data.num_nodes * 0.6) : int(data.num_nodes * 0.8)]] = True
    test_mask[node_id[int(data.num_nodes * 0.8) :]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    if not use_text:
        return data, None

    texts = []
    for idx, row in enumerate(data.x.cpu()):
        nonzero_idx = torch.nonzero(row, as_tuple=False).view(-1)
        top_idx = nonzero_idx[:32].tolist() if nonzero_idx.numel() else [0]
        feature_tokens = " ".join(f"feature_{token_idx}" for token_idx in top_idx)
        texts.append(f"Title: PubMed node {idx}\nAbstract: {feature_tokens}")
    return data, texts
