import ast
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


CSTAG_DIR_NAMES = {
    "children": "Children",
    "history": "History",
    "photo": "Photo",
}


def _candidate_roots():
    return [
        os.path.normpath("../dataset"),
        os.path.normpath("../datasets"),
        os.path.normpath("E:/dataset"),
        os.path.normpath("E:/\u6570\u636e\u96c6"),
    ]


def resolve_cstag_csv_path(dataset_name):
    canonical_dir = CSTAG_DIR_NAMES[dataset_name]
    file_name = f"{canonical_dir}.csv"
    candidates = []
    for root in _candidate_roots():
        candidates.append(os.path.join(root, "CSTAG", canonical_dir, file_name))
        candidates.append(os.path.join(root, canonical_dir, file_name))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot find CSTAG dataset file for {dataset_name}. Tried: {candidates}"
    )


def _normalize_edge_index(edge_index):
    if edge_index.numel() == 0:
        return edge_index
    reversed_edges = edge_index[[1, 0]]
    edge_index = torch.cat([edge_index, reversed_edges], dim=1)
    return torch.unique(edge_index, dim=1)


def _parse_neighbor_list(raw_neighbors):
    if pd.isna(raw_neighbors):
        return []
    if isinstance(raw_neighbors, (list, tuple)):
        return [int(item) for item in raw_neighbors]
    return [int(item) for item in ast.literal_eval(str(raw_neighbors))]


def _build_split_masks(num_nodes, seed):
    rng = np.random.default_rng(seed)
    node_id = np.arange(num_nodes)
    rng.shuffle(node_id)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[node_id[: int(num_nodes * 0.6)]] = True
    val_mask[node_id[int(num_nodes * 0.6) : int(num_nodes * 0.8)]] = True
    test_mask[node_id[int(num_nodes * 0.8) :]] = True
    return train_mask, val_mask, test_mask


def _load_cstag_dataset(dataset_name, use_text=False, seed=0):
    csv_path = resolve_cstag_csv_path(dataset_name)
    df = pd.read_csv(csv_path).sort_values("node_id").reset_index(drop=True)

    required_columns = {"node_id", "text", "label", "neighbour"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing_columns)}")

    expected_node_ids = np.arange(len(df))
    if not np.array_equal(df["node_id"].to_numpy(), expected_node_ids):
        raise ValueError(f"{csv_path} has non-contiguous node_id values")

    rows = []
    cols = []
    for source_id, raw_neighbors in enumerate(df["neighbour"].tolist()):
        for target_id in _parse_neighbor_list(raw_neighbors):
            rows.append(source_id)
            cols.append(target_id)

    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_index = _normalize_edge_index(edge_index)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    x = torch.ones((len(df), 1), dtype=torch.float32)
    train_mask, val_mask, test_mask = _build_split_masks(len(df), seed)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    if not use_text:
        return data, None

    texts = [f"Title: {dataset_name}\nAbstract: {text}" for text in df["text"].fillna("").astype(str)]
    return data, texts


def get_raw_text_children(use_text=False, seed=0):
    return _load_cstag_dataset("children", use_text=use_text, seed=seed)


def get_raw_text_history(use_text=False, seed=0):
    return _load_cstag_dataset("history", use_text=use_text, seed=seed)


def get_raw_text_photo_csv(use_text=False, seed=0):
    return _load_cstag_dataset("photo", use_text=use_text, seed=seed)
