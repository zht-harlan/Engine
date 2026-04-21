import csv
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm
from yaml import SafeLoader

from data.load import load_data
from models import load_model
from utils.args import Arguments
from utils.sampling import subsampling


def macro_f1_score(y_true, y_pred, num_classes):
    f1_scores = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            f1_scores.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / len(f1_scores))


def save_dataset_results(config, rows, summary):
    os.makedirs(config.results_dir, exist_ok=True)
    dataset_path = os.path.join(config.results_dir, f"{config.dataset}_results.csv")
    summary_path = os.path.join(config.results_dir, "summary.csv")
    fieldnames = ["dataset", "seed", "acc", "f1_macro"]

    with open(dataset_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow(summary)

    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    print(f"Saved dataset results to {dataset_path}")
    print(f"Updated summary file at {summary_path}")


def train_subgraph(model, optimizer, criterion, config, train_loader, val_loader, test_loader, device):
    if config.earlystop:
        cnt = 0
        patience = config.patience
        best_val = 0
        best_test_fromval = {"acc": 0.0, "f1_macro": 0.0}

    for epoch in tqdm(range(config.epochs)):
        model.train()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        if config.earlystop:
            val_metrics = eval_subgraph(model, val_loader, device)
            if val_metrics["acc"] > best_val:
                cnt = 0
                best_test_fromval = eval_subgraph(model, test_loader, device)
                best_val = val_metrics["acc"]
            else:
                cnt += 1
                if cnt >= patience:
                    print(f"early stop at epoch {epoch}")
                    break
    if not config.earlystop:
        best_test_fromval = eval_subgraph(model, test_loader, device)
    return best_test_fromval


def eval_subgraph(model, data_loader, device):
    model.eval()

    preds_all = []
    labels_all = []
    for batch in data_loader:
        batch = batch.to(device)
        preds = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_index).argmax(dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(batch.y.cpu())
    y_pred = torch.cat(preds_all)
    y_true = torch.cat(labels_all)
    acc = float((y_pred == y_true).float().mean().item())
    f1_macro = macro_f1_score(y_true, y_pred, int(y_true.max().item()) + 1)
    return {"acc": acc, "f1_macro": f1_macro}


def train_fullgraph(model, optimizer, criterion, config, data, device):
    if config.earlystop:
        cnt = 0
        patience = config.patience
        best_val = 0
        best_test_fromval = {"acc": 0.0, "f1_macro": 0.0}
    model.train()

    data = data.to(device)
    for epoch in tqdm(range(config.epochs)):
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if config.earlystop:
            val_metrics = eval_fullgraph(model, data, device, config)
            if val_metrics["acc"] > best_val:
                cnt = 0
                best_test_fromval = eval_fullgraph(model, data, device, config, eval="test")
                best_val = val_metrics["acc"]
            else:
                cnt += 1
                if cnt >= patience:
                    print(f"early stop at epoch {epoch}")
                    break
    if not config.earlystop:
        best_test_fromval = eval_fullgraph(model, data, device, config, eval="test")
    return best_test_fromval


def eval_fullgraph(model, data, device, config, eval="valid"):
    assert eval in ["valid", "test"]
    model.eval()
    data = data.to(device)
    pred = model(data.x, data.edge_index).argmax(dim=1)
    if eval == "test":
        y_pred = pred[data.test_mask]
        y_true = data.y[data.test_mask]
    else:
        y_pred = pred[data.val_mask]
        y_true = data.y[data.val_mask]
    acc = float((y_pred == y_true).float().mean().item())
    f1_macro = macro_f1_score(y_true.detach().cpu(), y_pred.detach().cpu(), int(data.y.max().item()) + 1)
    return {"acc": acc, "f1_macro": f1_macro}


def train_eval(model, optimizer, criterion, config, data, train_loader, val_loader, test_loader, device):
    if config.subsampling:
        test_metrics = train_subgraph(model, optimizer, criterion, config, train_loader, val_loader, test_loader, device)
    else:
        test_metrics = train_fullgraph(model, optimizer, criterion, config, data, device)
    return test_metrics


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    acc_list = []
    f1_list = []
    for i in range(5):
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=i)
        data.y = data.y.squeeze()

        model = load_model(data.x.shape[1], num_classes, config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader, val_loader, test_loader = None, None, None
        if config.subsampling:
            train_loader, val_loader, test_loader = subsampling(data, config, sampler=config.sampler)
        test_metrics = train_eval(model, optimizer, criterion, config, data, train_loader, val_loader, test_loader, device)
        print(i, test_metrics)
        acc_list.append(test_metrics["acc"])
        f1_list.append(test_metrics["f1_macro"])
        rows.append(
            {
                "dataset": config.dataset,
                "seed": i,
                "acc": f"{test_metrics['acc']:.6f}",
                "f1_macro": f"{test_metrics['f1_macro']:.6f}",
            }
        )

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    final_f1, final_f1_std = np.mean(f1_list), np.std(f1_list)
    summary = {
        "dataset": config.dataset,
        "seed": "mean+-std",
        "acc": f"{final_acc:.6f}+-{final_acc_std:.6f}",
        "f1_macro": f"{final_f1:.6f}+-{final_f1_std:.6f}",
    }
    save_dataset_results(config, rows, summary)
    print(f"# final_acc: {final_acc*100:.2f}+-{final_acc_std*100:.2f}")
    print(f"# final_f1_macro: {final_f1*100:.2f}+-{final_f1_std*100:.2f}")


if __name__ == "__main__":
    args = Arguments().parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)
    for k, v in config.items():
        args.__setattr__(k, v)
    print(args)
    main(args)
