"""M2: train the relational GNN, evaluate it while hiding a growing share of the graph, save the weights.

--train_reveal full    trains on complete neighborhoods (the standard RelBench recipe)
--train_reveal random  hides a random share of rows in every training batch (partial-observation training)
--hide drop|mask       hidden rows are removed from message passing, or kept as [MASK] stubs
Evaluation reveals each distinct row with probability p, for every p in --eval_p (1.0 = full graph).
Writes outputs/<dataset>/<task>/rdl_<train_reveal>_<hide>_ep<epochs>_seed<seed>.json (metrics) and .pt (weights).
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from relbench.base import TaskType 
from torch.nn import BCEWithLogitsLoss
from torch_geometric.seed import seed_everything

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.metrics import call_list_metrics
from wtl.model import RDLModel
from wtl.reveal import random_reveal

@torch.no_grad()
def predict(model, loader, entity_table: str, p: float, hide: str, device):
    """Probabilities for every row of the loader's table, in table order, seeing a share p of the rows."""
    model.eval()
    probs = []
    for batch in loader:
        batch = batch.to(device)
        revealed = None if p >= 1.0 else random_reveal(batch, entity_table, p)
        probs.append(torch.sigmoid(model(batch, entity_table, revealed=revealed, hide=hide)).cpu())
    return torch.cat(probs).numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    parser.add_argument("--train_reveal", choices=["full", "random"], default="full")
    parser.add_argument("--hide", choices=["drop", "mask"], default="drop")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_p", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    args = parser.parse_args()


    # Python, numpy and torch RNGs
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, task, db = load(args.dataset, args.task)
    assert task.task_type == TaskType.BINARY_CLASSIFICATION, "this script handles binary tasks"
    data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
    train_loader, _ = make_loader(data, task, "train", batch_size=args.batch_size, shuffle=True)
    val_loader, val_table = make_loader(data, task, "val", batch_size=args.batch_size)

    model = RDLModel(data, col_stats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEWithLogitsLoss()
    start = time.time()


    # start the training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count= 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            revealed = None
            if args.train_reveal == "random":
                # a new share each batch
                revealed = random_reveal(batch, task.entity_table, p = random.random()) 

            logits = model(batch, task.entity_table, revealed=revealed, hide=args.hide)
            loss = loss_fn(logits, batch[task.entity_table].y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * logits.numel()
            count += logits.numel()

        print(f"epoch {epoch:02d} train loss {loss_sum / count:.4f}", flush=True)


    # Evaluate the final model at every revelation level (fixed seed for the reveal draw; the sampler is unseeded)
    results = {"args": vars(args), "minutes": None, "val": {}}

    y_val = val_table.df[task.target_col].astype(int).to_numpy()

    for p in args.eval_p:
        torch.manual_seed(1234)
        scores = predict(model, val_loader, task.entity_table, p, args.hide, device)

        # RelBench's own metrics (roc_auc, etc)
        official = task.evaluate(scores, val_table)

        lists = call_list_metrics(y_val, scores, list_size= min(128, len(y_val)))

        results["val"][str(p)] = {"roc_auc": official["roc_auc"], **{f"list_{k}": v for k, v in lists.items()}}
        print(f"reveal p={p:.2f} val AUROC {official['roc_auc']:.4f} call-list {lists}")

    results["minutes"] = (time.time() - start) / 60


    out_dir = Path("outputs") / args.dataset / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rdl_{args.train_reveal}_{args.hide}_ep{args.epochs}_seed{args.seed}"
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), out_dir / f"{stem}.pt")       # trained weights, reloaded by M7-M9
    print("saved", out_dir / f"{stem}.json", "and", out_dir / f"{stem}.pt")


if __name__ == "__main__":
    main()


                                  