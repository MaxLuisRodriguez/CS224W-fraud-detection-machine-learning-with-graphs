"""M2: train the relational GNN, evaluate it while hiding a growing share of the graph, save the weights.

What this script answers:
    How well does the graph model rank prospects when it can see every row, and how much of that quality
    is left when we hide more and more of the rows it could have looked up?

The big picture:
    Every database row is a graph node and every foreign key is an edge. For each target (a driver on
    rel-f1, a user on rel-event) a sampler collects its 2-hop neighborhood, using only rows dated at or
    before the target's prediction time, so the model never sees the future. The GNN passes messages
    along those edges and outputs one score per target. This script trains that model once, then tests
    it while pretending that some rows were never looked up.

Options:
--train_reveal full    trains on complete neighborhoods (the standard RelBench recipe)
--train_reveal random  hides a random share of rows in every training batch (partial-observation training)
--hide drop|mask       hidden rows are removed from message passing, or kept as [MASK] stubs
Evaluation reveals each distinct row with probability p, for every p in --eval_p (1.0 = full graph).

How to run (from the repo root):
    python scripts/train_rdl.py --dataset rel-event --task user-ignore --train_reveal full --seed 0

Reads the RelBench dataset and task (downloaded on first use, cached after that).
Writes outputs/<dataset>/<task>/rdl_<train_reveal>_<hide>_ep<epochs>_seed<seed>.json (metrics) and .pt (weights).
The .pt file holds the frozen weights that every later lookup experiment reloads.

How to read the output:
    One "epoch NN train loss" line per epoch (average loss per training target; lower = better fit).
    Then one line per reveal level p: RelBench's validation AUROC over the whole validation table, and
    the call-list metrics (AUROC, average precision and precision@K, averaged over random lists of 128
    prospects). Reading down the lines shows how much quality is lost as fewer rows are visible.
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
from wtl.runs import graph_for
from wtl.metrics import call_list_metrics
from wtl.model import RDLModel
from wtl.reveal import random_reveal

# @torch.no_grad(): skip gradient bookkeeping inside this function. We only predict here, never train,
# so this saves memory and time.
@torch.no_grad()
def predict(model, loader, entity_table: str, p: float, hide: str, device):
    """Probabilities for every row of the loader's table, in table order, seeing a share p of the rows.

    Args:
        model: the trained RDLModel.
        loader: a NeighborLoader over one split. It must NOT shuffle, so batches come out in table order.
        entity_table: name of the table whose rows are being scored (the targets).
        p: share of distinct rows to reveal, from 0 to 1. At 1.0 nothing is hidden. The targets' own
           rows are always visible.
        hide: "drop" removes hidden rows from message passing; "mask" keeps them as blank stubs.
        device: where the model lives (cpu or cuda).

    Returns:
        A numpy array of probabilities between 0 and 1, one per row of the table, in table order.
    """
    # Evaluation mode: switches off training-only behavior such as dropout.
    model.eval()
    probs = []
    for batch in loader:
        batch = batch.to(device)
        # p = 1 means "see everything", so skip the reveal step. Otherwise flip one coin per distinct row
        # (by its original id), so every copy of a shared row is shown or hidden together.
        revealed = None if p >= 1.0 else random_reveal(batch, entity_table, p)
        # The model returns logits (raw scores); sigmoid squashes them into probabilities.
        # .cpu() moves them off the GPU so they can be joined into one numpy array at the end.
        probs.append(torch.sigmoid(model(batch, entity_table, revealed=revealed, hide=hide)).cpu())
    # Stitch the per-batch results into one array, still in table order.
    return torch.cat(probs).numpy()

def main():
    """Parse the options, train the model, evaluate it at every reveal level, and save metrics and weights."""
    # Command-line options. Every one has a default, so the script runs with no flags (on rel-f1).
    parser = argparse.ArgumentParser()
    # Which RelBench database and which prediction task (rel-event / user-ignore is the main pair).
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    # How to train: "full" shows every sampled row; "random" hides a fresh random share in every batch.
    parser.add_argument("--train_reveal", choices=["full", "random"], default="full")
    # What "hidden" means: "drop" deletes hidden rows from the graph; "mask" keeps them as blank stubs.
    parser.add_argument("--hide", choices=["drop", "mask"], default="drop")
    # Training length, Adam's step size (learning rate), and how many targets go into one mini-batch.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=512)
    # Seed for the random numbers (weight initialization, batch order, the random reveals in training).
    parser.add_argument("--seed", type=int, default=0)
    # Reveal levels to evaluate at: 1.0 = every row visible, 0.0 = only the targets' own rows.
    parser.add_argument("--eval_p", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    # Feature columns to leave out, each "table.column" (e.g. users.joinedAt, which can postdate predictions).
    parser.add_argument("--drop_cols", nargs="*", default=[])
    # Give the targets' own rows no "how recent" vector (their timestamp can postdate the prediction).
    parser.add_argument("--no_target_time", action="store_true")
    # Extra label appended to the output file names, to keep variants apart (e.g. "fix").
    parser.add_argument("--tag", default="")
    args = parser.parse_args()


    # Step 1: make the run repeatable. seed_everything seeds the
    # Python, numpy and torch RNGs (random number generators).
    seed_everything(args.seed)
    # Use the GPU when there is one; otherwise the CPU (slower, but it works).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: load the data and turn it into a graph.
    # load() returns the dataset, the task (which table to predict for, the label, the split tables) and
    # the database (every table as a DataFrame).
    dataset, task, db = load(args.dataset, args.task)
    # This script only handles yes/no tasks: it uses a binary loss and AUROC.
    assert task.task_type == TaskType.BINARY_CLASSIFICATION, "this script handles binary tasks"
    # build_graph: each row becomes a node with its columns encoded as tensors; each foreign key becomes
    # an edge. column_stypes says how each column is encoded (number, category, text, date).
    # col_stats holds per-column statistics (means, category lists) that the model's encoders need.
    # graph_for (wtl/runs.py) builds the graph without any --drop_cols, with its own cache folder in that case.
    _, data, col_stats = graph_for(args.dataset, args.task, args.drop_cols)
    # Loaders hand out mini-batches of targets, each with its sampled 2-hop neighborhood from before its
    # prediction time. Training batches are shuffled. Validation keeps table order, so the predictions
    # line up with the labels in val_table.
    train_loader, _ = make_loader(data, task, "train", batch_size=args.batch_size, shuffle=True)
    val_loader, val_table = make_loader(data, task, "val", batch_size=args.batch_size)

    # Step 3: the model, the optimizer and the loss.
    # RDLModel encodes each row's columns, then runs a GNN over the sampled graph: one score per target.
    model = RDLModel(data, col_stats, target_time=not args.no_target_time).to(device)
    # Adam nudges the weights after every batch; lr sets how big each nudge is.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # BCEWithLogitsLoss: binary cross-entropy for 0/1 labels. It takes raw scores (logits) and applies the
    # sigmoid itself, which is more numerically stable than doing the two steps separately.
    loss_fn = BCEWithLogitsLoss()
    # Start the clock; the total minutes go into the results file.
    start = time.time()


    # Step 4: start the training loop. One epoch = one pass over every training target.
    for epoch in range(1, args.epochs + 1):
        # Training mode: turns on training-only behavior such as dropout.
        model.train()
        # Running totals for this epoch's average loss per target.
        loss_sum, count= 0.0, 0

        for batch in train_loader:
            # Move the mini-batch (node features, edges, labels) to the GPU or CPU.
            batch = batch.to(device)
            # None means "show the model every sampled row".
            revealed = None
            if args.train_reveal == "random":
                # a new share each batch: p is drawn uniformly between 0 and 1, then each distinct row is
                # revealed with probability p. Seeing many amounts of missing data teaches the model to
                # cope with partial views.
                revealed = random_reveal(batch, task.entity_table, p = random.random()) 

            # Forward pass: one logit (raw score) per target in the batch.
            logits = model(batch, task.entity_table, revealed=revealed, hide=args.hide)
            # Compare with the true 0/1 labels, which the loader attached as .y.
            loss = loss_fn(logits, batch[task.entity_table].y.float())

            # The standard three steps: clear old gradients, compute new ones, take one Adam step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss is the batch's mean, so weight it by the batch size; the last batch can be smaller.
            loss_sum += loss.item() * logits.numel()
            count += logits.numel()

        # flush=True prints right away, so progress shows up in a log file while the run is going.
        print(f"epoch {epoch:02d} train loss {loss_sum / count:.4f}", flush=True)


    # Step 5: evaluate the final model at every revelation level (fixed seed for the reveal draw; the
    # sampler is unseeded). "minutes" is filled in at the end; "val" gets one entry per reveal level.
    results = {"args": vars(args), "minutes": None, "val": {}}

    # The true 0/1 labels of the validation table, in table order (the same order as the predictions).
    y_val = val_table.df[task.target_col].astype(int).to_numpy()

    for p in args.eval_p:
        # Reset torch's seed so the reveal coin flips start from the same point at every level and in every
        # run. The neighbor sampler is not controlled by this seed, so sampled neighborhoods can still vary.
        torch.manual_seed(1234)
        scores = predict(model, val_loader, task.entity_table, p, args.hide, device)

        # RelBench's own metrics (roc_auc, etc): task.evaluate compares the scores with the labels of the
        # whole validation table and returns a dict of RelBench's standard metrics for this task.
        official = task.evaluate(scores, val_table)

        # Our call-list metrics: draw many random lists of 128 validation targets (fewer if the table is
        # small) and average AUROC, average precision and precision@K over them, the way a rep sees quality.
        lists = call_list_metrics(y_val, scores, list_size= min(128, len(y_val)))

        # Store this level. JSON keys must be strings, hence str(p); call-list keys get a "list_" prefix.
        results["val"][str(p)] = {"roc_auc": official["roc_auc"], **{f"list_{k}": v for k, v in lists.items()}}
        print(f"reveal p={p:.2f} val AUROC {official['roc_auc']:.4f} call-list {lists}")

    # Wall-clock time for training plus evaluation, in minutes.
    results["minutes"] = (time.time() - start) / 60


    # Step 6: save. The file name records the settings, so runs with different settings get different files.
    out_dir = Path("outputs") / args.dataset / args.task
    # Create the folder (and any missing parent folders); fine if it already exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rdl_{args.train_reveal}_{args.hide}_ep{args.epochs}_seed{args.seed}"
    # A --tag is appended, so variants don't overwrite each other.
    if args.tag:
        stem += f"_{args.tag}"
    # The metrics, as readable JSON.
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    # state_dict() is only the learned weights (a dict of tensors). To reload them, build RDLModel the same
    # way and call load_state_dict, as run_frontier.py does.
    torch.save(model.state_dict(), out_dir / f"{stem}.pt")       # trained weights, reloaded by M7-M9
    print("saved", out_dir / f"{stem}.json", "and", out_dir / f"{stem}.pt")


# Run main() only when this file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()


                                  