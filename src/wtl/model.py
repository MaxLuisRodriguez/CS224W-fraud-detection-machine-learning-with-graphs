"""Relational deep learning model: per-table encoders -> time encoding -> hetero GraphSAGE -> linear head.

Training and evaluation scripts build one RDLModel and call it on the mini-batches from
wtl.data.make_loader; it returns one score (a logit) per target, e.g. per prospect on a call list.
In short: every sampled row first becomes a vector of 128 numbers, then two rounds of message passing
let each row mix in its neighbors' vectors, and finally the target's own vector is turned into a score.

The model can also see only part of the graph (M1), which is what makes lookups matter:
  hide="drop": unrevealed rows are cut out of message passing (their edges are removed)
  hide="mask": unrevealed rows stay in the graph as stubs, represented by a learned per-table [MASK] vector
"""

import torch
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.nn import MLP


# =====================================================================================================
# RDLModel: scores each target row from its sampled relational neighborhood
# =====================================================================================================
class RDLModel(torch.nn.Module):
    """Scores target rows from their sampled relational neighborhood (RelBench's baseline design).

    Args:
        data:       the full HeteroData graph (used here only to read node types, edge types, columns)
        col_stats:  per-column statistics from build_graph, for normalizing inputs
        channels:   size of every row's vector (128)
        num_layers: rounds of message passing = hops of information reaching the target (2)
        aggr:       how a node combines its neighbors' messages within one edge type ("sum")
        target_time: if False, the targets' own rows get no "how recent" vector. Use it when a target row's
                    timestamp can postdate the prediction (rel-event's users.joinedAt does for some users),
                    so the model can't learn from that leak. Neighbors' recency is unaffected.
    """

    def __init__(self, data, col_stats, channels: int = 128, num_layers: int = 2, aggr: str = "sum",
                 target_time: bool = True):
        super().__init__()
        self.target_time = target_time
        # Part 1, row encoder. Turns each row's raw column values into one vector.
        # HeteroEncoder: takes, per table, its column names (grouped by type) and their statistics;
        # builds one torch_frame ResNet per table; at run time maps each table's rows -> [rows, channels].
        # Inside, each column type has its own small encoder: categories -> learned embeddings,
        # numbers -> a linear layer, text (already GloVe vectors) -> a linear layer, timestamps -> date features.
        # data[nt].tf.col_names_dict -> {stype: [column names]} for table nt.
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats,
        )
        # Part 2, recency. Tells the model how old each row is relative to the prediction time.
        # HeteroTemporalEncoder: for tables that have a time column, maps (prediction time - row time)
        # -> a `channels`-d vector, so the model knows how recent each row is. (It converts the gap to
        # days, applies a sine/cosine position encoding, then a linear layer per table.)
        # "time" in data[nt] -> True if table nt has a time attribute.
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels,
        )
        # Part 3, message passing. Each row updates its vector from its neighbors' vectors.
        # HeteroGraphSAGE: num_layers rounds of message passing; in each round, one SAGEConv per edge type
        # (its own weights), results summed per destination node, then LayerNorm + ReLU.
        # SAGEConv also keeps a separate weight for the row's own vector, so a row with no neighbors
        # still has its own information after each round.
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )

        # One learned [mask] vector per table, for rows that exist but haven't been fetched (mask mode)
        # It starts at all zeros and is trained like any other weight. ParameterDict is a dict that
        # registers its values as model weights, so the optimizer and saved checkpoints include them.
        self.mask_emb = torch.nn.ParameterDict(
            {nt: torch.nn.Parameter(torch.zeros(channels)) for nt in data.node_types}
        )

        # Part 4, the head: turns the target's final vector into a score.
        # MLP(in, out_channels=1, num_layers=1): a single linear layer, 128 numbers -> 1 logit per target.
        # (With one layer, PyG's MLP has no hidden layer, so norm="batch_norm" is never applied.)
        self.head = MLP(channels, out_channels=1, norm="batch_norm", num_layers=1)

    def forward(self, batch, entity_table: str, revealed: dict = None, hide: str = "drop") -> torch.Tensor:
        """Returns one logit per target in the batch (sigmoid of it = probability).

        Args:
            batch:        one HeteroData mini-batch from wtl.data.make_loader.
            entity_table: name of the table the targets come from (e.g. "users").
            revealed:     {table: bool tensor, one entry per sampled row, True = fetched}. None = see every row.
            hide:         what happens to unfetched rows: "drop" (removed) or "mask" ([MASK] stubs).

        Returns:
            Tensor of shape [num_targets]: one logit per target, in the loader's order.
        """
        # Step 1: one vector per sampled row, plus a "how recent" vector for dated rows.
        # batch[entity_table].seed_time: prediction time of each target, shape [num_targets]
        seed_time = batch[entity_table].seed_time
        # batch.tf_dict: {table: TensorFrame of the sampled rows}; encoder -> {table: [rows, 128]}
        x_dict = self.encoder(batch.tf_dict)
        # batch.time_dict:  {table: each sampled row's timestamp}
        # batch.batch_dict: {table: which target each sampled row was sampled for}
        #   (needed because each target has its own prediction time)
        rel_time = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        if not self.target_time and entity_table in rel_time:
            # Hide the targets' own row ages (the first len(seed_time) rows of the entity table are the
            # targets). clone() first, so we don't edit a tensor autograd still needs.
            t = rel_time[entity_table].clone()
            t[: seed_time.size(0)] = 0
            rel_time[entity_table] = t
        for node_type, t in rel_time.items():
            x_dict[node_type] = x_dict[node_type] + t            # Eq. (1): row vector + recency vector

        # batch.edge_index_dict: {edge type: [2, num_edges] index pairs} for the sampled subgraph
        # Row 0 of each pair list is the source row, row 1 the destination row (positions in the batch).
        edge_index_dict = batch.edge_index_dict

        # Step 2: hide what has not been fetched yet (skipped when revealed is None).
        if revealed is not None and hide == "mask":
            # Mask mode: keep every row and edge, but replace each unfetched row's vector with its
            # table's learned [MASK] vector. The model then knows "a row is here" but nothing about
            # its contents or its date (the recency vector was added above, so it is replaced too).
            # Because the edges stay, a fetched row two hops away can still reach the target through
            # a masked row in the middle.
            for node_type, seen in revealed.items():
                # expand_as: repeats the [channels] mask vector to [rows, channels] without copying
                # memory. torch.where would broadcast it anyway; this just makes the shapes explicit.
                stub = self.mask_emb[node_type].expand_as(x_dict[node_type])

                # torch.where(cond, a, b): row by row, take a where cond is True, else b.
                # seen.unsqueeze(-1) turns [rows] into [rows, 1] so one True/False covers a whole row.
                x_dict[node_type] = torch.where(seen.unsqueeze(-1), x_dict[node_type], stub)

        elif revealed is not None and hide == "drop":
            # Drop mode: rows keep their vectors, but every edge touching an unfetched row is removed,
            # so information only flows along paths where every row has been fetched.
            # revealed[src][ei[0]] looks up "is the source row fetched?" for every edge at once
            # (one True/False per edge); same for the destination; & keeps edges where both are True.
            # ei[:, keep] then selects those edge columns.
            edge_index_dict = {
                # keep edge only if both ends are revealed
                (src, rel, dst) : ei[:, revealed[src][ei[0]] & revealed[dst][ei[1]]]
                for (src, rel, dst), ei in edge_index_dict.items()
            }


        # Step 3: message passing over the (possibly filtered) edges, then score the targets.
        # edge_index_dict here is the batch's edges, minus any removed by drop mode above.
        x_dict = self.gnn(x_dict, edge_index_dict)          # Eq. (2), twice: 2 hops reach the target
        # The first len(seed_time) rows of the entity table are the targets themselves (the loader puts
        # them first). head -> [num_targets, 1]; .view(-1) flattens to [num_targets].
        return self.head(x_dict[entity_table][: seed_time.size(0)]).view(-1)
