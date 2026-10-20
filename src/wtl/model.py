"""Relational deep learning model: per-table encoders -> time encoding -> hetero GraphSAGE -> MLP head."""
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
    """

    def __init__(self, data, col_stats, channels: int = 128, num_layers: int = 2, aggr: str = "sum"):
        super().__init__()
        # HeteroEncoder: takes, per table, its column names (grouped by type) and their statistics;
        # builds one torch_frame ResNet per table; at run time maps each table's rows -> [rows, channels].
        # data[nt].tf.col_names_dict -> {stype: [column names]} for table nt.
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats,
        )
        # HeteroTemporalEncoder: for tables that have a time column, maps (prediction time - row time)
        # -> a `channels`-d vector, so the model knows how recent each row is.
        # "time" in data[nt] -> True if table nt has a time attribute.
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels,
        )
        # HeteroGraphSAGE: num_layers rounds of message passing; in each round, one SAGEConv per edge type
        # (its own weights), results summed per destination node, then LayerNorm + ReLU.
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        # MLP(in, out_channels=1, num_layers=1): a single linear layer, 128 numbers -> 1 logit per target.
        self.head = MLP(channels, out_channels=1, norm="batch_norm", num_layers=1)

    def forward(self, batch, entity_table: str) -> torch.Tensor:
        """Returns one logit per target in the batch (sigmoid of it = probability)."""
        # batch[entity_table].seed_time: prediction time of each target, shape [num_targets]
        seed_time = batch[entity_table].seed_time
        # batch.tf_dict: {table: TensorFrame of the sampled rows}; encoder -> {table: [rows, 128]}
        x_dict = self.encoder(batch.tf_dict)
        # batch.time_dict:  {table: each sampled row's timestamp}
        # batch.batch_dict: {table: which target each sampled row was sampled for}
        #   (needed because each target has its own prediction time)
        rel_time = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, t in rel_time.items():
            x_dict[node_type] = x_dict[node_type] + t            # Eq. (1): row vector + recency vector
        # batch.edge_index_dict: {edge type: [2, num_edges] index pairs} for the sampled subgraph
        x_dict = self.gnn(x_dict, batch.edge_index_dict)          # Eq. (2), twice: 2 hops reach the target
        # The first len(seed_time) rows of the entity table are the targets themselves (the loader puts
        # them first). head -> [num_targets, 1]; .view(-1) flattens to [num_targets].
        return self.head(x_dict[entity_table][: seed_time.size(0)]).view(-1)
