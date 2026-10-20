"""Load a RelBench database and task, and turn it into a PyG heterogeneous temporal graph.

Graph layout (RDL):
- one node type per table, one node per row; row features live in data[table].tf (a TensorFrame)
- one edge type per foregin key, in both directions: (child, "f2p_<fkey>", parent) and its "rev_"
reverse
"""
import json
from pathlib import Path

from relbench import load_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader

from wtl.text import GloveTextEmbedding

# Path.home() -> user folder
# cache for column types and materialized tensors
CACHE_ROOT = Path.home() / ".cache" / "wtl"


#
# 1. Load: the dataset, the task, and the database
#
def load(dataset_name: str, task_name: str):
    """Download (first time only) and return the dataset, the task, and the database.
    
    Returns:
        dataset: RelBench dataset object (knows its cutoff dates and its tasks)
        task: the prediction task (which entity, which label, which horizon, the split tables)
        db: Database object; db.table_dict maps table name -> Table (a pandas DataFrame plus its keys)
    """

    dataset = load_dataset(dataset_name)
    task = dataset.load_task(task_name)

    db = dataset.get_db()
    return dataset, task, db


#
# Column types: what kind of data is in each column?
#
def column_stypes(db, dataset_name: str) -> dict:
    """Semantic type of every column (numerical, categorical, text, timestamp, ...), cached as JSON.
    
        The type decides how a column is encoded: numbers are normalized, categories get learned
        embeddings, text goes through GloVe, timestamps become date features.
        
        Returns:
        {table name: {column name: torch_frame.stype}}
    """
    path = CACHE_ROOT / dataset_name / "stypes.json"

    if path.exists():
        raw = json.loads(path.read_text())
        return {table: {col: stype(s) for col, s in cols.items()} for table, cols in raw.items()}

    # get_stype_proposal(db): takes the Database; looks at each column's dtype and contents;
    # returns RelBench's best guess
    proposal = get_stype_proposal(db)

    path.parent.mkdir(parents=True, exist_ok=True)

    # stype is a plain Enum: save .value ("numerical") not str()
    serializable = {table: {col:st.value for col, st in cols.items()} for table, cols in proposal.items()}
    path.write_text(json.dumps(serializable, indent=2)) # json.dumps: Python dict -> JSON text
    return proposal

#
## Build the graph: rows -> nodes with tensor features, foreign keys -> edges
#
def build_graph(db, dataset_name: str, task, col_to_stype:dict):
    """Turn every table into tensors and connect rows through their primary/foreign keys.
    Returns:
        data: torch_geometric HeteroData, the whole database as one graph

        col_stats: per-column statistics (means, std devs, category lists) the encoders normalize with
    """
    data, col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype,
        # TextEmbedderConfig(text_embedder, batch_size): tells torch_frame which callable embeds text
        # and how many strings to send per call (256).
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device="cpu"), batch_size=256),
        cache_dir=str(CACHE_ROOT / dataset_name / "materialized"),
        # task.hidden_columns(): columns that would leak this task's label, e.g. a field computed from
        # the future. They're dropped after loading from the cache, so one cache serves every task.
        remove_columns=task.hidden_columns(),
    )
    return data, col_stats


# =====================================================================================================
# 4. Sample mini-batches: each target's time-respecting 2-hop neighborhood
# =====================================================================================================
def make_loader(data, task, split: str, num_neighbors=(128, 64), batch_size: int = 512, shuffle: bool = False):
    """Mini-batches of targets, each with its own sampled 2-hop neighborhood from before its prediction time.

    Returns:
        loader: iterate over it to get HeteroData mini-batches
        table:  the split's task table (entity ids, prediction times, labels), in loader order
    """
    table = task.get_table(split)         # keep this table: get_table re-reads from disk on every call
    # get_node_train_table_input(table, task): converts the task table into what the loader needs.
    # Returns a named tuple:
    #   .nodes     = (entity table name, tensor of the targets' row indices)
    #   .time      = tensor of each target's prediction time (Unix seconds)
    #   .transform = a function that attaches labels to each batch as batch[entity_table].y
    table_input = get_node_train_table_input(table=table, task=task)
    # NeighborLoader: takes the full graph and a list of target nodes; for each mini-batch of targets,
    # samples their neighborhoods hop by hop and returns that subgraph as a HeteroData batch.
    loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),  # max neighbors per edge type: 128 at hop 1, 64 at hop 2
        time_attr="time",                   # only rows with time <= the target's prediction time (needs pyg-lib)
        input_nodes=table_input.nodes,      # which nodes are the targets
        input_time=table_input.time,        # each target's prediction time
        transform=table_input.transform,    # attaches labels as batch[entity_table].y
        batch_size=batch_size,              # targets per mini-batch
        temporal_strategy="uniform",        # when there are too many past neighbors, sample uniformly
        shuffle=shuffle,                    # shuffle targets each epoch (training only)
        num_workers=0,                      # load in the main process (simplest on Windows)
    )
    return loader, table