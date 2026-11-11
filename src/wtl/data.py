"""Load a RelBench database and task, and turn the database into one PyG heterogeneous temporal graph.

This is the first step of the pipeline. Scripts call load() -> column_stypes() -> build_graph() once;
training and evaluation scripts then call make_loader() per split, and the model in wtl.model runs on
the mini-batches the loader yields.

Graph layout (RDL, "relational deep learning"):
- one node type per table, one node per row; row features live in data[table].tf (a TensorFrame:
  the table's columns already converted to tensors)
- one edge type per foreign key, in both directions: (child, "f2p_<fkey>", parent) points from the
  row that holds the key to the row it names, and (parent, "rev_f2p_<fkey>", child) points back

The one idea to hold on to: a foreign key is not a feature, it is an edge. "This attendance row
belongs to user 42" becomes a link between two nodes, and the GNN learns by passing messages along
those links.
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

# Path.home() -> the user's home folder.
# Everything slow to compute is cached here: the guessed column types and the materialized tensors.
CACHE_ROOT = Path.home() / ".cache" / "wtl"


#
# 1. Load: the dataset, the task, and the database
#
def load(dataset_name: str, task_name: str):
    """Download (first time only) and return the dataset, the task, and the database.

    Args:
        dataset_name: RelBench dataset name, e.g. "rel-event" (main) or "rel-f1" (small, for development).
        task_name: a prediction task defined on that dataset, e.g. "user-ignore".

    Returns:
        dataset: RelBench dataset object (knows its cutoff dates and its tasks)
        task: the prediction task (which entity, which label, which horizon, the split tables)
        db: Database object; db.table_dict maps table name -> Table (a pandas DataFrame plus its keys)
    """

    # load_dataset(name): finds the dataset (downloading its files if they are not on disk yet) and
    # returns an object that knows its tables and tasks. load_task(name) does the same for one task.
    dataset = load_dataset(dataset_name)
    task = dataset.load_task(task_name)

    # get_db(): builds the full database as pandas tables. By default it drops every row dated after
    # the test cutoff, so the graph can never contain the future the test labels are about.
    # It is not cached, so we call it once and pass the result around.
    db = dataset.get_db()
    return dataset, task, db


#
# 2. Column types: what kind of data is in each column?
#
def column_stypes(db, dataset_name: str) -> dict:
    """Semantic type of every column (numerical, categorical, text, timestamp, ...), cached as JSON.

    The type decides how a column is encoded: numbers are normalized, categories get learned
    embeddings, text goes through GloVe, timestamps become date features.

    Args:
        db: the Database from load().
        dataset_name: used only to name the cache folder (one stypes.json per dataset).

    Returns:
        {table name: {column name: torch_frame.stype}}
    """
    path = CACHE_ROOT / dataset_name / "stypes.json"

    # Fast path: reuse the saved guess. JSON stores plain strings ("numerical"), so stype(s) turns
    # each one back into the torch_frame enum value. Delete the file to force a fresh guess.
    if path.exists():
        raw = json.loads(path.read_text())
        return {table: {col: stype(s) for col, s in cols.items()} for table, cols in raw.items()}

    # get_stype_proposal(db): takes the Database; looks at each column's dtype and contents;
    # returns RelBench's best guess. It inspects a random sample of up to 1,000 rows per table,
    # which is one more reason to cache the answer: every later run then uses the same types.
    proposal = get_stype_proposal(db)

    path.parent.mkdir(parents=True, exist_ok=True)

    # stype is a plain Enum: save .value ("numerical") not str()
    serializable = {table: {col:st.value for col, st in cols.items()} for table, cols in proposal.items()}
    path.write_text(json.dumps(serializable, indent=2)) # json.dumps: Python dict -> JSON text
    return proposal


#
# 3. Build the graph: rows -> nodes with tensor features, foreign keys -> edges
#
def drop_columns(col_to_stype: dict, drop) -> dict:
    """Return a copy of the column types without the listed feature columns.

    Use it to stop the model from reading a column, for example rel-event's users.joinedAt, which for some
    users is dated after the prediction time. A dropped column is no longer a feature, but a table's time
    column still decides which rows the sampler may use.

    Args:
        col_to_stype: {table: {column: stype}} from column_stypes()
        drop:         columns to remove, each written "table.column" (e.g. ["users.joinedAt"])

    Returns:
        {table: {column: stype}} without those columns
    """
    out = {table: dict(cols) for table, cols in col_to_stype.items()}
    for name in drop or []:
        table, col = name.split(".", 1)
        out[table].pop(col, None)
    return out


def build_graph(db, dataset_name: str, task, col_to_stype:dict, cache_name: str = "materialized"):
    """Turn every table into tensors and connect rows through their primary/foreign keys.

    Key columns are not used as features; they only decide which rows get linked. Turning text into
    vectors is the slow part, so the result is cached on disk per dataset.

    Args:
        db: the Database from load().
        dataset_name: used only to name the cache folder.
        task: the task from load(); used only to know which columns would leak its label.
        col_to_stype: column types from column_stypes().
        cache_name: the cache folder's name. Use a different name whenever col_to_stype changes (for
            example after drop_columns), because a cached table is reloaded as-is, whatever its columns.

    Returns:
        data: torch_geometric HeteroData, the whole database as one graph

        col_stats: per-column statistics (means, std devs, category lists) the encoders normalize with
    """
    # make_pkey_fkey_graph: takes the database and the column types. For each table it converts
    # the non-key columns into a TensorFrame (data[table].tf), stores the row timestamps as
    # data[table].time (tables with a time column only), and adds the f2p / rev_f2p edge pairs for
    # each foreign key. Rows whose foreign key is empty simply get no edge for that key.
    # Returns (HeteroData graph, {table: column statistics}).
    data, col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype,
        # TextEmbedderConfig(text_embedder, batch_size): tells torch_frame which callable embeds text
        # and how many strings to send per call (256).
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device="cpu"), batch_size=256),
        # One <table>.pt file per table; if it already exists it is loaded instead of recomputed.
        cache_dir=str(CACHE_ROOT / dataset_name / cache_name),
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

    A "target" is one row of the task table: one entity (e.g. a user) at one prediction time. Hop 1
    is the rows directly linked to it (e.g. that user's attendance rows); hop 2 is the rows linked to
    those (e.g. the events those attendance rows point to).

    Args:
        data: the HeteroData graph from build_graph().
        task: the task from load().
        split: "train", "val" or "test".
        num_neighbors: max neighbors kept per edge type at each hop, (hop 1, hop 2).
        batch_size: targets per mini-batch.
        shuffle: shuffle the targets each epoch (use True for training only).

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
    #                (None if this split's table has no labels)
    table_input = get_node_train_table_input(table=table, task=task)
    # NeighborLoader: takes the full graph and a list of target nodes; for each mini-batch of targets,
    # samples their neighborhoods hop by hop and returns that subgraph as a HeteroData batch.
    # Because sampling is time-aware, each target gets its own separate copy of its neighborhood, so
    # a row shared by two targets appears twice in the batch (batch[table].n_id gives each copy's
    # original row id). The targets themselves are always the first rows of the entity table.
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