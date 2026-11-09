"""Call lists and the lookup environment (M6, M7).

A call list is N prospects scored on one day. Their 2-hop neighborhoods are sampled ONCE into one batch, and
an agent then fetches rows with two kinds of lookup. Each lookup costs 1 if it reveals anything new, and 0 if
everything it would reveal is already fetched (a cache hit):
  pull(prospect, table): reveal all of that prospect's own rows in one table (hop 1), and list the entities
                         those rows point to (their ids become known, their contents stay hidden)
  enrich(table, n_id):   reveal one entity that pulled rows point to (hop 2), in every sampled copy
Everything is keyed on n_id (the row's id in the full database), so a row fetched once is visible everywhere:
that is the shared cache. The key idea: prospects can share rows (two users at the same event), so one
lookup can inform several prospects at once.
"""
import numpy as np 
import torch
from relbench.base import Table
from relbench.modeling.graph import get_node_train_table_input
from torch_geometric.loader import NeighborLoader

from wtl.metrics import list_metrics


#
# M6 : One call list, one subgraph
#
def sample_call_list(data, task, split: str, size: int, seed: int, num_neighbors=(128, 64)):
    """Draw `size` prospects scored at one prediction time and sample all their neighborhoods into one batch.

    Picks a prediction time with at least `size` rows (rel-event: its single validation day). If no time has
    that many (rel-f1), it draws from all times instead and reports one_day=False.

    Args:
        data:          the whole database as one PyG HeteroData graph (from wtl.data.build_graph)
        task:          the RelBench task (knows the entity table and the split tables)
        split:         which task table to draw from, e.g. "val"
        size:          N, how many prospects go on the list
        seed:          fixes which day and which prospects are drawn
        num_neighbors: max neighbors sampled per edge type at hop 1 and hop 2

    Returns:
        batch:   HeteroData; the prospects are the first `size` rows of the entity table, labels in .y
        rows:    the task-table rows drawn, in batch order (DataFrame)
        one_day: True if every prospect shares one prediction time
    """
    # The task table has one row per (prospect, prediction time), plus the label.
    table = task.get_table(split)
    df = table.df

    # Step 1: choose a day. Count rows per prediction time and keep the times with at least `size` rows.
    # If any exist, pick one at random and draw only from that day (a real call list is made on one day).
    rng = np.random.default_rng(seed)
    counts = df[table.time_col].value_counts()
    days = counts[counts >= size].index
    if len(days) > 0:
        day = days[rng.integers(len(days))]
        pool = np.flatnonzero((df[table.time_col] == day).to_numpy())
    else: 
        pool = np.arange(len(df))
    # Step 2: draw `size` distinct rows. Sorting keeps them in the task table's original order.
    picked = np.sort(rng.choice(pool, size=size, replace=False))
    rows = df.iloc[picked].reset_index(drop=True)


    # Step 3: wrap the drawn rows in a small RelBench Table (same keys and time column as the full one),
    # so the standard RelBench helper below can read it.
    sub = Table(df=rows, fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                pkey_col=table.pkey_col, time_col=table.time_col)

    # get_node_train_table_input(table, task): converts a task table into what the loader needs.
    # Returns a named tuple: .nodes (entity table name, the targets' row indices), .time (each target's
    # prediction time), and .transform (attaches labels to the batch as batch[entity_table].y).
    table_input = get_node_train_table_input(table=sub, task=task)

    # Step 4: sample every prospect's 2-hop neighborhood in ONE batch (batch_size=size, no shuffling, so the
    # batch order matches `rows`). Sampling is time-aware: only rows dated at or before the prediction time.
    # Time-aware sampling in PyG is also disjoint: each prospect gets its own copy of its neighborhood, and
    # batch[table].batch says which prospect each sampled row belongs to.
    loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors), time_attr="time",
        input_nodes=table_input.nodes, input_time=table_input.time, transform=table_input.transform,
        batch_size=size, temporal_strategy="uniform", shuffle=False,
    )

    # The loader holds exactly one batch: take it.
    return next(iter(loader)), rows, len(days) > 0

def find_hops(batch, entity_table: str, num_prospects: int):
    """Read the call list's structure off the sampled edges.

    Row indices here are positions inside batch[table] (one per sampled copy), not database ids.

    Args:
        batch:         the call list's sampled HeteroData batch (from sample_call_list)
        entity_table:  the prospects' table, e.g. "users" on rel-event
        num_prospects: N; the prospects are the first N rows of batch[entity_table]

    Returns:
        hop1:  {table: row indices}  each prospect's own rows (rows with an edge into a prospect)
        links: [(s_table, s_rows, t_table, t_rows)]  edges from a hop-2 row into a hop-1 row
    """
    # Step 1: hop-1 rows. edge_index_dict maps each edge type (source table, relation, destination table)
    # to a 2 x E tensor: row 0 holds source row indices, row 1 destination row indices. For every edge type
    # that points into the prospects' table from another table, keep the edges that end at a prospect
    # (index < num_prospects). Their sources are the prospects' own rows, e.g. their attendance rows.
    hop1 = {}
    for (src, rel, dst), ei in batch.edge_index_dict.items():
        if dst == entity_table and src != entity_table:
            rows = ei[0][ei[1]  < num_prospects]
            # rows[:0] is an empty tensor of the right type to start from. unique() drops repeats when a
            # table reaches the prospects through more than one foreign key.
            hop1[src] = torch.cat([hop1.get(src, rows[:0]), rows]).unique()

    # Step 2: hop-2 links. For every edge that ends at a hop-1 row, its source is a hop-2 row: typically an
    # entity the hop-1 row points to (e.g. the event an attendance row is about).
    # torch.isin(a, b) is True for each element of a that appears anywhere in b.
    links = []
    for (src, rel, dst), ei in batch.edge_index_dict.items():
        if dst in hop1:
            keep = torch.isin(ei[1], hop1[dst])
            if keep.any():
                links.append((src, ei[0][keep], dst, ei[1][keep]))

    return hop1, links


#
# The lookup environment
#

class CallList:
    """One call list's lookup state: what has been fetched, what can be fetched next, and what it cost.

    Policies read this state to choose lookups, call apply() to perform them, and call scores() or metrics()
    to see how well the frozen model ranks the list with only the fetched rows.
    """

    def __init__(self, batch, entity_table: str):
        """Set up an empty lookup state for one sampled call list.

        Args:
            batch:        the call list's sampled HeteroData batch (from sample_call_list)
            entity_table: the prospects' table, e.g. "users" on rel-event
        """
        # The batch and its shape: seed_time has one entry per prospect, so its length is N.
        self.batch = batch
        self.entity = entity_table
        self.n = batch[entity_table].seed_time.size(0)                       # prospects on the list
        self.y = batch[entity_table].y[: self.n].float()                     # their labels (for metrics only)
        # The list's structure: hop-1 rows per table and the hop-2 links into them.
        self.hop1, self.links = find_hops(batch, entity_table, self.n)
        # batch[t].batch gives, for each sampled row, the index of the prospect it was sampled for.
        self.owner = {t: batch[t].batch[rows] for t, rows in self.hop1.items()}  # prospect of each hop-1 row
        # Every (prospect, table) pair where the prospect has rows. Sorted, so each prospect's tables come in
        # alphabetical order.
        self.pull_options = sorted({(int(i), t) for t, own in self.owner.items() for i in own.tolist()})
        # The cache starts with only the prospects themselves: the agent knows who is on its list.
        self.revealed = {nt: set() for nt in batch.node_types}             # the cache: fetched n_ids
        self.revealed[entity_table].update(batch[entity_table].n_id[: self.n].tolist())
        # Bookkeeping for the budget.
        self.pulled = set()                                                  # (prospect, table) done
        self.cost = 0
        self.spent = np.zeros(self.n, dtype=int)                             # lookups charged per prospect
        self.fetched = {}                                                    # paid lookups per table


    def _isin(self, n_id, table: str) -> torch.Tensor:
        """True where an n_id is already in the cache for `table`.

        Args:
            n_id:  1-D tensor of database ids
            table: which table's cache to check

        Returns:
            bool tensor, same shape as n_id
        """

        # Turn the cached id set into a tensor on the same device, then test membership element by element.
        known = torch.tensor(sorted(self.revealed[table]), dtype=n_id.dtype, device=n_id.device)

        return torch.isin(n_id, known)

    def legal_pulls(self) -> list:
        """(prospect, table) pairs not pulled yet, in prospect order, each prospect's tables alphabetical."""
        return [a for a in self.pull_options if a not in self.pulled]

    def enrich_candidates(self) -> dict:
        """Entities that a pulled row points to and that aren't fetched yet (list before get).

        Returns {table: (n_id, prospect, mentions)}, one entry per (entity, prospect) pair: `mentions` counts that
        prospect's pulled rows that point to the entity. An entity 3 prospects point to appears 3 times.
        All three are 1-D tensors of the same length.
        """

        # Step 1: walk every hop-2 link and keep the ones that are listed but not yet fetched.
        pairs = {}
        for s_table, s_rows, t_table, t_rows in self.links:
            # Links whose hop-1 row is fetched: the agent can see what that row points to.
            pulled = self._isin(self.batch[t_table].n_id[t_rows], t_table)
            # The database ids of the entities those rows point to, minus the ones already in the cache.
            s_nid = self.batch[s_table].n_id[s_rows[pulled]]
            new = ~self._isin(s_nid, s_table)
            # The prospect each remaining link belongs to.
            owner = self.batch[t_table].batch[t_rows[pulled]][new]
            # Stack into a 2 x E tensor: row 0 = entity id, row 1 = prospect.
            pairs.setdefault(s_table, []).append(torch.stack([s_nid[new], owner]))
        # Step 2: per table, merge the chunks and count repeats. unique(dim=1, return_counts=True) treats each
        # column (entity, prospect) as one item, keeps one copy of each distinct column, and counts how many
        # times it appeared: that count is how many of the prospect's rows mention the entity.
        out = {}
        for table, chunks in pairs.items():
            both = torch.cat(chunks, dim=1)
            if both.size(1) > 0:
                uniq, mentions = both.unique(dim=1, return_counts=True)
                out[table] = (uniq[0], uniq[1], mentions)
        return out


    def _new_ids(self, action) -> tuple:
        """The table and the n_ids an action would add to the cache (empty set = cache hit).

        Does not change the state. Raises ValueError for an enrich that no fetched row points to.
        """
        kind, a, b = action
        # A pull ("pull", prospect, table): that prospect's hop-1 rows in the table, minus what is cached.
        if kind == "pull":
            rows = self.hop1[b][self.owner[b] == a]
            return b, set(self.batch[b].n_id[rows].tolist()) - self.revealed[b]
        # An enrich ("enrich", table, n_id) is legal only if some fetched hop-1 row points to that entity.
        # Look for a link from the entity (hits) whose hop-1 end is in the cache.
        for s_table, s_rows, t_table, t_rows in self.links:                  # enrich: check list before get
            if s_table == a:
                hits = self.batch[s_table].n_id[s_rows] == b
                if hits.any() and self._isin(self.batch[t_table].n_id[t_rows[hits]], t_table).any():
                    return a, {b} - self.revealed[a]
        raise ValueError(f"list before get: no pulled row points to {a} {b}")


    def apply(self, action, prospect=None) -> int:
        """Perform one lookup, charged to `prospect` if given. Returns its cost: 1, or 0 on a cache hit."""
        # A repeated pull is free. A first pull is recorded even if it turns out to reveal nothing new.
        if action[0] == "pull":
            if (action[1], action[2]) in self.pulled:
                return 0
            self.pulled.add((action[1], action[2]))
        # Nothing new to reveal: a cache hit, no charge.
        table, new = self._new_ids(action)
        if not new:
            return 0
        # A paid lookup: add the ids to the cache and charge it to the list, the table and the prospect.
        self.revealed[table] |= new
        self.cost += 1
        self.fetched[table] = self.fetched.get(table, 0) + 1
        if prospect is not None:
            self.spent[prospect] += 1
        return 1

    def masks(self, extra=None) -> dict:
        """{table: bool per sampled row}, True where fetched: the model's `revealed`. `extra` adds one action.

        `extra` is a what-if: the masks show the state as if that action had been taken, without taking it.
        """
        # A sampled row is visible if its database id is in the cache, so every copy of a fetched row is visible.
        out = {nt: self._isin(self.batch[nt].n_id, nt) for nt in self.batch.node_types}
        if extra is not None:
            table, new = self._new_ids(extra)
            if new:
                ids = torch.tensor(sorted(new), device=self.batch[table].n_id.device)
                out[table] |= torch.isin(self.batch[table].n_id, ids)
        return out

    @torch.no_grad()
    def scores(self, model, extra=None) -> torch.Tensor:
        """Probabilities for the N prospects, seeing only fetched rows (drop mode).

        Args:
            model: the frozen, trained GNN (wtl.model.RDLModel)
            extra: optional what-if action, passed on to masks()

        Returns:
            float tensor of shape [N], one probability per prospect
        """

        # eval() puts any layers that act differently during training into inference mode. In drop mode, the
        # model removes every edge that touches an unfetched row, so unfetched rows send no messages.
        model.eval()
        return torch.sigmoid(model(self.batch, self.entity, revealed=self.masks(extra), hide="drop"))

    def metrics(self, model) -> dict:
        """AUROC, AP and P@K of the list as it stands (see wtl.metrics.list_metrics)"""
        return list_metrics(self.y.cpu().numpy(), self.scores(model).cpu().numpy())

def fetch_all(cl: CallList) -> float:
    """Full information: every pull, then every entity the pulled rows list, until nothing is left.

    Returns the share of listed entities that two or more prospects point to (the list's sharing).
    """
    # Step 1: pull every (prospect, table) pair, each charged to its prospect.
    for i, t in cl.legal_pulls():
        cl.apply(("pull", i, t), i)
    # Step 2: measure sharing. Each candidate entry is one (entity, prospect) pair, so counting how often an
    # entity id appears gives the number of prospects that point to it.
    cands = cl.enrich_candidates()
    reach = [ent.unique(return_counts=True)[1] for ent, _, _ in cands.values()]   # prospects per entity
    shared = (torch.cat(reach) >= 2).float().mean().item() if reach else 0.0
    # Step 3: enrich every listed entity once (charged to the list, not a prospect), and repeat until no
    # unfetched entity is left.
    while cands:
        for table, (n_id, _, _) in cands.items():
            for nid in n_id.unique().tolist():
                cl.apply(("enrich", table, nid))
        cands = cl.enrich_candidates()
    return shared