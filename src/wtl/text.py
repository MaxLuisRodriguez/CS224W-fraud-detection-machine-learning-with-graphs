"""Text embedder used when RelBench text columns are turned into tensors.

A neural network needs numbers, not strings. wtl.data.build_graph hands this embedder to torch_frame,
which calls it on the values of every text column while building the graph. The resulting vectors are
cached on disk, so the embedding work happens only the first time a dataset is processed (the GloVe
model itself is still loaded on every run, because build_graph always creates this object).
"""
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from torch import Tensor


#
# GloVe text embedding: list of strings -> one 300 number vector per string
#
class GloveTextEmbedding:
    """Callable that maps a batch of strings to 300-d averaged GloVe word vectors (fast, CPU-friendly).

    torch_frame calls this object like a function on batches of a text column's values while it
    "materializes" a table into tensors (P5 step 2; the call happens inside wtl.data.build_graph).
    GloVe is a fixed table of pre-trained word vectors: words used in similar contexts get similar
    vectors. Averaging them gives one rough "meaning" vector per string.

    Args:
        device: where the model runs, e.g. "cpu" or "cuda". None lets sentence_transformers choose.
    """

    def __init__(self, device: Optional[str] = None):
        # This particular model has no neural network: it looks up each word's GloVe vector and averages them.
        # SentenceTransformer(name): downloads the named model the first time, then loads it from cache.
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device
        )

    # __call__ makes an instance callable like a function: embedder(["a", "b"]) runs this method.
    def __call__(self, sentences:List[str]) -> Tensor:
        """Embed a batch of strings.

        Args:
            sentences: one text value per row. Missing values may arrive as None or NaN.

        Returns:
            Tensor of shape [len(sentences), 300].
        """
        # Missing cells come in as None or as float NaN (pandas' "empty" marker). "s != s" is True
        # only for NaN, so both become the empty string; everything else is turned into a string.
        cleaned = ["" if s is None or (isinstance(s,float) and s!= s) else str(s) for s in sentences]

        # returns a torch Tensor of shape [N, 300] (convert_to_tensor=True; otherwise a NumPy array).
        return self.model.encode(cleaned, convert_to_tensor=True)