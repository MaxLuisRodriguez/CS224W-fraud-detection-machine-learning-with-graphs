"""Test embedder used when RelBench text columns are turned into tensors."""
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from torch import Tensor


#
# GloVe text embedding: list of strings -> one 300 number vector per string
#
class GloveTextEmbedding:
    """Callable that maps a batch of strings to 300-d averaged GloVe word vectors (fast, CPU-friendly).
    
    torch_frame calls this object like a function on batches of a text column's values while it "materializes a table into 
    tensors (P5 step 2).
    """

    def __init__(self, device: Optional[str] = None):
        # This particular model has no neural network: it looks up each word's GloVe vector and averages them.
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device
        )

    # __call__ makes an instance callable like a function: embedder(["a", "b"]) runs this method.
    def __call__(self, sentences:List[str]) -> Tensor:
        cleaned = ["" if s is None or (isinstance(s,float) and s!= s) else str(s) for s in sentences]

        # returns a torch Tensor of shape [N, 300] (convert_to_tensor=True; otherwise a NumPy array).
        return self.model.encode(cleaned, convert_to_tensor=True)