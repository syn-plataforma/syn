from typing import Any, Union, Iterable

from numpy.core._multiarray_umath import ndarray


class PretrainedWordEmbedding:

    def __init__(
            self,
            size: int = 100,
            reversed_vocabulary: dict = None,
            embeddings: Union[Union[ndarray, Iterable, int, float, tuple, dict], Any] = None
    ):
        # Inicializa los atributos.
        self.size = size
        self.reversed_vocabulary = reversed_vocabulary
        self.embeddings = embeddings
