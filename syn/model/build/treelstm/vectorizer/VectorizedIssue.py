class VectorizedIssue:

    def __init__(
            self,
            constituency_trees_raw_data=None,
            pretrained_embeddings=None,
            attention_vector_raw_data=None
    ):
        # Inicializa los atributos.
        self.constituency_trees_raw_data = constituency_trees_raw_data
        self.pretrained_embeddings = pretrained_embeddings
        self.attention_vector_raw_data = attention_vector_raw_data
