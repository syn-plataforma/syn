class NLPParams:

    def __init__(
            self,
            java_class_name="UpdateMongoDBNLPFields",
            max_num_tokens=150,
            parser_model="corenlp",
            get_tokens=True,
            get_trees=True,
            get_embeddings=True,
            get_coherence=True

    ):
        self.java_class_name = java_class_name
        self.max_num_tokens = max_num_tokens
        self.parser_model = parser_model
        self.get_tokens = get_tokens
        self.get_trees = get_trees
        self.get_embeddings = get_embeddings
        self.get_coherence = get_coherence
