from app.utils.parameters.NlpStepQueryStringParameters import NlpStepQueryStringParameters


class EmbeddingsStepQueryStringParameters(NlpStepQueryStringParameters):

    def __init__(
            self,
            lemmatizer='Snowball',
            lemmatizer_text='Lematizador',
            language='Inglés',
            language_text='Idioma del Stemmer y de las stop words',
            corpus='Bugzilla',
            corpus_text='Corpus',
            emb_type='Word2vec',
            emb_type_text='Modelo utilizado para entrenar los word embeddings',
            emb_size=100,
            emb_size_text='Dimensión del vector de características',
            min_count=1,
            min_count_text='Frecuencia total mínima de aparición de las palabras',
    ):
        super().__init__(lemmatizer, lemmatizer_text, language, language_text)
        self.corpus = corpus
        self.corpus_text = corpus_text
        self.emb_type = emb_type
        self.emb_type_text = emb_type_text
        self.emb_size = emb_size
        self.emb_size_text = emb_size_text
        self.min_count = min_count
        self.min_count_text = min_count_text
