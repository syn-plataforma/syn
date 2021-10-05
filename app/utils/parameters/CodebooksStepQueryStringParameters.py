from app.utils.parameters.EmbeddingsStepQueryStringParameters import EmbeddingsStepQueryStringParameters


class CodebooksStepQueryStringParameters(EmbeddingsStepQueryStringParameters):

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
            cb_size=50,
            cb_size_text='Tamaño del codebook'
    ):
        super().__init__(lemmatizer, lemmatizer_text, language, language_text, corpus, corpus_text, emb_type,
                         emb_type_text, emb_size, emb_size_text, min_count, min_count_text)
        self.cb_size = cb_size
        self.cb_size_text = cb_size_text
