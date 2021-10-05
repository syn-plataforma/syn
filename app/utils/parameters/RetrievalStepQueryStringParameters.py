from app.utils.parameters.VectorizerStepQueryStringParameters import VectorizerStepQueryStringParameters


class RetrievalStepQueryStringParameters(VectorizerStepQueryStringParameters):

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
            cb_size_text='Tamaño del codebook',
            vec_desc='TF-IDF y valor medio de los word embeddings',
            vec_desc_text='Técnica de vectorización del texto existente en la descripción',
            sim_desc='Similitud del coseno',
            sim_desc_text='Medida de similitud existente entre los textos',
            sim_prod_comp='Similitud de Jaccard',
            sim_prod_comp_text='Medida de similitud existente entre los productos y componenetes',
            topk=5,
            topk_text='Número de incidencias similares recuperadas'
    ):
        super().__init__(lemmatizer, lemmatizer_text, language, language_text, corpus, corpus_text, emb_type,
                         emb_type_text, emb_size, emb_size_text, min_count, min_count_text, cb_size, cb_size_text)
        self.vec_desc = vec_desc
        self.vec_desc_text = vec_desc_text
        self.sim_desc = sim_desc
        self.sim_desc_text = sim_desc_text
        self.sim_prod_comp = sim_prod_comp
        self.sim_prod_comp_text = sim_prod_comp_text
        self.topk = topk
        self.topk_text = topk_text
