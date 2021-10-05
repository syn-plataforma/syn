from app.utils.parameters.VectorizerStepQueryStringParameters import VectorizerStepQueryStringParameters


class DuplicatesStepQueryStringParameters(VectorizerStepQueryStringParameters):

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
            method='DecisionTreeClassifier',
            method_text='Algoritmo de machine learning',
            method_param='Valores por defecto de la clase',
            method_param_text='Parámetros del algoritmo de machine learning',
            hyperopt=False,
            hyperopt_text='Optimización de hiperparámetros',
            opt_method='N/A',
            opt_method_text='Algoritmo de optimización de hiperparámetros',
            opt_iter='N/A',
            opt_iter_text='Número de iteraciones de la optimización de hiperparámetros',
            opt_param='N/A',
            opt_param_text='Hiperparámetros del algoritmo de machine learning a optimizar',
            topk=5,
            topk_text='Número de incidencias duplicadas recuperadas'
    ):
        super().__init__(lemmatizer, lemmatizer_text, language, language_text, corpus, corpus_text, emb_type,
                         emb_type_text, emb_size, emb_size_text, min_count, min_count_text, cb_size, cb_size_text)
        self.method = method
        self.method_text = method_text
        self.method_param = method_param
        self.method_param_text = method_param_text
        self.hyperopt = hyperopt
        self.hyperopt_text = hyperopt_text
        self.opt_method = opt_method
        self.opt_method_text = opt_method_text
        self.opt_iter = opt_iter
        self.opt_iter_text = opt_iter_text
        self.opt_param = opt_param
        self.opt_param_text = opt_param_text
        self.topk = topk
        self.topk_text = topk_text
