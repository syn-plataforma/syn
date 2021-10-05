from app.utils.parameters.BaseParameters import BaseParameters


class NlpStepQueryStringParameters(BaseParameters):

    def __init__(
            self,
            lemmatizer='Snowball',
            lemmatizer_text='Lematizador',
            language='Inglés',
            language_text='Idioma del Stemmer y de las stop words',
    ):
        self.lemmatizer = lemmatizer
        self.lemmatizer_text = lemmatizer_text
        self.language = language
        self.language_text = language_text
