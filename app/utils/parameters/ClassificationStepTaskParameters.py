from app.utils.parameters.BaseStepTaskParameters import BaseStepTaskParameters
from app.utils.parameters.ModelDumpParameters import ModelDumpParameters
from app.utils.parameters.ClassificationStepQueryStringParameters import ClassificationStepQueryStringParameters


class ClassificationStepTaskParameters(BaseStepTaskParameters):

    def __init__(
            self,
            name='duplicates',
            model_dump=None,
            execution_time_in_minutes=0.0,
            endpoint='duplicates/',
            query_string='',
            query_string_parameters=ClassificationStepQueryStringParameters()
    ):
        if model_dump is None:
            model_dump = [ModelDumpParameters()]
        super().__init__(name, model_dump, execution_time_in_minutes, endpoint, query_string)
        self.query_string_parameters = query_string_parameters

